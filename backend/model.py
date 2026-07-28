

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report, log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler

from features import build_feature_dataframe, validate_feature_dataframe, NON_FEATURE_COLUMNS
from nli_checker import ContradictionChecker

ARTIFACT_DIR = Path("model_artifacts")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        cases = json.load(f)
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def three_way_split(X: pd.DataFrame, y: np.ndarray, case_ids: pd.Series,
                     val_size: float = 0.15, test_size: float = 0.15,
                     random_state: int = 42):
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X, y, case_ids, test_size=(val_size + test_size),
        stratify=y, random_state=random_state,
    )
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp, test_size=relative_test_size,
        stratify=y_temp, random_state=random_state,
    )
    return {
        "train": (X_train, y_train, ids_train),
        "val": (X_val, y_val, ids_val),
        "test": (X_test, y_test, ids_test),
    }


# ---------------------------------------------------------------------------
# Base model training — one function per library, each with early
# stopping on the validation fold.
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
    reg_lambda=1.5, reg_alpha=0.1,
    objective="multi:softprob", eval_metric="mlogloss",
    early_stopping_rounds=25, random_state=42, n_jobs=-1,
)

DEFAULT_LGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_samples=5,
    reg_lambda=1.5, reg_alpha=0.1,
    objective="multiclass", random_state=42, n_jobs=-1, verbosity=-1,
)

DEFAULT_CATBOOST_PARAMS = dict(
    iterations=300, depth=5, learning_rate=0.05,
    l2_leaf_reg=3.0, loss_function="MultiClass",
    random_seed=42, verbose=False, early_stopping_rounds=25,
)


def train_xgboost(X_train, y_train, X_val, y_val, num_class, params: Optional[Dict] = None):
    params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    model = xgb.XGBClassifier(num_class=num_class, **params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, num_class, params: Optional[Dict] = None):
    params = {**DEFAULT_LGB_PARAMS, **(params or {})}
    model = lgb.LGBMClassifier(num_class=num_class, **params)
    try:
        # Newer LightGBM sklearn API
        model.fit(
            X_train, y_train,
            eval_X=X_val, eval_y=y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )
    except TypeError:
        # Older LightGBM sklearn API
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )
    return model


def train_catboost(X_train, y_train, X_val, y_val, num_class, params: Optional[Dict] = None):
    params = {**DEFAULT_CATBOOST_PARAMS, **(params or {})}
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model


def calibrate_model(fitted_model, X_val, y_val):
    """
    Isotonic calibration on top of an already-fitted model, using the
    validation fold. sklearn >=1.6 replaced cv="prefit" with the
    FrozenEstimator wrapper; older sklearn still expects cv="prefit"
    directly — support both.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(fitted_model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def optimize_ensemble_weights(
    proba_by_model: Dict[str, np.ndarray], y_val: np.ndarray, min_weight: float = 0.15
) -> Dict[str, float]:
    """
    Finds weights (summing to 1) that minimize multiclass log loss of
    the weighted-average ensemble on the validation fold, subject to
    each weight being at least `min_weight`.

    The floor matters: on a small validation set (a few dozen rows,
    typical for a hackathon-scale dataset), unconstrained optimization
    reliably collapses to ~100% weight on whichever model got slightly
    lucky on that particular val split — that's overfitting to val
    noise, not a real signal that the other two models are worthless.
    A floor keeps this an actual ensemble while still letting the
    optimizer prefer the stronger validator.

    Falls back to equal weights if scipy isn't available or the
    optimization fails for any reason.
    """
    names = list(proba_by_model.keys())
    n = len(names)
    equal = {name: round(1.0 / n, 4) for name in names}

    if min_weight * n > 1.0:
        min_weight = 1.0 / n  # floor too high to be feasible, relax to equal

    try:
        from scipy.optimize import minimize

        n_classes = proba_by_model[names[0]].shape[1]

        def loss_for_weights(w):
            w = np.clip(w, 0, None)
            w = w / w.sum()
            blended = sum(w[i] * proba_by_model[names[i]] for i in range(n))
            blended = np.clip(blended, 1e-9, 1 - 1e-9)
            return log_loss(y_val, blended, labels=list(range(n_classes)))

        result = minimize(
            loss_for_weights,
            x0=np.array([1.0 / n] * n),
            method="SLSQP",
            bounds=[(min_weight, 1.0) for _ in range(n)],
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 200, "ftol": 1e-6},
        )
        if not result.success:
            print("[warn] weight optimization did not converge, using equal weights")
            return equal

        w = np.clip(result.x, min_weight, None)
        w = w / w.sum()
        return {name: round(float(wi), 4) for name, wi in zip(names, w)}

    except Exception as e:
        print(f"[warn] weight optimization failed ({e}), using equal weights")
        return equal


class EnsembleModel:
    """
    Soft-voting ensemble over calibrated XGBoost / LightGBM / CatBoost
    models. Weights are fixed at construction time (from
    optimize_ensemble_weights). Exposes a sklearn-like predict /
    predict_proba interface so it drops into the same evaluate()
    function as any single model.
    """

    def __init__(self, calibrated_models: Dict[str, object], weights: Dict[str, float], classes_: np.ndarray):
        self.calibrated_models = calibrated_models
        self.weights = weights
        self.classes_ = classes_

    def predict_proba(self, X) -> np.ndarray:
        blended = None
        for name, model in self.calibrated_models.items():
            p = model.predict_proba(X) * self.weights[name]
            blended = p if blended is None else blended + p
        return blended

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# Baseline comparison (kept from the single-model version — cheap,
# useful context alongside the ensemble numbers)
# ---------------------------------------------------------------------------

def train_baselines(X_train, y_train, X_val, y_val) -> Dict[str, float]:
    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_scaled, y_train)
    results["logistic_regression_val_accuracy"] = accuracy_score(y_val, lr.predict(X_val_scaled))

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest_val_accuracy"] = accuracy_score(y_val, rf.predict(X_val))

    return results


# ---------------------------------------------------------------------------
# Evaluation — works identically for the ensemble or any single
# calibrated base model, since both expose predict / predict_proba.
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, label_encoder: LabelEncoder) -> Dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(label_encoder.classes_)), zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    try:
        y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        roc_auc_per_class = roc_auc_score(y_test_bin, y_proba, average=None, multi_class="ovr")
    except ValueError as e:
        roc_auc_macro = None
        roc_auc_per_class = None
        print(f"[warn] ROC-AUC could not be computed: {e}")

    cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))

    per_class = {}
    for i, cls in enumerate(label_encoder.classes_):
        per_class[cls] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
            "roc_auc": round(float(roc_auc_per_class[i]), 4) if roc_auc_per_class is not None else None,
        }

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "roc_auc_macro": round(float(roc_auc_macro), 4) if roc_auc_macro is not None else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_encoder.classes_.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
        ),
    }


def get_ensemble_feature_importance(raw_models: Dict[str, object], feature_names: List[str], top_n: int = 25) -> List[Dict]:
    """
    Averages normalized feature importance across the three raw
    (uncalibrated) base models. Each library reports importance on a
    different scale, so each is normalized to sum to 1 before averaging
    — otherwise whichever library happens to produce larger raw numbers
    would dominate the ranking for no meaningful reason.
    """
    normalized = []
    for name, model in raw_models.items():
        imp = np.asarray(model.feature_importances_, dtype=float)
        total = imp.sum()
        normalized.append(imp / total if total > 0 else imp)

    avg_importance = np.mean(normalized, axis=0)
    order = np.argsort(avg_importance)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": round(float(avg_importance[i]), 5)}
        for i in order
    ]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_artifacts(
    raw_models: Dict[str, object], calibrated_models: Dict[str, object],
    ensemble: EnsembleModel, label_encoder: LabelEncoder,
    feature_names: List[str], ensemble_metrics: Dict,
    per_model_metrics: Dict[str, Dict], baseline_results: Optional[Dict],
    ensemble_weights: Dict[str, float], dataset_path: str, n_cases: int,
    out_dir: Path = ARTIFACT_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in raw_models.items():
        joblib.dump(model, out_dir / f"{name}_raw.joblib")
    for name, model in calibrated_models.items():
        joblib.dump(model, out_dir / f"{name}_calibrated.joblib")
    joblib.dump(ensemble, out_dir / "ensemble_model.joblib")
    joblib.dump(label_encoder, out_dir / "label_encoder.joblib")

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": dataset_path,
        "n_cases": n_cases,
        "n_features": len(feature_names),
        "classes": label_encoder.classes_.tolist(),
        "ensemble_weights": ensemble_weights,
        "ensemble_test_metrics": {k: v for k, v in ensemble_metrics.items() if k != "classification_report"},
        "per_model_test_metrics": {
            name: {k: v for k, v in m.items() if k != "classification_report"}
            for name, m in per_model_metrics.items()
        },
        "baseline_comparison": baseline_results,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(out_dir / "classification_report_ensemble.txt", "w") as f:
        f.write(ensemble_metrics["classification_report"])
    for name, m in per_model_metrics.items():
        with open(out_dir / f"classification_report_{name}.txt", "w") as f:
            f.write(m["classification_report"])

    return out_dir


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> Dict:
    model_names = ["xgboost", "lightgbm", "catboost"]
    return {
        "raw_models": {name: joblib.load(artifact_dir / f"{name}_raw.joblib") for name in model_names},
        "calibrated_models": {name: joblib.load(artifact_dir / f"{name}_calibrated.joblib") for name in model_names},
        "ensemble": joblib.load(artifact_dir / "ensemble_model.joblib"),
        "label_encoder": joblib.load(artifact_dir / "label_encoder.joblib"),
        "feature_names": json.load(open(artifact_dir / "feature_names.json")),
        "metadata": json.load(open(artifact_dir / "metadata.json")),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    dataset_path: str,
    run_baselines: bool = True,
    xgb_params: Optional[Dict] = None,
    lgb_params: Optional[Dict] = None,
    catboost_params: Optional[Dict] = None,
    out_dir: Path = ARTIFACT_DIR,
) -> Dict:
    print(f"Loading dataset from {dataset_path} ...")
    cases = load_dataset(dataset_path)
    print(f"Loaded {len(cases)} cases.")

    print("Initialising NLI checker (loaded once, reused for the whole batch) ...")
    nli_checker = ContradictionChecker()

    print("Batch parsing + batch NLI inference + feature engineering ...")
    df = build_feature_dataframe(cases, nli_checker)
    validate_feature_dataframe(df)
    print(f"Feature matrix: {df.shape[0]} rows x {df.shape[1] - len(NON_FEATURE_COLUMNS)} features.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    feature_names = X.columns.tolist()
    case_ids = df["case_id"]
    num_class = len(label_encoder.classes_)

    print("Splitting train/val/test (stratified) ...")
    splits = three_way_split(X, y, case_ids)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    print("Training base models (XGBoost, LightGBM, CatBoost) ...")
    raw_models = {
        "xgboost": train_xgboost(X_train, y_train, X_val, y_val, num_class, xgb_params),
        "lightgbm": train_lightgbm(X_train, y_train, X_val, y_val, num_class, lgb_params),
        "catboost": train_catboost(X_train, y_train, X_val, y_val, num_class, catboost_params),
    }

    print("Calibrating each base model on the validation fold ...")
    calibrated_models = {
        name: calibrate_model(model, X_val, y_val) for name, model in raw_models.items()
    }

    print("Optimizing ensemble weights on the validation fold ...")
    val_proba_by_model = {name: m.predict_proba(X_val) for name, m in calibrated_models.items()}
    weights = optimize_ensemble_weights(val_proba_by_model, y_val)
    print(f"  Weights: {weights}")

    ensemble = EnsembleModel(calibrated_models, weights, classes_=label_encoder.transform(label_encoder.classes_))

    baseline_results = None
    if run_baselines:
        print("Training baseline models for comparison ...")
        baseline_results = train_baselines(X_train, y_train, X_val, y_val)
        print(f"  Baselines: {baseline_results}")

    print("Evaluating ensemble + each base model on held-out test set ...")
    ensemble_metrics = evaluate(ensemble, X_test, y_test, label_encoder)
    per_model_metrics = {
        name: evaluate(model, X_test, y_test, label_encoder)
        for name, model in calibrated_models.items()
    }
    print(f"  Ensemble   -> Accuracy: {ensemble_metrics['accuracy']}  F1(macro): {ensemble_metrics['f1_macro']}  ROC-AUC(macro): {ensemble_metrics['roc_auc_macro']}")
    for name, m in per_model_metrics.items():
        print(f"  {name:10s} -> Accuracy: {m['accuracy']}  F1(macro): {m['f1_macro']}  ROC-AUC(macro): {m['roc_auc_macro']}")

    importance = get_ensemble_feature_importance(raw_models, feature_names)

    out = save_artifacts(
        raw_models, calibrated_models, ensemble, label_encoder, feature_names,
        ensemble_metrics, per_model_metrics, baseline_results, weights,
        dataset_path, len(cases), out_dir=out_dir,
    )
    with open(out / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    print(f"\nArtifacts saved to: {out.resolve()}")
    return {
        "ensemble_metrics": ensemble_metrics,
        "per_model_metrics": per_model_metrics,
        "ensemble_weights": weights,
        "baseline_results": baseline_results,
        "feature_importance": importance,
        "artifact_dir": str(out),
    }


if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dispute_dataset_300.json"
    results = run_training_pipeline(dataset_path)

    print("\nTop 10 features by averaged importance:")
    for f in results["feature_importance"][:10]:
        print(f"  {f['feature']:35s} {f['importance']}")
