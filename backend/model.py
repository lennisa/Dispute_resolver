# """
# train_model.py — Trains an XGBoost classifier on the feature matrix,
# evaluates performance, and saves the trained model artifact to disk.
# """

# import json
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder

# from features import build_feature_dataframe, get_training_matrix
# from nli_checker import ContradictionChecker

# MODEL_PATH = "dispute_xgboost_model.json"
# ENCODER_PATH = "label_encoder.json"

# def generate_dummy_training_data(n_samples=200):
#     """
#     Generates synthetic training samples if you don't have your full 
#     dataset JSON ready yet, so you can test the training loop end-to-end.
#     """
#     import random
#     reasons = [
#         "delivered_but_disputed", "duplicate_charge", 
#         "subscription_cancellation", "item_not_received"
#     ]
#     labels = ["merchant_wins", "card_member_wins"]
    
#     dataset = []
#     for i in range(n_samples):
#         r = random.choice(reasons)
#         lbl = random.choice(labels)
#         dataset.append({
#             "case_id": f"DC-SYNTH-{i:03d}",
#             "dispute_reason_category": r,
#             "customer_statement": f"I never received order {i} or it was an unauthorized charge.",
#             "merchant_statement": f"Order {i} was successfully delivered with tracking and signature.",
#             "evidence_customer": ["Photo of doorstep"] if random.random() > 0.5 else [],
#             "evidence_merchant": ["Signed delivery confirmation", "Carrier GPS log"],
#             "label": lbl
#         })
#     return dataset

# def train():
#     print("Initializing NLI checker for feature extraction...")
#     checker = ContradictionChecker()
    
#     print("Generating/Loading dataset...")
#     # Replace this with your actual loaded JSON dataset later (e.g. from disputes_dataset.json)
#     raw_data = generate_dummy_training_data(250)
    
#     print("Extracting feature matrix...")
#     df = build_feature_dataframe(raw_data, checker)
    
#     print("Splitting training matrix...")
#     X, y_raw, case_ids = get_training_matrix(df)
    
#     # Encode text labels ("merchant_wins" / "card_member_wins") to integers (0 / 1)
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y_raw)
    
#     # Stratified 80/20 train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
#     # Train XGBoost classifier
#     clf = xgb.XGBClassifier(
#         n_estimators=100,
#         max_depth=4,
#         learning_rate=0.1,
#         random_state=42,
#         eval_metric="logloss"
#     )
    
#     print("Fitting XGBoost model...")
#     clf.fit(X_train, y_train)
    
#     # Evaluate model
#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"\nTest Accuracy: {acc:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
    
#     # Save model artifact and label mappings
#     clf.save_model(MODEL_PATH)
    
#     mapping_info = {
#         "classes": label_encoder.classes_.tolist()
#     }
#     with open(ENCODER_PATH, "w") as f:
#         json.dump(mapping_info, f)
        
#     print(f"\nModel successfully saved to {MODEL_PATH}")
#     print(f"Label classes saved to {ENCODER_PATH}")

# if __name__ == "__main__":
#     train()

































# """
# model.py — Training pipeline for the dispute resolution classifier.

# Dataset -> Batch Parser -> Batch NLI -> Feature Engineering ->
# Feature Matrix -> Train/Val/Test Split -> XGBoost (calibrated) ->
# Evaluation + Feature Importance -> Save Model

# Enhancements over the base pseudocode:
#   - Three-way split (train/val/test), not just train/test. The
#     validation fold is used for early stopping AND for probability
#     calibration — reusing the test fold for both would leak test
#     performance into your reported metrics.
#   - Probability calibration (CalibratedClassifierCV, isotonic) on top
#     of the trained XGBoost. Raw XGBoost softmax outputs are usually
#     not well-calibrated, and "confidence" is a headline number your
#     decision engine and UI both depend on directly.
#   - Optional baseline comparison (Logistic Regression, Random Forest)
#     reported alongside XGBoost — cheap to compute, and a real
#     comparison table is more convincing to judges than a single
#     unverified number.
#   - ROC-AUC computed one-vs-rest per class plus macro-averaged, since
#     plain "ROC-AUC" isn't well-defined for >2 classes without saying
#     which averaging scheme you mean.
#   - Everything needed to reproduce or serve the model — the model
#     itself, the label encoder, the exact feature column order, and a
#     metadata JSON with metrics/hyperparams/dataset info — is saved
#     together, since a model file alone isn't enough to serve safely.
# """

# import json
# import time
# from pathlib import Path
# from typing import Dict, List, Optional

# import joblib
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score, precision_recall_fscore_support,
#     roc_auc_score, confusion_matrix, classification_report,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, label_binarize

# from features import build_feature_dataframe, validate_feature_dataframe, NON_FEATURE_COLUMNS
# from nli_checker import ContradictionChecker

# ARTIFACT_DIR = Path("model_artifacts")


# # ---------------------------------------------------------------------------
# # Data loading
# # ---------------------------------------------------------------------------

# def load_dataset(path: str) -> List[Dict]:
#     with open(path) as f:
#         cases = json.load(f)
#     if not cases:
#         raise ValueError(f"No cases found in {path}")
#     return cases


# # ---------------------------------------------------------------------------
# # Splitting
# # ---------------------------------------------------------------------------

# def three_way_split(X: pd.DataFrame, y: np.ndarray, case_ids: pd.Series,
#                      val_size: float = 0.15, test_size: float = 0.15,
#                      random_state: int = 42):
#     """
#     Train / val / test, stratified by label at each split.
#     val is used for early stopping + calibration.
#     test is held out entirely until final evaluation.
#     """
#     X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
#         X, y, case_ids, test_size=(val_size + test_size),
#         stratify=y, random_state=random_state,
#     )
#     relative_test_size = test_size / (val_size + test_size)
#     X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
#         X_temp, y_temp, ids_temp, test_size=relative_test_size,
#         stratify=y_temp, random_state=random_state,
#     )
#     return {
#         "train": (X_train, y_train, ids_train),
#         "val": (X_val, y_val, ids_val),
#         "test": (X_test, y_test, ids_test),
#     }


# # ---------------------------------------------------------------------------
# # Training
# # ---------------------------------------------------------------------------

# DEFAULT_XGB_PARAMS = dict(
#     n_estimators=300,
#     max_depth=5,
#     learning_rate=0.05,
#     subsample=0.85,
#     colsample_bytree=0.85,
#     min_child_weight=2,
#     reg_lambda=1.5,
#     reg_alpha=0.1,
#     objective="multi:softprob",
#     eval_metric="mlogloss",
#     early_stopping_rounds=25,
#     random_state=42,
#     n_jobs=-1,
# )


# def train_xgboost(X_train, y_train, X_val, y_val, params: Optional[Dict] = None):
#     params = {**DEFAULT_XGB_PARAMS, **(params or {})}
#     num_class = len(set(y_train) | set(y_val))
#     model = xgb.XGBClassifier(num_class=num_class, **params)
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=False,
#     )
#     return model


# def calibrate_model(fitted_model, X_val, y_val):
#     """
#     Wraps the already-fitted XGBoost model with isotonic calibration on
#     the validation fold — we don't refit the base model, we only learn
#     a mapping from raw scores to calibrated probabilities.

#     sklearn >=1.6 replaced CalibratedClassifierCV(cv="prefit") with the
#     FrozenEstimator wrapper; older sklearn still expects cv="prefit"
#     directly. Support both so this doesn't break on either version.
#     """
#     try:
#         from sklearn.frozen import FrozenEstimator
#         calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method="isotonic")
#     except ImportError:
#         calibrated = CalibratedClassifierCV(fitted_model, method="isotonic", cv="prefit")
#     calibrated.fit(X_val, y_val)
#     return calibrated


# def train_baselines(X_train, y_train, X_val, y_val) -> Dict[str, float]:
#     """Optional rigor pass: quick baseline comparison. Cheap, and gives
#     you a real "why XGBoost" answer if a judge asks."""
#     results = {}

#     # LR needs scaling to converge cleanly; XGBoost and RF don't.
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)

#     lr = LogisticRegression(max_iter=2000)
#     lr.fit(X_train_scaled, y_train)
#     results["logistic_regression_val_accuracy"] = accuracy_score(y_val, lr.predict(X_val_scaled))

#     rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     results["random_forest_val_accuracy"] = accuracy_score(y_val, rf.predict(X_val))

#     return results


# # ---------------------------------------------------------------------------
# # Evaluation
# # ---------------------------------------------------------------------------

# def evaluate(model, X_test, y_test, label_encoder: LabelEncoder) -> Dict:
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision, recall, f1, support = precision_recall_fscore_support(
#         y_test, y_pred, average=None, labels=range(len(label_encoder.classes_)), zero_division=0
#     )
#     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
#         y_test, y_pred, average="macro", zero_division=0
#     )

#     try:
#         y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
#         roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
#         roc_auc_per_class = roc_auc_score(y_test_bin, y_proba, average=None, multi_class="ovr")
#     except ValueError as e:
#         # Can happen if a class is entirely absent from the test fold
#         # on a small dataset — report it rather than crashing training.
#         roc_auc_macro = None
#         roc_auc_per_class = None
#         print(f"[warn] ROC-AUC could not be computed: {e}")

#     cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))

#     per_class = {}
#     for i, cls in enumerate(label_encoder.classes_):
#         per_class[cls] = {
#             "precision": round(float(precision[i]), 4),
#             "recall": round(float(recall[i]), 4),
#             "f1": round(float(f1[i]), 4),
#             "support": int(support[i]),
#             "roc_auc": round(float(roc_auc_per_class[i]), 4) if roc_auc_per_class is not None else None,
#         }

#     return {
#         "accuracy": round(float(accuracy), 4),
#         "precision_macro": round(float(precision_macro), 4),
#         "recall_macro": round(float(recall_macro), 4),
#         "f1_macro": round(float(f1_macro), 4),
#         "roc_auc_macro": round(float(roc_auc_macro), 4) if roc_auc_macro is not None else None,
#         "per_class": per_class,
#         "confusion_matrix": cm.tolist(),
#         "confusion_matrix_labels": label_encoder.classes_.tolist(),
#         "classification_report": classification_report(
#             y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
#         ),
#     }


# def get_feature_importance(model, feature_names: List[str], top_n: int = 25) -> List[Dict]:
#     importances = model.feature_importances_
#     order = np.argsort(importances)[::-1][:top_n]
#     return [
#         {"feature": feature_names[i], "importance": round(float(importances[i]), 5)}
#         for i in order
#     ]


# # ---------------------------------------------------------------------------
# # Save / load
# # ---------------------------------------------------------------------------

# def save_artifacts(
#     model, calibrated_model, label_encoder: LabelEncoder,
#     feature_names: List[str], metrics: Dict, baseline_results: Optional[Dict],
#     params: Dict, dataset_path: str, n_cases: int,
#     out_dir: Path = ARTIFACT_DIR,
# ) -> Path:
#     out_dir.mkdir(parents=True, exist_ok=True)

#     joblib.dump(model, out_dir / "xgboost_model.joblib")
#     joblib.dump(calibrated_model, out_dir / "xgboost_model_calibrated.joblib")
#     joblib.dump(label_encoder, out_dir / "label_encoder.joblib")

#     with open(out_dir / "feature_names.json", "w") as f:
#         json.dump(feature_names, f, indent=2)

#     metadata = {
#         "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "dataset_path": dataset_path,
#         "n_cases": n_cases,
#         "n_features": len(feature_names),
#         "classes": label_encoder.classes_.tolist(),
#         "xgb_params": {k: v for k, v in params.items() if k != "early_stopping_rounds"},
#         "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
#         "baseline_comparison": baseline_results,
#     }
#     with open(out_dir / "metadata.json", "w") as f:
#         json.dump(metadata, f, indent=2, default=str)

#     with open(out_dir / "classification_report.txt", "w") as f:
#         f.write(metrics["classification_report"])

#     return out_dir


# def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> Dict:
#     return {
#         "model": joblib.load(artifact_dir / "xgboost_model.joblib"),
#         "calibrated_model": joblib.load(artifact_dir / "xgboost_model_calibrated.joblib"),
#         "label_encoder": joblib.load(artifact_dir / "label_encoder.joblib"),
#         "feature_names": json.load(open(artifact_dir / "feature_names.json")),
#         "metadata": json.load(open(artifact_dir / "metadata.json")),
#     }


# # ---------------------------------------------------------------------------
# # Full pipeline
# # ---------------------------------------------------------------------------

# def run_training_pipeline(
#     dataset_path: str,
#     run_baselines: bool = True,
#     xgb_params: Optional[Dict] = None,
#     out_dir: Path = ARTIFACT_DIR,
# ) -> Dict:
#     print(f"Loading dataset from {dataset_path} ...")
#     cases = load_dataset(dataset_path)
#     print(f"Loaded {len(cases)} cases.")

#     print("Initialising NLI checker (loaded once, reused for the whole batch) ...")
#     nli_checker = ContradictionChecker()

#     print("Batch parsing + batch NLI inference + feature engineering ...")
#     df = build_feature_dataframe(cases, nli_checker)
#     validate_feature_dataframe(df)
#     print(f"Feature matrix: {df.shape[0]} rows x {df.shape[1] - len(NON_FEATURE_COLUMNS)} features.")

#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["label"])
#     drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
#     X = df.drop(columns=drop_cols)
#     feature_names = X.columns.tolist()
#     case_ids = df["case_id"]

#     print("Splitting train/val/test (stratified) ...")
#     splits = three_way_split(X, y, case_ids)
#     X_train, y_train, _ = splits["train"]
#     X_val, y_val, _ = splits["val"]
#     X_test, y_test, _ = splits["test"]
#     print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

#     print("Training XGBoost ...")
#     model = train_xgboost(X_train, y_train, X_val, y_val, params=xgb_params)

#     print("Calibrating probabilities on the validation fold ...")
#     calibrated_model = calibrate_model(model, X_val, y_val)

#     baseline_results = None
#     if run_baselines:
#         print("Training baseline models for comparison ...")
#         baseline_results = train_baselines(X_train, y_train, X_val, y_val)
#         print(f"  Baselines: {baseline_results}")

#     print("Evaluating on held-out test set ...")
#     metrics = evaluate(calibrated_model, X_test, y_test, label_encoder)
#     print(f"  Accuracy: {metrics['accuracy']}  F1(macro): {metrics['f1_macro']}  ROC-AUC(macro): {metrics['roc_auc_macro']}")

#     importance = get_feature_importance(model, feature_names)

#     out = save_artifacts(
#         model, calibrated_model, label_encoder, feature_names, metrics,
#         baseline_results, {**DEFAULT_XGB_PARAMS, **(xgb_params or {})},
#         dataset_path, len(cases), out_dir=out_dir,
#     )
#     with open(out / "feature_importance.json", "w") as f:
#         json.dump(importance, f, indent=2)

#     print(f"\nArtifacts saved to: {out.resolve()}")
#     return {
#         "metrics": metrics,
#         "baseline_results": baseline_results,
#         "feature_importance": importance,
#         "artifact_dir": str(out),
#     }


# if __name__ == "__main__":
#     import sys

#     dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dispute_dataset_300.json"
#     results = run_training_pipeline(dataset_path)

#     print("\nTop 10 features by importance:")
#     for f in results["feature_importance"][:10]:
#         print(f"  {f['feature']:35s} {f['importance']}")











"""
model.py — Training pipeline for the dispute resolution classifier.

Dataset -> Batch Parser -> Batch NLI -> Feature Engineering ->
Feature Matrix -> Train/Val/Test Split ->
XGBoost + LightGBM + CatBoost, each calibrated individually ->
Weight-optimized soft-voting ensemble ->
Evaluation (ensemble + each base model) + Feature Importance -> Save.

Why an ensemble instead of a single model:
  XGBoost, LightGBM, and CatBoost differ in how they grow trees
  (level-wise vs leaf-wise), how they handle categorical-looking
  integer features, and their regularization defaults. On a small
  tabular dataset like this one, each tends to make somewhat different
  mistakes, so averaging their predictions usually generalizes better
  than any single model — and gives you a genuine "why an ensemble"
  answer if a judge asks, backed by a real comparison table.

Enhancements over a single-model pipeline:
  - Each base model is trained with its own early stopping on the
    validation fold, then calibrated INDIVIDUALLY (CalibratedClassifierCV,
    isotonic) before being combined — calibrating after ensembling would
    require a custom multiclass calibration routine; calibrating each
    base learner first and then averaging calibrated probabilities is
    the standard, well-behaved way to do this.
  - Ensemble weights are NOT just 1/3 each — they're optimized on the
    validation fold by minimizing multiclass log loss (scipy.optimize,
    falls back to equal weights if scipy isn't available). A model that
    validates worse gets down-weighted automatically.
  - Final evaluation reports the ensemble AND each individual calibrated
    base model on the held-out test set side by side, so you can show
    concretely that the ensemble does at least as well as any single
    model — don't just assert it, the metadata.json has the numbers.
  - Three-way train/val/test split, as before: val is used for early
    stopping + calibration + weight optimization, test is touched only
    once, at the very end.
"""

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
