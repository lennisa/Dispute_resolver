# # """
# # explainability.py — Computes SHAP values for trained XGBoost dispute model
# # and translates top feature attributions into plain English explanations.
# # """

# # import shap
# # import xgboost as xgb
# # import pandas as pd
# # from features import build_feature_dataframe, get_training_matrix
# # from nli_checker import ContradictionChecker

# # class DisputeExplainer:
# #     def __init__(self, model_path="dispute_xgboost_model.json"):
# #         self.model = xgb.XGBClassifier()
# #         self.model.load_model(model_path)
# #         self.explainer = shap.TreeExplainer(self.model)

# #     def explain_case(self, case_data: dict, checker: ContradictionChecker, feature_columns: list):
# #         """
# #         Computes SHAP values for a single case and extracts top 3 factors 
# #         pushing for/against the outcome.
# #         """
# #         # Build features for this single case
# #         df = build_feature_dataframe([case_data], checker)
# #         X, _, _ = get_training_matrix(df)
        
# #         # Ensure column alignment with training matrix
# #         X = X.reindex(columns=feature_columns, fill_value=0)
        
# #         shap_values = self.explainer(X)
        
# #         # For binary classification, shap_values.values is shape (n_samples, n_features) or (n_samples, n_features, n_classes)
# #         vals = shap_values.values[0]
# #         if len(vals.shape) > 1:
# #             vals = vals[:, 1] # Take positive class attribution
            
# #         feature_names = X.columns.tolist()
        
# #         # Zip feature names with their SHAP attribution values
# #         attributions = sorted(zip(feature_names, vals, X.iloc[0]), key=lambda x: abs(x[1]), reverse=True)
        
# #         top_reasons = []
# #         for feat, val, raw_val in attributions[:5]:
# #             direction = "increases likelihood of merchant win" if val > 0 else "increases likelihood of card member win"
# #             top_reasons.append({
# #                 "feature": feat,
# #                 "shap_value": round(float(val), 4),
# #                 "feature_value": float(raw_val),
# #                 "impact": direction
# #             })
            
# #         return top_reasons

# # if __name__ == "__main__":
# #     checker = ContradictionChecker()
# #     explainer = DisputeExplainer()
    
# #     # Test sample case
# #     sample_case = {
# #         "case_id": "DC-SHAP-001",
# #         "dispute_reason_category": "delivered_but_disputed",
# #         "customer_statement": "I never received my package, tracking TRK999888777.",
# #         "merchant_statement": "Delivered with signature confirmation.",
# #         "evidence_customer": [],
# #         "evidence_merchant": ["Signed delivery confirmation"],
# #     }
    
# #     # We load feature columns from training setup
# #     # For quick testing, we can extract using feature builder
# #     df_dummy = build_feature_dataframe([sample_case], checker)
# #     X_dummy, _, _ = get_training_matrix(df_dummy)
    
# #     reasons = explainer.explain_case(sample_case, checker, X_dummy.columns.tolist())
# #     import json
# #     print(json.dumps(reasons, indent=2))



























# """
# explainability.py — SHAP explainability pipeline for a single dispute.

# Load trained model -> Load feature names -> Create SHAP TreeExplainer ->
# Extract features for one dispute -> Align columns with training ->
# Predict class + probability -> Compute SHAP values ->
# Sort by absolute contribution -> Select Top-K ->
# Translate into human-readable explanations ->
# Return prediction, confidence, positive reasons, negative reasons.

# Enhancements over the base pseudocode:
#   - Explains relative to the CALIBRATED model's probabilities (so the
#     confidence number shown here matches the confidence used by the
#     decision engine and the UI — explaining an uncalibrated score
#     while showing a calibrated confidence elsewhere would be
#     inconsistent), while computing SHAP values from the underlying
#     XGBoost model (SHAP's TreeExplainer needs a tree model, not the
#     CalibratedClassifierCV wrapper).
#   - Dynamic, value-aware explanations instead of static canned
#     sentences. "merchant_evidence_count: The merchant supplied
#     multiple supporting documents" is fine, but "The merchant supplied
#     3 supporting documents versus the customer's 1" is more concrete
#     and auditable — the pseudocode's examples are used as the
#     fallback template when a feature doesn't have a dynamic template.
#   - explain_many() for batch explanation (fairness dashboard, bulk
#     audit export) instead of only single-case explanation.
#   - Column alignment is defensive: any feature present at inference
#     but missing at training (or vice versa) is logged, not silently
#     dropped — a silent mismatch here is the kind of bug that only
#     shows up as an unexplained accuracy drop days later.
# """

# import json
# from pathlib import Path
# from typing import Dict, List, Optional

# import numpy as np
# import pandas as pd
# import shap

# from features import extract_features_single, NON_FEATURE_COLUMNS
# from model import ARTIFACT_DIR, load_artifacts


# # ---------------------------------------------------------------------------
# # Human-readable explanation templates
# # ---------------------------------------------------------------------------
# # Each entry is a function(feature_name, value, case_row) -> str, so the
# # sentence can reference the actual value rather than being generic.
# # Falls back to a static template (matching the pseudocode's examples)
# # for any feature not covered here.

# def _fmt_amount(v):
#     try:
#         return f"${float(v):.2f}"
#     except (TypeError, ValueError):
#         return str(v)


# DYNAMIC_TEMPLATES = {
#     "tracking_match": lambda v, row: (
#         "The customer and merchant reference the same tracking number."
#         if v else "The customer and merchant do not reference a matching tracking number."
#     ),
#     "contradiction_score": lambda v, row: (
#         f"The customer and merchant narratives strongly contradict each other "
#         f"(contradiction score {v:.2f})." if v >= 0.6 else
#         f"The customer and merchant narratives are broadly consistent (contradiction score {v:.2f})."
#     ),
#     "merchant_evidence_count": lambda v, row: (
#         f"The merchant supplied {int(v)} supporting document{'s' if v != 1 else ''}"
#         + (f", versus {int(row.get('customer_evidence_count', 0))} from the customer." if row is not None else ".")
#     ),
#     "customer_evidence_count": lambda v, row: (
#         f"The customer supplied {int(v)} supporting document{'s' if v != 1 else ''}"
#         + (f", versus {int(row.get('merchant_evidence_count', 0))} from the merchant." if row is not None else ".")
#     ),
#     "amount_difference": lambda v, row: (
#         "The customer and merchant reported the same payment amount."
#         if v < 0.01 else f"The customer and merchant reported payment amounts differing by {_fmt_amount(v)}."
#     ),
#     "customer_evidence_reliability": lambda v, row: (
#         f"The customer's evidence has an average reliability score of {v:.2f}."
#     ),
#     "merchant_evidence_reliability": lambda v, row: (
#         f"The merchant's evidence has an average reliability score of {v:.2f}."
#     ),
#     "evidence_reliability_gap": lambda v, row: (
#         "The customer's evidence is substantially more reliable than the merchant's."
#         if v > 0.15 else
#         "The merchant's evidence is substantially more reliable than the customer's."
#         if v < -0.15 else
#         "Both sides' evidence is of broadly similar reliability."
#     ),
#     "either_side_missing_evidence": lambda v, row: (
#         "At least one side provided no supporting evidence at all."
#         if v else "Both sides provided at least some supporting evidence."
#     ),
#     "organisation_overlap": lambda v, row: (
#         "The customer and merchant reference the same organisation(s)."
#         if v > 0 else "The customer and merchant do not reference the same organisation(s)."
#     ),
#     "customer_negations": lambda v, row: (
#         f"The customer's statement contains {int(v)} negation word{'s' if v != 1 else ''} "
#         f"(e.g. 'never', 'not'), often indicating a denial of receipt or service."
#     ),
#     "nli_confidence": lambda v, row: (
#         f"The contradiction-detection model is {v:.0%} confident in its assessment of the two statements."
#     ),
#     "reason_category": lambda v, row: (
#         "The dispute reason category is a strong prior for this type of outcome."
#     ),
# }

# # Static fallback templates — directly from the provided pseudocode,
# # used when a feature isn't covered by a dynamic template above.
# STATIC_FALLBACK_TEMPLATES = {
#     "tracking_match": "The customer and merchant reference the same tracking number.",
#     "contradiction_score": "The customer and merchant narratives strongly contradict each other.",
#     "merchant_evidence_count": "The merchant supplied multiple supporting documents.",
#     "date_overlap": "The reported dates differ.",
#     "amount_difference": "The customer and merchant reported different payment amounts.",
# }


# def humanize_feature(feature_name: str, value, row: Optional[pd.Series] = None) -> str:
#     if feature_name in DYNAMIC_TEMPLATES:
#         try:
#             return DYNAMIC_TEMPLATES[feature_name](value, row)
#         except Exception:
#             pass  # fall through to generic below on any formatting edge case
#     if feature_name in STATIC_FALLBACK_TEMPLATES:
#         return STATIC_FALLBACK_TEMPLATES[feature_name]
#     # Generic fallback: turn snake_case into a plain phrase with the value.
#     plain = feature_name.replace("_", " ")
#     return f"{plain.capitalize()} = {value}."


# # ---------------------------------------------------------------------------
# # Explainer
# # ---------------------------------------------------------------------------

# class DisputeExplainer:
#     def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
#         artifacts = load_artifacts(artifact_dir)
#         self.model = artifacts["model"]                       # raw XGBoost — SHAP needs the tree model
#         self.calibrated_model = artifacts["calibrated_model"]  # used for the reported confidence/prediction
#         self.label_encoder = artifacts["label_encoder"]
#         self.feature_names = artifacts["feature_names"]
#         self.metadata = artifacts["metadata"]

#         self.explainer = shap.TreeExplainer(self.model)

#     def _align_columns(self, feature_dict: Dict) -> pd.DataFrame:
#         """Build a single-row DataFrame matching the exact training column
#         order. Logs (doesn't silently swallow) any mismatch."""
#         row = {}
#         missing = []
#         for col in self.feature_names:
#             if col in feature_dict:
#                 row[col] = feature_dict[col]
#             else:
#                 row[col] = 0
#                 missing.append(col)

#         extra = [
#             k for k in feature_dict
#             if k not in self.feature_names and k not in NON_FEATURE_COLUMNS
#         ]

#         if missing:
#             print(f"[explainability warn] {len(missing)} training features missing at inference, filled with 0: {missing}")
#         if extra:
#             print(f"[explainability warn] {len(extra)} inference features not seen at training, dropped: {extra}")

#         return pd.DataFrame([row], columns=self.feature_names)

#     def explain_case(self, case: Dict, nli_checker, top_k: int = 5) -> Dict:
#         feature_dict = extract_features_single(case, nli_checker)
#         X_row = self._align_columns(feature_dict)

#         proba = self.calibrated_model.predict_proba(X_row)[0]
#         predicted_idx = int(np.argmax(proba))
#         predicted_label = self.label_encoder.classes_[predicted_idx]
#         confidence = float(proba[predicted_idx])

#         # SHAP values from the underlying (uncalibrated) tree model.
#         # For multiclass XGBoost, shap_values has shape
#         # (n_samples, n_features, n_classes) in recent shap versions.
#         shap_values = self.explainer.shap_values(X_row)
#         shap_row = self._extract_class_shap_row(shap_values, predicted_idx)

#         contributions = list(zip(self.feature_names, shap_row, X_row.iloc[0].tolist()))
#         contributions.sort(key=lambda t: abs(t[1]), reverse=True)

#         positive = [(f, v, val) for f, v, val in contributions if v > 0][:top_k]
#         negative = [(f, v, val) for f, v, val in contributions if v < 0][:top_k]

#         row_series = X_row.iloc[0]
#         top_positive_reasons = [
#             {"feature": f, "shap_value": round(float(v), 4), "value": val,
#              "explanation": humanize_feature(f, val, row_series)}
#             for f, v, val in positive
#         ]
#         top_negative_reasons = [
#             {"feature": f, "shap_value": round(float(v), 4), "value": val,
#              "explanation": humanize_feature(f, val, row_series)}
#             for f, v, val in negative
#         ]

#         return {
#             "case_id": case.get("case_id", ""),
#             "prediction": predicted_label,
#             "confidence": round(confidence, 4),
#             "class_probabilities": {
#                 cls: round(float(p), 4) for cls, p in zip(self.label_encoder.classes_, proba)
#             },
#             "top_positive_reasons": top_positive_reasons,
#             "top_negative_reasons": top_negative_reasons,
#         }

#     def explain_many(self, cases: List[Dict], nli_checker, top_k: int = 5) -> List[Dict]:
#         """Batch version — still calls extract_features_single per case
#         internally (SHAP explanation is inherently per-instance), but
#         avoids re-creating the explainer or re-loading artifacts."""
#         return [self.explain_case(c, nli_checker, top_k=top_k) for c in cases]

#     @staticmethod
#     def _extract_class_shap_row(shap_values, class_idx: int) -> np.ndarray:
#         """Normalizes across the shap-version differences in how
#         multiclass TreeExplainer output is shaped."""
#         if isinstance(shap_values, list):
#             # Older shap: list of (n_samples, n_features) arrays, one per class
#             return np.asarray(shap_values[class_idx])[0]
#         arr = np.asarray(shap_values)
#         if arr.ndim == 3:
#             # (n_samples, n_features, n_classes)
#             return arr[0, :, class_idx]
#         # Binary / already single-class: (n_samples, n_features)
#         return arr[0]


# # ---------------------------------------------------------------------------
# # Simple JSON-shaped output matching the pseudocode's recommended format
# # ---------------------------------------------------------------------------

# def to_simple_json(explanation: Dict) -> Dict:
#     """Collapses the richer explain_case() output down to exactly the
#     shape shown in the pseudocode, for callers that just want the
#     compact version."""
#     return {
#         "prediction": explanation["prediction"],
#         "confidence": explanation["confidence"],
#         "top_positive_reasons": [r["feature"] for r in explanation["top_positive_reasons"]],
#         "top_negative_reasons": [r["feature"] for r in explanation["top_negative_reasons"]],
#     }


# if __name__ == "__main__":
#     from nli_checker import ContradictionChecker

#     explainer = DisputeExplainer()
#     checker = ContradictionChecker()

#     test_case = {
#         "case_id": "DC-DEMO-001",
#         "dispute_reason_category": "delivered_but_disputed",
#         "customer_statement": "I never received my package. No one was home and nothing was left at the door.",
#         "merchant_statement": "Delivered with signature confirmation on July 10th at 2:14 PM.",
#         "evidence_customer": ["Photo of empty doorstep"],
#         "evidence_merchant": ["Signed delivery confirmation", "Carrier GPS delivery log"],
#     }

#     full = explainer.explain_case(test_case, checker)
#     print(json.dumps(full, indent=2))

#     print("\n--- Compact form ---")
#     print(json.dumps(to_simple_json(full), indent=2))



"""
explainability.py — SHAP explainability pipeline for a single dispute.

Load trained model -> Load feature names -> Create SHAP TreeExplainer ->
Extract features for one dispute -> Align columns with training ->
Predict class + probability -> Compute SHAP values ->
Sort by absolute contribution -> Select Top-K ->
Translate into human-readable explanations ->
Return prediction, confidence, positive reasons, negative reasons.

Enhancements over the base pseudocode:
  - Explains relative to the CALIBRATED model's probabilities (so the
    confidence number shown here matches the confidence used by the
    decision engine and the UI — explaining an uncalibrated score
    while showing a calibrated confidence elsewhere would be
    inconsistent), while computing SHAP values from the underlying
    XGBoost model (SHAP's TreeExplainer needs a tree model, not the
    CalibratedClassifierCV wrapper).
  - Dynamic, value-aware explanations instead of static canned
    sentences. "merchant_evidence_count: The merchant supplied
    multiple supporting documents" is fine, but "The merchant supplied
    3 supporting documents versus the customer's 1" is more concrete
    and auditable — the pseudocode's examples are used as the
    fallback template when a feature doesn't have a dynamic template.
  - explain_many() for batch explanation (fairness dashboard, bulk
    audit export) instead of only single-case explanation.
  - Column alignment is defensive: any feature present at inference
    but missing at training (or vice versa) is logged, not silently
    dropped — a silent mismatch here is the kind of bug that only
    shows up as an unexplained accuracy drop days later.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from features import extract_features_single, NON_FEATURE_COLUMNS
from model import ARTIFACT_DIR, load_artifacts,EnsembleModel


# ---------------------------------------------------------------------------
# Human-readable explanation templates
# ---------------------------------------------------------------------------
# Each entry is a function(feature_name, value, case_row) -> str, so the
# sentence can reference the actual value rather than being generic.
# Falls back to a static template (matching the pseudocode's examples)
# for any feature not covered here.

def _fmt_amount(v):
    try:
        return f"${float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


DYNAMIC_TEMPLATES = {
    "tracking_match": lambda v, row: (
        "The customer and merchant reference the same tracking number."
        if v else "The customer and merchant do not reference a matching tracking number."
    ),
    "contradiction_score": lambda v, row: (
        f"The customer and merchant narratives strongly contradict each other "
        f"(contradiction score {v:.2f})." if v >= 0.6 else
        f"The customer and merchant narratives are broadly consistent (contradiction score {v:.2f})."
    ),
    "merchant_evidence_count": lambda v, row: (
        f"The merchant supplied {int(v)} supporting document{'s' if v != 1 else ''}"
        + (f", versus {int(row.get('customer_evidence_count', 0))} from the customer." if row is not None else ".")
    ),
    "customer_evidence_count": lambda v, row: (
        f"The customer supplied {int(v)} supporting document{'s' if v != 1 else ''}"
        + (f", versus {int(row.get('merchant_evidence_count', 0))} from the merchant." if row is not None else ".")
    ),
    "amount_difference": lambda v, row: (
        "The customer and merchant reported the same payment amount."
        if v < 0.01 else f"The customer and merchant reported payment amounts differing by {_fmt_amount(v)}."
    ),
    "customer_evidence_reliability": lambda v, row: (
        f"The customer's evidence has an average reliability score of {v:.2f}."
    ),
    "merchant_evidence_reliability": lambda v, row: (
        f"The merchant's evidence has an average reliability score of {v:.2f}."
    ),
    "evidence_reliability_gap": lambda v, row: (
        "The customer's evidence is substantially more reliable than the merchant's."
        if v > 0.15 else
        "The merchant's evidence is substantially more reliable than the customer's."
        if v < -0.15 else
        "Both sides' evidence is of broadly similar reliability."
    ),
    "either_side_missing_evidence": lambda v, row: (
        "At least one side provided no supporting evidence at all."
        if v else "Both sides provided at least some supporting evidence."
    ),
    "organisation_overlap": lambda v, row: (
        "The customer and merchant reference the same organisation(s)."
        if v > 0 else "The customer and merchant do not reference the same organisation(s)."
    ),
    "customer_negations": lambda v, row: (
        f"The customer's statement contains {int(v)} negation word{'s' if v != 1 else ''} "
        f"(e.g. 'never', 'not'), often indicating a denial of receipt or service."
    ),
    "nli_confidence": lambda v, row: (
        f"The contradiction-detection model is {v:.0%} confident in its assessment of the two statements."
    ),
    "reason_category": lambda v, row: (
        "The dispute reason category is a strong prior for this type of outcome."
    ),
}

# Static fallback templates — directly from the provided pseudocode,
# used when a feature isn't covered by a dynamic template above.
STATIC_FALLBACK_TEMPLATES = {
    "tracking_match": "The customer and merchant reference the same tracking number.",
    "contradiction_score": "The customer and merchant narratives strongly contradict each other.",
    "merchant_evidence_count": "The merchant supplied multiple supporting documents.",
    "date_overlap": "The reported dates differ.",
    "amount_difference": "The customer and merchant reported different payment amounts.",
}


def humanize_feature(feature_name: str, value, row: Optional[pd.Series] = None) -> str:
    if feature_name in DYNAMIC_TEMPLATES:
        try:
            return DYNAMIC_TEMPLATES[feature_name](value, row)
        except Exception:
            pass  # fall through to generic below on any formatting edge case
    if feature_name in STATIC_FALLBACK_TEMPLATES:
        return STATIC_FALLBACK_TEMPLATES[feature_name]
    # Generic fallback: turn snake_case into a plain phrase with the value.
    plain = feature_name.replace("_", " ")
    return f"{plain.capitalize()} = {value}."


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class DisputeExplainer:
    """
    Explains predictions from the XGBoost + LightGBM + CatBoost ensemble.

    SHAP TreeExplainer supports all three libraries individually. To
    explain the ENSEMBLE's decision (not just one base model's), this
    computes SHAP values from each raw base model separately and
    combines them using the same weights the ensemble uses for
    prediction — so if CatBoost is weighted 0.5 and XGBoost 0.2, its
    attributions count for proportionally more in the explanation too.

    Caveat worth knowing: each library's TreeExplainer output is on
    that library's own raw margin/log-odds scale, which aren't
    perfectly identical across libraries. Weighted-averaging them is a
    standard, widely-used practical approximation for ensemble
    explainability, not an exact decomposition of the calibrated
    ensemble probability. It's the right level of rigor for this
    project — flag it as a known simplification if asked, don't
    oversell it as mathematically exact.
    """

    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        artifacts = load_artifacts(artifact_dir)
        self.raw_models = artifacts["raw_models"]            # for SHAP
        self.ensemble = artifacts["ensemble"]                 # for prediction/confidence
        self.label_encoder = artifacts["label_encoder"]
        self.feature_names = artifacts["feature_names"]
        self.metadata = artifacts["metadata"]
        self.weights = self.metadata["ensemble_weights"]

        self.explainers = {
            name: shap.TreeExplainer(model) for name, model in self.raw_models.items()
        }

    def _align_columns(self, feature_dict: Dict) -> pd.DataFrame:
        row = {}
        missing = []
        for col in self.feature_names:
            if col in feature_dict:
                row[col] = feature_dict[col]
            else:
                row[col] = 0
                missing.append(col)

        extra = [
            k for k in feature_dict
            if k not in self.feature_names and k not in NON_FEATURE_COLUMNS
        ]

        if missing:
            print(f"[explainability warn] {len(missing)} training features missing at inference, filled with 0: {missing}")
        if extra:
            print(f"[explainability warn] {len(extra)} inference features not seen at training, dropped: {extra}")

        return pd.DataFrame([row], columns=self.feature_names)

    def explain_case(self, case: Dict, nli_checker, top_k: int = 5) -> Dict:
        feature_dict = extract_features_single(case, nli_checker)
        X_row = self._align_columns(feature_dict)

        proba = self.ensemble.predict_proba(X_row)[0]
        predicted_idx = int(np.argmax(proba))
        predicted_label = self.label_encoder.classes_[predicted_idx]
        confidence = float(proba[predicted_idx])

        combined_shap_row = self._combined_shap_row(X_row, predicted_idx)

        contributions = list(zip(self.feature_names, combined_shap_row, X_row.iloc[0].tolist()))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)

        positive = [(f, v, val) for f, v, val in contributions if v > 0][:top_k]
        negative = [(f, v, val) for f, v, val in contributions if v < 0][:top_k]

        row_series = X_row.iloc[0]
        top_positive_reasons = [
            {"feature": f, "shap_value": round(float(v), 4), "value": val,
             "explanation": humanize_feature(f, val, row_series)}
            for f, v, val in positive
        ]
        top_negative_reasons = [
            {"feature": f, "shap_value": round(float(v), 4), "value": val,
             "explanation": humanize_feature(f, val, row_series)}
            for f, v, val in negative
        ]

        return {
            "case_id": case.get("case_id", ""),
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(self.label_encoder.classes_, proba)
            },
            "ensemble_weights": self.weights,
            "top_positive_reasons": top_positive_reasons,
            "top_negative_reasons": top_negative_reasons,
        }

    def explain_many(self, cases: List[Dict], nli_checker, top_k: int = 5) -> List[Dict]:
        return [self.explain_case(c, nli_checker, top_k=top_k) for c in cases]

    def _combined_shap_row(self, X_row: pd.DataFrame, class_idx: int) -> np.ndarray:
        """Weighted average of per-model SHAP rows for the predicted class,
        using the same weights the ensemble uses for prediction."""
        combined = None
        for name, explainer in self.explainers.items():
            shap_values = explainer.shap_values(X_row)
            row = self._extract_class_shap_row(shap_values, class_idx)
            weighted = row * self.weights[name]
            combined = weighted if combined is None else combined + weighted
        return combined

    @staticmethod
    def _extract_class_shap_row(shap_values, class_idx: int) -> np.ndarray:
        """Normalizes across the shap-version / library differences in
        how multiclass TreeExplainer output is shaped."""
        if isinstance(shap_values, list):
            return np.asarray(shap_values[class_idx])[0]
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            return arr[0, :, class_idx]
        return arr[0]


# ---------------------------------------------------------------------------
# Simple JSON-shaped output matching the pseudocode's recommended format
# ---------------------------------------------------------------------------

def to_simple_json(explanation: Dict) -> Dict:
    """Collapses the richer explain_case() output down to exactly the
    shape shown in the pseudocode, for callers that just want the
    compact version."""
    return {
        "prediction": explanation["prediction"],
        "confidence": explanation["confidence"],
        "top_positive_reasons": [r["feature"] for r in explanation["top_positive_reasons"]],
        "top_negative_reasons": [r["feature"] for r in explanation["top_negative_reasons"]],
    }


if __name__ == "__main__":
    from nli_checker import ContradictionChecker

    explainer = DisputeExplainer()
    checker = ContradictionChecker()

    test_case = {
        "case_id": "DC-DEMO-001",
        "dispute_reason_category": "delivered_but_disputed",
        "customer_statement": "I never received my package. No one was home and nothing was left at the door.",
        "merchant_statement": "Delivered with signature confirmation on July 10th at 2:14 PM.",
        "evidence_customer": ["Photo of empty doorstep"],
        "evidence_merchant": ["Signed delivery confirmation", "Carrier GPS delivery log"],
    }

    full = explainer.explain_case(test_case, checker)
    print(json.dumps(full, indent=2))

    print("\n--- Compact form ---")
    print(json.dumps(to_simple_json(full), indent=2))






















