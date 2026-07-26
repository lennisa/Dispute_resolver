"""
features.py — Converts raw dispute cases into a numeric feature matrix
for the XGBoost classifier.

Sits between the NLP modules (parser.py, nli_checker.py) and the model:

    Dispute Case → parser.py + nli_checker.py → feature groups →
    merged feature dict → pandas DataFrame → XGBoost

Enhancements over the base spec:
  - Evidence reliability scoring (bank statement > signature > email >
    customer statement, etc.) folded in as its own feature group, since
    it's the single most load-bearing signal in the whole model.
  - build_feature_dataframe() does REAL batch processing — one
    parse_many() call and one check_batch() call for the entire
    dataset, not a per-case loop calling parse_case()/check() (which
    is what extract_features_single() does, and is fine for one-off
    calls but would be slow across a few hundred cases).
  - validate_feature_dataframe() implements the Step 22 checks
    (no NaN, no inf, no duplicate columns, all numeric) as a callable
    function you run right before training, not just a checklist.
  - Fixed-mapping categorical encoding for reason_category and the NLI
    predicted_label, instead of a freshly-fit LabelEncoder per run —
    a LabelEncoder fit separately at train time and inference time can
    silently assign different integers to the same category, which is
    a nasty bug in a served model. A fixed dict is deterministic.
"""

import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from parser import parse_case, parse_many

# ContradictionChecker/NLIResult are imported lazily inside functions that
# need them, so this module can still be imported (and its non-NLI helpers
# tested) in an environment where torch/transformers aren't installed yet.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEGATION_WORDS = {
    "never", "not", "no", "without", "didn't", "wasn't", "don't", "won't",
    "can't", "doesn't", "isn't", "aren't", "weren't", "couldn't", "shouldn't",
    "wouldn't", "hasn't", "haven't", "hadn't", "none", "nobody", "nothing",
}

KEYWORDS = [
    "refund", "delivery", "charge", "fraud", "tracking", "cancel",
    "duplicate", "damaged", "unauthorized", "signature", "policy",
    "chargeback", "returned",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "was", "were", "to", "of",
    "in", "on", "for", "with", "at", "by", "from", "it", "this", "that",
}

AMOUNT_REGEX = re.compile(r'\$?\s?\d[\d,]*(?:\.\d{1,2})?')
NUMBER_REGEX = re.compile(r'\d+')
SENTENCE_SPLIT_REGEX = re.compile(r'[.!?]+')

# Evidence reliability lookup — substring-matched against the free-text
# evidence descriptions in evidence_customer / evidence_merchant.
# Order matters: first match wins, so put more specific phrases first.
EVIDENCE_RELIABILITY_TABLE = [
    ("bank statement", 0.99),
    ("signed delivery confirmation", 0.96),
    ("signature", 0.96),
    ("gps", 0.92),
    ("carrier gps", 0.92),
    ("invoice", 0.94),
    ("fulfillment", 0.90),
    ("warehouse scan", 0.90),
    ("shipping label", 0.75),
    ("photo", 0.78),
    ("email", 0.80),
    ("chat", 0.72),
    ("text message", 0.70),
    ("call transcript", 0.68),
    ("cancellation confirmation", 0.85),
    ("screenshot", 0.72),
    ("billing log", 0.70),
    ("return policy", 0.60),
    ("product listing", 0.55),
    ("order history", 0.85),
    ("refund processing record", 0.85),
]
DEFAULT_EVIDENCE_RELIABILITY = 0.5

# Fixed category encodings — deterministic across train/serve, unlike a
# freshly-fit LabelEncoder. "unknown" is the fallback for any category
# seen at inference time that wasn't in the training set.
REASON_CATEGORY_ENCODING = {
    "item_not_received": 0,
    "delivered_but_disputed": 1,
    "duplicate_charge": 2,
    "subscription_cancellation": 3,
    "service_not_rendered": 4,
    "not_as_described": 5,
    "unauthorized_transaction": 6,
    "ambiguous_conflicting": 7,
    "missing_evidence": 8,
    "unknown": 9,
}

NLI_LABEL_ENCODING = {"contradiction": 0, "entailment": 1, "neutral": 2}

NON_FEATURE_COLUMNS = ["case_id", "label", "reason_category_raw"]


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------

def clean_text(text: str, remove_punct: bool = False) -> str:
    """Lowercase + collapse whitespace; punctuation removal is optional
    since sentence counting needs punctuation to still be present."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)
    return text


def extract_text_statistics(statement: str) -> Dict:
    if not statement or not statement.strip():
        return {
            "word_count": 0, "character_count": 0, "sentence_count": 0,
            "average_word_length": 0.0, "uppercase_ratio": 0.0,
            "numeric_token_count": 0,
        }

    words = statement.split()
    word_count = len(words)
    character_count = len(statement)

    sentences = [s for s in SENTENCE_SPLIT_REGEX.split(statement) if s.strip()]
    sentence_count = max(1, len(sentences))

    clean_words = [re.sub(r'[^\w]', '', w) for w in words]
    clean_words = [w for w in clean_words if w]
    average_word_length = (
        sum(len(w) for w in clean_words) / len(clean_words) if clean_words else 0.0
    )

    alpha_chars = [c for c in statement if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )

    numeric_token_count = sum(1 for w in words if NUMBER_REGEX.search(w))

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "average_word_length": round(average_word_length, 3),
        "uppercase_ratio": round(uppercase_ratio, 3),
        "numeric_token_count": numeric_token_count,
    }


# ---------------------------------------------------------------------------
# Evidence features (from parser.py output — dates/amounts/tracking/orgs
# extracted from the statement text itself)
# ---------------------------------------------------------------------------

def extract_evidence_features(parsed_side: Dict) -> Dict:
    """parsed_side is one side of parser.parse_case()'s output, e.g.
    parsed['customer'] — a dict with extracted_dates/amounts/orgs/tracking."""
    date_count = len(parsed_side.get("extracted_dates", []))
    amount_count = len(parsed_side.get("extracted_amounts", []))
    tracking_count = len(parsed_side.get("extracted_tracking_numbers", []))
    organisation_count = len(parsed_side.get("extracted_orgs", []))
    total_evidence = date_count + amount_count + tracking_count + organisation_count

    word_count = max(1, len(parsed_side.get("raw_text", "").split()))
    evidence_density = round(total_evidence / word_count, 4)

    return {
        "date_count": date_count,
        "amount_count": amount_count,
        "tracking_count": tracking_count,
        "organisation_count": organisation_count,
        "total_evidence": total_evidence,
        "evidence_density": evidence_density,
    }


def parse_amount(text: str) -> Optional[float]:
    """'$50.00' / '50,000' -> float. Returns None if not parseable —
    caller is responsible for turning that into a 0.0 default feature,
    per the "never NaN in the feature matrix" rule."""
    if not text:
        return None
    cleaned = re.sub(r'[^\d.]', '', text.replace(',', ''))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_amount_features(customer_parsed: Dict, merchant_parsed: Dict) -> Dict:
    cust_amounts = [parse_amount(a) for a in customer_parsed.get("extracted_amounts", [])]
    merch_amounts = [parse_amount(a) for a in merchant_parsed.get("extracted_amounts", [])]
    cust_amounts = [a for a in cust_amounts if a is not None]
    merch_amounts = [a for a in merch_amounts if a is not None]

    customer_amount = cust_amounts[0] if cust_amounts else 0.0
    merchant_amount = merch_amounts[0] if merch_amounts else 0.0
    amount_difference = abs(customer_amount - merchant_amount)
    amount_match = int(amount_difference < 0.01 and (customer_amount > 0 or merchant_amount > 0))

    return {
        "customer_amount": round(customer_amount, 2),
        "merchant_amount": round(merchant_amount, 2),
        "amount_difference": round(amount_difference, 2),
        "amount_match": amount_match,
    }


def overlap_score(list1: List[str], list2: List[str]) -> float:
    """Jaccard similarity: |intersection| / |union|, case-insensitive."""
    set1 = {x.strip().lower() for x in list1 if x and x.strip()}
    set2 = {x.strip().lower() for x in list2 if x and x.strip()}
    if not set1 and not set2:
        return 0.0
    union = set1 | set2
    if not union:
        return 0.0
    return round(len(set1 & set2) / len(union), 4)


def extract_tracking_features(customer_parsed: Dict, merchant_parsed: Dict) -> Dict:
    cust_tracking = customer_parsed.get("extracted_tracking_numbers", [])
    merch_tracking = merchant_parsed.get("extracted_tracking_numbers", [])

    tracking_overlap = overlap_score(cust_tracking, merch_tracking)
    tracking_match = int(tracking_overlap > 0)
    customer_has_tracking = int(len(cust_tracking) > 0)
    merchant_has_tracking = int(len(merch_tracking) > 0)

    return {
        "tracking_overlap": tracking_overlap,
        "tracking_match": tracking_match,
        "customer_has_tracking": customer_has_tracking,
        "merchant_has_tracking": merchant_has_tracking,
    }


def extract_organisation_features(customer_parsed: Dict, merchant_parsed: Dict) -> Dict:
    cust_orgs = customer_parsed.get("extracted_orgs", [])
    merch_orgs = merchant_parsed.get("extracted_orgs", [])

    organisation_overlap = overlap_score(cust_orgs, merch_orgs)
    organisation_match = int(organisation_overlap > 0)

    return {
        "customer_org_count": len(cust_orgs),
        "merchant_org_count": len(merch_orgs),
        "organisation_overlap": organisation_overlap,
        "organisation_match": organisation_match,
    }


# ---------------------------------------------------------------------------
# Keyword / negation features
# ---------------------------------------------------------------------------

def count_keywords(statement: str) -> Dict:
    text = clean_text(statement)
    counts = {}
    total = 0
    for kw in KEYWORDS:
        n = len(re.findall(re.escape(kw), text))
        counts[f"{kw}_count"] = n
        total += n
    counts["total_keyword_count"] = total
    return counts


def count_negations(statement: str) -> int:
    text = clean_text(statement)
    words = set(re.findall(r"[\w']+", text))
    return sum(1 for neg in NEGATION_WORDS if neg in words)


# ---------------------------------------------------------------------------
# NLI features
# ---------------------------------------------------------------------------

def extract_nli_features(nli_result) -> Dict:
    """nli_result is an NLIResult (or any duck-typed object exposing the
    same attributes) from nli_checker.py."""
    contradiction = nli_result.contradiction_score
    entailment = nli_result.entailment_score
    neutral = nli_result.neutral_score

    confidence = max(contradiction, entailment, neutral)
    margin = contradiction - entailment
    predicted_label_encoded = NLI_LABEL_ENCODING.get(nli_result.label, -1)

    return {
        "contradiction_score": round(contradiction, 4),
        "entailment_score": round(entailment, 4),
        "neutral_score": round(neutral, 4),
        "nli_predicted_label": predicted_label_encoded,
        "nli_confidence": round(confidence, 4),
        "nli_margin": round(margin, 4),
    }


# ---------------------------------------------------------------------------
# Evidence reliability features (enhancement — not in the original spec,
# but this is arguably the highest-signal feature group in the whole
# model, so it earns its own function rather than being an afterthought)
# ---------------------------------------------------------------------------

def _reliability_for(evidence_item: str) -> float:
    item_lower = evidence_item.lower()
    for phrase, score in EVIDENCE_RELIABILITY_TABLE:
        if phrase in item_lower:
            return score
    return DEFAULT_EVIDENCE_RELIABILITY


def extract_evidence_reliability_features(
    evidence_customer: List[str], evidence_merchant: List[str]
) -> Dict:
    evidence_customer = evidence_customer or []
    evidence_merchant = evidence_merchant or []

    cust_scores = [_reliability_for(e) for e in evidence_customer]
    merch_scores = [_reliability_for(e) for e in evidence_merchant]

    customer_evidence_reliability = (
        round(sum(cust_scores) / len(cust_scores), 4) if cust_scores else 0.0
    )
    merchant_evidence_reliability = (
        round(sum(merch_scores) / len(merch_scores), 4) if merch_scores else 0.0
    )

    return {
        "customer_evidence_count": len(evidence_customer),
        "merchant_evidence_count": len(evidence_merchant),
        "customer_evidence_reliability": customer_evidence_reliability,
        "merchant_evidence_reliability": merchant_evidence_reliability,
        "evidence_reliability_gap": round(
            customer_evidence_reliability - merchant_evidence_reliability, 4
        ),
        "customer_has_evidence": int(len(evidence_customer) > 0),
        "merchant_has_evidence": int(len(evidence_merchant) > 0),
        "either_side_missing_evidence": int(
            len(evidence_customer) == 0 or len(evidence_merchant) == 0
        ),
    }


# ---------------------------------------------------------------------------
# Difference features — tree models often learn relative differences
# better than raw counts, so these get computed explicitly rather than
# left for XGBoost to infer from the raw pairs.
# ---------------------------------------------------------------------------

def extract_difference_features(
    text_stats_c: Dict, text_stats_m: Dict,
    evidence_c: Dict, evidence_m: Dict,
    amount_features: Dict, tracking_features: Dict,
    keyword_c: Dict, keyword_m: Dict,
    negation_c: int, negation_m: int,
) -> Dict:
    return {
        "word_difference": abs(text_stats_c["word_count"] - text_stats_m["word_count"]),
        "sentence_difference": abs(text_stats_c["sentence_count"] - text_stats_m["sentence_count"]),
        "evidence_difference": abs(evidence_c["total_evidence"] - evidence_m["total_evidence"]),
        "amount_difference": amount_features["amount_difference"],
        "tracking_difference": abs(evidence_c["tracking_count"] - evidence_m["tracking_count"]),
        "keyword_difference": abs(keyword_c["total_keyword_count"] - keyword_m["total_keyword_count"]),
        "negation_difference": abs(negation_c - negation_m),
    }


# ---------------------------------------------------------------------------
# Metadata features
# ---------------------------------------------------------------------------

def extract_metadata(case: Dict) -> Dict:
    raw_category = case.get("dispute_reason_category", "unknown")
    encoded = REASON_CATEGORY_ENCODING.get(raw_category, REASON_CATEGORY_ENCODING["unknown"])
    return {
        "reason_category": encoded,
        "reason_category_raw": raw_category,  # kept for debugging, dropped before training
    }


# ---------------------------------------------------------------------------
# Missing-value sanitization (Step 21) — never let None/NaN into the
# numeric matrix.
# ---------------------------------------------------------------------------

def _sanitize(features: Dict) -> Dict:
    clean = {}
    for k, v in features.items():
        if k in NON_FEATURE_COLUMNS:
            clean[k] = v
            continue
        if v is None:
            clean[k] = 0
        elif isinstance(v, bool):
            clean[k] = int(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            clean[k] = 0.0
        else:
            clean[k] = v
    return clean


# ---------------------------------------------------------------------------
# Single-case extraction (convenience — not batched, fine for ad-hoc /
# single live-inference calls where batching one case has no benefit)
# ---------------------------------------------------------------------------

def extract_features_single(case: Dict, nli_checker) -> Dict:
    """
    nli_checker: an instance of nli_checker.ContradictionChecker (or
    anything exposing .check(customer_statement, merchant_statement)
    returning an NLIResult-shaped object).
    """
    customer_statement = case.get("customer_statement", "") or ""
    merchant_statement = case.get("merchant_statement", "") or ""

    parsed = parse_case(customer_statement, merchant_statement)
    nli_result = nli_checker.check(customer_statement, merchant_statement)

    return _build_feature_dict(case, customer_statement, merchant_statement, parsed, nli_result)


def _build_feature_dict(case, customer_statement, merchant_statement, parsed, nli_result) -> Dict:
    text_stats_c = extract_text_statistics(customer_statement)
    text_stats_m = extract_text_statistics(merchant_statement)

    evidence_c = extract_evidence_features(parsed["customer"])
    evidence_m = extract_evidence_features(parsed["merchant"])

    amount_features = extract_amount_features(parsed["customer"], parsed["merchant"])
    tracking_features = extract_tracking_features(parsed["customer"], parsed["merchant"])
    org_features = extract_organisation_features(parsed["customer"], parsed["merchant"])

    keyword_c = count_keywords(customer_statement)
    keyword_m = count_keywords(merchant_statement)

    negation_c = count_negations(customer_statement)
    negation_m = count_negations(merchant_statement)

    nli_features = extract_nli_features(nli_result)

    diff_features = extract_difference_features(
        text_stats_c, text_stats_m, evidence_c, evidence_m,
        amount_features, tracking_features, keyword_c, keyword_m,
        negation_c, negation_m,
    )

    evidence_reliability_features = extract_evidence_reliability_features(
        case.get("evidence_customer", []), case.get("evidence_merchant", [])
    )

    metadata_features = extract_metadata(case)

    features = {"case_id": case.get("case_id", ""), "label": case.get("label", "")}
    features.update({f"customer_{k}": v for k, v in text_stats_c.items()})
    features.update({f"merchant_{k}": v for k, v in text_stats_m.items()})
    features.update({f"customer_{k}": v for k, v in evidence_c.items()})
    features.update({f"merchant_{k}": v for k, v in evidence_m.items()})
    features.update(amount_features)
    features.update(tracking_features)
    features.update(org_features)
    features.update({f"customer_{k}": v for k, v in keyword_c.items()})
    features.update({f"merchant_{k}": v for k, v in keyword_m.items()})
    features["customer_negations"] = negation_c
    features["merchant_negations"] = negation_m
    features.update(nli_features)
    features.update(diff_features)
    features.update(evidence_reliability_features)
    features.update(metadata_features)

    return _sanitize(features)


# ---------------------------------------------------------------------------
# Batch processing (Step 19) — the function you actually want to call
# for a dataset of any real size. Runs parser and NLI ONCE across all
# cases instead of per-case.
# ---------------------------------------------------------------------------

def build_feature_dataframe(cases: List[Dict], nli_checker) -> pd.DataFrame:
    customer_statements = [c.get("customer_statement", "") or "" for c in cases]
    merchant_statements = [c.get("merchant_statement", "") or "" for c in cases]

    parsed_customer = parse_many(customer_statements)
    parsed_merchant = parse_many(merchant_statements)
    nli_results = nli_checker.check_batch(list(zip(customer_statements, merchant_statements)))

    rows = []
    for case, p_cust, p_merch, nli_result, cust_stmt, merch_stmt in zip(
        cases, parsed_customer, parsed_merchant, nli_results,
        customer_statements, merchant_statements,
    ):
        parsed = {"customer": p_cust, "merchant": p_merch}
        rows.append(_build_feature_dict(case, cust_stmt, merch_stmt, parsed, nli_result))

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Feature validation (Step 22) — run this right before training.
# ---------------------------------------------------------------------------

def validate_feature_dataframe(df: pd.DataFrame) -> Dict:
    """Returns a report dict; raises AssertionError on hard failures
    (duplicate columns, NaN/inf in feature columns) so a bad matrix
    can't silently reach model.fit()."""
    issues = []

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        issues.append(f"Duplicate columns: {dupes}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    non_numeric_cols = [
        c for c in feature_cols
        if not np.issubdtype(df[c].dropna().infer_objects().dtype, np.number)
        and df[c].dtype == object
    ]
    if non_numeric_cols:
        issues.append(f"Non-numeric feature columns: {non_numeric_cols}")

    nan_cols = numeric_df.columns[numeric_df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values in: {nan_cols}")

    inf_cols = numeric_df.columns[np.isinf(numeric_df.to_numpy(dtype=float, na_value=0)).any(axis=0)].tolist()
    if inf_cols:
        issues.append(f"Infinite values in: {inf_cols}")

    if issues:
        raise AssertionError("Feature validation failed:\n" + "\n".join(issues))

    return {
        "n_rows": len(df),
        "n_feature_columns": len(feature_cols),
        "feature_columns": feature_cols,
        "status": "ok",
    }


def get_training_matrix(df: pd.DataFrame):
    """Convenience: split a validated feature dataframe into (X, y, case_ids)
    ready for XGBoost, dropping the non-feature/debug columns."""
    validate_feature_dataframe(df)
    y = df["label"]
    case_ids = df["case_id"]
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y, case_ids


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use the real ContradictionChecker if torch/transformers are available
    # in this environment; otherwise fall back to a lightweight stub so the
    # rest of the feature pipeline can still be exercised and reviewed.
    try:
        from nli_checker import ContradictionChecker
        checker = ContradictionChecker()
        print("Using real ContradictionChecker.\n")
    except Exception as e:
        print(f"[stub] Could not load real NLI model ({e}); using a stub for this test run.\n")

        from dataclasses import dataclass

        @dataclass
        class _StubResult:
            label: str
            contradiction_score: float
            entailment_score: float
            neutral_score: float

        class _StubChecker:
            def check(self, cust, merch):
                return self.check_batch([(cust, merch)])[0]

            def check_batch(self, pairs):
                results = []
                for cust, merch in pairs:
                    contradiction = 0.8 if "never" in cust.lower() and "confirm" in merch.lower() else 0.35
                    results.append(_StubResult(
                        label="contradiction" if contradiction > 0.5 else "neutral",
                        contradiction_score=contradiction,
                        entailment_score=1 - contradiction - 0.1,
                        neutral_score=0.1,
                    ))
                return results

        checker = _StubChecker()

    test_cases = [
        {
            "case_id": "DC-TEST-001",
            "dispute_reason_category": "delivered_but_disputed",
            "customer_statement": "I never received my package. No one was home and nothing was left at the door.",
            "merchant_statement": "Delivered with signature confirmation on July 10th at 2:14 PM.",
            "evidence_customer": ["Photo of empty doorstep"],
            "evidence_merchant": ["Signed delivery confirmation", "Carrier GPS delivery log"],
            "label": "merchant_wins",
        },
        {
            "case_id": "DC-TEST-002",
            "dispute_reason_category": "duplicate_charge",
            "customer_statement": "I was charged $18.40 twice by Ferro & Wick Coffee Co. on the same day.",
            "merchant_statement": "We show one fulfilled order for $18.40.",
            "evidence_customer": ["Bank statement showing two line items"],
            "evidence_merchant": [],
            "label": "card_member_wins",
        },
    ]

    df = build_feature_dataframe(test_cases, checker)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(df.T)

    report = validate_feature_dataframe(df)
    print("\nValidation report:", report)
