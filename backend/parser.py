# import spacy
# import re

# nlp = spacy.load("en_core_web_sm")


# def extract_tracking_numbers(text):
#     # Simple regex pattern for tracking numbers or order IDs (e.g., TRK followed by numbers, or generic uppercase alphanumeric IDs)
#     pattern = r'\b(?:TRK|ORD|INV)[-_\s]?[A-Z0-9]{6,}\b'
#     return re.findall(pattern, text, re.IGNORECASE)

# def parse_dispute_evidence(statement_text):
#     doc = nlp(statement_text)
    
#     dates = []
#     amounts = []
#     orgs = []
    
#     for ent in doc.ents:
#         if ent.label_ == "DATE":
#             dates.append(ent.text)
#         elif ent.label_ == "MONEY":
#             amounts.append(ent.text)
#         elif ent.label_ == "ORG":
#             orgs.append(ent.text)
            
#     # Run regex for structured IDs
#     tracking_numbers = extract_tracking_numbers(statement_text)
    
#     return {
#         "raw_text": statement_text,
#         "extracted_dates": list(set(dates)),
#         "extracted_amounts": list(set(amounts)),
#         "extracted_orgs": list(set(orgs)),
#         "extracted_tracking_numbers": tracking_numbers
#     }

# # Test the main function
# test_case = "I was charged $50.00 by FastShop on 2026-02-10. Tracking is TRK123456789."
# parsed_output = parse_dispute_evidence(test_case)

# import json
# print(json.dumps(parsed_output, indent=2))






"""
parser.py — Evidence extraction from dispute statements.

Uses spaCy NER for dates/amounts/orgs, plus regex for structured
IDs (tracking numbers, order numbers) that NER won't reliably catch.
"""

import re
from typing import Dict, List

import spacy

nlp = spacy.load("en_core_web_sm")

# Matches TRK-123456, ORD_ABC123, INV1234567, etc.
TRACKING_ID_PATTERN = re.compile(
    r'\b(?:TRK|ORD|INV)[-_\s]?[A-Z0-9]{6,}\b', re.IGNORECASE
)

# Fallback for dollar amounts spaCy's MONEY entity occasionally misses
# (e.g. "$50" with no decimal, or mid-sentence amounts after a symbol).
DOLLAR_AMOUNT_PATTERN = re.compile(r'\$\s?\d[\d,]*(?:\.\d{1,2})?')


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _normalize_amount_key(amount_text: str) -> str:
    """Strip currency symbols/commas so '$50.00' and '50.00' dedup as one."""
    return re.sub(r'[^\d.]', '', amount_text)


def _dedup_amounts(items: List[str]) -> List[str]:
    """
    Like _dedup_preserve_order, but treats '$50.00' and '50.00' as the
    same amount. Prefers the version with a '$' sign when both forms
    appear, since that's clearer downstream.
    """
    by_value: Dict[str, str] = {}
    order: List[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        key = _normalize_amount_key(item)
        if not key:
            continue
        if key not in by_value:
            by_value[key] = item
            order.append(key)
        elif "$" in item and "$" not in by_value[key]:
            by_value[key] = item
    return [by_value[k] for k in order]


def extract_tracking_numbers(text: str) -> List[str]:
    return _dedup_preserve_order(TRACKING_ID_PATTERN.findall(text))


def extract_dollar_amounts_fallback(text: str) -> List[str]:
    return _dedup_preserve_order(DOLLAR_AMOUNT_PATTERN.findall(text))


def parse_dispute_evidence(statement_text: str, doc=None) -> Dict:
    """
    Parse a single statement. Pass a pre-computed spaCy `doc` when
    batch-processing many statements (see parse_many below) to avoid
    re-running the pipeline per call.
    """
    if not statement_text or not statement_text.strip():
        return {
            "raw_text": statement_text,
            "extracted_dates": [],
            "extracted_amounts": [],
            "extracted_orgs": [],
            "extracted_tracking_numbers": [],
        }

    if doc is None:
        doc = nlp(statement_text)

    dates, amounts, orgs = [], [], []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ == "MONEY":
            amounts.append(ent.text)
        elif ent.label_ == "ORG":
            orgs.append(ent.text)

    amounts += extract_dollar_amounts_fallback(statement_text)
    tracking_numbers = extract_tracking_numbers(statement_text)

    return {
        "raw_text": statement_text,
        "extracted_dates": _dedup_preserve_order(dates),
        "extracted_amounts": _dedup_amounts(amounts),
        "extracted_orgs": _dedup_preserve_order(orgs),
        "extracted_tracking_numbers": tracking_numbers,
    }


def parse_many(statements: List[str]) -> List[Dict]:
    """
    Batch-parse many statements efficiently using nlp.pipe instead of
    calling nlp() one at a time. This is the function to use when
    running the parser over your full 300-case dataset — it's
    meaningfully faster than a per-statement loop once you're past
    a few dozen cases.
    """
    # nlp.pipe can't handle empty strings gracefully in all spaCy
    # versions, so guard them out and reinsert empty results after.
    non_empty_idx = [i for i, s in enumerate(statements) if s and s.strip()]
    docs = list(nlp.pipe([statements[i] for i in non_empty_idx]))

    results = [None] * len(statements)
    for idx, doc in zip(non_empty_idx, docs):
        results[idx] = parse_dispute_evidence(statements[idx], doc=doc)

    for i, r in enumerate(results):
        if r is None:
            results[i] = parse_dispute_evidence("")

    return results


def parse_case(customer_statement: str, merchant_statement: str) -> Dict:
    """
    Parse both sides of a dispute case in one call — this is the
    function your pipeline should actually call per case, since
    every downstream feature (evidence counts, contradiction check,
    etc.) needs both sides together.
    """
    parsed = parse_many([customer_statement, merchant_statement])
    return {
        "customer": parsed[0],
        "merchant": parsed[1],
    }


if __name__ == "__main__":
    import json

    test_case = {
        "customer_statement": "I was charged $50.00 by FastShop on 2026-02-10. Tracking is TRK123456789.",
        "merchant_statement": "Order ORD-778899 was shipped on Feb 10th for $50.00 via standard carrier.",
    }

    result = parse_case(
        test_case["customer_statement"], test_case["merchant_statement"]
    )
    print(json.dumps(result, indent=2))
