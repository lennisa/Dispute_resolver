# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# model_name = "cross-encoder/nli-deberta-v3-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # Print the model's official label mapping to be 100% sure
# print("Model id2label mapping:", model.config.id2label)

# def check_contradiction(premise, hypothesis):
#     inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
        
#     probs = torch.softmax(logits, dim=1).tolist()[0]
    
#     # Map dynamically using model.config.id2label (e.g. {0: 'contradiction', 1: 'entailment', 2: 'neutral'} or similar)
#     scores = {}
#     for idx, prob in enumerate(probs):
#         label_name = model.config.id2label.get(idx, f"label_{idx}")
#         scores[f"{label_name}_score"] = round(prob, 4)
        
#     return scores

# # Test cases
# customer_claim = "I never received the package I ordered on January 10th."
# merchant_claim_matching = "The package was successfully delivered to the customer's front porch on January 12th."
# merchant_claim_contradicting = "The customer picked up the item directly from our retail store on January 10th."

# print("--- Test 1 (Contradicting/Conflicting Stories) ---")
# print(check_contradiction(customer_claim, merchant_claim_contradicting))

# print("--- Test 2 (Plausible/Neutral Alignment) ---")
# print(check_contradiction(customer_claim, merchant_claim_matching))





"""
nli_checker.py — Contradiction detection between customer and merchant
statements, using a pretrained NLI (Natural Language Inference) model.

No API calls, no training required — this is off-the-shelf statistical
model inference, run locally.

Model: cross-encoder/nli-deberta-v3-small
  - Small enough to run on CPU at hackathon scale (a few hundred cases).
  - Outputs contradiction / entailment / neutral for a premise-hypothesis
    pair, which we treat as (customer_statement, merchant_statement).

NOTE: label order for NLI models varies by checkpoint. This module reads
id2label directly from the model config instead of hardcoding an index
order, which is the safe way to do this and avoids a class of silent
bugs where "contradiction" and "entailment" scores get swapped.
"""

from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


@dataclass
class NLIResult:
    label: str                 # "contradiction" | "entailment" | "neutral"
    contradiction_score: float  # 0-1, this is the feature your pipeline wants
    entailment_score: float
    neutral_score: float

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "contradiction_score": round(self.contradiction_score, 4),
            "entailment_score": round(self.entailment_score, 4),
            "neutral_score": round(self.neutral_score, 4),
        }


class ContradictionChecker:
    """
    Loads the NLI model once (loading it is the expensive part — never
    construct this class per-request in your API; instantiate one instance
    at backend startup and reuse it).
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Read the label order from the model itself rather than assuming
        # a fixed [contradiction, entailment, neutral] order.
        id2label = {k: v.lower() for k, v in self.model.config.id2label.items()}
        self.label_order = [id2label[i] for i in range(len(id2label))]
        for required in ("contradiction", "entailment", "neutral"):
            if required not in self.label_order:
                raise ValueError(
                    f"Unexpected label set from model config: {self.label_order}. "
                    f"Expected contradiction/entailment/neutral."
                )

    @torch.no_grad()
    def check(self, customer_statement: str, merchant_statement: str) -> NLIResult:
        """
        Single pair. For anything beyond a handful of ad-hoc calls, use
        check_batch instead — it's far more efficient than calling this
        in a loop.
        """
        return self.check_batch([(customer_statement, merchant_statement)])[0]

    @torch.no_grad()
    def check_batch(self, pairs: List[tuple]) -> List[NLIResult]:
        """
        pairs: list of (customer_statement, merchant_statement) tuples.
        This is the function to call when scoring your full dataset —
        batching through the model is dramatically faster than a
        per-case Python loop calling check() one at a time.
        """
        premises = [p[0] for p in pairs]
        hypotheses = [p[1] for p in pairs]

        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for row in probs:
            scores = {label: float(row[i]) for i, label in enumerate(self.label_order)}
            top_label = max(scores, key=scores.get)
            results.append(
                NLIResult(
                    label=top_label,
                    contradiction_score=scores["contradiction"],
                    entailment_score=scores["entailment"],
                    neutral_score=scores["neutral"],
                )
            )
        return results


if __name__ == "__main__":
    import json

    checker = ContradictionChecker()

    test_pairs = [
        (
            "I never received my package.",
            "Delivered with signature confirmation on file.",
        ),
        (
            "The item arrived a day late but otherwise as expected.",
            "Shipment was delivered on schedule per our tracking records.",
        ),
        (
            "I cancelled my subscription before the renewal date.",
            "We have no record of a cancellation request before billing.",
        ),
    ]

    results = checker.check_batch(test_pairs)
    for (cust, merch), result in zip(test_pairs, results):
        print(json.dumps({
            "customer_statement": cust,
            "merchant_statement": merch,
            **result.to_dict(),
        }, indent=2))
