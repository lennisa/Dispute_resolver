"""
main.py — FastAPI backend for the dispute resolution system.

Wires together everything built so far into one live service:

    Request (case) -> parser.py + nli_checker.py (inside features.py) ->
    features.py -> model.py's trained ensemble -> explainability.py (SHAP) ->
    decision engine -> similar case retrieval + policy matching ->
    JSON response shaped for the React frontend (submission, pipeline,
    explainability, decision, fairness screens).

Run:
    uvicorn main:app --reload --port 8000

Requires model_artifacts/ to already exist — run `python3 model.py
dispute_dataset_300.json` first.

Design notes:
  - Heavy objects (NLI model, the 3-model ensemble, SHAP explainers,
    the similar-case index) are loaded ONCE at startup via FastAPI's
    lifespan, not per-request. Reloading any of these per-request would
    make every call take seconds instead of milliseconds.
  - Similar-case retrieval uses TF-IDF + cosine similarity over the
    training dataset as a lightweight, dependency-light stand-in for
    the sentence-embedding version described in the build plan — swap
    in sentence-transformers later without changing the response shape
    if embedding-quality similarity turns out to matter.
  - The decision engine's confidence thresholds and the
    card-member/merchant percentage split are pure functions, unit-
    testable in isolation from the web layer — see decision_engine()
    and to_two_sided_split() below.
  - /fairness reads directly from model_artifacts/metadata.json (real
    test-set metrics saved by model.py), not a fabricated number.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from model import EnsembleModel, ARTIFACT_DIR
import __main__
setattr(__main__, "EnsembleModel", EnsembleModel)

from model import ARTIFACT_DIR
from explainability import DisputeExplainer, to_simple_json
from nli_checker import ContradictionChecker

DATASET_PATH = Path("disputes_dataset_300.json")

# ---------------------------------------------------------------------------
# Decision engine — pure functions, no I/O, easy to test/tune in isolation.
# ---------------------------------------------------------------------------

AUTO_RESOLVE_THRESHOLD = 0.75
RECOMMEND_THRESHOLD = 0.55

OUTCOME_DISPLAY_NAMES = {
    "card_member_wins": "Card member",
    "merchant_wins": "Merchant",
    "partial": "Partial",
    "escalate": "Escalated",
}


def decision_engine(predicted_label: str, confidence: float) -> Dict:
    """
    Maps a 4-class prediction (card_member_wins / merchant_wins /
    partial / escalate) + its confidence into the auto-resolve /
    recommend / human-review action the UI's stamp badge shows.

    The model can predict "escalate" directly (it's a real training
    label), OR any other class can still get escalated here if
    confidence is too low — a low-confidence "merchant_wins" shouldn't
    be auto-resolved just because the model happened to lean that way.
    """
    if predicted_label == "escalate" or confidence < RECOMMEND_THRESHOLD:
        action = "HUMAN REVIEW"
    elif confidence >= AUTO_RESOLVE_THRESHOLD:
        action = "AUTO-RESOLVED"
    else:
        action = "RECOMMENDED"

    outcome = OUTCOME_DISPLAY_NAMES.get(predicted_label, predicted_label)
    return {"action": action, "outcome": outcome}


def to_two_sided_split(class_probabilities: Dict[str, float]) -> Dict[str, float]:
    """
    The UI's scale bar wants a single card-member-vs-merchant split.
    Renormalizes just the card_member_wins / merchant_wins probability
    mass (excluding partial/escalate) so the bar always reads as a
    clean two-sided comparison. Falls back to 50/50 if both are ~0
    (e.g. the case is dominated by escalate/partial probability).
    """
    cust = class_probabilities.get("card_member_wins", 0.0)
    merch = class_probabilities.get("merchant_wins", 0.0)
    total = cust + merch
    if total < 1e-6:
        return {"custPct": 0.5, "merchPct": 0.5}
    return {"custPct": round(cust / total, 4), "merchPct": round(merch / total, 4)}


# ---------------------------------------------------------------------------
# SHAP -> frontend feature-bar shape
# ---------------------------------------------------------------------------

def to_frontend_features(explanation: Dict) -> List[Dict]:
    """
    Converts explain_case()'s top_positive_reasons/top_negative_reasons
    into the {label, side, weight} shape the React FeatureBar component
    already expects (see dispute-console.jsx).
    """
    out = []
    for r in explanation["top_positive_reasons"]:
        out.append({"label": r["explanation"], "side": "customer", "weight": abs(r["shap_value"])})
    for r in explanation["top_negative_reasons"]:
        out.append({"label": r["explanation"], "side": "merchant", "weight": abs(r["shap_value"])})
    out.sort(key=lambda f: f["weight"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Similar case retrieval — TF-IDF + cosine similarity (lightweight
# stand-in for sentence-embedding similarity; same output shape either way)
# ---------------------------------------------------------------------------

class SimilarCaseIndex:
    def __init__(self, cases: List[Dict]):
        self.cases = cases
        self.texts = [
            f"{c.get('customer_statement', '')} {c.get('merchant_statement', '')}"
            for c in cases
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None

    def query(self, customer_statement: str, merchant_statement: str, top_k: int = 3) -> List[Dict]:
        if self.matrix is None:
            return []
        query_text = f"{customer_statement} {merchant_statement}"
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "id": self.cases[i].get("case_id", f"case-{i}"),
                "sim": round(float(sims[i]), 4),
                "outcome": OUTCOME_DISPLAY_NAMES.get(self.cases[i].get("label", ""), self.cases[i].get("label", "")),
            }
            for i in top_idx if sims[i] > 0
        ]


# ---------------------------------------------------------------------------
# Policy matching — hardcoded snippets, simple keyword scoring against
# the dispute's category and evidence flags (Tier 1, per the build plan)
# ---------------------------------------------------------------------------

POLICY_SNIPPETS = [
    {
        "id": "REG-E-1005.11-signature",
        "keywords": ["signature", "signed", "delivered", "delivery confirmation"],
        "text": "Reg. E §1005.11 — signed proof of delivery is dispositive absent contrary evidence.",
    },
    {
        "id": "REG-E-1005.11-noproof",
        "keywords": ["not received", "never received", "no tracking", "no delivery"],
        "text": "Reg. E §1005.11 — burden shifts to merchant when delivery cannot be substantiated.",
    },
    {
        "id": "DUPLICATE-CHARGE",
        "keywords": ["duplicate", "charged twice", "two charges"],
        "text": "Standard chargeback reason code 12.6 — duplicate processing entitles the card member to a refund of the redundant charge.",
    },
    {
        "id": "SUBSCRIPTION-CANCEL",
        "keywords": ["cancel", "cancelled", "subscription", "renewal"],
        "text": "Merchant cancellation policy applies — billing after a confirmed cancellation date is refundable regardless of internal processing delay.",
    },
    {
        "id": "NOT-AS-DESCRIBED",
        "keywords": ["not as described", "different", "material", "listing"],
        "text": "Chargeback reason code 13.3 — merchandise not as described places the burden of proof on the merchant to substantiate listing accuracy.",
    },
    {
        "id": "UNAUTHORIZED",
        "keywords": ["unauthorized", "did not authorize", "fraud", "compromised"],
        "text": "Reg. E §1005.6 — card member liability for unauthorized transactions is limited pending merchant verification of authorization.",
    },
    {
        "id": "MISSING-EVIDENCE",
        "keywords": [],
        "text": "Absent sufficient evidence from either party, standard practice routes the case to manual review rather than an automated determination.",
    },
]


def match_policy(case: Dict) -> str:
    text = f"{case.get('customer_statement', '')} {case.get('merchant_statement', '')}".lower()
    best, best_score = None, 0
    for policy in POLICY_SNIPPETS:
        score = sum(1 for kw in policy["keywords"] if kw in text)
        if score > best_score:
            best, best_score = policy, score
    if best is None or best_score == 0:
        return next(p["text"] for p in POLICY_SNIPPETS if p["id"] == "MISSING-EVIDENCE")
    return best["text"]


# ---------------------------------------------------------------------------
# App state (loaded once at startup)
# ---------------------------------------------------------------------------

class AppState:
    explainer: Optional[DisputeExplainer] = None
    nli_checker: Optional[ContradictionChecker] = None
    similar_case_index: Optional[SimilarCaseIndex] = None
    metadata: Optional[Dict] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (ARTIFACT_DIR / "ensemble_model.joblib").exists():
        raise RuntimeError(
            f"No trained model found in {ARTIFACT_DIR.resolve()}. "
            f"Run `python3 model.py dispute_dataset_300.json` first."
        )

    print("Loading NLI checker ...")
    state.nli_checker = ContradictionChecker()

    print("Loading trained ensemble + SHAP explainers ...")
    state.explainer = DisputeExplainer(ARTIFACT_DIR)
    state.metadata = state.explainer.metadata

    print("Building similar-case index ...")
    if DATASET_PATH.exists():
        with open(DATASET_PATH) as f:
            dataset_cases = json.load(f)
        state.similar_case_index = SimilarCaseIndex(dataset_cases)
    else:
        print(f"[warn] {DATASET_PATH} not found — similar-case retrieval will return nothing.")
        state.similar_case_index = SimilarCaseIndex([])

    print("Startup complete.")
    yield
    print("Shutting down.")


app = FastAPI(title="Dispute Resolution API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite / CRA dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DisputeCaseRequest(BaseModel):
    case_id: Optional[str] = Field(default=None, description="If omitted, one is generated.")
    dispute_reason_category: str = Field(default="unknown")
    customer_statement: str
    merchant_statement: str
    evidence_customer: List[str] = Field(default_factory=list)
    evidence_merchant: List[str] = Field(default_factory=list)


class ResolveResponse(BaseModel):
    case_id: str
    prediction: str
    outcome: str
    action: str
    confidence: float
    custPct: float
    merchPct: float
    class_probabilities: Dict[str, float]
    ensemble_weights: Dict[str, float]
    features: List[Dict]
    similar: List[Dict]
    policy: str
    audit: Dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.explainer is not None,
        "classes": state.metadata["classes"] if state.metadata else None,
        "n_training_cases": state.metadata["n_cases"] if state.metadata else None,
    }


@app.post("/resolve", response_model=ResolveResponse)
def resolve(request: DisputeCaseRequest):
    if state.explainer is None or state.nli_checker is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    case = request.model_dump()
    if not case.get("case_id"):
        case["case_id"] = f"DC-LIVE-{abs(hash(case['customer_statement'])) % 100000}"

    try:
        explanation = state.explainer.explain_case(case, state.nli_checker, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    decision = decision_engine(explanation["prediction"], explanation["confidence"])
    split = to_two_sided_split(explanation["class_probabilities"])
    features = to_frontend_features(explanation)
    similar = state.similar_case_index.query(
        request.customer_statement, request.merchant_statement, top_k=3
    )
    policy = match_policy(case)

    return ResolveResponse(
        case_id=case["case_id"],
        prediction=explanation["prediction"],
        outcome=decision["outcome"],
        action=decision["action"],
        confidence=explanation["confidence"],
        custPct=split["custPct"],
        merchPct=split["merchPct"],
        class_probabilities=explanation["class_probabilities"],
        ensemble_weights=explanation["ensemble_weights"],
        features=features,
        similar=similar,
        policy=policy,
        audit={
            "case": case,
            "explanation": explanation,
            "decision": decision,
        },
    )


@app.get("/fairness")
def fairness():
    """Real numbers from model.py's held-out test evaluation, saved at
    training time — not recomputed live, and not fabricated."""
    if state.metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return {
        "n_training_cases": state.metadata["n_cases"],
        "classes": state.metadata["classes"],
        "ensemble_weights": state.metadata["ensemble_weights"],
        "ensemble_test_metrics": state.metadata["ensemble_test_metrics"],
        "per_model_test_metrics": state.metadata["per_model_test_metrics"],
        "baseline_comparison": state.metadata["baseline_comparison"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
