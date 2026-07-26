// components/PipelineScreen.jsx
// /resolve runs the whole pipeline server-side in one call — there's no
// stage-by-stage streaming to listen to. This screen ticks through the real
// stage list on a timer to keep the docket legible while the request is in
// flight, then snaps to "done" the instant the response (or an error)
// actually arrives, rather than pretending to know server-side timing.

import { useEffect, useState } from "react";
import Card from "./Card.jsx";
import Stepper from "./Stepper.jsx";

const STEPS = [
  { key: "parse", label: "Parsing statements", detail: "spaCy NER + regex evidence extraction" },
  { key: "nli", label: "Checking for contradictions", detail: "NLI cross-encoder, local inference" },
  { key: "features", label: "Building feature matrix", detail: "85 features across 6 groups" },
  { key: "ensemble", label: "Scoring with the ensemble", detail: "XGBoost + LightGBM + CatBoost" },
  { key: "shap", label: "Computing SHAP attributions", detail: "weighted across all three models" },
  { key: "decision", label: "Applying decision thresholds", detail: "auto-resolve / recommend / escalate" },
  { key: "similar", label: "Retrieving similar cases", detail: "TF-IDF + cosine similarity" },
  { key: "policy", label: "Matching policy references", detail: "" },
];

const TICK_MS = 550;

export default function PipelineScreen({ status, error, caseId, onRetry, onBack, onContinue }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (status !== "running") return;
    const id = setInterval(() => {
      setIndex((i) => Math.min(i + 1, STEPS.length - 1));
    }, TICK_MS);
    return () => clearInterval(id);
  }, [status]);

  useEffect(() => {
    if (status === "done") setIndex(STEPS.length);
  }, [status]);

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <p className="font-mono text-xs tracking-[0.18em] uppercase text-amber-bright mb-1">
          Case {caseId}
        </p>
        <h1 className="font-display text-3xl font-semibold text-paper">
          {status === "error" ? "Pipeline Interrupted" : "Running Resolution Pipeline"}
        </h1>
      </div>

      <Card>
        <Stepper steps={STEPS} currentIndex={index} />

        {status === "done" && (
          <div className="mt-5 pt-4 border-t border-paper-line flex justify-end">
            <button
              onClick={onContinue}
              className="px-5 py-2.5 rounded-sm bg-ink-950 text-paper font-medium text-sm tracking-wide hover:bg-ink-800 transition-colors"
            >
              View Explainability →
            </button>
          </div>
        )}

        {status === "error" && (
          <div className="mt-5 pt-4 border-t border-paper-line">
            <p className="font-mono text-xs text-stamp-red bg-stamp-red/10 border border-stamp-red/40 rounded-sm px-3 py-2 mb-3">
              {error || "The pipeline did not complete."}
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={onBack}
                className="px-4 py-2 rounded-sm border border-ink-950/20 text-ink-800 text-sm hover:bg-ink-950/5"
              >
                Back to Case
              </button>
              <button
                onClick={onRetry}
                className="px-4 py-2 rounded-sm bg-ink-950 text-paper text-sm hover:bg-ink-800"
              >
                Retry
              </button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
