// components/PipelineScreen.jsx
// /resolve runs the whole pipeline server-side in one call — there's no
// stage-by-stage streaming to listen to. This screen still plays the real
// stage list back one at a time (spinner → "Done" → next stage) to keep the
// wait legible, then holds on the last stage's spinner until the actual
// response arrives, and hands off to the explainability dossier on its own.

import { useEffect, useState } from "react";
import Card from "./Card.jsx";
import Stepper from "./Stepper.jsx";

const STEPS = [
  {
    key: "parse",
    label: "Parsing statements",
    detail: "spaCy NER + regex evidence extraction",
  },
  {
    key: "nli",
    label: "Checking for contradictions",
    detail: "NLI cross-encoder, local inference",
  },
  {
    key: "features",
    label: "Building feature matrix",
    detail: "85 features across 6 groups",
  },
  {
    key: "ensemble",
    label: "Scoring with the ensemble",
    detail: "XGBoost + LightGBM + CatBoost",
  },
  {
    key: "shap",
    label: "Computing SHAP attributions",
    detail: "weighted across all three models",
  },
  {
    key: "decision",
    label: "Applying decision thresholds",
    detail: "auto-resolve / recommend / escalate",
  },
  {
    key: "similar",
    label: "Retrieving similar cases",
    detail: "TF-IDF + cosine similarity",
  },
  { key: "policy", label: "Matching policy references", detail: "" },
];

const RUN_MS = 900; // how long each stage's spinner shows before flipping to "Done"
const DONE_MS = 450; // how long "Done" holds before the next stage starts
const HANDOFF_MS = 700; // how long the final "Done" holds before leaving the screen

export default function PipelineScreen({
  status,
  error,
  caseId,
  onRetry,
  onBack,
  onContinue,
}) {
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState("running"); // "running" | "done"
  const isLastStep = index === STEPS.length - 1;

  // Steps before the last one: cycle running -> done -> next, purely on a
  // timer, independent of the actual request.
  useEffect(() => {
    if (status === "error" || isLastStep) return;
    if (phase === "running") {
      const t = setTimeout(() => setPhase("done"), RUN_MS);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setIndex((i) => i + 1);
      setPhase("running");
    }, DONE_MS);
    return () => clearTimeout(t);
  }, [phase, isLastStep, status]);

  // Last step: keep spinning for real until the response actually arrives,
  // then show "Done" and auto-continue.
  useEffect(() => {
    if (status === "error" || !isLastStep) return;
    if (status === "running") {
      setPhase("running");
      return;
    }
    // status === "done"
    setPhase("done");
    const t = setTimeout(() => onContinue?.(), HANDOFF_MS);
    return () => clearTimeout(t);
  }, [status, isLastStep, onContinue]);

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <p className="font-mono text-xs tracking-[0.18em] uppercase text-amber-bright mb-1">
          Case {caseId}
        </p>
        <h1 className="font-display text-3xl font-semibold text-paper">
          {status === "error"
            ? "Pipeline Interrupted"
            : "Running Resolution Pipeline"}
        </h1>
      </div>

      <Card photo="/assets/pic2.avif" matchPhotoSize photoOverlay={false}>
        {status === "error" ? (
          <div className="py-10">
            <p className="font-mono text-xs text-paper bg-stamp-red/25 border border-stamp-red/50 rounded-sm px-3 py-2 mb-4 text-center">
              {error || "The pipeline did not complete."}
            </p>
            <div className="flex gap-2 justify-center">
              <button
                onClick={onBack}
                className="px-4 py-2 rounded-sm border border-paper/30 text-paper text-sm hover:bg-paper/10"
              >
                Back to Case
              </button>
              <button
                onClick={onRetry}
                className="px-4 py-2 rounded-sm bg-blue-500 text-white text-sm hover:bg-blue-400"
              >
                Retry
              </button>
            </div>
          </div>
        ) : (
          <Stepper steps={STEPS} index={index} phase={phase} />
        )}
      </Card>
    </div>
  );
}
