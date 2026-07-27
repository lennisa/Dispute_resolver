// components/DisputeConsole.jsx
// Owns all console state and talks to the real backend via api/disputeApi.js.
// This replaces the old mock `runPipeline()` — the shape /resolve returns
// (`custPct`, `merchPct`, `features`, `action`, `outcome`, `similar`,
// `policy`, `audit`) is passed straight through to the screens unchanged.

import { useState, useCallback } from "react";
import { resolveDispute, DisputeApiError } from "../api/disputeApi.js";
import SubmissionScreen from "./SubmissionScreen.jsx";
import PipelineScreen from "./PipelineScreen.jsx";
import ExplainabilityScreen from "./ExplainabilityScreen.jsx";
import DecisionScreen from "./DecisionScreen.jsx";

const SCREENS = {
  SUBMISSION: "submission",
  PIPELINE: "pipeline",
  EXPLAINABILITY: "explainability",
  DECISION: "decision",
};

export default function DisputeConsole() {
  const [screen, setScreen] = useState(SCREENS.SUBMISSION);
  const [caseInput, setCaseInput] = useState(null);
  const [result, setResult] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState("running"); // running | done | error
  const [pipelineError, setPipelineError] = useState(null);
  const [submitError, setSubmitError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const runResolution = useCallback(async (input) => {
    setPipelineStatus("running");
    setPipelineError(null);
    try {
      const response = await resolveDispute(input);
      setResult(response);
      setPipelineStatus("done");
    } catch (err) {
      const message =
        err instanceof DisputeApiError
          ? `${err.message}${err.detail ? ` — ${err.detail}` : ""}`
          : "Unexpected error while resolving the case.";
      setPipelineError(message);
      setPipelineStatus("error");
    }
  }, []);

  const handleSubmit = async (input) => {
    setSubmitting(true);
    setSubmitError(null);
    setCaseInput(input);
    setScreen(SCREENS.PIPELINE);
    setSubmitting(false);
    await runResolution(input);
  };

  const handleRetry = () => {
    if (caseInput) runResolution(caseInput);
  };

  const handleNewCase = () => {
    setCaseInput(null);
    setResult(null);
    setPipelineStatus("running");
    setPipelineError(null);
    setSubmitError(null);
    setScreen(SCREENS.SUBMISSION);
  };

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="max-w-5xl mx-auto mb-8 flex items-center justify-end">
        <div className="flex items-center gap-4">
          {screen !== SCREENS.SUBMISSION && (
            <button
              onClick={handleNewCase}
              className="font-mono text-xs text-ink-faint hover:text-paper underline underline-offset-2"
            >
              new case
            </button>
          )}
          <img src="/assets/logo2.webp" alt="Logo" className="h-16 w-auto" />
        </div>
      </div>

      {screen === SCREENS.SUBMISSION && (
        <SubmissionScreen onSubmit={handleSubmit} submitting={submitting} error={submitError} />
      )}

      {screen === SCREENS.PIPELINE && (
        <PipelineScreen
          status={pipelineStatus}
          error={pipelineError}
          caseId={caseInput?.case_id}
          onRetry={handleRetry}
          onBack={() => setScreen(SCREENS.SUBMISSION)}
          onContinue={() => setScreen(SCREENS.EXPLAINABILITY)}
        />
      )}

      {screen === SCREENS.EXPLAINABILITY && (
        <ExplainabilityScreen
          result={result}
          caseId={caseInput?.case_id}
          onBack={() => setScreen(SCREENS.PIPELINE)}
          onContinue={() => setScreen(SCREENS.DECISION)}
        />
      )}

      {screen === SCREENS.DECISION && (
        <DecisionScreen result={result} caseId={caseInput?.case_id} onNewCase={handleNewCase} />
      )}
    </div>
  );
}
