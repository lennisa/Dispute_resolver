// components/ExplainabilityScreen.jsx
// The evidence page of the dossier: how the ensemble split its vote between
// the two sides, and the human-readable features that drove it. Ensemble
// SHAP here is a weighted average across three TreeExplainer outputs, not
// an exact decomposition — the caveat is stated plainly rather than implying
// more precision than the number has.

import Card from "./Card.jsx";
import ScaleBar from "./ScaleBar.jsx";
import FeatureList from "./FeatureList.jsx";

export default function ExplainabilityScreen({ result, caseId, onContinue, onBack }) {
  const { custPct = 0, merchPct = 0, features = [] } = result || {};

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6 flex items-end justify-between gap-4">
        <div>
          <p className="font-mono text-xs tracking-[0.18em] uppercase text-amber-bright mb-1">
            Case {caseId}
          </p>
          <h1 className="font-display text-3xl font-semibold text-paper">Explainability Dossier</h1>
        </div>
        <button
          onClick={onBack}
          className="font-mono text-xs text-ink-faint hover:text-paper underline underline-offset-2"
        >
          ← back
        </button>
      </div>

      <div className="space-y-5">
        <Card eyebrow="Ensemble Split" title="Which side the evidence favors">
          <ScaleBar custPct={custPct} merchPct={merchPct} size="lg" />
        </Card>

        <Card
          eyebrow={`${features.length} Contributing Factors`}
          title="What drove this split"
        >
          <FeatureList features={features} />
          <p className="mt-4 pt-3 border-t border-paper-line font-mono text-[11px] leading-relaxed text-ink-600">
            These weights are a weighted average of SHAP attributions across XGBoost,
            LightGBM, and CatBoost — a practical approximation of each model's influence,
            not an exact decomposition of the calibrated ensemble probability.
          </p>
        </Card>

        <div className="flex justify-end">
          <button
            onClick={onContinue}
            className="px-5 py-2.5 rounded-sm bg-ink-950 text-paper font-medium text-sm tracking-wide hover:bg-ink-800 transition-colors"
          >
            View Decision →
          </button>
        </div>
      </div>
    </div>
  );
}
