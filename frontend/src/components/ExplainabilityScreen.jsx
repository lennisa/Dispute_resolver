// components/ExplainabilityScreen.jsx
// The evidence page of the dossier: how the ensemble split its vote between
// the two sides, and the human-readable features that drove it. Ensemble
// SHAP here is a weighted average across three TreeExplainer outputs, not
// an exact decomposition — the caveat is stated plainly rather than implying
// more precision than the number has.
//
// The 10ish factors are clubbed into a handful of accordion categories
// (Evidence & Documents, Customer/Merchant Text Analysis, Case Context)
// rather than one long flat list. main.py doesn't send a category field, so
// this groups client-side by keyword in the feature label — reasonable for
// display, but if the backend ever adds a real `group` field, swap the
// categorize() heuristic below for that field directly.

import Card from "./Card.jsx";
import ScaleBar from "./ScaleBar.jsx";
import FeatureList from "./FeatureList.jsx";
import Accordion from "./Accordion.jsx";

const CATEGORY_ORDER = [
  "Evidence & Documents",
  "Customer Text Analysis",
  "Merchant Text Analysis",
  "Case Context",
];

function categorize(label = "") {
  if (/evidence|document|reliab/i.test(label)) return "Evidence & Documents";
  if (/customer/i.test(label)) return "Customer Text Analysis";
  if (/merchant/i.test(label)) return "Merchant Text Analysis";
  return "Case Context";
}

function groupFeatures(features) {
  const groups = {};
  for (const f of features) {
    const cat = categorize(f.label);
    (groups[cat] ||= []).push(f);
  }
  return groups;
}

export default function ExplainabilityScreen({
  result,
  caseId,
  onContinue,
  onBack,
}) {
  const { custPct = 0, merchPct = 0, features = [] } = result || {};
  const groups = groupFeatures(features);
  const maxWeight = Math.max(
    ...features.map((f) => Math.abs(f.weight)),
    0.0001,
  );
  const presentCategories = CATEGORY_ORDER.filter((cat) => groups[cat]?.length);

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6 flex items-end justify-between gap-4">
        <div>
          <p className="font-mono text-xs tracking-[0.18em] uppercase text-amber-bright mb-1">
            Case {caseId}
          </p>
          <h1 className="font-display text-3xl font-semibold text-paper">
            Explainability Dossier
          </h1>
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
          <div>
            {presentCategories.map((cat, i) => (
              <Accordion
                key={cat}
                title={cat}
                count={groups[cat].length}
                defaultOpen={i === 0}
              >
                <FeatureList features={groups[cat]} maxWeight={maxWeight} />
              </Accordion>
            ))}
          </div>
          <p className="mt-4 pt-3 border-t border-paper/15 font-mono text-[11px] leading-relaxed text-paper/50">
            These weights are a weighted average of SHAP attributions across
            XGBoost, LightGBM, and CatBoost .
          </p>
        </Card>

        <div className="flex justify-end">
          <button
            onClick={onContinue}
            className="px-5 py-2.5 rounded-sm bg-blue-500 text-white font-medium text-sm tracking-wide hover:bg-blue-400 transition-colors"
          >
            View Decision →
          </button>
        </div>
      </div>
    </div>
  );
}
