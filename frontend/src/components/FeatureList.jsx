// components/FeatureList.jsx
// Renders /resolve's `features: [{label, side, weight}]` as dossier line
// entries — each a short explanation with an amber-tinted bar sized to its
// SHAP weight, colored by which side of the case it favors. This is the
// weighted average across all three base models' TreeExplainer output, so
// treat weight as "relative influence," not a precise probability delta.

import { sides } from "../theme/tokens.js";

export default function FeatureList({ features = [], limit }) {
  if (!features.length) {
    return <p className="font-mono text-sm text-ink-faint">No contributing factors returned.</p>;
  }

  const maxWeight = Math.max(...features.map((f) => Math.abs(f.weight)), 0.0001);
  const shown = limit ? features.slice(0, limit) : features;

  return (
    <ul className="divide-y divide-paper-line">
      {shown.map((f, i) => {
        const side = f.side === "merchant" ? sides.merchant : sides.customer;
        const barPct = (Math.abs(f.weight) / maxWeight) * 100;
        return (
          <li key={`${f.label}-${i}`} className="py-2.5 first:pt-0 last:pb-0">
            <div className="flex items-start justify-between gap-3 mb-1">
              <p className="text-sm text-ink-950 leading-snug">{f.label}</p>
              <span
                className="shrink-0 font-mono text-[11px] tabular px-1.5 py-0.5 rounded-sm border"
                style={{
                  color: side.color,
                  borderColor: `${side.color}55`,
                  backgroundColor: `${side.color}14`,
                }}
              >
                {side.label}
              </span>
            </div>
            <div className="w-full h-1.5 rounded-sm bg-paper-dim overflow-hidden">
              <div
                className="h-full bg-amber transition-all duration-500 ease-out"
                style={{ width: `${barPct}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
