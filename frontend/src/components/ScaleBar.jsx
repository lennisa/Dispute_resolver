// components/ScaleBar.jsx
// Renders custPct/merchPct as a single split bar — a ledger "balance" between
// the two sides of the case, rather than two separate progress bars, so the
// tension between them reads at a glance.

import { sides } from "../theme/tokens.js";

export default function ScaleBar({ custPct, merchPct, size = "md" }) {
  const total = custPct + merchPct || 1;
  const custWidth = (custPct / total) * 100;
  const height = size === "lg" ? "h-8" : "h-4";

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5 font-mono text-xs">
        <span className="text-ink-700">
          {sides.customer.label} <span className="tabular font-semibold">{custPct.toFixed(1)}%</span>
        </span>
        <span className="text-ink-700">
          {sides.merchant.label} <span className="tabular font-semibold">{merchPct.toFixed(1)}%</span>
        </span>
      </div>
      <div className="relative">
        <div
          className={`w-full ${height} rounded-sm overflow-hidden border border-ink-950/15 flex bg-paper-dim`}
          role="img"
          aria-label={`${sides.customer.label} ${custPct.toFixed(1)} percent, ${sides.merchant.label} ${merchPct.toFixed(1)} percent`}
        >
          <div
            className="h-full transition-all duration-500 ease-out"
            style={{ width: `${custWidth}%`, backgroundColor: sides.customer.color }}
          />
          <div
            className="h-full flex-1 transition-all duration-500 ease-out"
            style={{ backgroundColor: sides.merchant.color }}
          />
        </div>
        {/* fulcrum mark at the split point */}
        <div
          className="absolute top-full w-px h-2 bg-ink-950/40 transition-all duration-500 ease-out"
          style={{ left: `${custWidth}%` }}
          aria-hidden="true"
        />
      </div>
    </div>
  );
}
