// components/Stepper.jsx
// Renders the pipeline as a docket: numbered entries in a register, each
// stamped "done" / actively ticking / pending. The numbers earn their place
// here — /resolve really does run these stages in this order server-side.

const STATUS_STYLE = {
  done: "text-stamp-green border-stamp-green/50 bg-stamp-green/10",
  active: "text-amber-bright border-amber/60 bg-amber/10",
  pending: "text-ink-faint border-ink-600/40 bg-transparent",
};

export default function Stepper({ steps, currentIndex }) {
  return (
    <ol className="divide-y divide-paper-line">
      {steps.map((step, i) => {
        const status = i < currentIndex ? "done" : i === currentIndex ? "active" : "pending";
        return (
          <li key={step.key} className="flex items-center gap-4 py-3 first:pt-0 last:pb-0">
            <span
              className={`shrink-0 w-7 h-7 rounded-full border font-mono text-xs flex items-center justify-center transition-colors duration-300 ${STATUS_STYLE[status]}`}
              aria-hidden="true"
            >
              {status === "done" ? "✓" : String(i + 1).padStart(2, "0")}
            </span>
            <div className="flex-1 min-w-0">
              <p
                className={`font-body text-sm leading-tight ${
                  status === "pending" ? "text-ink-600" : "text-ink-950"
                }`}
              >
                {step.label}
              </p>
              {status === "active" && step.detail && (
                <p className="font-mono text-[11px] text-amber-dim mt-0.5">{step.detail}</p>
              )}
            </div>
            {status === "active" && (
              <span
                className="w-1.5 h-1.5 rounded-full bg-amber animate-pulse shrink-0"
                aria-label="in progress"
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}
