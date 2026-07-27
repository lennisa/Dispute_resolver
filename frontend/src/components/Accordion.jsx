// components/Accordion.jsx
// A plain <details>/<summary> accordion — no JS state needed, and it's
// accessible by default. `group`/`group-open:` drives the chevron rotation
// off the element's own open attribute.

export default function Accordion({ title, count, defaultOpen = false, children }) {
  return (
    <details className="group border-b border-paper/15 last:border-b-0" open={defaultOpen}>
      <summary className="flex items-center justify-between gap-3 cursor-pointer select-none list-none py-3 [&::-webkit-details-marker]:hidden">
        <span className="font-display text-base font-semibold text-paper">{title}</span>
        <span className="flex items-center gap-2 shrink-0">
          {count != null && (
            <span className="font-mono text-[11px] tabular text-paper/50">{count}</span>
          )}
          <svg
            width="14"
            height="14"
            viewBox="0 0 16 16"
            fill="none"
            className="text-paper/70 transition-transform duration-200 group-open:rotate-180"
            aria-hidden="true"
          >
            <path
              d="M4 6l4 4 4-4"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </summary>
      <div className="pb-4">{children}</div>
    </details>
  );
}
