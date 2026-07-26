// components/Card.jsx
// Base "dossier page" panel: paper-colored surface, hairline rule under an
// optional eyebrow+title header, ink-navy body around it. Every screen is
// built from one or more of these so the ledger identity stays consistent
// without each screen re-deriving its own card chrome.

export default function Card({
  eyebrow,
  title,
  action,
  children,
  className = "",
  bodyClassName = "",
}) {
  return (
    <section
      className={`bg-paper text-ink-950 rounded-sm shadow-ledger border border-paper-line/60 ${className}`}
    >
      {(eyebrow || title || action) && (
        <header className="flex items-start justify-between gap-4 px-6 pt-5 pb-4 border-b border-paper-line">
          <div>
            {eyebrow && (
              <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-600/70 mb-1">
                {eyebrow}
              </p>
            )}
            {title && (
              <h2 className="font-display text-xl font-semibold text-ink-950 leading-snug">
                {title}
              </h2>
            )}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </header>
      )}
      <div className={`px-6 py-5 ${bodyClassName}`}>{children}</div>
    </section>
  );
}
