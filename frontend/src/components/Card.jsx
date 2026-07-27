// components/Card.jsx
// Base "dossier page" panel — plain dark glass by default (ink-900 base +
// a light blur wash, no photo). Pass `photo` to put a picture behind a card
// (only Submission and Pipeline do this); everything else stays glass-only.
//
// Two photo modes:
//   - default: CSS background, `bg-contain` so nothing crops, box height
//     driven by the card's own content (used on the submission form, which
//     needs to flow to fit a long form).
//   - `matchPhotoSize`: the photo is a real <img> laid out in normal flow,
//     so the card's box is exactly the image's own size/aspect ratio — zero
//     cropping, guaranteed — with children centered on top as an overlay
//     (used on the pipeline screen, whose content is short and centered).
//
// `photoOverlay={false}` drops the dark wash entirely so the photo shows at
// full brightness.
//
// Every card also tilts toward the cursor on hover (see hooks/useTilt.js).

import useTilt from "../hooks/useTilt.js";

export default function Card({
  eyebrow,
  title,
  action,
  children,
  className = "",
  bodyClassName = "",
  photo,
  photoOverlay = true,
  matchPhotoSize = false,
}) {
  const tilt = useTilt({ maxDeg: 5 });
  const wrapperProps = {
    ref: tilt.ref,
    onMouseMove: tilt.onMouseMove,
    onMouseLeave: tilt.onMouseLeave,
    style: { ...tilt.style, transformStyle: "preserve-3d" },
  };

  if (photo && matchPhotoSize) {
    // The image sets the box's size itself (normal flow, w-full h-auto) —
    // the card can never be bigger or smaller than the picture, so nothing
    // gets cropped. Content floats centered on top instead of pushing the
    // box's height around.
    return (
      <section
        {...wrapperProps}
        className={`relative overflow-hidden rounded-sm shadow-ledger border border-paper/15 text-paper will-change-transform ${className}`}
      >
        <img src={photo} alt="" className="block w-full h-auto" />
        {photoOverlay && (
          <div className="absolute inset-0 bg-ink-950/45 backdrop-blur-[2px]" aria-hidden="true" />
        )}
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <div className={`w-full ${bodyClassName}`}>{children}</div>
        </div>
      </section>
    );
  }

  return (
    <section
      {...wrapperProps}
      className={`relative overflow-hidden rounded-sm shadow-ledger border border-paper/15 text-paper bg-ink-900 will-change-transform ${className}`}
    >
      {photo && (
        <div
          className="absolute inset-0 bg-contain bg-center bg-no-repeat"
          style={{ backgroundImage: `url(${photo})` }}
          aria-hidden="true"
        />
      )}
      {photoOverlay && (
        <div className="absolute inset-0 bg-ink-950/45 backdrop-blur-[2px]" aria-hidden="true" />
      )}
      <div className="relative z-10">
        {(eyebrow || title || action) && (
          <header className="flex items-start justify-between gap-4 px-6 pt-5 pb-4 border-b border-paper/15">
            <div>
              {eyebrow && (
                <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-amber-bright/80 mb-1">
                  {eyebrow}
                </p>
              )}
              {title && (
                <h2 className="font-display text-xl font-semibold text-paper leading-snug">
                  {title}
                </h2>
              )}
            </div>
            {action && <div className="shrink-0">{action}</div>}
          </header>
        )}
        <div className={`px-6 py-5 ${bodyClassName}`}>{children}</div>
      </div>
    </section>
  );
}
