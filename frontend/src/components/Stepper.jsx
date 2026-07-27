// components/Stepper.jsx
// Shows exactly one pipeline stage at a time, centered in the card: a
// spinning blue ring with the stage's label while it runs, then just the
// word "Done" once it finishes, before the next stage takes its place.
//
// This sits directly on the raw, un-tinted pic.png (Card's photoOverlay is
// off here), so every bit of text carries its own drop-shadow instead of
// relying on a dark wash for contrast.

const TEXT_SHADOW = { textShadow: "0 1px 3px rgba(0,0,0,0.9), 0 0 12px rgba(0,0,0,0.5)" };

export default function Stepper({ steps, index, phase }) {
  const step = steps[index];
  const isDone = phase === "done";

  return (
    <div className="flex flex-col items-center justify-center text-center py-8 min-h-[160px]">
      {isDone ? (
        <>
          <span
            className="w-12 h-12 rounded-full bg-blue-500 shadow-[0_0_16px_rgba(59,130,246,0.85)]"
            aria-hidden="true"
          />
          <p className="mt-5 font-display text-2xl font-semibold text-paper" style={TEXT_SHADOW}>
            Done
          </p>
        </>
      ) : (
        <>
          <span
            className="w-12 h-12 rounded-full border-[3px] border-blue-400 border-t-transparent animate-spin shadow-[0_0_10px_rgba(59,130,246,0.55)]"
            aria-hidden="true"
          />
          <p className="mt-5 font-body text-lg text-paper" style={TEXT_SHADOW}>
            {step.label}
          </p>
          {step.detail && (
            <p className="mt-1.5 font-mono text-xs text-blue-300" style={TEXT_SHADOW}>
              {step.detail}
            </p>
          )}
        </>
      )}
      <p className="mt-6 font-mono text-[11px] tracking-[0.16em] text-paper/70" style={TEXT_SHADOW}>
        STEP {String(index + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}
      </p>
    </div>
  );
}
