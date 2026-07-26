// components/DecisionBadge.jsx
// The signature element of the ledger identity: a rubber-stamp badge, as if
// a caseworker inked the outcome onto the dossier page. Built as inline SVG
// (not a background image) so its ink color can be driven directly by the
// decision engine's `action` field, and so it stays crisp at any size.

import { actionStyles } from "../theme/tokens.js";

const FALLBACK = { label: "Pending", color: "#7385a6", ink: "#f3ecda" };

export default function DecisionBadge({ action, outcome, size = 128, rotation = -6 }) {
  const style = actionStyles[action] || FALLBACK;
  const uid = `${action || "pending"}`;

  return (
    <div
      className="inline-flex flex-col items-center justify-center select-none"
      style={{ width: size, height: size, transform: `rotate(${rotation}deg)` }}
      role="img"
      aria-label={`Decision: ${style.label}${outcome ? `, outcome ${outcome}` : ""}`}
    >
      <svg viewBox="0 0 200 200" width={size} height={size}>
        <defs>
          <filter id={`stamp-rough-${uid}`}>
            <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" result="noise" />
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="3.5" />
          </filter>
        </defs>
        <g filter={`url(#stamp-rough-${uid})`} opacity="0.92">
          <circle cx="100" cy="100" r="92" fill="none" stroke={style.color} strokeWidth="5" />
          <circle cx="100" cy="100" r="78" fill="none" stroke={style.color} strokeWidth="2" />
          <path
            id={`stamp-arc-top-${uid}`}
            d="M 30 100 A 70 70 0 0 1 170 100"
            fill="none"
            stroke="none"
          />
        </g>
        <text
          x="100"
          y="94"
          textAnchor="middle"
          fill={style.color}
          fontFamily="'IBM Plex Serif', serif"
          fontWeight="700"
          fontSize="21"
          letterSpacing="0.5"
          filter={`url(#stamp-rough-${uid})`}
        >
          {style.label.split(" ").map((word, i, arr) => (
            <tspan key={i} x="100" dy={i === 0 ? (arr.length > 1 ? "-8" : "0") : "22"}>
              {word.toUpperCase()}
            </tspan>
          ))}
        </text>
        {outcome && (
          <text
            x="100"
            y="132"
            textAnchor="middle"
            fill={style.color}
            fontFamily="'IBM Plex Mono', monospace"
            fontSize="9"
            letterSpacing="1.5"
            filter={`url(#stamp-rough-${uid})`}
          >
            {outcome.replace(/_/g, " ").toUpperCase()}
          </text>
        )}
      </svg>
    </div>
  );
}
