// theme/tokens.js
// The raw values behind tailwind.config.js's `theme.extend`. Tailwind classes
// (bg-ink-900, text-amber, etc.) are the primary way components use these —
// reach for this file only where you need a literal hex/rem value: inline
// SVG fills, canvas/chart gradients, or computed styles (e.g. a bar's width
// driven by a SHAP weight). Keeping one file means the palette never drifts
// between the CSS and the handful of places that can't use Tailwind classes.

export const colors = {
  ink: {
    950: "#0b1220",
    900: "#101a2c",
    800: "#182338",
    700: "#212f49",
    600: "#334463",
    faint: "#7385a6",
  },
  paper: {
    DEFAULT: "#f3ecda",
    dim: "#e8dfc7",
    line: "#d8cba7",
  },
  amber: {
    DEFAULT: "#b8863b",
    bright: "#d6a355",
    dim: "#8a6529",
  },
  stamp: {
    green: "#3f6b4a",
    red: "#8b3a3a",
    navy: "#26405f",
  },
};

export const fonts = {
  display: "'IBM Plex Serif', Georgia, serif",
  body: "'IBM Plex Sans', system-ui, sans-serif",
  mono: "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace",
};

// Canonical labels/colors for the two ledger sides. custPct/merchPct from the
// API always refer to these two — components should pull the label/color
// from here rather than hardcoding "Card Member" / "Merchant" strings.
export const sides = {
  customer: { label: "Card Member", color: colors.stamp.navy, key: "custPct" },
  merchant: { label: "Merchant", color: colors.amber.dim, key: "merchPct" },
};

// Decision engine actions (main.py) → stamp badge styling.
export const actionStyles = {
  auto_resolve: { label: "Auto-Resolved", color: colors.stamp.green, ink: "#e9f2ea" },
  recommend: { label: "Recommended", color: colors.amber.dim, ink: "#f8eedd" },
  escalate: { label: "Escalated", color: colors.stamp.red, ink: "#f6e8e8" },
};

export const disputeCategories = [
  { value: "item_not_received", label: "Item Not Received" },
  { value: "delivered_but_disputed", label: "Delivered But Disputed" },
  { value: "duplicate_charge", label: "Duplicate Charge" },
  { value: "subscription_cancellation", label: "Subscription Cancellation" },
  { value: "service_not_rendered", label: "Service Not Rendered" },
  { value: "not_as_described", label: "Not As Described" },
  { value: "unauthorized_transaction", label: "Unauthorized Transaction" },
  { value: "ambiguous_conflicting", label: "Ambiguous / Conflicting" },
  { value: "missing_evidence", label: "Missing Evidence" },
];
