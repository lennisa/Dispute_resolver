// components/SubmissionScreen.jsx
// First screen: open a new case. Collects exactly the fields /resolve's
// pipeline needs (parser + NLI run on the two statements, evidence lists
// feed the reliability-scored features). Nothing here is decorative — every
// field maps to a real feature upstream. Sits on the same photo card as the
// pipeline screen, so inputs use light-on-dark styling to stay legible.

import { useState } from "react";
import Card from "./Card.jsx";
import { disputeCategories } from "../theme/tokens.js";

const FIELD =
  "w-full bg-white/10 border border-paper/25 rounded-sm px-3 py-1.5 text-sm text-paper placeholder:text-paper/35 focus:bg-white/15 focus:border-paper/45 outline-none transition-colors";
const LABEL =
  "block font-mono text-[11px] tracking-[0.14em] uppercase text-paper/55 mb-1.5";

function EvidenceEditor({ label, items, onChange, placeholder }) {
  const [draft, setDraft] = useState("");

  const add = () => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    onChange([...items, trimmed]);
    setDraft("");
  };

  return (
    <div>
      <label className={LABEL}>{label}</label>
      <div className="flex gap-2 mb-2">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              add();
            }
          }}
          placeholder={placeholder}
          className={FIELD}
        />
        <button
          type="button"
          onClick={add}
          className="px-3 py-1.5 text-sm font-medium rounded-sm border border-paper/25 text-paper hover:bg-paper/10"
        >
          Add
        </button>
      </div>
      {items.length > 0 && (
        <ul className="flex flex-wrap gap-1.5">
          {items.map((item, i) => (
            <li
              key={`${item}-${i}`}
              className="group flex items-center gap-1.5 font-mono text-xs bg-blue-500/15 border border-blue-400/40 text-blue-200 rounded-sm px-2 py-1"
            >
              {item}
              <button
                type="button"
                onClick={() => onChange(items.filter((_, idx) => idx !== i))}
                aria-label={`Remove ${item}`}
                className="text-blue-200/70 hover:text-blue-100"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const nextCaseId = () =>
  `DC-${Date.now().toString(36).toUpperCase().slice(-6)}`;

export default function SubmissionScreen({ onSubmit, submitting, error }) {
  const [form, setForm] = useState({
    case_id: nextCaseId(),
    dispute_reason_category: disputeCategories[0].value,
    amount: "",
    customer_statement: "",
    merchant_statement: "",
    evidence_customer: [],
    evidence_merchant: [],
  });

  const update = (field) => (e) =>
    setForm((f) => ({ ...f, [field]: e.target.value }));

  const canSubmit =
    form.customer_statement.trim().length > 0 &&
    form.merchant_statement.trim().length > 0;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canSubmit || submitting) return;
    onSubmit({
      ...form,
      amount: form.amount === "" ? undefined : Number(form.amount),
    });
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <h1 className="font-display text-3xl font-semibold text-paper">
          Open a Dispute Case
        </h1>
        <p className="text-ink-faint text-sm mt-1.5">Fill in the details.</p>
      </div>

      <Card photo="/assets/pic2.avif">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={LABEL}>Case ID</label>
              <input
                type="text"
                value={form.case_id}
                onChange={update("case_id")}
                className={`${FIELD} font-mono`}
              />
            </div>
            <div>
              <label className={LABEL}>Dispute Reason</label>
              <select
                value={form.dispute_reason_category}
                onChange={update("dispute_reason_category")}
                className={FIELD}
              >
                {disputeCategories.map((c) => (
                  <option
                    key={c.value}
                    value={c.value}
                    className="bg-ink-900 text-paper"
                  >
                    {c.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className={LABEL}>Disputed Amount (optional)</label>
            <input
              type="number"
              step="0.01"
              min="0"
              value={form.amount}
              onChange={update("amount")}
              placeholder="0.00"
              className={`${FIELD} w-40 font-mono`}
            />
          </div>

          <div>
            <label className={LABEL}>Card Member Statement</label>
            <textarea
              value={form.customer_statement}
              onChange={update("customer_statement")}
              required
              rows={3}
              placeholder="Describe what the card member reported…"
              className={`${FIELD} resize-none`}
            />
          </div>

          <div>
            <label className={LABEL}>Merchant Statement</label>
            <textarea
              value={form.merchant_statement}
              onChange={update("merchant_statement")}
              required
              rows={3}
              placeholder="Describe the merchant's response…"
              className={`${FIELD} resize-none`}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <EvidenceEditor
              label="Card Member Evidence"
              items={form.evidence_customer}
              onChange={(items) =>
                setForm((f) => ({ ...f, evidence_customer: items }))
              }
              placeholder="e.g. Photo of doorstep"
            />
            <EvidenceEditor
              label="Merchant Evidence"
              items={form.evidence_merchant}
              onChange={(items) =>
                setForm((f) => ({ ...f, evidence_merchant: items }))
              }
              placeholder="e.g. Signed delivery confirmation"
            />
          </div>

          {error && (
            <p className="font-mono text-xs text-paper bg-stamp-red/25 border border-stamp-red/50 rounded-sm px-3 py-2">
              {error}
            </p>
          )}

          <div className="pt-1 flex justify-end">
            <button
              type="submit"
              disabled={!canSubmit || submitting}
              className="px-5 py-2.5 rounded-sm bg-blue-500 text-white font-medium text-sm tracking-wide hover:bg-blue-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? "Submitting…" : "Submit Case for Resolution"}
            </button>
          </div>
        </form>
      </Card>
    </div>
  );
}
