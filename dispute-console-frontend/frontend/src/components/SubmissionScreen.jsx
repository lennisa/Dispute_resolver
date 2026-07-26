// components/SubmissionScreen.jsx
// First screen: open a new case. Collects exactly the fields /resolve's
// pipeline needs (parser + NLI run on the two statements, evidence lists
// feed the reliability-scored features). Nothing here is decorative — every
// field maps to a real feature upstream.

import { useState } from "react";
import Card from "./Card.jsx";
import { disputeCategories } from "../theme/tokens.js";

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
      <label className="block font-mono text-[11px] tracking-[0.14em] uppercase text-ink-600/70 mb-1.5">
        {label}
      </label>
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
          className="flex-1 bg-white/40 border border-paper-line rounded-sm px-3 py-1.5 text-sm text-ink-950 placeholder:text-ink-faint focus:bg-white/70"
        />
        <button
          type="button"
          onClick={add}
          className="px-3 py-1.5 text-sm font-medium rounded-sm border border-ink-950/20 text-ink-800 hover:bg-ink-950/5"
        >
          Add
        </button>
      </div>
      {items.length > 0 && (
        <ul className="flex flex-wrap gap-1.5">
          {items.map((item, i) => (
            <li
              key={`${item}-${i}`}
              className="group flex items-center gap-1.5 font-mono text-xs bg-amber/10 border border-amber/40 text-amber-dim rounded-sm px-2 py-1"
            >
              {item}
              <button
                type="button"
                onClick={() => onChange(items.filter((_, idx) => idx !== i))}
                aria-label={`Remove ${item}`}
                className="text-amber-dim/70 hover:text-amber-dim"
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

const nextCaseId = () => `DC-${Date.now().toString(36).toUpperCase().slice(-6)}`;

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

  const update = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }));

  const canSubmit =
    form.customer_statement.trim().length > 0 && form.merchant_statement.trim().length > 0;

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
        <p className="font-mono text-xs tracking-[0.18em] uppercase text-amber-bright mb-1">
          New Docket Entry
        </p>
        <h1 className="font-display text-3xl font-semibold text-paper">Open a Dispute Case</h1>
        <p className="text-ink-faint text-sm mt-1.5">
          Enter both statements and any evidence on file. The pipeline runs entirely on
          submit — parsing, contradiction detection, feature extraction, and the ensemble
          decision all happen in one pass.
        </p>
      </div>

      <Card>
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block font-mono text-[11px] tracking-[0.14em] uppercase text-ink-600/70 mb-1.5">
                Case ID
              </label>
              <input
                type="text"
                value={form.case_id}
                onChange={update("case_id")}
                className="w-full bg-white/40 border border-paper-line rounded-sm px-3 py-1.5 font-mono text-sm text-ink-950 focus:bg-white/70"
              />
            </div>
            <div>
              <label className="block font-mono text-[11px] tracking-[0.14em] uppercase text-ink-600/70 mb-1.5">
                Dispute Reason
              </label>
              <select
                value={form.dispute_reason_category}
                onChange={update("dispute_reason_category")}
                className="w-full bg-white/40 border border-paper-line rounded-sm px-3 py-1.5 text-sm text-ink-950 focus:bg-white/70"
              >
                {disputeCategories.map((c) => (
                  <option key={c.value} value={c.value}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block font-mono text-[11px] tracking-[0.14em] uppercase text-ink-600/70 mb-1.5">
              Disputed Amount (optional)
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              value={form.amount}
              onChange={update("amount")}
              placeholder="0.00"
              className="w-40 bg-white/40 border border-paper-line rounded-sm px-3 py-1.5 font-mono text-sm text-ink-950 placeholder:text-ink-faint focus:bg-white/70"
            />
          </div>

          <div>
            <label className="block font-mono text-[11px] tracking-[0.14em] uppercase text-ink-600/70 mb-1.5">
              Card Member Statement
            </label>
            <textarea
              value={form.customer_statement}
              onChange={update("customer_statement")}
              required
              rows={3}
              placeholder="Describe what the card member reported…"
              className="w-full bg-white/40 border border-paper-line rounded-sm px-3 py-2 text-sm text-ink-950 placeholder:text-ink-faint focus:bg-white/70 resize-none"
            />
          </div>

          <div>
            <label className="block font-mono text-[11px] tracking-[0.14em] uppercase text-ink-600/70 mb-1.5">
              Merchant Statement
            </label>
            <textarea
              value={form.merchant_statement}
              onChange={update("merchant_statement")}
              required
              rows={3}
              placeholder="Describe the merchant's response…"
              className="w-full bg-white/40 border border-paper-line rounded-sm px-3 py-2 text-sm text-ink-950 placeholder:text-ink-faint focus:bg-white/70 resize-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <EvidenceEditor
              label="Card Member Evidence"
              items={form.evidence_customer}
              onChange={(items) => setForm((f) => ({ ...f, evidence_customer: items }))}
              placeholder="e.g. Photo of doorstep"
            />
            <EvidenceEditor
              label="Merchant Evidence"
              items={form.evidence_merchant}
              onChange={(items) => setForm((f) => ({ ...f, evidence_merchant: items }))}
              placeholder="e.g. Signed delivery confirmation"
            />
          </div>

          {error && (
            <p className="font-mono text-xs text-stamp-red bg-stamp-red/10 border border-stamp-red/40 rounded-sm px-3 py-2">
              {error}
            </p>
          )}

          <div className="pt-1 flex justify-end">
            <button
              type="submit"
              disabled={!canSubmit || submitting}
              className="px-5 py-2.5 rounded-sm bg-ink-950 text-paper font-medium text-sm tracking-wide hover:bg-ink-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? "Submitting…" : "Submit Case for Resolution"}
            </button>
          </div>
        </form>
      </Card>
    </div>
  );
}
