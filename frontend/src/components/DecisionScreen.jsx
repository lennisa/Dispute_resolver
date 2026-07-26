// components/DecisionScreen.jsx
// The final page of the dossier: the stamped decision, the similar cases and
// policy reference that support it, and the audit export. main.py's
// /resolve already returns a complete `audit` object — this just serializes
// it to a file, no extra request needed.
//
// NOTE ON SHAPE: main.py wasn't available when this was built, so `similar`
// and `policy` are read defensively (a few reasonable field names tried per
// item) rather than assuming one exact schema. If the real response uses
// different keys, adjust the two small accessor helpers below — everything
// else keeps working.

import Card from "./Card.jsx";
import DecisionBadge from "./DecisionBadge.jsx";

function pick(obj, keys, fallback = undefined) {
  for (const k of keys) {
    if (obj && obj[k] !== undefined && obj[k] !== null && obj[k] !== "") return obj[k];
  }
  return fallback;
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function SimilarCaseRow({ item }) {
  const id = pick(item, ["case_id", "id"], "—");
  const outcome = pick(item, ["outcome", "label"]);
  const similarity = pick(item, ["similarity", "score"]);
  const excerpt = pick(item, ["excerpt", "summary", "customer_statement"]);

  return (
    <li className="flex items-start justify-between gap-4 py-2.5 first:pt-0 last:pb-0">
      <div className="min-w-0">
        <p className="font-mono text-sm text-ink-950">{id}</p>
        {excerpt && <p className="text-xs text-ink-600 mt-0.5 truncate">{excerpt}</p>}
      </div>
      <div className="shrink-0 text-right">
        {outcome && (
          <p className="font-mono text-[11px] uppercase tracking-wide text-ink-700">
            {String(outcome).replace(/_/g, " ")}
          </p>
        )}
        {similarity !== undefined && (
          <p className="font-mono text-xs tabular text-amber-dim">
            {typeof similarity === "number" ? `${(similarity * 100).toFixed(0)}% match` : similarity}
          </p>
        )}
      </div>
    </li>
  );
}

export default function DecisionScreen({ result, caseId, onNewCase }) {
  const { action, outcome, similar = [], policy, audit } = result || {};

  const policyItems = Array.isArray(policy) ? policy : policy ? [policy] : [];

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <p className="font-mono text-xs tracking-[0.18em] uppercase text-amber-bright mb-1">
          Case {caseId}
        </p>
        <h1 className="font-display text-3xl font-semibold text-paper">Resolution</h1>
      </div>

      <div className="space-y-5">
        <Card bodyClassName="flex items-center gap-6">
          <DecisionBadge action={action} outcome={outcome} />
          <div>
            <p className="font-mono text-[11px] tracking-[0.14em] uppercase text-ink-600/70 mb-1">
              Decision Engine Output
            </p>
            <p className="font-display text-lg font-semibold text-ink-950 capitalize">
              {(action || "pending").replace(/_/g, " ")}
            </p>
            {outcome && (
              <p className="text-sm text-ink-700 mt-0.5 capitalize">
                Favors: {outcome.replace(/_/g, " ")}
              </p>
            )}
          </div>
        </Card>

        {similar.length > 0 && (
          <Card eyebrow="Precedent" title="Similar Cases">
            <ul className="divide-y divide-paper-line">
              {similar.map((item, i) => (
                <SimilarCaseRow key={pick(item, ["case_id", "id"], i)} item={item} />
              ))}
            </ul>
            <p className="mt-3 pt-3 border-t border-paper-line font-mono text-[11px] text-ink-600">
              Retrieved via TF-IDF + cosine similarity — matches on shared vocabulary, not
              paraphrased meaning.
            </p>
          </Card>
        )}

        {policyItems.length > 0 && (
          <Card eyebrow="Reference" title="Policy Match">
            <ul className="space-y-3">
              {policyItems.map((p, i) => (
                <li key={i} className="text-sm">
                  <p className="font-mono text-xs text-amber-dim mb-0.5">
                    {pick(p, ["reference", "id", "code"], `Policy ${i + 1}`)}
                  </p>
                  <p className="text-ink-800 leading-snug">
                    {pick(p, ["text", "summary", "description"], String(p))}
                  </p>
                </li>
              ))}
            </ul>
          </Card>
        )}

        <div className="flex justify-between items-center pt-1">
          <button
            onClick={() => downloadJson(`${caseId || "case"}-audit.json`, audit ?? result)}
            disabled={!result}
            className="px-4 py-2 rounded-sm border border-paper/30 text-paper text-sm hover:bg-paper/10 disabled:opacity-40 transition-colors"
          >
            Export Audit JSON
          </button>
          <button
            onClick={onNewCase}
            className="px-5 py-2.5 rounded-sm bg-ink-950 text-paper font-medium text-sm tracking-wide hover:bg-ink-800 transition-colors border border-paper/10"
          >
            Open New Case
          </button>
        </div>
      </div>
    </div>
  );
}
