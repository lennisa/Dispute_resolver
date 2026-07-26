// api/disputeApi.js
// Thin wrapper around the three endpoints main.py exposes. Nothing in here
// shapes or renames the response — components read `custPct`, `merchPct`,
// `features`, `action`, `outcome`, `similar`, `policy`, `audit` exactly as
// /resolve sends them, because main.py was deliberately built to already
// match what the UI expects. If that ever stops being true, fix it here
// once rather than patching every component that reads the response.

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

class DisputeApiError extends Error {
  constructor(message, { status, detail } = {}) {
    super(message);
    this.name = "DisputeApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (networkErr) {
    throw new DisputeApiError(
      `Could not reach the backend at ${BASE_URL}. Is main.py running?`,
      { detail: networkErr.message }
    );
  }

  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = await response.text().catch(() => undefined);
    }
    throw new DisputeApiError(`${path} failed (${response.status})`, {
      status: response.status,
      detail,
    });
  }

  return response.json();
}

/**
 * Runs the full pipeline: parse → NLI → features → ensemble → SHAP →
 * decision engine → similar cases → policy match.
 *
 * @param {Object} caseInput
 * @param {string} caseInput.case_id
 * @param {string} caseInput.dispute_reason_category
 * @param {string} caseInput.customer_statement
 * @param {string} caseInput.merchant_statement
 * @param {string[]} [caseInput.evidence_customer]
 * @param {string[]} [caseInput.evidence_merchant]
 * @param {number} [caseInput.amount]
 * @returns {Promise<Object>} { custPct, merchPct, features, action, outcome, similar, policy, audit }
 */
export function resolveDispute(caseInput) {
  return request("/resolve", {
    method: "POST",
    body: JSON.stringify(caseInput),
  });
}

/**
 * Real test-set metrics read from metadata.json at training time: ensemble
 * weights, per-class metrics, confusion matrix, baseline comparison.
 */
export function getFairnessMetrics() {
  return request("/fairness", { method: "GET" });
}

/** Model-loaded health check. */
export function getHealth() {
  return request("/health", { method: "GET" });
}

export { DisputeApiError, BASE_URL };
