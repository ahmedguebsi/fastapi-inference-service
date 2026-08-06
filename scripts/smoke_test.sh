#!/usr/bin/env bash
#
# Smoke test a running inference service.
#
# Verifies the contract a deploy must satisfy before traffic is shifted onto it:
# probes respond, predictions work, errors use the standard envelope, metrics
# are exposed, and request correlation flows through.
#
# Usage:
#   ./scripts/smoke_test.sh [BASE_URL] [API_KEY]
#   ./scripts/smoke_test.sh http://localhost:8000
#
# Exits non-zero on the first failure, so it drops straight into a deployment
# pipeline as a gate.

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-${INFERENCE_API_KEY:-}}"

PASS=0
FAIL=0

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[90m'; RESET=$'\033[0m'
else
  GREEN=""; RED=""; DIM=""; RESET=""
fi

command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }

auth_args=()
if [[ -n "${API_KEY}" ]]; then
  auth_args=(-H "X-API-Key: ${API_KEY}")
fi

ok()   { printf '%s  ok  %s%s\n' "${GREEN}" "${RESET}" "$1"; PASS=$((PASS + 1)); }
bad()  { printf '%s fail %s%s\n' "${RED}" "${RESET}" "$1"; FAIL=$((FAIL + 1)); }
note() { printf '%s       %s%s\n' "${DIM}" "$1" "${RESET}"; }

# Returns the HTTP status of a request; body is written to $BODY_FILE.
BODY_FILE="$(mktemp)"
trap 'rm -f "${BODY_FILE}"' EXIT

request() {
  local method="$1" path="$2" data="${3:-}"
  local args=(-sS -o "${BODY_FILE}" -w '%{http_code}' -X "${method}" "${auth_args[@]}")
  if [[ -n "${data}" ]]; then
    args+=(-H 'Content-Type: application/json' -d "${data}")
  fi
  curl "${args[@]}" "${BASE_URL}${path}" 2>/dev/null || echo "000"
}

expect_status() {
  local label="$1" expected="$2" actual="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    ok "${label}"
  else
    bad "${label} (expected ${expected}, got ${actual})"
    note "$(head -c 300 "${BODY_FILE}")"
  fi
}

echo "smoke testing ${BASE_URL}"
echo

# -- probes -----------------------------------------------------------------
expect_status "liveness /health" 200 "$(request GET /health)"
expect_status "readiness /ready" 200 "$(request GET /ready)"

# -- prediction -------------------------------------------------------------
status="$(request POST /v1/predict '{"text":"this is excellent and works perfectly"}')"
expect_status "POST /v1/predict" 200 "${status}"
if [[ "${status}" == "200" ]]; then
  if grep -q '"label"' "${BODY_FILE}" && grep -q '"scores"' "${BODY_FILE}"; then
    ok "prediction body has label and scores"
  else
    bad "prediction body is missing expected fields"
  fi
fi

expect_status "POST /v1/predict/batch" 200 \
  "$(request POST /v1/predict/batch '{"texts":["great","awful"]}')"

# -- validation and error envelope ------------------------------------------
status="$(request POST /v1/predict '{}')"
expect_status "empty body rejected" 422 "${status}"
if grep -q '"code"' "${BODY_FILE}" && grep -q '"request_id"' "${BODY_FILE}"; then
  ok "errors use the standard envelope"
else
  bad "error envelope is missing code/request_id"
fi

expect_status "unknown path is 404" 404 "$(request GET /nope)"
expect_status "wrong method is 405" 405 "$(request GET /v1/predict)"

# -- observability ----------------------------------------------------------
status="$(request GET /metrics)"
expect_status "GET /metrics" 200 "${status}"
if [[ "${status}" == "200" ]]; then
  if grep -q 'inference_http_requests_total' "${BODY_FILE}"; then
    ok "metrics include request counters"
  else
    bad "metrics are missing inference_http_requests_total"
  fi
  if grep -q 'inference_ready 1' "${BODY_FILE}"; then
    ok "service reports ready in metrics"
  else
    bad "inference_ready is not 1"
  fi
fi

expect_status "GET /v1/meta" 200 "$(request GET /v1/meta)"
if grep -q 'artifact_sha256' "${BODY_FILE}"; then
  ok "meta reports the model artifact hash"
else
  bad "meta is missing artifact_sha256"
fi

# -- request correlation ----------------------------------------------------
supplied="smoke-test-$(date +%s)"
returned="$(curl -sS -o /dev/null -D - -X POST \
  "${auth_args[@]}" \
  -H 'Content-Type: application/json' \
  -H "X-Request-ID: ${supplied}" \
  -d '{"text":"correlation check"}' \
  "${BASE_URL}/v1/predict" 2>/dev/null \
  | tr -d '\r' | awk -F': ' 'tolower($1)=="x-request-id" {print $2}')"

if [[ "${returned}" == "${supplied}" ]]; then
  ok "client-supplied X-Request-ID is honoured"
else
  bad "X-Request-ID not echoed (sent '${supplied}', got '${returned}')"
fi

# -- summary ----------------------------------------------------------------
echo
if (( FAIL > 0 )); then
  printf '%s%d passed, %d FAILED%s\n' "${RED}" "${PASS}" "${FAIL}" "${RESET}"
  exit 1
fi
printf '%s%d checks passed%s\n' "${GREEN}" "${PASS}" "${RESET}"
