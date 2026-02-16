#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-smoke"
REPORT_DIR="${ROOT_DIR}/.hydra/reports"

log() {
  printf '%s\n' "$1"
}

fail() {
  log "FAIL: $1"
  exit 1
}

run_allow_doctor_codes() {
  set +e
  "$@"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ] && [ "$rc" -ne 2 ]; then
    fail "Command exited with unexpected code ${rc}: $*"
  fi
}

log "== Smoke install/run =="
log "Repo: ${ROOT_DIR}"

rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}" || fail "Unable to create venv"

# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

python -m pip install -U pip || fail "pip upgrade failed"
python -m pip install -e "${ROOT_DIR}" || fail "editable install failed"

log "Step 1: continuum doctor --list-checks"
continuum doctor --list-checks >/tmp/continuum_doctor_list_checks.out 2>/tmp/continuum_doctor_list_checks.err \
  || fail "--list-checks failed"

log "Step 2: deterministic JSON no-write run"
run_allow_doctor_codes continuum doctor --deterministic --json --no-write >/tmp/continuum_doctor_det_json.out 2>/tmp/continuum_doctor_det_json.err

mkdir -p "${REPORT_DIR}" || fail "Could not create report directory"
before_count="$(find "${REPORT_DIR}" -maxdepth 1 -type f -name 'doctor_*.json' | wc -l | tr -d ' ')"

log "Step 3: default doctor run (should write JSON report)"
run_allow_doctor_codes continuum doctor >/tmp/continuum_doctor_default.out 2>/tmp/continuum_doctor_default.err

after_count="$(find "${REPORT_DIR}" -maxdepth 1 -type f -name 'doctor_*.json' | wc -l | tr -d ' ')"

if [ "${after_count}" -le "${before_count}" ]; then
  fail "No new doctor_*.json report found in ${REPORT_DIR}"
fi

latest_report="$(ls -1t "${REPORT_DIR}"/doctor_*.json 2>/dev/null | head -n 1 || true)"
[ -n "${latest_report}" ] || fail "Report path resolution failed"

log "PASS: smoke install/run completed"
log "Latest report: ${latest_report}"
