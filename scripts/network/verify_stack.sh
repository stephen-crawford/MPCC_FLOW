#!/usr/bin/env bash
# Smoke-test Mahimahi + Nimbus wiring from mpcc_flow (no pytest required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="${HOME}/.cargo/bin:${PATH}"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

echo "== mpcc-network check =="
mpcc-network check || true

echo ""
echo "== mm-link + cellular trace + /bin/true =="
MAHI="${MAHIMAHI_ROOT:-$ROOT/third_party/mahimahi}"
UP="${MAHI}/traces/Verizon-LTE-short.up"
DOWN="${MAHI}/traces/Verizon-LTE-short.down"
if [[ ! -f "$UP" ]]; then
  echo "Skip mm-link: traces not at $UP (set MAHIMAHI_ROOT or run third_party/setup_symlinks.sh)"
else
  mm-link "$UP" "$DOWN" -- /bin/true && echo "mm-link e2e: OK"
fi

echo ""
echo "== iperf3 =="
command -v iperf3 >/dev/null && iperf3 --version | head -1 || echo "iperf3 not installed (optional)"

echo ""
echo "== CCP module (optional for live Nimbus) =="
if [[ -r /proc/modules ]] && grep -q '^ccp ' /proc/modules 2>/dev/null; then
  echo "ccp module loaded"
else
  echo "ccp not loaded (expected without CCP kernel setup)"
fi

echo ""
echo "== pytest integration =="
if [[ -x "$ROOT/.venv/bin/pytest" ]]; then
  "$ROOT/.venv/bin/pytest" "$ROOT/tests/test_mahimahi_nimbus_integration.py" \
    "$ROOT/tests/test_network_stack_e2e.py" -q --tb=no
else
  echo "Run: make venv && .venv/bin/pytest tests/..."
fi
