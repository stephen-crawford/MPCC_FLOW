#!/usr/bin/env bash
# Standalone fairness sweep. Runs the multi-flow fairness experiments against
# the most recent OUTPUT_ROOT (or a given one), then re-aggregates paper
# metrics and re-copies tables/figures into ../paper/.
#
# Use this after run_paper_matrix.sh has finished, if the fairness phase
# inside it failed (or if you want to re-do fairness with different
# parameters).
#
# Env overrides:
#   OUTPUT_ROOT     path to the timestamped results dir to extend
#   FAIRNESS_FLOWS  number of concurrent flows (default 4)
#   DURATION_S      per-run duration (default 60)
#   SEEDS           repeats (default 3)
#   CCAS            subset of "mpcc cubic bbr" (default all three)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
    OUTPUT_ROOT="$(ls -dt results/paper/*/ 2>/dev/null | head -1 | sed 's:/$::')"
    if [[ -z "$OUTPUT_ROOT" ]]; then
        echo "no results/paper/*/ directory found; run the main matrix first" >&2
        exit 1
    fi
    OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
fi

FAIRNESS_FLOWS="${FAIRNESS_FLOWS:-4}"
DURATION_S="${DURATION_S:-60}"
SEEDS="${SEEDS:-3}"
CCAS="${CCAS:-mpcc cubic bbr}"

echo "== fairness sweep =="
echo "  OUTPUT_ROOT:    $OUTPUT_ROOT"
echo "  FAIRNESS_FLOWS: $FAIRNESS_FLOWS"
echo "  DURATION_S:     $DURATION_S"
echo "  SEEDS:          $SEEDS"
echo "  CCAS:           $CCAS"

if [[ $EUID -eq 0 ]]; then
    echo "do not run under sudo (mahimahi rejects root)" >&2
    exit 1
fi
if ! sudo -n /usr/sbin/sysctl -n net.ipv4.ip_forward >/dev/null 2>&1; then
    echo "passwordless sudo not configured; see scripts/network/sudoers-mpcc-paper" >&2
    exit 1
fi

if ! grep -q '^ccp_cong ' /proc/modules; then
    echo "-- loading ccp_cong.ko --"
    CCP_KO="$REPO_ROOT/../nimbus-measurement/ccp-kernel/ccp-cong.ko"
    sudo -n /usr/sbin/insmod "$CCP_KO" ipc=0
    sleep 0.5
else
    echo "   ccp_cong already loaded"
fi
ALLOWED="$(cat /proc/sys/net/ipv4/tcp_allowed_congestion_control)"
case " $ALLOWED " in
    *" ccp "*) ;;
    *) sudo -n /usr/sbin/sysctl -w "net.ipv4.tcp_allowed_congestion_control=$ALLOWED ccp" >/dev/null ;;
esac
sudo -n /usr/sbin/sysctl -q -w net.ipv4.ip_forward=1

for CCA in $CCAS; do
    echo "-- fairness: $CCA --"
    python3 -m scripts.network.run_cc_experiments fairness \
        --ccas "$CCA" \
        --n-flows "$FAIRNESS_FLOWS" \
        --duration "$DURATION_S" \
        --seeds "$SEEDS" \
        --output "$OUTPUT_ROOT/matrix"
done

echo "-- re-aggregating paper metrics --"
python3 -m scripts.network.paper_metrics "$OUTPUT_ROOT/matrix"

echo "-- re-syncing tables + figures into paper/ --"
PAPER_ROOT="${PAPER_ROOT:-$REPO_ROOT/../paper}"
mkdir -p "$PAPER_ROOT/tables" "$PAPER_ROOT/figures"
[[ -d "$OUTPUT_ROOT/matrix/paper_tables"  ]] && cp -f "$OUTPUT_ROOT/matrix/paper_tables"/*.tex  "$PAPER_ROOT/tables/"  2>/dev/null || true
[[ -d "$OUTPUT_ROOT/matrix/paper_figures" ]] && cp -f "$OUTPUT_ROOT/matrix/paper_figures"/*.png "$PAPER_ROOT/figures/" 2>/dev/null || true

echo "done."
