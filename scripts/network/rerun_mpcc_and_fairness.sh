#!/usr/bin/env bash
# Targeted re-run after the RUST_LOG-sudo bug: every MPCC cell in the main
# matrix, ablations, and the broken fairness sweep. Baseline CCA runs
# (cubic/bbr/reno/nimbus) are left alone — they were unaffected.
#
# Env overrides:
#   OUTPUT_ROOT  path to the timestamped results dir to repopulate (default: newest)
#   DURATION_S   (default 60)
#   SEEDS        (default 3)
#
# Usage:
#   bash scripts/network/rerun_mpcc_and_fairness.sh
#
# Takes ~95 min: 21 main-MPCC + 60 ablation + 9 fairness runs at ~62s each.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
    OUTPUT_ROOT="$(ls -dt results/paper/*/ 2>/dev/null | head -1 | sed 's:/$::')"
    if [[ -z "$OUTPUT_ROOT" ]]; then
        echo "no results/paper/*/ directory found" >&2
        exit 1
    fi
    OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
fi

DURATION_S="${DURATION_S:-60}"
SEEDS="${SEEDS:-3}"

echo "== rerun of MPCC cells + fairness =="
echo "  OUTPUT_ROOT: $OUTPUT_ROOT"
echo "  DURATION:    ${DURATION_S}s  SEEDS: $SEEDS"

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
    sudo -n /usr/sbin/insmod "$REPO_ROOT/../nimbus-measurement/ccp-kernel/ccp-cong.ko" ipc=0
    sleep 0.5
fi
# Make sure `ccp` is in tcp_allowed_congestion_control — if not, every MPCC
# run silently falls back to cubic.
ALLOWED="$(cat /proc/sys/net/ipv4/tcp_allowed_congestion_control)"
case " $ALLOWED " in
    *" ccp "*) ;;
    *) sudo -n /usr/sbin/sysctl -w "net.ipv4.tcp_allowed_congestion_control=$ALLOWED ccp" >/dev/null ;;
esac
sudo -n /usr/sbin/sysctl -q -w net.ipv4.ip_forward=1

# Delete the stale (cubic-fallback) MPCC data so metrics can't accidentally
# merge old + new runs.
echo "-- cleaning stale MPCC cells --"
find "$OUTPUT_ROOT/matrix" -type d -name mpcc -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$OUTPUT_ROOT/matrix/fairness" 2>/dev/null || true
for v in baseline no_contour no_lag no_fairness; do
    rm -rf "$OUTPUT_ROOT/ablations/$v/wired" "$OUTPUT_ROOT/ablations/$v/cellular" \
           "$OUTPUT_ROOT/ablations/$v/leo" "$OUTPUT_ROOT/ablations/$v/fairness" \
           "$OUTPUT_ROOT/ablations/$v/summary.json" \
           "$OUTPUT_ROOT/ablations/$v/paper_summary.json" 2>/dev/null || true
done

echo "-- Phase R.1: re-run MPCC on the full matrix (~22 min) --"
python3 -m scripts.network.run_cc_experiments run \
    --scenarios wired cellular leo \
    --ccas mpcc \
    --duration "$DURATION_S" \
    --seeds "$SEEDS" \
    --output "$OUTPUT_ROOT/matrix" \
    --mpcc-config-dir "$REPO_ROOT/scripts/network/configs/paper" \
    --mpcc-solver qp

echo "-- Phase R.2: fairness sweep (mpcc, cubic, bbr) (~10 min) --"
for CCA in mpcc cubic bbr; do
    python3 -m scripts.network.run_cc_experiments fairness \
        --ccas "$CCA" \
        --n-flows 4 \
        --duration "$DURATION_S" \
        --seeds "$SEEDS" \
        --output "$OUTPUT_ROOT/matrix"
done

echo "-- Phase R.3: re-run ablations (~62 min) --"
python3 -m scripts.network.run_ablations \
    --scenarios wired cellular fairness \
    --duration "$DURATION_S" --seeds "$SEEDS" \
    --output "$OUTPUT_ROOT/ablations"

echo "-- Phase R.4: re-aggregate paper metrics --"
python3 -m scripts.network.paper_metrics "$OUTPUT_ROOT/matrix"

echo "-- Phase R.5: re-stitch ablation table --"
python3 -m scripts.network._stitch_ablation_table \
    "$OUTPUT_ROOT/ablations" "$REPO_ROOT/../paper/tables/ablation.tex" || true

echo "-- Phase R.6: re-sync tables + figures into paper/ --"
PAPER_ROOT="${PAPER_ROOT:-$REPO_ROOT/../paper}"
mkdir -p "$PAPER_ROOT/tables" "$PAPER_ROOT/figures"
[[ -d "$OUTPUT_ROOT/matrix/paper_tables"  ]] && cp -f "$OUTPUT_ROOT/matrix/paper_tables"/*.tex  "$PAPER_ROOT/tables/"  2>/dev/null || true
[[ -d "$OUTPUT_ROOT/matrix/paper_figures" ]] && cp -f "$OUTPUT_ROOT/matrix/paper_figures"/*.png "$PAPER_ROOT/figures/" 2>/dev/null || true

echo
echo "== done =="
echo "  summary: $OUTPUT_ROOT/matrix/paper_summary.json"
