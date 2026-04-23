#!/usr/bin/env bash
# Paper matrix orchestrator (Delivery 1 — no ABC).
#
# IMPORTANT: Run this WITHOUT sudo, as your regular user. Mahimahi (`mm-link`)
# refuses to operate when the invoking process is root — it uses a setuid
# binary + per-user network namespaces, so any root-uid caller fails silently.
# We therefore `sudo` only the commands that genuinely need root:
#   - insmod ccp_cong.ko
#   - sysctl net.ipv4.ip_forward / tcp_congestion_control
#   - the portus-mpcc / nimbus CCP user-space daemons (netlink to ccp_cong)
#   - pkill for those daemons at the end of each run
#
# Environment overrides:
#   SCENARIOS, CCAS, DURATION_S, SEEDS, OUTPUT_ROOT,
#   RUN_FAIRNESS, RUN_ABLATIONS, FAIRNESS_FLOWS, INCLUDE_NLP
#
# Usage:
#   bash scripts/network/run_paper_matrix.sh
#
# Smoke:
#   DURATION_S=10 SEEDS=1 CCAS="mpcc cubic bbr" SCENARIOS=wired \
#     RUN_FAIRNESS=0 RUN_ABLATIONS=0 \
#     bash scripts/network/run_paper_matrix.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%dT%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/paper/$TIMESTAMP}"
SCENARIOS="${SCENARIOS:-wired cellular leo}"
CCAS="${CCAS:-cubic bbr reno mpcc nimbus sprout verus}"
DURATION_S="${DURATION_S:-60}"
SEEDS="${SEEDS:-3}"
RUN_FAIRNESS="${RUN_FAIRNESS:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
FAIRNESS_FLOWS="${FAIRNESS_FLOWS:-4}"
INCLUDE_NLP="${INCLUDE_NLP:-0}"

CCP_KERNEL_DIR="${CCP_KERNEL_DIR:-$REPO_ROOT/../nimbus-measurement/ccp-kernel}"

if [[ $EUID -eq 0 ]]; then
    echo "ERROR: do not run this script with sudo. Mahimahi (mm-link) will" >&2
    echo "       silently fail when invoked as root. The script will sudo" >&2
    echo "       only the specific prereq commands that need it."         >&2
    exit 1
fi

echo "== MPCC paper matrix =="
echo "  output:    $OUTPUT_ROOT"
echo "  scenarios: $SCENARIOS"
echo "  ccas:      $CCAS"
echo "  duration:  ${DURATION_S}s seeds=$SEEDS"
echo "  fairness:  $RUN_FAIRNESS (flows=$FAIRNESS_FLOWS)"
echo "  ablations: $RUN_ABLATIONS (include NLP = $INCLUDE_NLP)"

echo "-- checking passwordless sudo for required commands --"
if ! sudo -n /usr/sbin/sysctl -n net.ipv4.ip_forward >/dev/null 2>&1; then
    echo "ERROR: passwordless sudo not configured for this matrix." >&2
    echo "       Install the sudoers drop-in once (asks for password):" >&2
    echo "           sudo install -m 0440 $(pwd)/scripts/network/sudoers-mpcc-paper /etc/sudoers.d/mpcc-paper" >&2
    echo "           sudo visudo -c -f /etc/sudoers.d/mpcc-paper" >&2
    echo "       Then re-run this script." >&2
    exit 1
fi

# Self-heal: a prior sudo invocation of this script may have left
# results/paper/ owned by root. Take ownership back before we try to mkdir.
if [[ -d "$REPO_ROOT/results/paper" ]]; then
    sudo -n /usr/bin/chown -R "$USER":"$USER" "$REPO_ROOT/results" 2>/dev/null || true
fi

mkdir -p "$OUTPUT_ROOT"

cleanup() {
    echo "-- cleanup: killing CCP daemons, unloading module --"
    # Use -x (exact command name) not -f (full cmdline). pkill -f on substrings
    # like "nimbus" or "mpcc" would also match the Python matrix runner itself,
    # whose argv includes those CCAs as arguments — killing its own parent.
    sudo -n /usr/bin/pkill -9 -x mpcc_cca     2>/dev/null || true
    sudo -n /usr/bin/pkill -9 -x nimbus       2>/dev/null || true
    sudo -n /usr/bin/pkill -9 -x sproutbt2    2>/dev/null || true
    sudo -n /usr/bin/pkill -9 -x verus_server 2>/dev/null || true
    sudo -n /usr/bin/pkill -9 -x verus_client 2>/dev/null || true
    sudo -n /usr/bin/pkill -9 -x iperf3       2>/dev/null || true
    # Python pyportus fallback runs under the module path; -f here is safe
    # because the matrix runner is `scripts.network.run_cc_experiments`, not
    # `network_emulation.mpcc_cca`.
    sudo -n /usr/bin/pkill -9 -f network_emulation.mpcc_cca 2>/dev/null || true
    sudo -n /usr/sbin/sysctl -q -w net.ipv4.tcp_congestion_control=cubic 2>/dev/null || true
    sudo -n /usr/sbin/rmmod ccp_cong 2>/dev/null || true
}
trap cleanup EXIT

echo "-- loading ccp_cong.ko --"
CCP_KO="$CCP_KERNEL_DIR/ccp-cong.ko"
if [[ ! -f "$CCP_KO" ]]; then
    echo "ccp-cong.ko missing at $CCP_KO — build it first:" >&2
    echo "  (cd $CCP_KERNEL_DIR && make)" >&2
    exit 1
fi
# Detect via /proc/modules (more reliable than `lsmod | grep`). If the module
# is already loaded (e.g., left over from a previous run whose sockets are
# still in TIME_WAIT holding references), reuse it — we can't rmmod while the
# refcount is > 0 anyway, and the module is stateless across flows.
if grep -q '^ccp_cong ' /proc/modules; then
    echo "   ccp_cong already loaded — reusing"
else
    sudo -n /usr/sbin/insmod "$CCP_KO" ipc=0
    sleep 0.5
fi

# The ccp_cong module registers `ccp` as a TCP CCA, but Linux requires it to
# be explicitly added to tcp_allowed_congestion_control before sockets can
# opt in via `sysctl tcp_congestion_control=ccp` or `iperf3 -C ccp`. Skipping
# this is the reason MPCC and Nimbus runs silently fell back to cubic.
ALLOWED="$(cat /proc/sys/net/ipv4/tcp_allowed_congestion_control)"
case " $ALLOWED " in
    *" ccp "*) ;;
    *) sudo -n /usr/sbin/sysctl -w "net.ipv4.tcp_allowed_congestion_control=$ALLOWED ccp" >/dev/null ;;
esac

echo "-- kernel IP forwarding (required by mahimahi) --"
sudo -n /usr/sbin/sysctl -q -w net.ipv4.ip_forward=1

echo "-- preflight : mm-link round-trip with /bin/true --"
TRACE_ROOT="$(python3 -c 'from network_emulation.paths import mahimahi_root; print(mahimahi_root() / "traces")')"
PREFLIGHT_UP="$TRACE_ROOT/ATT-LTE-driving.up"
PREFLIGHT_DN="$TRACE_ROOT/ATT-LTE-driving.down"
if [[ ! -f "$PREFLIGHT_UP" || ! -f "$PREFLIGHT_DN" ]]; then
    echo "preflight skipped: traces not at $TRACE_ROOT" >&2
else
    if ! timeout 10s mm-link "$PREFLIGHT_UP" "$PREFLIGHT_DN" -- /bin/true >/dev/null 2>&1; then
        echo "ERROR: mm-link preflight failed. Likely causes:" >&2
        echo "  - this script was invoked with sudo (it must not be; see top of file)" >&2
        echo "  - kernel IP forwarding is off (we just set it; check dmesg)" >&2
        echo "  - mahimahi's setuid bit is missing; re-run 'sudo make install' in mahimahi" >&2
        exit 2
    fi
    echo "   ok"
fi

echo "-- Phase E.1 : main matrix (running as $USER, not root) --"
python3 -m scripts.network.run_cc_experiments run \
    --scenarios $SCENARIOS \
    --ccas $CCAS \
    --duration "$DURATION_S" \
    --seeds "$SEEDS" \
    --output "$OUTPUT_ROOT/matrix" \
    --mpcc-config-dir "$REPO_ROOT/scripts/network/configs/paper" \
    --mpcc-solver qp

if [[ "$RUN_FAIRNESS" == "1" ]]; then
    echo "-- Phase E.2 : fairness --"
    for CCA in mpcc cubic bbr; do
        python3 -m scripts.network.run_cc_experiments fairness \
            --ccas $CCA \
            --n-flows "$FAIRNESS_FLOWS" \
            --duration "$DURATION_S" \
            --seeds "$SEEDS" \
            --output "$OUTPUT_ROOT/matrix"
    done
fi

if [[ "$RUN_ABLATIONS" == "1" ]]; then
    echo "-- Phase E.3 : ablations --"
    ABLATION_ARGS=(
        --scenarios wired cellular fairness
        --duration "$DURATION_S" --seeds "$SEEDS"
        --output "$OUTPUT_ROOT/ablations"
    )
    if [[ "$INCLUDE_NLP" == "1" ]]; then
        ABLATION_ARGS+=(--include-nlp)
    fi
    python3 -m scripts.network.run_ablations "${ABLATION_ARGS[@]}"
fi

echo "-- Phase E.4 : aggregate paper metrics --"
python3 -m scripts.network.paper_metrics "$OUTPUT_ROOT/matrix"

echo "-- Phase E.5 : sync tables + figures into paper/ --"
PAPER_ROOT="${PAPER_ROOT:-$REPO_ROOT/../paper}"
mkdir -p "$PAPER_ROOT/tables" "$PAPER_ROOT/figures"
SRC_TABLES="$OUTPUT_ROOT/matrix/paper_tables"
SRC_FIGURES="$OUTPUT_ROOT/matrix/paper_figures"
[[ -d "$SRC_TABLES"  ]] && cp -f "$SRC_TABLES"/*.tex  "$PAPER_ROOT/tables/"  2>/dev/null || true
[[ -d "$SRC_FIGURES" ]] && cp -f "$SRC_FIGURES"/*.png "$PAPER_ROOT/figures/" 2>/dev/null || true

# Stitch the ablation runs into a single LaTeX table for §V.C.
if [[ -d "$OUTPUT_ROOT/ablations" ]]; then
    python3 -m scripts.network._stitch_ablation_table \
        "$OUTPUT_ROOT/ablations" "$PAPER_ROOT/tables/ablation.tex" || true
fi

echo
echo "== done =="
echo "  tables:  $OUTPUT_ROOT/matrix/paper_tables/  ->  $PAPER_ROOT/tables/"
echo "  figures: $OUTPUT_ROOT/matrix/paper_figures/ ->  $PAPER_ROOT/figures/"
echo "  summary: $OUTPUT_ROOT/matrix/paper_summary.json"
echo
echo "rebuild paper:"
echo "  cd $REPO_ROOT/.. && pdflatex mpcc_flow.tex && bibtex mpcc_flow && pdflatex mpcc_flow.tex && pdflatex mpcc_flow.tex"
