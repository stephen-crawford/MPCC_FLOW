#!/usr/bin/env bash
# Typical congestion-control style experiment: emulate a cellular bottleneck with
# Mahimahi mm-link (NSDI traces), run iperf3, save mm-link logs for mm-graph.
#
# Prerequisites: mahimahi installed (mm-link on PATH), iperf3, root for mahimahi namespaces.
#
# Usage:
#   sudo ./cellular_iperf_experiment.sh Verizon-LTE-short 10.0.0.2
#
# SERVER (outside mm-link):  iperf3 -s
# This script runs the client inside: mm-link UP DOWN -- iperf3 -c HOST ...

set -euo pipefail

TRACE_STEM="${1:?trace stem e.g. Verizon-LTE-short}"
SERVER_IP="${2:?iperf server IP (often the host on the other side of the namespace)}"
MAHIMAHI_ROOT="${MAHIMAHI_ROOT:-$HOME/mahimahi}"
UP="${MAHIMAHI_ROOT}/traces/${TRACE_STEM}.up"
DOWN="${MAHIMAHI_ROOT}/traces/${TRACE_STEM}.down"
OUTDIR="${OUTDIR:-./cellular_exp_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

if [[ ! -f "$UP" || ! -f "$DOWN" ]]; then
  echo "Missing trace files: $UP $DOWN" >&2
  echo "Set MAHIMAHI_ROOT or copy traces from your Mahimahi checkout." >&2
  exit 1
fi

command -v mm-link >/dev/null || { echo "mm-link not in PATH"; exit 1; }
command -v iperf3 >/dev/null || { echo "iperf3 not installed"; exit 1; }

UPLINK_LOG="${OUTDIR}/uplink.log"
DOWNLINK_LOG="${OUTDIR}/downlink.log"

echo "Logs -> $OUTDIR"
echo "Starting mm-link with iperf3 client; ensure iperf3 -s is running on $SERVER_IP"

exec mm-link "$UP" "$DOWN" \
  --uplink-log="$UPLINK_LOG" \
  --downlink-log="$DOWNLINK_LOG" \
  -- iperf3 -c "$SERVER_IP" -t 60 -i 1
