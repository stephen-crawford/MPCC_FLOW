# Cellular / congestion-control experiments

This package wires **Mahimahi** (emulated cellular links with real traces) and **Nimbus** (CCP congestion control) into your workflow without copying those trees into `mpcc_flow`.

## Motivation: cellular bottlenecks

Traces record time-varying capacity of U.S. cellular networks (Saturator / NSDI 2013). `mm-link` replays uplink/downlink schedules so TCP flows (or any traffic) see realistic time-varying rates—standard for congestion-control evaluation.

## Commands

```bash
# Paths and tools (set MAHIMAHI_ROOT / NIMBUS_ROOT if not ~/mahimahi and ~/nimbus)
mpcc-network check

# List cellular trace pairs discovered under $MAHIMAHI_ROOT/traces
mpcc-network list-traces

# Print a full mm-link argv for a trace stem (then run manually or from shell scripts)
mpcc-network mm-link-cmd Verizon-LTE-short iperf3 -c 10.0.0.2 -t 30
```

## Typical flow (iperf + mm-link)

1. Start server: `iperf3 -s` (on the peer or host, depending on your topology).
2. Run client under emulation: see `scripts/network/cellular_iperf_experiment.sh` (requires `sudo` for mahimahi namespaces).
3. Plot logs: `mm-graph <log> <ms_per_bin>` from Mahimahi (or use `network_emulation.mm_graph_argv`).

## Nimbus (CCP)

Nimbus is not invoked from Python directly; it runs as a CCP algorithm process. See `mpcc-network nimbus-help` and the [CCP guide](https://ccp-project.github.io/guide).

## Connecting to MPC / MPCC experiments

Use the same **metrics** you would for CC studies (throughput, delay, queueing from mm-link logs) as **observations** for your controller, or run **control** in a separate process while traffic traverses `mm-link` + optional `mm-delay`.
