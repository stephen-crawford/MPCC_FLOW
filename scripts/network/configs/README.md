# Experiment configs

Two harness options drive real Mahimahi experiments in this project:

## 1. Native MPCC_FLOW matrix runner

The **primary** entry point — no TOML needed:

```
python -m scripts.network.run_cc_experiments run \
    --scenarios wired cellular leo \
    --ccas cubic bbr mpcc nimbus sprout verus \
    --duration 60 --seeds 3 --output results/real
```

See `scripts/network/run_cc_experiments.py`. Runs everything via nested
`mm-delay` + `mm-link` directly, parses mm-link's uplink log for
throughput/loss, and iperf3's `--json` output for RTT. No external framework
needed.

## 2. nimbus-measurement harness

For head-to-head comparisons against the Nimbus paper's own experiments
(Goyal et al.), you can drive the same CCAs through the `nimbus-measurement`
framework. The TOML files in this directory are drop-in configs for
`nimbus-measurement/script/experiment.py`:

```
cd /path/to/nimbus-measurement
python script/experiment.py \
    --config /path/to/mpcc_flow/scripts/network/configs/mpcc-cellular.toml \
    --outdir /path/to/out
```

Configs:

| File | Scenario |
|------|----------|
| `mpcc-wired.toml` | Constant 50 Mbps, 40 ms RTT, 1 MPCC flow + 1 CUBIC flow |
| `mpcc-cellular.toml` | Mimics the ATT-LTE driving profile (same shaping profile as `nimbus-cubic-etg.toml`) |
| `mpcc-leo.toml` | LEO-inspired (bw jumps between 50 and 5 Mbps) |
| `mpcc-fairness.toml` | 4 MPCC flows sharing 96 Mbps, 20 ms RTT |
| `mpcc-vs-baselines.toml` | 1 MPCC flow alongside 1 iperf flow that cycles CUBIC, BBR, and Reno |

All of these assume the Rust `portus-mpcc` binary (or `pyportus` fallback) is
available — the `mpcc_bin` field in each `[[traffic]]` block can be edited to
point at your build, or left blank to use the Python pyportus stub.
