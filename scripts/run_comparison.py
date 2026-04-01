"""Run all CCAs on all available Mahimahi traces and report comparison metrics."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from congestion_control.emulation.runner import run_comparison
from congestion_control.emulation.link_emulator import LinkConfig
from congestion_control.emulation.flow import FlowConfig
from congestion_control.emulation.traces import (
    create_constant_trace,
    create_variable_trace,
    create_cellular_like_trace,
    find_mahimahi_traces_dir,
)
from congestion_control.sprout import Sprout, SproutEWMA
from congestion_control.verus import Verus
from congestion_control.abc_cc import ABC
from congestion_control.mpcc_cc import MPCCController

FACTORIES = {
    "Sprout": lambda: Sprout(),
    "SproutEWMA": lambda: SproutEWMA(),
    "Verus": lambda: Verus(R=4.0),
    "ABC": lambda: ABC(eta=0.7),
    "MPCC": lambda: MPCCController(horizon=8, target_delay_ms=50.0),
}

LINK = LinkConfig(propagation_delay_ms=20, queue_size_bytes=150_000)
FLOW = FlowConfig(duration_s=10.0, tick_ms=5.0)


def header():
    print(f"{'Trace':<30s} {'CCA':<12s} {'Tput Mbps':>9s} "
          f"{'RTT p50ms':>9s} {'RTT p95ms':>9s} {'Util%':>6s} "
          f"{'Loss%':>6s} {'QDelay ms':>9s} {'AvgCwnd':>9s}")
    print("-" * 110)


def run_on_trace(name: str, path: Path):
    result = run_comparison(FACTORIES, path, LINK, FLOW)
    for cca, m in sorted(result.metrics.items()):
        print(f"{name:<30s} {cca:<12s} {m.avg_throughput_mbps:>9.2f} "
              f"{m.p50_rtt_ms:>9.1f} {m.p95_rtt_ms:>9.1f} "
              f"{m.link_utilization*100:>5.1f}% "
              f"{m.loss_rate*100:>5.2f}% {m.avg_queue_delay_ms:>9.1f} "
              f"{m.avg_cwnd_bytes:>9.0f}")


def main():
    header()
    print()

    # Synthetic traces
    print("=== SYNTHETIC TRACES ===")
    synth = [
        ("Constant 10 Mbps", create_constant_trace(10.0, 30_000)),
        ("Constant 50 Mbps", create_constant_trace(50.0, 30_000)),
        ("Variable 2-20 Mbps", create_variable_trace([10, 2, 20, 5, 15, 3, 25, 8], 3000)),
        ("Cellular-like 5 Mbps", create_cellular_like_trace(5.0, 30_000)),
        ("Cellular-like 20 Mbps", create_cellular_like_trace(20.0, 30_000)),
    ]
    for name, path in synth:
        run_on_trace(name, path)
        print()

    # Real Mahimahi traces
    traces_dir = find_mahimahi_traces_dir()
    if traces_dir:
        print("\n=== REAL MAHIMAHI CELLULAR TRACES ===")
        for trace in sorted(traces_dir.glob("*.down")):
            run_on_trace(trace.stem, trace)
            print()

    # Cleanup synthetic
    for _, p in synth:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
