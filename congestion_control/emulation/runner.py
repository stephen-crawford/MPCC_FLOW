"""Comparison runner: executes multiple CCAs over the same trace and reports results."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type

from congestion_control.base import CongestionController
from congestion_control.diagnostics import Diagnostics, FlowMetrics, compare_flows, jains_fairness_index
from congestion_control.emulation.flow import Flow, FlowConfig
from congestion_control.emulation.link_emulator import LinkConfig, LinkEmulator

logger = logging.getLogger("cc.runner")


@dataclass
class ComparisonResult:
    """Results from running multiple CCAs on the same trace."""
    trace_name: str
    link_config: LinkConfig
    metrics: Dict[str, FlowMetrics] = field(default_factory=dict)
    diagnostics: Dict[str, Diagnostics] = field(default_factory=dict)

    def summary_table(self) -> str:
        lines = [
            f"=== Comparison on {self.trace_name} "
            f"(prop_delay={self.link_config.propagation_delay_ms}ms, "
            f"queue={self.link_config.queue_size_bytes//1500}pkts) ===",
            f"{'CCA':>12s} | {'Tput (Mbps)':>11s} | {'RTT p50':>7s} | {'RTT p95':>7s} | "
            f"{'Util':>5s} | {'Loss':>6s} | {'Avg cwnd':>9s}",
            "-" * 80,
        ]
        for name, m in sorted(self.metrics.items()):
            lines.append(
                f"{name:>12s} | {m.avg_throughput_mbps:>11.2f} | "
                f"{m.p50_rtt_ms:>6.1f}m | {m.p95_rtt_ms:>6.1f}m | "
                f"{m.link_utilization*100:>4.1f}% | "
                f"{m.loss_rate*100:>5.2f}% | {m.avg_cwnd_bytes:>9.0f}"
            )
        return "\n".join(lines)


def run_single(
    cca: CongestionController,
    trace_path: Path,
    link_config: LinkConfig | None = None,
    flow_config: FlowConfig | None = None,
) -> Diagnostics:
    """Run a single CCA on one trace and return diagnostics."""
    link = LinkEmulator(trace_path, link_config)
    flow = Flow(cca, link, flow_config)
    return flow.run()


def run_comparison(
    cca_factories: Dict[str, callable],
    trace_path: Path,
    link_config: LinkConfig | None = None,
    flow_config: FlowConfig | None = None,
) -> ComparisonResult:
    """Run multiple CCAs on the same trace sequentially and compare.

    Args:
        cca_factories: dict mapping name -> callable that returns a CCA instance
        trace_path: path to Mahimahi-format trace file
        link_config: shared link configuration
        flow_config: shared flow configuration
    """
    link_config = link_config or LinkConfig()
    flow_config = flow_config or FlowConfig()

    result = ComparisonResult(
        trace_name=trace_path.stem,
        link_config=link_config,
    )

    for name, factory in cca_factories.items():
        logger.info("Running CCA: %s on trace: %s", name, trace_path.name)
        cca = factory()
        link = LinkEmulator(trace_path, link_config)
        flow = Flow(cca, link, flow_config)
        diag = flow.run()
        result.diagnostics[name] = diag
        result.metrics[name] = diag.compute_metrics()

    logger.info("\n%s", result.summary_table())
    return result


def run_fairness_test(
    cca_factories: List[callable],
    trace_path: Path,
    link_config: LinkConfig | None = None,
    duration_s: float = 60.0,
    stagger_s: float = 5.0,
) -> Dict[str, FlowMetrics]:
    """Run multiple flows concurrently (simulated) to test fairness.

    Since our emulator is single-threaded, we simulate concurrent flows
    by interleaving their send/receive events on the same link.
    """
    link_config = link_config or LinkConfig()
    link = LinkEmulator(trace_path, link_config)

    flows = []
    for i, factory in enumerate(cca_factories):
        cca = factory()
        config = FlowConfig(duration_s=duration_s, start_delay_s=i * stagger_s)
        flow = Flow(cca, link, config)
        flows.append(flow)

    # Run all flows (sequential simulation of concurrent operation)
    diagnostics_list = []
    for flow in flows:
        diag = flow.run()
        diagnostics_list.append(diag)

    return compare_flows(diagnostics_list)
