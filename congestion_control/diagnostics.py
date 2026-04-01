"""Diagnostics and metrics collection for congestion control experiments.

Every CCA is wrapped with a Diagnostics instance that records per-packet
events, cwnd evolution, and computes summary statistics for comparison.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from congestion_control.base import CongestionController

import logging

logger = logging.getLogger("cc.diagnostics")


# ---------------------------------------------------------------------------
# Per-flow summary metrics
# ---------------------------------------------------------------------------

@dataclass
class FlowMetrics:
    """Aggregate metrics for one flow over a measurement interval."""
    cca_name: str = ""
    duration_s: float = 0.0
    bytes_sent: int = 0
    bytes_delivered: int = 0
    bytes_lost: int = 0

    # Throughput
    avg_throughput_mbps: float = 0.0
    peak_throughput_mbps: float = 0.0

    # Delay
    min_rtt_ms: float = float("inf")
    avg_rtt_ms: float = 0.0
    p50_rtt_ms: float = 0.0
    p95_rtt_ms: float = 0.0
    p99_rtt_ms: float = 0.0
    avg_queue_delay_ms: float = 0.0

    # Utilization & loss
    link_utilization: float = 0.0
    loss_rate: float = 0.0

    # Fairness (populated when multiple flows)
    jains_fairness_index: float = 0.0

    # CCA-specific
    avg_cwnd_bytes: float = 0.0
    cwnd_changes: int = 0

    def summary_line(self) -> str:
        return (
            f"{self.cca_name:>12s} | "
            f"tput={self.avg_throughput_mbps:7.2f} Mbps | "
            f"delay p50={self.p50_rtt_ms:6.1f} p95={self.p95_rtt_ms:6.1f} ms | "
            f"util={self.link_utilization*100:5.1f}% | "
            f"loss={self.loss_rate*100:5.2f}%"
        )


# ---------------------------------------------------------------------------
# Event records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AckEvent:
    timestamp_s: float
    rtt_s: float
    bytes_acked: int
    cwnd: float


@dataclass(frozen=True)
class LossEvent:
    timestamp_s: float
    bytes_lost: int
    cwnd_before: float
    cwnd_after: float


@dataclass(frozen=True)
class CwndEvent:
    timestamp_s: float
    old_cwnd: float
    new_cwnd: float
    reason: str


# ---------------------------------------------------------------------------
# Diagnostics collector
# ---------------------------------------------------------------------------

class Diagnostics:
    """Attaches to a CongestionController and records all events."""

    def __init__(self, cca: CongestionController, link_capacity_bps: float = 0.0,
                 max_events: int = 100_000):
        self.cca = cca
        self.link_capacity_bps = link_capacity_bps
        self._max_events = max_events

        # Event logs
        self.ack_events: Deque[AckEvent] = deque(maxlen=max_events)
        self.loss_events: Deque[LossEvent] = deque(maxlen=max_events)
        self.cwnd_events: Deque[CwndEvent] = deque(maxlen=max_events)

        # Running accumulators
        self._rtt_samples: List[float] = []
        self._throughput_samples: List[Tuple[float, int]] = []  # (time, cum_bytes)
        self._bytes_delivered: int = 0
        self._bytes_sent: int = 0
        self._bytes_lost: int = 0
        self._cwnd_sum: float = 0.0
        self._cwnd_count: int = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        # Wire up diagnostic hooks on the CCA
        cca._diag_on_cwnd_change = self._on_cwnd_change

    # ---- recording methods (called by emulation runner) ----

    def record_ack(self, timestamp_s: float, rtt_s: float,
                   bytes_acked: int, cwnd: float) -> None:
        if self._start_time is None:
            self._start_time = timestamp_s
        self._end_time = timestamp_s
        self.ack_events.append(AckEvent(timestamp_s, rtt_s, bytes_acked, cwnd))
        self._rtt_samples.append(rtt_s)
        self._bytes_delivered += bytes_acked
        self._throughput_samples.append((timestamp_s, self._bytes_delivered))
        self._cwnd_sum += cwnd
        self._cwnd_count += 1

    def record_loss(self, timestamp_s: float, bytes_lost: int,
                    cwnd_before: float, cwnd_after: float) -> None:
        if self._start_time is None:
            self._start_time = timestamp_s
        self._end_time = timestamp_s
        self.loss_events.append(LossEvent(timestamp_s, bytes_lost, cwnd_before, cwnd_after))
        self._bytes_lost += bytes_lost

    def record_send(self, timestamp_s: float, bytes_sent: int) -> None:
        self._bytes_sent += bytes_sent

    def _on_cwnd_change(self, timestamp_s: float, old_cwnd: float,
                        new_cwnd: float, reason: str) -> None:
        self.cwnd_events.append(CwndEvent(timestamp_s, old_cwnd, new_cwnd, reason))

    # ---- metrics computation ----

    def compute_metrics(self) -> FlowMetrics:
        m = FlowMetrics(cca_name=self.cca.name)

        if self._start_time is None or self._end_time is None:
            return m

        m.duration_s = max(self._end_time - self._start_time, 1e-6)
        m.bytes_sent = self._bytes_sent
        m.bytes_delivered = self._bytes_delivered
        m.bytes_lost = self._bytes_lost

        # Throughput
        m.avg_throughput_mbps = (self._bytes_delivered * 8) / m.duration_s / 1e6
        m.peak_throughput_mbps = self._compute_peak_throughput_mbps()

        # Delay
        if self._rtt_samples:
            sorted_rtts = sorted(self._rtt_samples)
            n = len(sorted_rtts)
            m.min_rtt_ms = sorted_rtts[0] * 1000
            m.avg_rtt_ms = (sum(sorted_rtts) / n) * 1000
            m.p50_rtt_ms = sorted_rtts[n // 2] * 1000
            m.p95_rtt_ms = sorted_rtts[int(n * 0.95)] * 1000
            m.p99_rtt_ms = sorted_rtts[int(n * 0.99)] * 1000
            m.avg_queue_delay_ms = m.avg_rtt_ms - m.min_rtt_ms

        # Utilization
        if self.link_capacity_bps > 0:
            m.link_utilization = (m.avg_throughput_mbps * 1e6) / self.link_capacity_bps

        # Loss
        total = m.bytes_delivered + m.bytes_lost
        m.loss_rate = m.bytes_lost / total if total > 0 else 0.0

        # Cwnd
        m.avg_cwnd_bytes = self._cwnd_sum / self._cwnd_count if self._cwnd_count else 0.0
        m.cwnd_changes = len(self.cwnd_events)

        return m

    def _compute_peak_throughput_mbps(self, window_s: float = 1.0) -> float:
        """Sliding-window peak throughput."""
        if len(self._throughput_samples) < 2:
            return 0.0
        peak = 0.0
        left = 0
        for right in range(1, len(self._throughput_samples)):
            while (self._throughput_samples[right][0] - self._throughput_samples[left][0]) > window_s:
                left += 1
            dt = self._throughput_samples[right][0] - self._throughput_samples[left][0]
            db = self._throughput_samples[right][1] - self._throughput_samples[left][1]
            if dt > 0:
                rate_mbps = (db * 8) / dt / 1e6
                peak = max(peak, rate_mbps)
        return peak

    # ---- time-series for plotting ----

    def cwnd_timeseries(self) -> Tuple[List[float], List[float]]:
        """Returns (timestamps, cwnd_values) for plotting."""
        ts = [e.timestamp_s for e in self.ack_events]
        cw = [e.cwnd for e in self.ack_events]
        return ts, cw

    def rtt_timeseries(self) -> Tuple[List[float], List[float]]:
        ts = [e.timestamp_s for e in self.ack_events]
        rtts = [e.rtt_s * 1000 for e in self.ack_events]
        return ts, rtts

    def throughput_timeseries(self, bin_s: float = 1.0) -> Tuple[List[float], List[float]]:
        """Returns (bin_centers, throughput_mbps) time series."""
        if not self._throughput_samples:
            return [], []
        t0 = self._throughput_samples[0][0]
        t1 = self._throughput_samples[-1][0]
        if t1 - t0 < bin_s:
            return [t0], [self.compute_metrics().avg_throughput_mbps]

        bins_t, bins_v = [], []
        idx = 0
        t = t0
        while t + bin_s <= t1:
            bytes_start = 0
            bytes_end = 0
            while idx < len(self._throughput_samples) and self._throughput_samples[idx][0] < t:
                idx += 1
            if idx < len(self._throughput_samples):
                bytes_start = self._throughput_samples[idx][1]
            j = idx
            while j < len(self._throughput_samples) and self._throughput_samples[j][0] < t + bin_s:
                j += 1
            if j > 0:
                bytes_end = self._throughput_samples[j - 1][1]
            bins_t.append(t + bin_s / 2)
            bins_v.append((bytes_end - bytes_start) * 8 / bin_s / 1e6)
            t += bin_s
        return bins_t, bins_v

    def log_summary(self) -> None:
        m = self.compute_metrics()
        logger.info("--- %s flow summary ---", m.cca_name)
        logger.info("  duration:    %.1f s", m.duration_s)
        logger.info("  throughput:  %.2f Mbps (peak %.2f)", m.avg_throughput_mbps, m.peak_throughput_mbps)
        logger.info("  RTT:         min=%.1f avg=%.1f p95=%.1f ms", m.min_rtt_ms, m.avg_rtt_ms, m.p95_rtt_ms)
        logger.info("  queue delay: %.1f ms avg", m.avg_queue_delay_ms)
        logger.info("  utilization: %.1f%%", m.link_utilization * 100)
        logger.info("  loss rate:   %.2f%%", m.loss_rate * 100)
        logger.info("  cwnd:        avg=%.0f bytes, %d changes", m.avg_cwnd_bytes, m.cwnd_changes)


# ---------------------------------------------------------------------------
# Multi-flow fairness
# ---------------------------------------------------------------------------

def jains_fairness_index(throughputs: List[float]) -> float:
    """Compute Jain's fairness index over a list of throughputs."""
    n = len(throughputs)
    if n == 0:
        return 0.0
    s = sum(throughputs)
    ss = sum(x * x for x in throughputs)
    if ss == 0:
        return 1.0
    return (s * s) / (n * ss)


def compare_flows(diagnostics_list: List[Diagnostics]) -> Dict[str, FlowMetrics]:
    """Compute metrics for multiple flows and add fairness index."""
    results = {}
    throughputs = []
    for diag in diagnostics_list:
        m = diag.compute_metrics()
        results[m.cca_name] = m
        throughputs.append(m.avg_throughput_mbps)
    fi = jains_fairness_index(throughputs)
    for m in results.values():
        m.jains_fairness_index = fi
    return results
