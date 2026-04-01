"""Network state and dynamics for congestion control.

Provides NetworkState (observable network variables) and helper functions
for bandwidth/delay estimation used across CCA implementations.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple


@dataclass
class NetworkState:
    """Observable network state at a point in time."""
    timestamp_s: float = 0.0
    cwnd_bytes: int = 0
    bytes_in_flight: int = 0
    rtt_s: float = 0.0
    min_rtt_s: float = float("inf")
    srtt_s: float = 0.0
    throughput_bps: float = 0.0
    delivery_rate_bps: float = 0.0
    loss_rate: float = 0.0
    queue_delay_s: float = 0.0       # estimated queuing delay = rtt - min_rtt


class RTTEstimator:
    """Windowed RTT tracker with min/max/smoothed estimates."""

    def __init__(self, window_s: float = 10.0):
        self._window_s = window_s
        self._samples: Deque[Tuple[float, float]] = deque()  # (timestamp, rtt)
        self.min_rtt_s: float = float("inf")
        self.max_rtt_s: float = 0.0
        self.latest_rtt_s: float = 0.0
        self.srtt_s: float = 0.0
        self.rttvar_s: float = 0.0

    def add_sample(self, timestamp_s: float, rtt_s: float) -> None:
        self.latest_rtt_s = rtt_s
        self._samples.append((timestamp_s, rtt_s))
        self._expire(timestamp_s)
        if rtt_s < self.min_rtt_s:
            self.min_rtt_s = rtt_s
        if rtt_s > self.max_rtt_s:
            self.max_rtt_s = rtt_s
        # RFC 6298 EWMA
        if self.srtt_s == 0.0:
            self.srtt_s = rtt_s
            self.rttvar_s = rtt_s / 2
        else:
            self.rttvar_s = 0.75 * self.rttvar_s + 0.25 * abs(self.srtt_s - rtt_s)
            self.srtt_s = 0.875 * self.srtt_s + 0.125 * rtt_s

    def windowed_min(self) -> float:
        if not self._samples:
            return self.min_rtt_s
        return min(rtt for _, rtt in self._samples)

    def windowed_max(self) -> float:
        if not self._samples:
            return self.max_rtt_s
        return max(rtt for _, rtt in self._samples)

    def queue_delay_s(self) -> float:
        return max(0.0, self.latest_rtt_s - self.min_rtt_s)

    def _expire(self, now_s: float) -> None:
        cutoff = now_s - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def reset(self) -> None:
        self._samples.clear()
        self.min_rtt_s = float("inf")
        self.max_rtt_s = 0.0
        self.latest_rtt_s = 0.0
        self.srtt_s = 0.0
        self.rttvar_s = 0.0


class DeliveryRateEstimator:
    """Estimates delivery rate from ACK arrivals (bytes/sec)."""

    def __init__(self, window_s: float = 1.0):
        self._window_s = window_s
        self._samples: Deque[Tuple[float, int]] = deque()  # (timestamp, cumulative_bytes)
        self.rate_bps: float = 0.0

    def on_ack(self, timestamp_s: float, cumulative_delivered: int) -> float:
        self._samples.append((timestamp_s, cumulative_delivered))
        self._expire(timestamp_s)
        if len(self._samples) >= 2:
            dt = self._samples[-1][0] - self._samples[0][0]
            db = self._samples[-1][1] - self._samples[0][1]
            if dt > 0:
                self.rate_bps = (db * 8) / dt
        return self.rate_bps

    def _expire(self, now_s: float) -> None:
        cutoff = now_s - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def reset(self) -> None:
        self._samples.clear()
        self.rate_bps = 0.0


class BandwidthEstimator:
    """Max-filter and Kalman-style bandwidth estimators (inspired by LeoCC)."""

    def __init__(self, window_s: float = 5.0, alpha: float = 0.3):
        self._window_s = window_s
        self._rate_samples: Deque[Tuple[float, float]] = deque()  # (timestamp, rate_bps)
        self.max_bw_bps: float = 0.0   # aggressive estimator
        self.smooth_bw_bps: float = 0.0  # moderate estimator (EWMA)
        self._alpha = alpha  # EWMA smoothing (higher = faster tracking)

    def add_sample(self, timestamp_s: float, rate_bps: float) -> None:
        self._rate_samples.append((timestamp_s, rate_bps))
        self._expire(timestamp_s)
        # Max filter (aggressive)
        self.max_bw_bps = max(r for _, r in self._rate_samples) if self._rate_samples else 0.0
        # EWMA (moderate)
        if self.smooth_bw_bps == 0.0:
            self.smooth_bw_bps = rate_bps
        else:
            self.smooth_bw_bps = (1 - self._alpha) * self.smooth_bw_bps + self._alpha * rate_bps

    def _expire(self, now_s: float) -> None:
        cutoff = now_s - self._window_s
        while self._rate_samples and self._rate_samples[0][0] < cutoff:
            self._rate_samples.popleft()

    def reset(self) -> None:
        self._rate_samples.clear()
        self.max_bw_bps = 0.0
        self.smooth_bw_bps = 0.0
