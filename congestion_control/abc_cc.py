"""ABC: A Simple Explicit Congestion Controller for Wireless Networks.

Reimplementation based on Goyal, Arun, Agarwal, Netravali, Alizadeh,
Balakrishnan (NSDI 2020).

ABC is an *explicit* congestion control scheme where the access point (AP)
computes a target rate based on observed link capacity and queue state, then
signals this rate to the sender via ACK modifications (ECN-like).  The sender
adjusts its window to match the signaled rate.

In our emulation, we simulate the AP-side computation and embed the rate
signal in the ACK path (since we control both sides).

ABC's core formula at the AP:
    target_rate = link_capacity * (1 - queue_fraction * eta)

where:
    link_capacity = observed link throughput
    queue_fraction = queue_occupancy / queue_capacity
    eta = aggressiveness parameter (0.5-0.9)

The sender interprets the target rate as:
    cwnd = target_rate * RTT
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional, Tuple

from congestion_control.base import AckInfo, CCAState, CongestionController, LossInfo
from congestion_control.network_model import BandwidthEstimator, DeliveryRateEstimator

logger = logging.getLogger("cc.abc")


class ABCAccessPoint:
    """Simulates the ABC access point logic.

    The AP observes:
    - Current link capacity (from delivery rate or configured)
    - Queue occupancy at the bottleneck
    And computes a target sending rate for the sender.
    """

    def __init__(
        self,
        queue_capacity_bytes: int = 150_000,
        eta: float = 0.7,
        link_capacity_bps: float = 0.0,
    ):
        self.queue_capacity_bytes = queue_capacity_bytes
        self.eta = eta
        self._link_capacity_bps = link_capacity_bps
        self._bw_estimator = BandwidthEstimator(window_s=2.0)
        self._queue_bytes: int = 0

    def update_link_capacity(self, timestamp_s: float, rate_bps: float) -> None:
        """Update observed link capacity."""
        self._bw_estimator.add_sample(timestamp_s, rate_bps)
        self._link_capacity_bps = self._bw_estimator.smooth_bw_bps

    def update_queue(self, queue_bytes: int) -> None:
        self._queue_bytes = queue_bytes

    def compute_target_rate_bps(self) -> float:
        """ABC target rate: capacity * (1 - queue_fraction * eta)."""
        capacity = self._link_capacity_bps
        if capacity <= 0:
            return 0.0
        q_frac = self._queue_bytes / max(self.queue_capacity_bytes, 1)
        q_frac = min(q_frac, 1.0)
        target = capacity * (1.0 - q_frac * self.eta)
        return max(target, 0.0)


class ABC(CongestionController):
    """ABC congestion controller (sender side).

    Receives rate signals from the simulated access point and adjusts
    cwnd accordingly: cwnd = target_rate * RTT.

    When no AP signal is available (e.g., legacy path), falls back to
    standard AIMD behavior.
    """

    def __init__(
        self,
        eta: float = 0.7,
        queue_capacity_bytes: int = 150_000,
        aimd_increase_bytes: int = 1500,
        aimd_decrease_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(name="ABC", **kwargs)
        self._ap = ABCAccessPoint(
            queue_capacity_bytes=queue_capacity_bytes,
            eta=eta,
        )
        self._aimd_increase = aimd_increase_bytes
        self._aimd_decrease = aimd_decrease_factor
        self._target_rate_bps: float = 0.0
        self._delivery_rate = DeliveryRateEstimator(window_s=1.0)
        self._has_ap_signal = False

        # For simulating AP feedback: track queue state
        self._estimated_queue_bytes: int = 0

        logger.debug("ABC initialized: eta=%.2f, queue_cap=%d bytes",
                      eta, queue_capacity_bytes)

    def on_ack(self, ack: AckInfo) -> None:
        self._update_rtt(ack.rtt_s)
        self._bytes_delivered += ack.bytes_acked
        self._bytes_in_flight = max(0, self._bytes_in_flight - ack.bytes_acked)

        # Update delivery rate (simulates AP observing link throughput)
        rate = self._delivery_rate.on_ack(ack.timestamp_s, ack.delivered_bytes)
        self._ap.update_link_capacity(ack.timestamp_s, rate)

        # Estimate queue occupancy from RTT inflation
        queue_delay_s = max(0, ack.rtt_s - self._min_rtt_s)
        if rate > 0:
            self._estimated_queue_bytes = int(queue_delay_s * rate / 8)
        self._ap.update_queue(self._estimated_queue_bytes)

        # Get AP target rate
        target_rate = self._ap.compute_target_rate_bps()
        self._target_rate_bps = target_rate

        old_cwnd = self._cwnd
        if target_rate > 0 and self._srtt_s > 0:
            self._has_ap_signal = True
            # cwnd = target_rate (bits/sec) / 8 * RTT (sec) = bytes deliverable per RTT
            target_cwnd = (target_rate / 8) * self._srtt_s
            # Smooth transition: don't jump more than 2x
            max_cwnd = max(self._cwnd * 2, self.initial_cwnd)
            self._cwnd = min(target_cwnd, max_cwnd)
            self._cwnd = max(self._cwnd, float(self.mtu))
            self._state = CCAState.STEADY
            self._notify_cwnd_change(old_cwnd, self._cwnd, ack.timestamp_s, "ap_signal")
        else:
            # Fallback: AIMD
            self._has_ap_signal = False
            if self._state == CCAState.SLOW_START:
                if self._cwnd < self._ssthresh:
                    self._cwnd += ack.bytes_acked
                else:
                    self._state = CCAState.STEADY
                    self._cwnd += self._aimd_increase * self.mtu / self._cwnd
            else:
                self._cwnd += self._aimd_increase * self.mtu / self._cwnd
            self._notify_cwnd_change(old_cwnd, self._cwnd, ack.timestamp_s, "aimd_ai")

    def on_loss(self, loss: LossInfo) -> None:
        self._bytes_lost += loss.bytes_lost
        old_cwnd = self._cwnd

        if self._has_ap_signal:
            # With AP signal, loss is likely stochastic (wireless), not congestion
            # Reduce less aggressively
            self._cwnd *= 0.8
        else:
            # Standard multiplicative decrease
            self._cwnd *= self._aimd_decrease

        self._cwnd = max(self._cwnd, float(self.mtu))
        self._ssthresh = self._cwnd
        self._state = CCAState.RECOVERY
        self._notify_cwnd_change(old_cwnd, self._cwnd, loss.timestamp_s, "loss_md")

    def get_cwnd(self) -> int:
        return max(int(self._cwnd), self.mtu)

    def get_pacing_rate(self) -> Optional[float]:
        if self._target_rate_bps > 0:
            return self._target_rate_bps / 8  # bytes per second
        return None

    def reset(self) -> None:
        super().reset()
        self._target_rate_bps = 0.0
        self._delivery_rate.reset()
        self._has_ap_signal = False
        self._estimated_queue_bytes = 0
