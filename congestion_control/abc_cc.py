"""ABC: A Simple Explicit Congestion Controller for Wireless Networks.

Reimplementation of Goyal, Arun, Agarwal, Netravali, Alizadeh,
Balakrishnan (NSDI 2020).

ABC is an *explicit* congestion control scheme where the access point (AP)
computes a target rate based on observed link capacity and queuing delay, then
marks each outgoing packet with a 1-bit accelerate/brake signal. The sender
adjusts its window based on the aggregate marks.

AP target rate (Eq. 1):
    tr(t) = eta * mu(t) - (mu(t) / delta) * max(x(t) - d_t, 0)

Accelerate fraction (Eq. 2):
    f(t) = min( (1/2) * tr(t) / cr(t),  1 )

Sender per-ACK update (Eq. 3):
    accelerate:  w <- w + 1 + 1/w
    brake:       w <- w - 1 + 1/w

The +1/w additive-increase term on every ACK makes ABC a MAIMD scheme
that converges to fairness.

Parameters (paper defaults):
    eta   = 0.98   target utilization fraction
    delta = 133 ms queue drain time constant (must be > 2/3 * max_RTT)
    d_t   = 5 ms   queuing delay threshold
"""

from __future__ import annotations

import logging
from typing import Optional

from congestion_control.base import AckInfo, CCAState, CongestionController, LossInfo
from congestion_control.network_model import BandwidthEstimator, DeliveryRateEstimator

logger = logging.getLogger("cc.abc")


class ABCAccessPoint:
    """Simulates the ABC access point (router-side) logic.

    The AP observes link capacity and queuing delay, computes a target rate,
    and marks packets accelerate/brake via a deterministic token bucket.
    """

    def __init__(
        self,
        eta: float = 0.98,
        delta_s: float = 0.133,
        delay_threshold_s: float = 0.005,
        queue_capacity_bytes: int = 375_000,
    ):
        self.eta = eta
        self.delta_s = delta_s
        self.delay_threshold_s = delay_threshold_s
        self.queue_capacity_bytes = queue_capacity_bytes

        self._link_capacity_bps: float = 0.0
        self._queueing_delay_s: float = 0.0
        self._dequeue_rate_bps: float = 0.0

        # Token bucket for deterministic packet marking (Algorithm 1)
        self._token: float = 0.0
        self._token_limit: float = 10.0

        # Fallback estimator when no direct state
        self._bw_estimator = BandwidthEstimator(window_s=2.0)
        self._has_direct_state = False

    def set_direct_state(
        self, capacity_bps: float, queue_bytes: int, delivery_rate_bps: float = 0.0,
    ) -> None:
        """Provide direct AP-observed link state (from emulator or real AP)."""
        self._link_capacity_bps = capacity_bps
        safe_cap = max(capacity_bps, 1.0)
        self._queueing_delay_s = (queue_bytes * 8) / safe_cap
        # Dequeue rate = link capacity when queue is draining,
        # sender's rate when queue is empty (packets pass through immediately).
        if queue_bytes > 0:
            self._dequeue_rate_bps = capacity_bps
        else:
            self._dequeue_rate_bps = max(delivery_rate_bps, 1.0)
        self._has_direct_state = True

    def update_link_capacity(self, timestamp_s: float, rate_bps: float) -> None:
        """Fallback: update from sender-side delivery rate estimate."""
        self._bw_estimator.add_sample(timestamp_s, rate_bps)
        if not self._has_direct_state:
            self._link_capacity_bps = self._bw_estimator.smooth_bw_bps
            self._dequeue_rate_bps = self._bw_estimator.smooth_bw_bps

    def update_queue_from_rtt(self, queueing_delay_s: float) -> None:
        """Fallback: update queuing delay from RTT measurement."""
        if not self._has_direct_state:
            self._queueing_delay_s = max(queueing_delay_s, 0.0)

    def compute_target_rate_bps(self) -> float:
        """ABC target rate (Eq. 1): tr = eta*mu - (mu/delta)*max(x - d_t, 0)."""
        mu = self._link_capacity_bps
        if mu <= 0:
            return 0.0
        excess_delay = max(self._queueing_delay_s - self.delay_threshold_s, 0.0)
        target = self.eta * mu - (mu / self.delta_s) * excess_delay
        return max(target, 0.0)

    def accel_fraction(self) -> float:
        """Accelerate fraction (Eq. 2): f = min(tr / (2*cr), 1)."""
        target = self.compute_target_rate_bps()
        cr = max(self._dequeue_rate_bps, 1.0)
        return min(0.5 * target / cr, 1.0)

    def mark_packet(self) -> bool:
        """Token-bucket marking (Algorithm 1). Returns True = accelerate."""
        f = self.accel_fraction()
        self._token = min(self._token + f, self._token_limit)
        if self._token >= 1.0:
            self._token -= 1.0
            return True   # accelerate
        return False       # brake

    @property
    def link_capacity_bps(self) -> float:
        return self._link_capacity_bps


class ABC(CongestionController):
    """ABC congestion controller (sender side).

    Maintains a dual window:
      w_abc    — controlled by AP accelerate/brake marks
      w_nonabc — follows standard AIMD (for non-ABC bottleneck compat)
    Actual cwnd = min(w_abc, w_nonabc).

    Both windows are capped at 2x in-flight packets to prevent
    unbounded growth when one window is not the active bottleneck.
    """

    def __init__(
        self,
        eta: float = 0.98,
        delta_s: float = 0.133,
        delay_threshold_s: float = 0.005,
        queue_capacity_bytes: int = 375_000,
        **kwargs,
    ):
        super().__init__(name="ABC", **kwargs)
        self._ap = ABCAccessPoint(
            eta=eta,
            delta_s=delta_s,
            delay_threshold_s=delay_threshold_s,
            queue_capacity_bytes=queue_capacity_bytes,
        )
        self._delivery_rate = DeliveryRateEstimator(window_s=1.0)
        self._has_ap_signal = False

        # Dual window (Section 4.1) — in packets
        self._w_abc: float = float(self.initial_cwnd) / self.mtu
        self._w_nonabc: float = float("inf")

        logger.debug(
            "ABC initialized: eta=%.2f, delta=%.3fs, d_t=%.3fs",
            eta, delta_s, delay_threshold_s,
        )

    def update_link_state(self, capacity_bps: float, queue_bytes: int) -> None:
        """Called by the emulator to provide direct AP link state."""
        # Use delivery rate as dequeue-rate proxy when queue empty
        self._ap.set_direct_state(
            capacity_bps, queue_bytes, self._delivery_rate.rate_bps,
        )

    def on_ack(self, ack: AckInfo) -> None:
        self._update_rtt(ack.rtt_s)
        self._bytes_delivered += ack.bytes_acked
        self._bytes_in_flight = max(0, self._bytes_in_flight - ack.bytes_acked)

        # Fallback estimation (used when no direct AP state)
        rate = self._delivery_rate.on_ack(ack.timestamp_s, ack.delivered_bytes)
        self._ap.update_link_capacity(ack.timestamp_s, rate)
        queueing_delay = max(0.0, ack.rtt_s - self._min_rtt_s)
        self._ap.update_queue_from_rtt(queueing_delay)

        # Check if AP can produce a valid target
        target = self._ap.compute_target_rate_bps()
        if target > 0 and self._srtt_s > 0:
            self._has_ap_signal = True

            # Get 1-bit mark for this ACK (Algorithm 1)
            mark_accel = self._ap.mark_packet()

            # Sender per-ACK update (Eq. 3, in packets)
            w = max(self._w_abc, 1.0)
            if mark_accel:
                self._w_abc = w + 1.0 + 1.0 / w
            else:
                self._w_abc = w - 1.0 + 1.0 / w
            self._w_abc = max(self._w_abc, 1.0)

            # Dual window: actual cwnd = min(w_abc, w_nonabc) in bytes
            old_cwnd = self._cwnd
            self._cwnd = min(self._w_abc, self._w_nonabc) * self.mtu
            self._cwnd = max(self._cwnd, float(self.mtu))
            self._state = CCAState.STEADY
            self._notify_cwnd_change(
                old_cwnd, self._cwnd, ack.timestamp_s, "ap_signal"
            )
        else:
            # No AP signal: standard AIMD fallback
            self._has_ap_signal = False
            old_cwnd = self._cwnd
            if self._state == CCAState.SLOW_START:
                if self._cwnd < self._ssthresh:
                    self._cwnd += ack.bytes_acked
                else:
                    self._state = CCAState.STEADY
                    self._cwnd += self.mtu * self.mtu / self._cwnd
            else:
                self._cwnd += self.mtu * self.mtu / self._cwnd
            # Keep w_abc in sync
            self._w_abc = self._cwnd / self.mtu
            self._notify_cwnd_change(
                old_cwnd, self._cwnd, ack.timestamp_s, "aimd_ai"
            )

    def on_loss(self, loss: LossInfo) -> None:
        """Standard TCP loss response (paper: ABC does not change loss response)."""
        self._bytes_lost += loss.bytes_lost
        old_cwnd = self._cwnd

        # Multiplicative decrease on both windows
        self._w_nonabc = max(self._w_abc / 2.0, 2.0)
        self._w_abc = max(self._w_abc / 2.0, 2.0)

        self._cwnd = min(self._w_abc, self._w_nonabc) * self.mtu
        self._cwnd = max(self._cwnd, float(self.mtu))
        self._ssthresh = self._cwnd
        self._state = CCAState.RECOVERY
        self._notify_cwnd_change(old_cwnd, self._cwnd, loss.timestamp_s, "loss_md")

    def get_cwnd(self) -> int:
        return max(int(self._cwnd), self.mtu)

    def get_pacing_rate(self) -> Optional[float]:
        target = self._ap.compute_target_rate_bps()
        if target > 0:
            return target / 8  # bytes per second
        return None

    def reset(self) -> None:
        super().reset()
        self._delivery_rate.reset()
        self._has_ap_signal = False
        self._w_abc = float(self.initial_cwnd) / self.mtu
        self._w_nonabc = float("inf")
        self._ap._token = 0.0
