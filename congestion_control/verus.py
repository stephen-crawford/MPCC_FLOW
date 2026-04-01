"""Verus: Adaptive Congestion Control for Unpredictable Cellular Networks.

Reimplementation based on Zaki, Potsch, Chen, Subramanian, Gorg (SIGCOMM 2015).

Core idea: instead of predicting cellular channel dynamics, Verus continuously
learns a *delay profile* -- a mapping from sending window size (W) to observed
delay (D). The protocol uses small epsilon-steps every epoch to explore the
(W,D) space and adjusts the window based on the delay profile and observed
delay changes. On loss, it applies multiplicative decrease.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from congestion_control.base import AckInfo, CCAState, CongestionController, LossInfo

logger = logging.getLogger("cc.verus")


class DelayProfile:
    """Learned mapping from sending window (packets) to expected delay (ms).

    Maintains (W, D) data points and provides interpolated lookup.
    Updated continuously via EWMA as new measurements arrive.
    """

    def __init__(self, ewma_alpha: float = 0.3, max_points: int = 500):
        self._alpha = ewma_alpha
        self._max_points = max_points
        # Sparse storage: window_pkts -> smoothed_delay_ms
        self._data: dict[int, float] = {}
        self._sorted_windows: List[int] = []
        self._sorted_delays: List[float] = []
        self._dirty = True

    def update(self, window_pkts: int, delay_ms: float) -> None:
        """Add or update a (window, delay) measurement with EWMA smoothing."""
        w = max(1, window_pkts)
        if w in self._data:
            self._data[w] = (1 - self._alpha) * self._data[w] + self._alpha * delay_ms
        else:
            self._data[w] = delay_ms
            # Prune if too many points
            if len(self._data) > self._max_points:
                # Remove the oldest/smallest window entry
                min_w = min(self._data.keys())
                del self._data[min_w]
        self._dirty = True

    def lookup_window(self, target_delay_ms: float) -> Optional[int]:
        """Given a target delay, find the corresponding window size.

        Returns the window W such that the interpolated D(W) = target_delay_ms.
        If target_delay is below all recorded delays, returns the smallest window.
        If above all, returns the largest window.
        """
        self._rebuild()
        if not self._sorted_windows:
            return None

        # Binary search for target delay in the (sorted by window) arrays
        # The delay profile is generally monotonically increasing
        for i in range(len(self._sorted_delays) - 1):
            d0, d1 = self._sorted_delays[i], self._sorted_delays[i + 1]
            w0, w1 = self._sorted_windows[i], self._sorted_windows[i + 1]
            if d0 <= target_delay_ms <= d1 and d1 > d0:
                # Linear interpolation
                frac = (target_delay_ms - d0) / (d1 - d0)
                return int(w0 + frac * (w1 - w0))
            elif d1 <= target_delay_ms <= d0 and d0 > d1:
                frac = (target_delay_ms - d1) / (d0 - d1)
                return int(w1 + frac * (w0 - w1))

        # Extrapolate
        if target_delay_ms <= self._sorted_delays[0]:
            return self._sorted_windows[0]
        return self._sorted_windows[-1]

    def lookup_delay(self, window_pkts: int) -> Optional[float]:
        """Given a window size, find the expected delay."""
        self._rebuild()
        if not self._sorted_windows:
            return None
        if window_pkts <= self._sorted_windows[0]:
            return self._sorted_delays[0]
        if window_pkts >= self._sorted_windows[-1]:
            return self._sorted_delays[-1]
        # Linear interpolation
        for i in range(len(self._sorted_windows) - 1):
            w0, w1 = self._sorted_windows[i], self._sorted_windows[i + 1]
            if w0 <= window_pkts <= w1 and w1 > w0:
                frac = (window_pkts - w0) / (w1 - w0)
                return self._sorted_delays[i] + frac * (self._sorted_delays[i + 1] - self._sorted_delays[i])
        return self._sorted_delays[-1]

    def has_data(self) -> bool:
        return len(self._data) >= 2

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        items = sorted(self._data.items())
        self._sorted_windows = [w for w, _ in items]
        self._sorted_delays = [d for _, d in items]
        self._dirty = False


class Verus(CongestionController):
    """Verus congestion controller with delay-profile-based adaptation.

    Parameters match the paper:
        epsilon:  5ms epoch length
        delta_1:  1ms conservative delay change
        delta_2:  2ms aggressive delay change
        R:        max tolerable D_max/D_min ratio (tradeoff knob)
        M:        multiplicative decrease factor on loss
    """

    def __init__(
        self,
        epsilon_ms: float = 5.0,
        delta_1_ms: float = 1.0,
        delta_2_ms: float = 2.0,
        R: float = 4.0,
        M: float = 0.5,
        profile_update_interval_s: float = 1.0,
        **kwargs,
    ):
        super().__init__(name="Verus", **kwargs)

        # Verus parameters
        self._epsilon_s = epsilon_ms / 1000.0
        self._delta_1_ms = delta_1_ms
        self._delta_2_ms = delta_2_ms
        self._R = R
        self._M = M
        self._profile_update_s = profile_update_interval_s

        # Delay tracking
        self._delay_profile = DelayProfile()
        self._d_max_ewma_ms: float = 0.0
        self._d_max_prev_ms: float = 0.0
        self._d_min_ms: float = float("inf")
        self._d_est_ms: float = 0.0  # estimated target delay
        self._epoch_delays: List[float] = []
        self._alpha = 0.7  # EWMA weight for D_max

        # Epoch tracking
        self._epoch_start_s: float = 0.0
        self._epoch_number: int = 0
        self._last_profile_update_s: float = 0.0

        # Window in packets
        self._window_pkts: int = self.initial_cwnd // self.mtu
        self._cwnd = float(self._window_pkts * self.mtu)

        # Slow start
        self._in_slow_start = True
        self._slow_start_threshold_delay_factor = 15  # exit when RTT > N * D_min

        # Loss recovery
        self._in_recovery = False

        self._state = CCAState.SLOW_START

        logger.debug("Verus initialized: eps=%.0fms, d1=%.1f, d2=%.1f, R=%.1f, M=%.2f",
                      epsilon_ms, delta_1_ms, delta_2_ms, R, M)

    def on_ack(self, ack: AckInfo) -> None:
        self._update_rtt(ack.rtt_s)
        self._bytes_delivered += ack.bytes_acked
        self._bytes_in_flight = max(0, self._bytes_in_flight - ack.bytes_acked)
        delay_ms = ack.rtt_s * 1000

        # Track minimum delay
        if delay_ms < self._d_min_ms:
            self._d_min_ms = delay_ms

        # Collect delay samples for this epoch
        self._epoch_delays.append(delay_ms)

        # Update delay profile with current (window, delay) pair
        self._delay_profile.update(self._window_pkts, delay_ms)

        # Slow start phase
        if self._in_slow_start:
            self._do_slow_start(ack, delay_ms)
            return

        # Recovery phase (after loss)
        if self._in_recovery:
            self._do_recovery(ack)
            return

        # Check epoch boundary
        elapsed = ack.timestamp_s - self._epoch_start_s
        if elapsed >= self._epsilon_s:
            self._end_epoch(ack.timestamp_s)

    def _do_slow_start(self, ack: AckInfo, delay_ms: float) -> None:
        """Verus slow start: exponential window growth until delay threshold."""
        old_cwnd = self._cwnd
        self._window_pkts += 1
        self._cwnd = float(self._window_pkts * self.mtu)
        self._notify_cwnd_change(old_cwnd, self._cwnd, ack.timestamp_s, "slow_start")

        # Exit condition: delay exceeds threshold or loss
        if self._d_min_ms > 0 and delay_ms > self._slow_start_threshold_delay_factor * self._d_min_ms:
            self._in_slow_start = False
            self._state = CCAState.STEADY
            self._epoch_start_s = ack.timestamp_s
            self._d_est_ms = delay_ms
            logger.debug("Verus: exiting slow start at W=%d, delay=%.1fms",
                          self._window_pkts, delay_ms)

    def _do_recovery(self, ack: AckInfo) -> None:
        """TCP-like additive increase during loss recovery."""
        old_cwnd = self._cwnd
        # Additive increase: 1/W per ACK (like TCP)
        self._window_pkts = max(1, self._window_pkts)
        increment = max(1, 1)  # at least 1 packet per W acks
        self._cwnd += (self.mtu * self.mtu) / self._cwnd  # Reno-style AI
        self._window_pkts = int(self._cwnd / self.mtu)

        # Exit recovery when window reaches pre-loss sending window
        # In practice: when we get an ACK with window <= current
        self._in_recovery = False
        self._state = CCAState.STEADY
        self._epoch_start_s = ack.timestamp_s
        self._notify_cwnd_change(old_cwnd, self._cwnd, ack.timestamp_s, "recovery_ai")

    def _end_epoch(self, timestamp_s: float) -> None:
        """Process the end of a Verus epoch: update D_max, adjust window."""
        if not self._epoch_delays:
            self._epoch_start_s = timestamp_s
            return

        # Compute max delay this epoch
        epoch_max_delay = max(self._epoch_delays)
        self._epoch_delays.clear()

        # EWMA update of D_max (Eq. 2 from paper)
        if self._d_max_ewma_ms == 0:
            self._d_max_ewma_ms = epoch_max_delay
        else:
            self._d_max_ewma_ms = (self._alpha * self._d_max_ewma_ms
                                   + (1 - self._alpha) * epoch_max_delay)

        # Delta_D: change in maximum delay (Eq. 3)
        delta_d = self._d_max_ewma_ms - self._d_max_prev_ms
        self._d_max_prev_ms = self._d_max_ewma_ms

        # Window estimation (Eq. 4 from paper)
        old_cwnd = self._cwnd
        if self._d_min_ms > 0 and self._d_max_ewma_ms / self._d_min_ms > self._R:
            # Aggressively decrease: delay ratio exceeded threshold
            self._d_est_ms = max(self._d_min_ms, self._d_est_ms - self._delta_2_ms)
        elif delta_d > 0:
            # Delay increasing: moderately decrease
            self._d_est_ms = max(self._d_min_ms, self._d_est_ms - self._delta_1_ms)
        else:
            # Delay stable or decreasing: increase
            self._d_est_ms += self._delta_2_ms

        # Look up target window from delay profile
        if self._delay_profile.has_data():
            target_w = self._delay_profile.lookup_window(self._d_est_ms)
            if target_w is not None and target_w > 0:
                # Smooth transition (Eq. 5 from paper)
                n = max(1, int(math.ceil(self._srtt_s / self._epsilon_s))) if self._srtt_s > 0 else 1
                new_w = max(1, int(self._window_pkts + (2 - n) / max(n - 1, 1) * (target_w - self._window_pkts)))
                self._window_pkts = max(1, new_w)
        else:
            # No profile yet: use simple additive adjustment
            if delta_d <= 0:
                self._window_pkts += 1
            else:
                self._window_pkts = max(1, self._window_pkts - 1)

        self._cwnd = float(self._window_pkts * self.mtu)
        self._epoch_start_s = timestamp_s
        self._epoch_number += 1
        self._notify_cwnd_change(old_cwnd, self._cwnd, timestamp_s,
                                 f"epoch_{self._epoch_number}_dd={delta_d:.1f}")

    def on_loss(self, loss: LossInfo) -> None:
        """Multiplicative decrease on loss (Eq. 6 from paper)."""
        self._bytes_lost += loss.bytes_lost
        old_cwnd = self._cwnd

        # W_{i+1} = M * W_loss
        self._window_pkts = max(1, int(self._window_pkts * self._M))
        self._cwnd = float(self._window_pkts * self.mtu)
        self._ssthresh = self._cwnd

        # Enter recovery
        self._in_recovery = True
        self._in_slow_start = False
        self._state = CCAState.RECOVERY

        # Freeze delay profile during recovery
        self._notify_cwnd_change(old_cwnd, self._cwnd, loss.timestamp_s,
                                 "loss_md")
        logger.debug("Verus: loss at seq=%d, MD to W=%d",
                      loss.seq_num, self._window_pkts)

    def get_cwnd(self) -> int:
        return max(int(self._cwnd), self.mtu)

    def reset(self) -> None:
        super().reset()
        self._delay_profile = DelayProfile()
        self._d_max_ewma_ms = 0.0
        self._d_max_prev_ms = 0.0
        self._d_min_ms = float("inf")
        self._d_est_ms = 0.0
        self._epoch_delays.clear()
        self._window_pkts = self.initial_cwnd // self.mtu
        self._cwnd = float(self._window_pkts * self.mtu)
        self._in_slow_start = True
        self._in_recovery = False
        self._epoch_number = 0
