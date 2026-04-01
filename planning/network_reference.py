"""
Reference trajectory for MPCC congestion control in throughput-delay space.

Generates a parametric curve Gamma(theta) where theta is utilization [0, 1]:
    x_ref(theta) = bw_est * theta          (throughput)
    y_ref(theta) = rtt_prop + alpha * theta^2  (RTT)

This maps directly into the existing ReferencePath type so ContouringObjective
computes contouring/lag errors without modification.
"""

from __future__ import annotations

import numpy as np
from planning.types import ReferencePath
from utils.math_tools import TKSpline


def generate_network_reference(
    bw_est: float,
    rtt_prop: float = 0.025,
    alpha: float = 0.150,
    num_points: int = 51,
) -> ReferencePath:
    ref = ReferencePath()

    theta = np.linspace(0.0, 1.0, num_points)
    tput_ref = bw_est * theta
    rtt_ref = rtt_prop + alpha * theta ** 2

    dx = np.diff(tput_ref)
    dy = np.diff(rtt_ref)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])

    dtput = np.gradient(tput_ref, s)
    drtt = np.gradient(rtt_ref, s)
    psi = np.arctan2(drtt, dtput)

    ref.x = tput_ref.tolist()
    ref.y = rtt_ref.tolist()
    ref.s = s.tolist()
    ref.psi = psi.tolist()
    ref.v = [1.0] * num_points

    ref.x_spline = TKSpline(s, tput_ref)
    ref.y_spline = TKSpline(s, rtt_ref)
    ref.v_spline = TKSpline(s, np.ones(num_points))

    return ref


class AdaptiveNetworkReference:

    def __init__(
        self,
        rtt_prop: float = 0.025,
        alpha: float = 0.150,
        num_points: int = 51,
        update_threshold: float = 0.10,
    ):
        self.rtt_prop = rtt_prop
        self.alpha = alpha
        self.num_points = num_points
        self.update_threshold = update_threshold
        self._last_bw_est = None
        self._reference = None

    def get_reference(self, bw_est: float) -> ReferencePath:
        if self._should_update(bw_est):
            self._reference = generate_network_reference(
                bw_est=bw_est,
                rtt_prop=self.rtt_prop,
                alpha=self.alpha,
                num_points=self.num_points,
            )
            self._last_bw_est = bw_est
        return self._reference

    def _should_update(self, bw_est: float) -> bool:
        if self._last_bw_est is None or self._reference is None:
            return True
        if self._last_bw_est == 0:
            return True
        frac_change = abs(bw_est - self._last_bw_est) / self._last_bw_est
        return frac_change > self.update_threshold

    @property
    def reference(self) -> ReferencePath | None:
        return self._reference
