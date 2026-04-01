"""
Bandwidth estimators for MPCC congestion control.

Two modes:
1. EMA: Exponential moving average of observed throughput.
2. Sprout-style: Stochastic forecast modelling packet inter-arrivals as a
   Poisson process with time-varying rate (Winstein & Balakrishnan, NSDI 2013).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class AckInfo:
    timestamp: float
    send_timestamp: float
    seq_num: int
    bytes_acked: int


class EMAEstimator:

    def __init__(self, alpha: float = 0.3, initial_bw: float = 1e6):
        self.alpha = alpha
        self.bw_est = initial_bw
        self._last_time = None
        self._bytes_since_last = 0

    def update(self, ack: AckInfo) -> float:
        if self._last_time is None:
            self._last_time = ack.timestamp
            self._bytes_since_last = ack.bytes_acked
            return self.bw_est

        dt = ack.timestamp - self._last_time
        self._bytes_since_last += ack.bytes_acked

        if dt >= 0.010:
            measured_bw = self._bytes_since_last / dt
            self.bw_est = self.alpha * measured_bw + (1 - self.alpha) * self.bw_est
            self._last_time = ack.timestamp
            self._bytes_since_last = 0

        return self.bw_est

    def get_estimate(self) -> float:
        return self.bw_est

    def get_rtt(self, ack: AckInfo) -> float:
        return ack.timestamp - ack.send_timestamp


class SproutEstimator:

    def __init__(
        self,
        initial_bw: float = 1e6,
        window_s: float = 1.0,
        tick_s: float = 0.010,
        brownian_sigma: float = 200.0,
        forecast_percentile: float = 0.05,
    ):
        self.tick_s = tick_s
        self.brownian_sigma = brownian_sigma
        self.forecast_percentile = forecast_percentile

        max_ticks = int(window_s / tick_s) + 1
        self._tick_bytes: deque[float] = deque(maxlen=max_ticks)
        self._tick_start: float | None = None
        self._current_tick_bytes: float = 0.0

        self._rate_mean = initial_bw
        self._rate_var = (initial_bw * 0.5) ** 2

        self.bw_est = initial_bw

    def update(self, ack: AckInfo) -> float:
        if self._tick_start is None:
            self._tick_start = ack.timestamp

        self._current_tick_bytes += ack.bytes_acked

        elapsed = ack.timestamp - self._tick_start
        if elapsed >= self.tick_s:
            n_ticks = max(1, int(elapsed / self.tick_s))
            bytes_per_tick = self._current_tick_bytes / n_ticks

            for _ in range(n_ticks):
                self._tick_bytes.append(bytes_per_tick)

            self._tick_start = ack.timestamp
            self._current_tick_bytes = 0.0
            self._update_posterior()

        self.bw_est = self._rate_mean
        return self.bw_est

    def _update_posterior(self):
        if len(self._tick_bytes) < 2:
            return

        observed_rates = np.array(self._tick_bytes) / self.tick_s
        obs_mean = np.mean(observed_rates)
        obs_var = np.var(observed_rates) + 1e-6

        dt = self.tick_s * len(self._tick_bytes)
        pred_var = self._rate_var + self.brownian_sigma ** 2 * dt

        K = pred_var / (pred_var + obs_var)
        self._rate_mean = self._rate_mean + K * (obs_mean - self._rate_mean)
        self._rate_var = max((1 - K) * pred_var, 1e3)

    def forecast(self, horizon_s: float) -> float:
        forecast_var = self._rate_var + self.brownian_sigma ** 2 * horizon_s
        forecast_std = math.sqrt(forecast_var)
        z = _normal_ppf(self.forecast_percentile)
        return max(self._rate_mean + z * forecast_std, 0.0)

    def get_estimate(self) -> float:
        return self.bw_est


def _normal_ppf(p: float) -> float:
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p == 0.5:
        return 0.0
    if p > 0.5:
        return -_normal_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


def create_estimator(config: dict) -> EMAEstimator | SproutEstimator:
    net_cfg = config.get("network", {})
    estimator_type = net_cfg.get("estimator", "ema")
    initial_bw = net_cfg.get("rate_max", 1e6) * 0.5

    if estimator_type == "sprout":
        return SproutEstimator(initial_bw=initial_bw)
    else:
        alpha = net_cfg.get("ema_alpha", 0.3)
        return EMAEstimator(alpha=alpha, initial_bw=initial_bw)
