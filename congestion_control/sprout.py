"""Sprout: Stochastic Forecasts Achieve High Throughput and Low Delay.

Reimplementation based on Winstein, Sivaraman, Balakrishnan (NSDI 2013).

Core idea: the receiver infers the varying Poisson rate (lambda) of the
bottleneck link from packet inter-arrival times using Bayesian inference.
It then makes a cautious (5th-percentile) forecast of how many bytes will
be delivered over the next several ticks, and the sender uses this forecast
as a window.

Simplification: since our emulation delivers ACKs with measured RTT, we
run the Bayesian inference at the sender side using ACK arrival patterns,
which is equivalent to receiver-side inference piggybacked on ACKs.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional

import numpy as np

from congestion_control.base import AckInfo, CCAState, CongestionController, LossInfo

logger = logging.getLogger("cc.sprout")


class Sprout(CongestionController):
    """Sprout congestion controller with Bayesian rate forecasting."""

    # --- Sprout parameters (from the paper) ---
    NUM_LAMBDA = 256               # discrete lambda values
    MAX_LAMBDA = 1000              # max packets-per-second
    TICK_S = 0.020                 # 20ms tick interval
    NUM_FORECAST_TICKS = 8         # forecast 8 ticks = 160ms ahead
    SIGMA = 200.0                  # Brownian motion noise (pps/sqrt(s))
    LAMBDA_ESCAPE = 1.0            # outage escape rate
    CONFIDENCE = 0.05              # 5th percentile (95% cautious)

    def __init__(self, **kwargs):
        super().__init__(name="Sprout", **kwargs)

        # Discrete lambda values: 0, MAX_LAMBDA/255, ..., MAX_LAMBDA
        self._lambdas = np.linspace(0, self.MAX_LAMBDA, self.NUM_LAMBDA)

        # Probability distribution over lambda (uniform prior)
        self._prob = np.ones(self.NUM_LAMBDA) / self.NUM_LAMBDA

        # Precompute Brownian transition matrix for one tick
        dt = self.TICK_S
        sigma_dt = self.SIGMA * math.sqrt(dt)
        self._transition = self._build_transition_matrix(sigma_dt)

        # Precompute Poisson PMF cache for common (lambda*tau, k) pairs
        self._poisson_cache = {}

        # State tracking
        self._last_tick_time_s: float = 0.0
        self._bytes_this_tick: int = 0
        self._queue_estimate_bytes: int = 0
        self._forecast_window_bytes: int = 0
        self._state = CCAState.STARTUP

        logger.debug("Sprout initialized: %d lambda values, sigma=%.0f, tick=%.0fms",
                      self.NUM_LAMBDA, self.SIGMA, self.TICK_S * 1000)

    def _build_transition_matrix(self, sigma_dt: float) -> np.ndarray:
        """Build Gaussian transition matrix for Brownian motion on lambda."""
        n = self.NUM_LAMBDA
        mat = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                # Outage state: escape with rate lambda_escape
                p_escape = 1.0 - math.exp(-self.LAMBDA_ESCAPE * self.TICK_S)
                mat[i, 0] = 1.0 - p_escape
                # Spread escape probability across non-zero lambdas
                if n > 1:
                    mat[i, 1:] = p_escape / (n - 1)
            else:
                lam = self._lambdas[i]
                # Gaussian centered at current lambda
                for j in range(n):
                    diff = self._lambdas[j] - lam
                    mat[i, j] = math.exp(-0.5 * (diff / max(sigma_dt, 1e-6)) ** 2)
                # Bias toward outage if close
                row_sum = mat[i].sum()
                if row_sum > 0:
                    mat[i] /= row_sum
        return mat

    def _evolve_distribution(self) -> None:
        """Apply Brownian motion transition to probability distribution."""
        self._prob = self._transition.T @ self._prob
        total = self._prob.sum()
        if total > 0:
            self._prob /= total

    def _observe(self, packets_arrived: int) -> None:
        """Bayesian update: multiply by Poisson likelihood of observed count."""
        tau = self.TICK_S
        for i in range(self.NUM_LAMBDA):
            lam = self._lambdas[i]
            rate = lam * tau
            # Poisson PMF: P(k | rate) = rate^k * exp(-rate) / k!
            if rate < 1e-10:
                # Lambda ~ 0: only P(k=0) is non-negligible
                likelihood = 1.0 if packets_arrived == 0 else 1e-30
            else:
                log_lik = (packets_arrived * math.log(rate) - rate
                           - math.lgamma(packets_arrived + 1))
                likelihood = math.exp(min(log_lik, 500))  # prevent overflow
            self._prob[i] *= likelihood
        total = self._prob.sum()
        if total > 1e-30:
            self._prob /= total
        else:
            # Distribution collapsed; reset to uniform
            self._prob[:] = 1.0 / self.NUM_LAMBDA

    def _make_forecast(self) -> int:
        """Forecast bytes deliverable in next NUM_FORECAST_TICKS ticks (5th pctile)."""
        # Evolve distribution forward without observation, accumulating
        # the distribution of cumulative deliveries
        prob = self._prob.copy()
        total_forecast_packets = 0.0

        for tick in range(self.NUM_FORECAST_TICKS):
            # Evolve
            prob = self._transition.T @ prob
            total = prob.sum()
            if total > 0:
                prob /= total
            # Expected delivery this tick (cautious: 5th percentile)
            # For efficiency, compute weighted 5th percentile of lambda
            cum = 0.0
            p5_lambda = 0.0
            for i in range(self.NUM_LAMBDA):
                cum += prob[i]
                if cum >= self.CONFIDENCE:
                    p5_lambda = self._lambdas[i]
                    break
            total_forecast_packets += p5_lambda * self.TICK_S

        return int(total_forecast_packets * self.mtu)

    def on_ack(self, ack: AckInfo) -> None:
        self._update_rtt(ack.rtt_s)
        self._bytes_delivered += ack.bytes_acked
        self._bytes_in_flight = max(0, self._bytes_in_flight - ack.bytes_acked)

        # Track packets arrived this tick
        self._bytes_this_tick += ack.bytes_acked

        # Check if a tick boundary has passed
        elapsed = ack.timestamp_s - self._last_tick_time_s
        if elapsed >= self.TICK_S:
            packets_this_tick = self._bytes_this_tick // self.mtu

            # 1. Evolve distribution
            self._evolve_distribution()
            # 2. Observe
            self._observe(packets_this_tick)

            # Update queue estimate
            self._queue_estimate_bytes = max(
                0, self._queue_estimate_bytes - self._bytes_this_tick)

            # Reset tick counters
            self._bytes_this_tick = 0
            self._last_tick_time_s = ack.timestamp_s

        # 3. Make forecast and set window
        forecast = self._make_forecast()
        safe_window = max(forecast - self._queue_estimate_bytes, self.mtu)

        old_cwnd = self._cwnd
        self._cwnd = float(safe_window)
        self._state = CCAState.STEADY
        self._notify_cwnd_change(old_cwnd, self._cwnd, ack.timestamp_s, "forecast")

    def on_loss(self, loss: LossInfo) -> None:
        self._bytes_lost += loss.bytes_lost
        # Sprout doesn't react to loss directly (it's forecast-driven)
        # but we update queue estimate
        self._queue_estimate_bytes = max(0, self._queue_estimate_bytes - loss.bytes_lost)

    def get_cwnd(self) -> int:
        return max(int(self._cwnd), self.mtu)

    def get_pacing_rate(self) -> Optional[float]:
        # Sprout paces based on forecast / time horizon
        if self._srtt_s > 0:
            return self._cwnd / self._srtt_s
        return None

    def reset(self) -> None:
        super().reset()
        self._prob[:] = 1.0 / self.NUM_LAMBDA
        self._last_tick_time_s = 0.0
        self._bytes_this_tick = 0
        self._queue_estimate_bytes = 0
        self._forecast_window_bytes = 0


class SproutEWMA(CongestionController):
    """Sprout-EWMA: simplified Sprout using EWMA rate estimate instead of Bayesian inference.

    Higher throughput but also higher delay than full Sprout.
    """

    EWMA_ALPHA = 0.3
    TICK_S = 0.020
    NUM_FORECAST_TICKS = 8

    def __init__(self, **kwargs):
        super().__init__(name="SproutEWMA", **kwargs)
        self._rate_estimate_pps: float = 0.0  # packets per second
        self._bytes_this_tick: int = 0
        self._last_tick_time_s: float = 0.0
        self._queue_estimate_bytes: int = 0

    def on_ack(self, ack: AckInfo) -> None:
        self._update_rtt(ack.rtt_s)
        self._bytes_delivered += ack.bytes_acked
        self._bytes_in_flight = max(0, self._bytes_in_flight - ack.bytes_acked)
        self._bytes_this_tick += ack.bytes_acked

        elapsed = ack.timestamp_s - self._last_tick_time_s
        if elapsed >= self.TICK_S:
            observed_pps = (self._bytes_this_tick / self.mtu) / max(elapsed, 1e-6)
            if self._rate_estimate_pps == 0.0:
                self._rate_estimate_pps = observed_pps
            else:
                self._rate_estimate_pps = (
                    (1 - self.EWMA_ALPHA) * self._rate_estimate_pps
                    + self.EWMA_ALPHA * observed_pps
                )
            self._queue_estimate_bytes = max(
                0, self._queue_estimate_bytes - self._bytes_this_tick)
            self._bytes_this_tick = 0
            self._last_tick_time_s = ack.timestamp_s

        # Forecast: constant rate for NUM_FORECAST_TICKS
        forecast_bytes = int(
            self._rate_estimate_pps * self.TICK_S * self.NUM_FORECAST_TICKS * self.mtu
        )
        safe_window = max(forecast_bytes - self._queue_estimate_bytes, self.mtu)

        old_cwnd = self._cwnd
        self._cwnd = float(safe_window)
        self._state = CCAState.STEADY
        self._notify_cwnd_change(old_cwnd, self._cwnd, ack.timestamp_s, "ewma_forecast")

    def on_loss(self, loss: LossInfo) -> None:
        self._bytes_lost += loss.bytes_lost
        self._queue_estimate_bytes = max(0, self._queue_estimate_bytes - loss.bytes_lost)

    def get_cwnd(self) -> int:
        return max(int(self._cwnd), self.mtu)

    def reset(self) -> None:
        super().reset()
        self._rate_estimate_pps = 0.0
        self._bytes_this_tick = 0
        self._last_tick_time_s = 0.0
        self._queue_estimate_bytes = 0
