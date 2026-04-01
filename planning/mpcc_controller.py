"""
MPCC congestion controller.

Wraps NetworkMPCCSolver (NLP) or NetworkMPCCQPSolver (QP) to provide
a step(observation) -> send_rate interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from planning.network_solver import NetworkMPCCSolver, NetworkMPCCQPSolver, NetworkMPCCResult


@dataclass
class Observation:
    tput: float
    rtt: float
    queue: float
    bw_est: float
    loss: int = 0


class MPCCController:

    def __init__(self, config: dict | None = None, solver_type: str = "nlp"):
        self.config = config or {}
        net_cfg = self.config.get("network", {})
        self.rtt_prop = net_cfg.get("rtt_prop", 0.025)
        self.rate_max = net_cfg.get("rate_max", 50e6)

        if solver_type == "qp":
            self.solver = NetworkMPCCQPSolver(self.config)
        else:
            self.solver = NetworkMPCCSolver(self.config)

        self._last_rate = self.rate_max * 0.5
        self._last_result: NetworkMPCCResult | None = None

    def step(self, obs: Observation) -> float:
        if obs.loss > 0:
            self._last_rate = max(self._last_rate * 0.5, self.rate_max * 0.05)
            return self._last_rate

        result = self.solver.solve(obs.tput, obs.rtt, obs.queue, obs.bw_est)
        self._last_result = result

        if result.success:
            self._last_rate = result.send_rate
        else:
            self._last_rate = self._fallback_rate(obs)

        return self._last_rate

    def _fallback_rate(self, obs: Observation) -> float:
        rtt_ratio = obs.rtt / max(self.rtt_prop, 0.001)
        if rtt_ratio > 3.0:
            return max(self._last_rate * 0.8, self.rate_max * 0.05)
        elif rtt_ratio > 1.5:
            return self._last_rate
        else:
            return min(self._last_rate * 1.05, self.rate_max * 0.95)

    @property
    def last_result(self) -> NetworkMPCCResult | None:
        return self._last_result
