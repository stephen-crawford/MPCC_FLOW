"""Thin wrapper around NetworkMPCCSolver / NetworkMPCCQPSolver.

Exposes a single `step(Observation) -> send_rate` entry point for CCP
datapath code and test harnesses. Keeps a light loss handler: on a loss
signal we halve the send rate (AIMD multiplicative decrease) and short-circuit
the solver to honour Chiu & Jain style fairness guarantees between solves.
"""

from __future__ import annotations

from dataclasses import dataclass

from planning.network_solver import (
    NetworkMPCCResult,
    NetworkMPCCSolver,
    NetworkMPCCQPSolver,
    create_solver,
)


@dataclass
class Observation:
    tput: float
    rtt: float
    queue: float
    bw_est: float
    loss: int = 0


class MPCCController:

    def __init__(self, config: dict | None = None, solver_type: str = "qp"):
        self.config = config or {}
        net_cfg = self.config.get("network", {})
        self.rtt_prop = net_cfg.get("rtt_prop", 0.025)
        self.rate_max = net_cfg.get("rate_max", 50e6)
        self.solver_type = solver_type.lower()

        self.solver = create_solver(self.config, mode=self.solver_type)

        self._last_rate = self.rate_max * 0.5
        self._last_result: NetworkMPCCResult | None = None

    def step(self, obs: Observation) -> float:
        if obs.loss > 0:
            # AIMD multiplicative decrease — see Sec. IV proof sketch.
            self._last_rate = max(self._last_rate * 0.5, self.rate_max * 0.05)
            return self._last_rate

        result = self.solver.solve(
            obs.tput,
            obs.rtt,
            obs.queue,
            obs.bw_est,
            u_prev_bps=self._last_rate,
        )
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
        if rtt_ratio > 1.5:
            return self._last_rate
        return min(self._last_rate * 1.05, self.rate_max * 0.95)

    @property
    def last_result(self) -> NetworkMPCCResult | None:
        return self._last_result


__all__ = [
    "MPCCController",
    "Observation",
    "NetworkMPCCResult",
    "NetworkMPCCSolver",
    "NetworkMPCCQPSolver",
]
