"""
MPCC NLP and QP solvers for congestion control.

NetworkMPCCSolver: full NLP via CasADi/IPOPT with contouring cost and fairness.
NetworkMPCCQPSolver: linearized QP approximation for real-time feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as cd
import numpy as np
from scipy import linalg as la


@dataclass
class NetworkMPCCResult:
    success: bool
    send_rate: float
    predicted_tput: np.ndarray | None = None
    predicted_rtt: np.ndarray | None = None
    solve_time_ms: float = 0.0


class NetworkMPCCSolver:

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        net = cfg.get("network", {})
        planner = cfg.get("planner", {})
        weights = cfg.get("weights", {})

        self.N = planner.get("horizon", 10)
        self.dt = planner.get("timestep", 0.05)

        self.rtt_prop = net.get("rtt_prop", 0.025)
        self.tau_rtt = net.get("tau_rtt", 0.1)
        self.tau_tput = net.get("tau_tput", 0.1)
        self.rate_max = net.get("rate_max", 50e6)
        self.alpha = net.get("alpha", 0.150)

        self.rtt_target = net.get("rtt_target", 0.050)
        self.q_max = net.get("q_max", 50_000)
        self.n_flows = net.get("n_flows", 1)

        self.w_contour = weights.get("contour_weight", 50.0)
        self.w_lag = weights.get("contouring_lag_weight", 1.0)
        self.w_delay = weights.get("delay_weight", 100.0)
        self.w_du = weights.get("acceleration_weight", 0.5)
        self.w_fair = weights.get("fairness_weight", 0.0)

        self._last_u = None

    def solve(self, tput: float, rtt: float, queue: float, bw_est: float) -> NetworkMPCCResult:
        import time
        t0 = time.monotonic()

        opti = cd.Opti()
        opti.solver("ipopt", {}, {
            "print_level": 0,
            "max_iter": 200,
            "warm_start_init_point": "yes",
        })

        U = opti.variable(self.N)
        X_tput = opti.variable(self.N + 1)
        X_rtt = opti.variable(self.N + 1)
        X_q = opti.variable(self.N + 1)

        opti.subject_to(X_tput[0] == tput)
        opti.subject_to(X_rtt[0] == rtt)
        opti.subject_to(X_q[0] == queue)

        bw_safe = max(bw_est, 1.0)
        fair_share = bw_safe / max(self.n_flows, 1)

        for k in range(self.N):
            eff_rate = cd.fmin(U[k], bw_safe)
            tput_next = X_tput[k] + self.dt * (eff_rate - X_tput[k]) / self.tau_tput
            rtt_eq = self.rtt_prop + X_q[k] / bw_safe
            rtt_next = X_rtt[k] + self.dt * (rtt_eq - X_rtt[k]) / self.tau_rtt
            q_next = X_q[k] + self.dt * (U[k] - bw_safe)

            opti.subject_to(X_tput[k + 1] == tput_next)
            opti.subject_to(X_rtt[k + 1] == rtt_next)
            opti.subject_to(X_q[k + 1] == q_next)

        for k in range(self.N + 1):
            opti.subject_to(X_tput[k] >= 0)
            opti.subject_to(X_rtt[k] >= self.rtt_prop * 0.5)
            opti.subject_to(X_rtt[k] <= self.rtt_target * 4)
            opti.subject_to(X_q[k] >= 0)
            opti.subject_to(X_q[k] <= self.q_max)

        for k in range(self.N):
            opti.subject_to(U[k] >= 0)
            opti.subject_to(U[k] <= self.rate_max)

        cost = 0
        for k in range(self.N + 1):
            theta = X_tput[k] / bw_safe
            theta_c = cd.fmin(cd.fmax(theta, 0), 1)

            ref_t = bw_safe * theta_c
            ref_r = self.rtt_prop + self.alpha * theta_c ** 2

            tt = bw_safe
            tr = 2 * self.alpha * theta_c
            tn = cd.sqrt(tt ** 2 + tr ** 2 + 1e-8)
            tx, ty = tt / tn, tr / tn

            dx = X_tput[k] - ref_t
            dy = X_rtt[k] - ref_r

            e_c = ty * dx - tx * dy
            e_l = tx * dx + ty * dy

            cost += self.w_contour * e_c ** 2
            cost += self.w_lag * e_l ** 2

            d_excess = (X_rtt[k] - self.rtt_target) / self.rtt_target
            cost += self.w_delay * cd.fmax(d_excess, 0) ** 2

            cost -= 0.1 * (X_tput[k] / bw_safe) / (X_rtt[k] / self.rtt_prop + 1e-6)

        for k in range(self.N):
            if k > 0:
                cost += self.w_du * (U[k] - U[k - 1]) ** 2 / (self.rate_max ** 2)
            if self.w_fair > 0 and self.n_flows > 1:
                cost += self.w_fair * (U[k] - fair_share) ** 2 / (self.rate_max ** 2)

        opti.minimize(cost)

        init_rate = self._last_u if self._last_u is not None else bw_safe * 0.5
        opti.set_initial(U, init_rate)
        opti.set_initial(X_tput, tput)
        opti.set_initial(X_rtt, rtt)
        opti.set_initial(X_q, queue)

        try:
            sol = opti.solve()
            send_rate = float(sol.value(U[0]))
            send_rate = max(0.0, min(send_rate, self.rate_max))
            self._last_u = send_rate

            pred_tput = np.array([float(sol.value(X_tput[k])) for k in range(self.N + 1)])
            pred_rtt = np.array([float(sol.value(X_rtt[k])) for k in range(self.N + 1)])

            solve_ms = (time.monotonic() - t0) * 1000
            return NetworkMPCCResult(True, send_rate, pred_tput, pred_rtt, solve_ms)
        except Exception:
            solve_ms = (time.monotonic() - t0) * 1000
            fallback = self._last_u if self._last_u else bw_safe * 0.5
            return NetworkMPCCResult(False, fallback, solve_time_ms=solve_ms)


class NetworkMPCCQPSolver:

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        net = cfg.get("network", {})
        planner = cfg.get("planner", {})
        weights = cfg.get("weights", {})

        self.N = planner.get("horizon", 10)
        self.dt = planner.get("timestep", 0.05)

        self.rtt_prop = net.get("rtt_prop", 0.025)
        self.tau_rtt = net.get("tau_rtt", 0.1)
        self.tau_tput = net.get("tau_tput", 0.1)
        self.rate_max = net.get("rate_max", 50e6)
        self.alpha = net.get("alpha", 0.150)
        self.rtt_target = net.get("rtt_target", 0.050)
        self.q_max = net.get("q_max", 50_000)
        self.n_flows = net.get("n_flows", 1)

        self.w_contour = weights.get("contour_weight", 50.0)
        self.w_lag = weights.get("contouring_lag_weight", 1.0)
        self.w_delay = weights.get("delay_weight", 100.0)
        self.w_du = weights.get("acceleration_weight", 0.5)
        self.w_fair = weights.get("fairness_weight", 0.0)

        self._last_u = None

    def _forward_sim(self, x0: np.ndarray, u_seq: np.ndarray, bw: float) -> np.ndarray:
        bw_safe = max(bw, 1.0)
        traj = [x0.copy()]
        for k in range(len(u_seq)):
            t, r, q = traj[-1]
            s = u_seq[k]
            eff = min(s, bw_safe)
            t_next = t + self.dt * (eff - t) / self.tau_tput
            r_eq = self.rtt_prop + max(q, 0) / bw_safe
            r_next = r + self.dt * (r_eq - r) / self.tau_rtt
            q_next = max(q + self.dt * (s - bw_safe), 0)
            traj.append(np.array([t_next, r_next, q_next]))
        return np.array(traj)

    def _cost(self, u_seq: np.ndarray, x0: np.ndarray, bw: float) -> float:
        bw_safe = max(bw, 1.0)
        fair_share = bw_safe / max(self.n_flows, 1)
        traj = self._forward_sim(x0, u_seq, bw)

        cost = 0.0
        for k in range(len(traj)):
            t, r, q = traj[k]
            theta = np.clip(t / bw_safe, 0, 1)

            ref_t = bw_safe * theta
            ref_r = self.rtt_prop + self.alpha * theta ** 2
            tt = bw_safe
            tr = 2 * self.alpha * theta
            tn = np.sqrt(tt ** 2 + tr ** 2 + 1e-8)
            tx, ty = tt / tn, tr / tn

            dx, dy = t - ref_t, r - ref_r
            e_c = ty * dx - tx * dy
            e_l = tx * dx + ty * dy

            cost += self.w_contour * e_c ** 2 + self.w_lag * e_l ** 2

            d_ex = max((r - self.rtt_target) / self.rtt_target, 0)
            cost += self.w_delay * d_ex ** 2

            cost -= 0.1 * (t / bw_safe) / (r / self.rtt_prop + 1e-6)

        for k in range(len(u_seq)):
            if k > 0:
                cost += self.w_du * (u_seq[k] - u_seq[k - 1]) ** 2 / (self.rate_max ** 2)
            if self.w_fair > 0 and self.n_flows > 1:
                cost += self.w_fair * (u_seq[k] - fair_share) ** 2 / (self.rate_max ** 2)

        return cost

    def solve(self, tput: float, rtt: float, queue: float, bw_est: float) -> NetworkMPCCResult:
        import time
        from scipy.optimize import minimize as sp_minimize
        t0 = time.monotonic()

        bw_safe = max(bw_est, 1.0)
        x0 = np.array([tput, rtt, queue])

        init = self._last_u if self._last_u is not None else bw_safe * 0.5
        u0 = np.full(self.N, init)

        bounds = [(0.0, self.rate_max)] * self.N

        try:
            res = sp_minimize(
                lambda u: self._cost(u, x0, bw_safe),
                u0,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 20, "ftol": 1e-6},
            )

            send_rate = float(np.clip(res.x[0], 0, self.rate_max))
            self._last_u = send_rate

            traj = self._forward_sim(x0, res.x, bw_safe)
            solve_ms = (time.monotonic() - t0) * 1000
            return NetworkMPCCResult(True, send_rate, traj[:, 0], traj[:, 1], solve_ms)

        except Exception:
            solve_ms = (time.monotonic() - t0) * 1000
            fallback = self._last_u if self._last_u else bw_safe * 0.5
            return NetworkMPCCResult(False, float(fallback), solve_time_ms=solve_ms)
