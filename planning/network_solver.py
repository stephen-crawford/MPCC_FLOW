"""MPCC NLP and QP solvers for congestion control.

Implements the paper's finite-horizon controller from Sec. IV:

    min_{s_k}  sum_k  w_c e_c(k)^2 + w_l e_l(k)^2 + w_d [R_k - R*]_+^2
                     - w_p * (T_k/C) / (R_k/R_0)
                     + sum_{k>=1} w_u * (s_k - s_{k-1})^2 / s_max^2
                     + sum_k     w_f * (s_k - C/n)^2
    s.t.  x_{k+1} = f(x_k, s_k)      [Eq. 4]
          0 <= s_k <= s_max          [Eq. 10b]
          0 <= q_k <= q_max          [Eq. 10c]
          R_k <= 4 R*                [Eq. 10d]

Two solver flavors:

* `NetworkMPCCSolver` — CasADi Opti + IPOPT nonlinear program. Used as ground
  truth / for ablation.
* `NetworkMPCCQPSolver` — Paper's QP approximation (Eq. 11–13): linearize the
  dynamics around the previous solve's nominal trajectory, Gauss-Newton on the
  contouring/lag costs, solve a single QP via CasADi `qpsol` (qrqp backend,
  ships with CasADi).

Both return a `NetworkMPCCResult` including per-step solve time so the paper
can report controller compute cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as cd
import numpy as np


@dataclass
class NetworkMPCCResult:
    success: bool
    send_rate: float
    predicted_tput: np.ndarray | None = None
    predicted_rtt: np.ndarray | None = None
    predicted_queue: np.ndarray | None = None
    control_sequence: np.ndarray | None = None
    solve_time_ms: float = 0.0


def _stage_cost_symbolic(
    X_tput, X_rtt, X_q, U, N, dt,
    *,
    bw, rtt_prop, rtt_target, alpha, n_flows, rate_max,
    w_contour, w_lag, w_delay, w_power, w_du, w_fair,
):
    """Build the MPCC stage cost + horizon penalties on CasADi symbolics.

    Used by both the NLP (Opti) and the QP condensing pass.
    """
    bw_safe = cd.fmax(bw, 1.0)
    fair_share = bw_safe / max(n_flows, 1)
    cost = 0

    for k in range(N + 1):
        theta = X_tput[k] / bw_safe
        theta_c = cd.fmin(cd.fmax(theta, 0), 1)

        ref_t = bw_safe * theta_c
        ref_r = rtt_prop + alpha * theta_c ** 2

        # Tangent to the reference curve at theta.
        tt = bw_safe
        tr = 2 * alpha * theta_c
        tn = cd.sqrt(tt ** 2 + tr ** 2 + 1e-8)
        tx, ty = tt / tn, tr / tn

        dx = X_tput[k] - ref_t
        dy = X_rtt[k] - ref_r

        e_c = ty * dx - tx * dy              # Eq. 6a — contouring error (normal)
        e_l = tx * dx + ty * dy              # Eq. 6b — lag error (tangent)

        cost += w_contour * e_c ** 2
        cost += w_lag * e_l ** 2

        # Soft over-delay penalty: [R - R*]_+^2, normalized by R*
        d_excess = (X_rtt[k] - rtt_target) / rtt_target
        cost += w_delay * cd.fmax(d_excess, 0) ** 2

        # Kleinrock power (reward, so subtract)
        cost -= w_power * (X_tput[k] / bw_safe) / (X_rtt[k] / rtt_prop + 1e-6)

    for k in range(N):
        if k > 0:
            cost += w_du * (U[k] - U[k - 1]) ** 2 / (rate_max ** 2)
        if w_fair > 0 and n_flows > 1:
            cost += w_fair * (U[k] - fair_share) ** 2 / (rate_max ** 2)

    return cost


class _SolverBase:
    """Shared config parsing for NLP and QP variants."""

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
        self.w_power = weights.get("power_weight", 0.1)
        self.w_du = weights.get("acceleration_weight", 0.5)
        self.w_fair = weights.get("fairness_weight", 0.0)

        self._last_u: np.ndarray | None = None

    def _fallback_rate(self, bw_safe: float) -> float:
        if self._last_u is None:
            return bw_safe * 0.5
        if hasattr(self._last_u, "__len__"):
            return float(self._last_u[0])
        return float(self._last_u)


class NetworkMPCCSolver(_SolverBase):
    """Nonlinear MPCC via CasADi Opti + IPOPT (paper Eq. 10)."""

    def solve(self, tput: float, rtt: float, queue: float,
              bw_est: float) -> NetworkMPCCResult:
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
            opti.subject_to(X_rtt[k] <= self.rtt_target * 4)       # Eq. 10d
            opti.subject_to(X_q[k] >= 0)
            opti.subject_to(X_q[k] <= self.q_max)                   # Eq. 10c

        for k in range(self.N):
            opti.subject_to(U[k] >= 0)                              # Eq. 10b
            opti.subject_to(U[k] <= self.rate_max)

        cost = _stage_cost_symbolic(
            X_tput, X_rtt, X_q, U, self.N, self.dt,
            bw=bw_safe, rtt_prop=self.rtt_prop, rtt_target=self.rtt_target,
            alpha=self.alpha, n_flows=self.n_flows, rate_max=self.rate_max,
            w_contour=self.w_contour, w_lag=self.w_lag, w_delay=self.w_delay,
            w_power=self.w_power, w_du=self.w_du, w_fair=self.w_fair,
        )
        opti.minimize(cost)

        init_u = self._last_u if self._last_u is not None else np.full(self.N, bw_safe * 0.5)
        if np.ndim(init_u) == 0:
            init_u = np.full(self.N, float(init_u))
        opti.set_initial(U, init_u)
        opti.set_initial(X_tput, tput)
        opti.set_initial(X_rtt, rtt)
        opti.set_initial(X_q, queue)

        try:
            sol = opti.solve()
            u_opt = np.array([float(sol.value(U[k])) for k in range(self.N)])
            u_opt = np.clip(u_opt, 0.0, self.rate_max)
            self._last_u = u_opt

            pred_t = np.array([float(sol.value(X_tput[k])) for k in range(self.N + 1)])
            pred_r = np.array([float(sol.value(X_rtt[k])) for k in range(self.N + 1)])
            pred_q = np.array([float(sol.value(X_q[k])) for k in range(self.N + 1)])

            return NetworkMPCCResult(
                success=True,
                send_rate=float(u_opt[0]),
                predicted_tput=pred_t,
                predicted_rtt=pred_r,
                predicted_queue=pred_q,
                control_sequence=u_opt,
                solve_time_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception:
            return NetworkMPCCResult(
                success=False,
                send_rate=self._fallback_rate(bw_safe),
                solve_time_ms=(time.monotonic() - t0) * 1000,
            )


class NetworkMPCCQPSolver(_SolverBase):
    """Linearized QP (paper Eq. 11–13).

    One SQP-style iteration per call:

    1. Warm-start nominal trajectory from last U_bar, else U_bar = C/2.
    2. Build dynamics (Eq. 4) and cost symbolically, linearize dynamics at the
       nominal, take the Gauss-Newton Hessian of the cost (via `cd.hessian`).
    3. Condense to decision variables = U only (states substituted via linear
       dynamics), assemble QP {H, g, box-bounds on U, linear state-inequality
       constraints}, solve via CasADi `qpsol("qrqp")` (ships with CasADi).
    4. Return U[0] for the current control, store U_opt as next warm start.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._qp_solver = None
        self._qp_N_built_for: int | None = None

    def _nominal_trajectory(
        self, x0: np.ndarray, u_bar: np.ndarray, bw: float,
    ) -> np.ndarray:
        """Forward simulate Eq. 4 with u_bar to obtain the linearization point."""
        bw_safe = max(bw, 1.0)
        traj = np.zeros((self.N + 1, 3))
        traj[0] = x0
        for k in range(self.N):
            t, r, q = traj[k]
            s = u_bar[k]
            eff = min(s, bw_safe)
            t_next = t + self.dt * (eff - t) / self.tau_tput
            r_eq = self.rtt_prop + max(q, 0.0) / bw_safe
            r_next = r + self.dt * (r_eq - r) / self.tau_rtt
            q_next = max(q + self.dt * (s - bw_safe), 0.0)
            traj[k + 1] = [t_next, r_next, q_next]
        return traj

    def solve(self, tput: float, rtt: float, queue: float,
              bw_est: float) -> NetworkMPCCResult:
        t0 = time.monotonic()
        bw_safe = max(bw_est, 1.0)

        x0 = np.array([tput, rtt, queue], dtype=float)
        if self._last_u is None:
            u_bar = np.full(self.N, bw_safe * 0.5)
        else:
            u_bar = self._last_u.copy()
            # Warm-start shift: s_k <- s_{k+1}, last entry duplicated.
            u_bar[:-1] = u_bar[1:]

        # Symbolic problem over U only (states substituted via linearized dynamics).
        U_sym = cd.MX.sym("U", self.N)

        # Trajectory with continuous dynamics (nonlinear), evaluated symbolically
        # in U. The min() in Eq. 4 is smoothed via a soft-min near the current
        # operating point so the Hessian is well defined.
        def soft_min(a, b, beta=1e-5):
            # -1/beta log(exp(-beta a) + exp(-beta b)) — but numerically stable.
            # For the tput dynamic, a = send rate, b = bw_safe. Since both are
            # non-negative and bw_safe is ~constant, plain min is fine: the
            # QP only needs the Jacobian at the nominal, which is 1 if s<C and
            # 0 if s>=C. Paper Eq. 13 matches that.
            return cd.fmin(a, b)

        X_tput_sym = [cd.MX(x0[0])]
        X_rtt_sym = [cd.MX(x0[1])]
        X_q_sym = [cd.MX(x0[2])]
        for k in range(self.N):
            t_k = X_tput_sym[-1]
            r_k = X_rtt_sym[-1]
            q_k = X_q_sym[-1]
            s_k = U_sym[k]

            eff = soft_min(s_k, bw_safe)
            t_next = t_k + self.dt * (eff - t_k) / self.tau_tput
            r_eq = self.rtt_prop + cd.fmax(q_k, 0.0) / bw_safe
            r_next = r_k + self.dt * (r_eq - r_k) / self.tau_rtt
            q_next = cd.fmax(q_k + self.dt * (s_k - bw_safe), 0.0)

            X_tput_sym.append(t_next)
            X_rtt_sym.append(r_next)
            X_q_sym.append(q_next)

        X_tput = cd.vertcat(*X_tput_sym)
        X_rtt = cd.vertcat(*X_rtt_sym)
        X_q = cd.vertcat(*X_q_sym)

        cost_sym = _stage_cost_symbolic(
            X_tput, X_rtt, X_q, U_sym, self.N, self.dt,
            bw=bw_safe, rtt_prop=self.rtt_prop, rtt_target=self.rtt_target,
            alpha=self.alpha, n_flows=self.n_flows, rate_max=self.rate_max,
            w_contour=self.w_contour, w_lag=self.w_lag, w_delay=self.w_delay,
            w_power=self.w_power, w_du=self.w_du, w_fair=self.w_fair,
        )

        # Gradient + Gauss-Newton Hessian of cost wrt U at u_bar.
        grad = cd.gradient(cost_sym, U_sym)
        H_sym, _ = cd.hessian(cost_sym, U_sym)

        grad_fn = cd.Function("grad", [U_sym], [grad])
        H_fn = cd.Function("H", [U_sym], [H_sym])

        g_val = np.asarray(grad_fn(u_bar)).flatten()
        H_val = np.asarray(H_fn(u_bar))

        # Symmetrize + ridge for numerical PSD-ness.
        H_val = 0.5 * (H_val + H_val.T) + 1e-6 * np.eye(self.N)

        # Linear constraints on the states: 0 <= q_k <= q_max,  R_k <= 4 R*.
        # Compute Jacobian of [q_1..q_N, R_1..R_N] wrt U at u_bar to form a
        # linear inequality constraint matrix.
        state_block = cd.vertcat(*X_q_sym[1:], *X_rtt_sym[1:])  # 2N x 1
        J_fn = cd.Function("J", [U_sym], [cd.jacobian(state_block, U_sym)])
        state_vec_fn = cd.Function("sv", [U_sym], [state_block])

        J = np.asarray(J_fn(u_bar))
        s_bar = np.asarray(state_vec_fn(u_bar)).flatten()

        # Stack linear constraint rows:
        #   q_k(U) = J_q dU + q_bar          <= q_max
        #  -q_k(U)                            <= 0
        #   R_k(U) = J_R dU + R_bar          <= 4 R*
        J_q = J[: self.N, :]
        J_R = J[self.N: 2 * self.N, :]
        q_bar = s_bar[: self.N]
        R_bar = s_bar[self.N: 2 * self.N]

        A = np.vstack([J_q, -J_q, J_R])
        ubA = np.concatenate([
            self.q_max - q_bar,
            q_bar,
            4 * self.rtt_target - R_bar,
        ])
        # No lower bound (qpsol accepts ubA only via -inf lba).
        lbA = np.full(ubA.shape, -np.inf)

        # Box bounds on U: translate to deltas dU = U - u_bar.
        lbU = np.maximum(-u_bar, -self.rate_max)
        ubU = np.minimum(self.rate_max - u_bar, self.rate_max)

        try:
            if self._qp_solver is None or self._qp_N_built_for != self.N:
                qp_struct = {
                    "h": cd.Sparsity.dense(self.N, self.N),
                    "a": cd.Sparsity.dense(A.shape[0], self.N),
                }
                self._qp_solver = cd.conic(
                    "qp", "qrqp", qp_struct,
                    {"print_iter": False, "print_header": False},
                )
                self._qp_N_built_for = self.N

            sol = self._qp_solver(
                h=cd.DM(H_val),
                g=cd.DM(g_val),
                a=cd.DM(A),
                lba=cd.DM(lbA),
                uba=cd.DM(ubA),
                lbx=cd.DM(lbU),
                ubx=cd.DM(ubU),
            )
            dU = np.asarray(sol["x"]).flatten()
            u_opt = np.clip(u_bar + dU, 0.0, self.rate_max)
            self._last_u = u_opt

            traj = self._nominal_trajectory(x0, u_opt, bw_safe)
            return NetworkMPCCResult(
                success=True,
                send_rate=float(u_opt[0]),
                predicted_tput=traj[:, 0],
                predicted_rtt=traj[:, 1],
                predicted_queue=traj[:, 2],
                control_sequence=u_opt,
                solve_time_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception:
            return NetworkMPCCResult(
                success=False,
                send_rate=self._fallback_rate(bw_safe),
                solve_time_ms=(time.monotonic() - t0) * 1000,
            )


def create_solver(config: dict | None, mode: str = "qp"):
    """Factory — choose between the NLP reference and the QP approximation."""
    mode = mode.lower()
    if mode == "nlp":
        return NetworkMPCCSolver(config)
    if mode == "qp":
        return NetworkMPCCQPSolver(config)
    raise ValueError(f"unknown solver mode {mode!r}; expected 'nlp' or 'qp'")
