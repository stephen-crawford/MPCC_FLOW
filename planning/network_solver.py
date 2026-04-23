"""MPCC NLP and QP solvers for congestion control.

Implements the paper's finite-horizon controller from Sec. IV, with the true
path-parameter MPCC formulation:

    Decision vars:
        s_0 ... s_{N-1}     sending rates
        v_0 ... v_{N-1}     path-speed (progress-rate) along the reference curve
        theta_1 ... theta_N derived: theta_{k+1} = theta_k + dt * v_k

    min   sum_k   w_c e_c(theta_k, x_k)^2 + w_l e_l(theta_k, x_k)^2
                + w_d [R_k - R*]_+^2
                - w_p * (T_k/C) / (R_k/R_0)
                - w_theta * v_k            <-- progress reward
          + sum_{k>=1}  w_u * (s_k - s_{k-1})^2 / s_max^2
          + sum_k       w_f * (s_k - C/n)^2
    s.t.  x_{k+1} = f(x_k, s_k)            Eq. 4
          0 <= s_k <= s_max                Eq. 10b
          0 <= q_k <= q_max                Eq. 10c
          R_k <= 4 R*                      Eq. 10d
          0 <= theta_k <= 1                (path-parameter bound)
          0 <= v_k <= v_theta_max          (progress-rate bound)

Crucially, the reference point Gamma(theta_k) = (C theta_k, R_0 + alpha theta_k^2)
is evaluated at the *symbolic* theta_k rather than theta_k = T_k/C. That makes
contouring/lag errors geometrically meaningful — the optimizer is free to trade
off "make progress along the frontier" (big v_k) vs "stay close to the frontier"
(small e_c, e_l), exactly as in the robotics MPCC of Liniger et al.

Two solver flavors:

* `NetworkMPCCSolver` — CasADi Opti + IPOPT nonlinear program.
* `NetworkMPCCQPSolver` — Linearized QP: forward-simulate nominal trajectory at
  (u_bar, v_bar), take Gauss-Newton Hessian of the cost, linearize the state
  inequality constraints, solve a single QP via CasADi `qpsol` (qrqp).

Both return a `NetworkMPCCResult` including per-step solve time.
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
    theta_sequence: np.ndarray | None = None
    v_theta_sequence: np.ndarray | None = None
    solve_time_ms: float = 0.0


def _stage_cost_symbolic(
    X_tput, X_rtt, X_q, U, N, dt,
    *,
    theta, v_theta,
    bw, rtt_prop, rtt_target, alpha, n_flows, rate_max,
    w_contour, w_lag, w_delay, w_power, w_du, w_fair, w_theta,
    delay_hinge_absolute: bool = False,
    fairness_unnormalized: bool = False,
    u_prev_bps: float | None = None,
):
    """Build the MPCC stage cost + horizon penalties on CasADi symbolics.

    ``theta``:   length N+1 sequence of symbolic path parameters.
    ``v_theta``: length N sequence of symbolic progress-rate controls.

    Contouring and lag errors are computed at the symbolic theta, not at
    theta = T/C. This is the classical MPCC formulation.

    ``w_theta * v_k`` is subtracted each step as the progress reward.
    """
    bw_safe = cd.fmax(bw, 1.0)
    fair_share = bw_safe / max(n_flows, 1)
    cost = 0

    # Normalize the reference plane to (T/C, (R-R0)/alpha) so contouring/lag
    # errors are O(1) instead of O(bps). Without this the Hessian spans ~13
    # orders of magnitude and the condensed QP becomes ill-conditioned.
    alpha_safe = max(alpha, 1e-9)

    for k in range(N + 1):
        theta_k = theta[k]

        # Reference point Gamma(theta) in normalized coordinates.
        ref_T_n = theta_k
        ref_R_n = theta_k ** 2

        # Unit tangent in normalized coordinates: d/dtheta (theta, theta^2) = (1, 2*theta).
        tt = 1.0
        tr = 2.0 * theta_k
        tn = cd.sqrt(tt ** 2 + tr ** 2 + 1e-8)
        tx, ty = tt / tn, tr / tn

        # State in normalized coordinates.
        T_n = X_tput[k] / bw_safe
        R_n = (X_rtt[k] - rtt_prop) / alpha_safe

        dx = T_n - ref_T_n
        dy = R_n - ref_R_n

        e_c = ty * dx - tx * dy              # contouring error (normal to curve)
        e_l = tx * dx + ty * dy              # lag error (tangent to curve)

        cost += w_contour * e_c ** 2
        cost += w_lag * e_l ** 2

        if delay_hinge_absolute:
            d_raw = X_rtt[k] - rtt_target
            cost += w_delay * cd.fmax(d_raw, 0) ** 2
        else:
            d_excess = (X_rtt[k] - rtt_target) / rtt_target
            cost += w_delay * cd.fmax(d_excess, 0) ** 2

        # Kleinrock power (reward, so subtract)
        cost -= w_power * (X_tput[k] / bw_safe) / (X_rtt[k] / rtt_prop + 1e-6)

    for k in range(N):
        # Progress reward: pay the controller to move along the reference.
        cost -= w_theta * v_theta[k]

        if u_prev_bps is not None and k == 0:
            cost += w_du * (U[k] - u_prev_bps) ** 2 / (rate_max ** 2)
        elif k > 0:
            cost += w_du * (U[k] - U[k - 1]) ** 2 / (rate_max ** 2)
        if w_fair > 0 and n_flows > 1:
            fair_dev = U[k] - fair_share
            if fairness_unnormalized:
                cost += w_fair * fair_dev ** 2
            else:
                cost += w_fair * fair_dev ** 2 / (rate_max ** 2)

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

        # Path-parameter bound on v_theta. Units: 1/s. With default dt=20 ms
        # and v_theta_max=5, per-step progress <= 0.1 and per-horizon (N=8)
        # <= 0.8 of the full curve.
        self.v_theta_max = net.get("v_theta_max", 5.0)

        self.w_contour = weights.get("contour_weight", 50.0)
        self.w_lag = weights.get("contouring_lag_weight", 1.0)
        self.w_delay = weights.get("delay_weight", 100.0)
        self.w_power = weights.get("power_weight", 0.1)
        self.w_du = weights.get("acceleration_weight", 0.5)
        self.w_fair = weights.get("fairness_weight", 0.0)
        # Progress reward weight. Must be >0 for the controller to have any
        # reason to advance along the curve instead of parking at theta=0.
        self.w_theta = weights.get("theta_progress_weight", 10.0)

        self.delay_hinge_absolute = bool(net.get("delay_hinge_absolute", False))
        self.fairness_unnormalized = bool(net.get("fairness_unnormalized", False))

        self._last_u: np.ndarray | None = None
        self._last_v_theta: np.ndarray | None = None
        # theta_0 to use on the *next* solve call. Updated at the end of each
        # successful solve by taking theta_1 from the current plan (receding
        # horizon shift).
        self._next_theta0: float = 0.0

    def _fallback_rate(self, bw_safe: float) -> float:
        if self._last_u is None:
            return bw_safe * 0.5
        if hasattr(self._last_u, "__len__"):
            return float(self._last_u[0])
        return float(self._last_u)


class NetworkMPCCSolver(_SolverBase):
    """Nonlinear MPCC via CasADi Opti + IPOPT (paper Eq. 10, now with proper θ)."""

    def solve(
        self,
        tput: float,
        rtt: float,
        queue: float,
        bw_est: float,
        u_prev_bps: float | None = None,
    ) -> NetworkMPCCResult:
        t0 = time.monotonic()

        opti = cd.Opti()
        opti.solver("ipopt", {}, {
            "print_level": 0,
            "max_iter": 200,
            "warm_start_init_point": "yes",
        })

        U = opti.variable(self.N)
        V = opti.variable(self.N)                    # progress rates
        Theta = opti.variable(self.N + 1)            # path parameter
        X_tput = opti.variable(self.N + 1)
        X_rtt = opti.variable(self.N + 1)
        X_q = opti.variable(self.N + 1)

        opti.subject_to(X_tput[0] == tput)
        opti.subject_to(X_rtt[0] == rtt)
        opti.subject_to(X_q[0] == queue)
        opti.subject_to(Theta[0] == self._next_theta0)

        bw_safe = max(bw_est, 1.0)

        for k in range(self.N):
            eff_rate = cd.fmin(U[k], bw_safe)
            tput_next = X_tput[k] + self.dt * (eff_rate - X_tput[k]) / self.tau_tput
            rtt_eq = self.rtt_prop + X_q[k] / bw_safe
            rtt_next = X_rtt[k] + self.dt * (rtt_eq - X_rtt[k]) / self.tau_rtt
            q_next = X_q[k] + self.dt * (U[k] - bw_safe)
            theta_next = Theta[k] + self.dt * V[k]

            opti.subject_to(X_tput[k + 1] == tput_next)
            opti.subject_to(X_rtt[k + 1] == rtt_next)
            opti.subject_to(X_q[k + 1] == q_next)
            opti.subject_to(Theta[k + 1] == theta_next)

        for k in range(self.N + 1):
            opti.subject_to(X_tput[k] >= 0)
            opti.subject_to(X_rtt[k] >= self.rtt_prop * 0.5)
            opti.subject_to(X_rtt[k] <= self.rtt_target * 4)       # Eq. 10d
            opti.subject_to(X_q[k] >= 0)
            opti.subject_to(X_q[k] <= self.q_max)                   # Eq. 10c
            opti.subject_to(Theta[k] >= 0)
            opti.subject_to(Theta[k] <= 1)

        for k in range(self.N):
            opti.subject_to(U[k] >= 0)                              # Eq. 10b
            opti.subject_to(U[k] <= self.rate_max)
            opti.subject_to(V[k] >= 0)
            opti.subject_to(V[k] <= self.v_theta_max)

        cost = _stage_cost_symbolic(
            X_tput, X_rtt, X_q, U, self.N, self.dt,
            theta=Theta, v_theta=V,
            bw=bw_safe, rtt_prop=self.rtt_prop, rtt_target=self.rtt_target,
            alpha=self.alpha, n_flows=self.n_flows, rate_max=self.rate_max,
            w_contour=self.w_contour, w_lag=self.w_lag, w_delay=self.w_delay,
            w_power=self.w_power, w_du=self.w_du, w_fair=self.w_fair,
            w_theta=self.w_theta,
            delay_hinge_absolute=self.delay_hinge_absolute,
            fairness_unnormalized=self.fairness_unnormalized,
            u_prev_bps=u_prev_bps,
        )
        opti.minimize(cost)

        init_u = self._last_u if self._last_u is not None else np.full(self.N, bw_safe * 0.5)
        if np.ndim(init_u) == 0:
            init_u = np.full(self.N, float(init_u))
        init_v = (self._last_v_theta if self._last_v_theta is not None
                  else np.full(self.N, self.v_theta_max * 0.5))
        init_theta = np.clip(
            self._next_theta0 + self.dt * np.cumsum(np.concatenate([[0.0], init_v])),
            0.0, 1.0,
        )
        opti.set_initial(U, init_u)
        opti.set_initial(V, init_v)
        opti.set_initial(Theta, init_theta)
        opti.set_initial(X_tput, tput)
        opti.set_initial(X_rtt, rtt)
        opti.set_initial(X_q, queue)

        try:
            sol = opti.solve()
            u_opt = np.array([float(sol.value(U[k])) for k in range(self.N)])
            u_opt = np.clip(u_opt, 0.0, self.rate_max)
            v_opt = np.array([float(sol.value(V[k])) for k in range(self.N)])
            v_opt = np.clip(v_opt, 0.0, self.v_theta_max)
            theta_opt = np.array([float(sol.value(Theta[k])) for k in range(self.N + 1)])
            theta_opt = np.clip(theta_opt, 0.0, 1.0)
            self._last_u = u_opt
            self._last_v_theta = v_opt
            # Receding-horizon shift: next call starts at theta_1.
            self._next_theta0 = float(theta_opt[1])

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
                theta_sequence=theta_opt,
                v_theta_sequence=v_opt,
                solve_time_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception:
            return NetworkMPCCResult(
                success=False,
                send_rate=self._fallback_rate(bw_safe),
                solve_time_ms=(time.monotonic() - t0) * 1000,
            )


class NetworkMPCCQPSolver(_SolverBase):
    """Linearized QP (paper Eq. 11–13) with the path-parameter MPCC formulation.

    Decision vector Z = [U; V] in R^{2N}:
      - U: sending rates (as before).
      - V: progress rates along the reference curve.

    Theta is built symbolically from Z via theta_k = theta_0 + dt * sum_{j<k} V_j
    and used to evaluate the reference curve Gamma(theta_k), the unit tangent,
    and the contouring/lag errors. That makes the geometric tracking problem
    genuinely 2D rather than collapsing to a vertical RTT penalty.
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

    def solve(
        self,
        tput: float,
        rtt: float,
        queue: float,
        bw_est: float,
        u_prev_bps: float | None = None,
    ) -> NetworkMPCCResult:
        t0 = time.monotonic()
        bw_safe = max(bw_est, 1.0)
        N = self.N

        x0 = np.array([tput, rtt, queue], dtype=float)
        theta_0 = float(np.clip(self._next_theta0, 0.0, 1.0))

        # Multi-start warm points. The condensed QP is only locally valid, and
        # this problem has a bad local minimum near (u = T_current, v ≈ 0)
        # where lag error e_l = 0 "by accident". Starting a QP linearization
        # from an aggressive point (u = bw, v = v_theta_max) reliably falls
        # into the good basin. We try both, run one QP step each, and keep
        # the solution with lower true cost.
        candidates: list[tuple[np.ndarray, np.ndarray]] = []

        # Candidate A: receding-horizon shift of the previous solve.
        if self._last_u is None:
            u_bar_a = np.full(N, bw_safe)          # aggressive default
            v_bar_a = np.full(N, self.v_theta_max)
        else:
            u_bar_a = self._last_u.copy()
            u_bar_a[:-1] = u_bar_a[1:]
            v_bar_a = self._last_v_theta.copy()
            v_bar_a[:-1] = v_bar_a[1:]
        candidates.append((u_bar_a, v_bar_a))

        # Candidate B: aggressive — full-rate, full-progress.
        if self._last_u is not None:
            candidates.append((
                np.full(N, bw_safe),
                np.full(N, self.v_theta_max),
            ))

        # Decision vector in *dimensionless* coordinates so the QP Hessian is
        # well-scaled: Z = [U / rate_max; V / v_theta_max]. Without this
        # Z is O(bps) in its first N components and O(v_theta_max) in the
        # last N, which makes any spectrum-based PSD shift either useless
        # on U or over-damping on V.
        v_max = max(self.v_theta_max, 1e-6)
        Z_sym = cd.MX.sym("Z", 2 * N)
        U_sym = Z_sym[:N] * self.rate_max         # back to bps for the dynamics
        V_sym = Z_sym[N:] * v_max                  # back to 1/s

        # Theta: theta_{k+1} = theta_k + dt * V_k, theta_0 fixed.
        Theta_sym = [cd.MX(theta_0)]
        for k in range(N):
            Theta_sym.append(Theta_sym[-1] + self.dt * V_sym[k])
        Theta = cd.vertcat(*Theta_sym)

        # Network dynamics driven by U.
        X_tput_sym = [cd.MX(x0[0])]
        X_rtt_sym = [cd.MX(x0[1])]
        X_q_sym = [cd.MX(x0[2])]
        for k in range(N):
            t_k = X_tput_sym[-1]
            r_k = X_rtt_sym[-1]
            q_k = X_q_sym[-1]
            s_k = U_sym[k]

            eff = cd.fmin(s_k, bw_safe)
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
            X_tput, X_rtt, X_q, U_sym, N, self.dt,
            theta=Theta_sym, v_theta=V_sym,
            bw=bw_safe, rtt_prop=self.rtt_prop, rtt_target=self.rtt_target,
            alpha=self.alpha, n_flows=self.n_flows, rate_max=self.rate_max,
            w_contour=self.w_contour, w_lag=self.w_lag, w_delay=self.w_delay,
            w_power=self.w_power, w_du=self.w_du, w_fair=self.w_fair,
            w_theta=self.w_theta,
            delay_hinge_absolute=self.delay_hinge_absolute,
            fairness_unnormalized=self.fairness_unnormalized,
            u_prev_bps=u_prev_bps,
        )

        grad_fn = cd.Function("grad", [Z_sym], [cd.gradient(cost_sym, Z_sym)])
        H_sym, _ = cd.hessian(cost_sym, Z_sym)
        H_fn = cd.Function("H", [Z_sym], [H_sym])
        cost_fn = cd.Function("c", [Z_sym], [cost_sym])

        state_block = cd.vertcat(
            *X_q_sym[1:], *X_rtt_sym[1:], *Theta_sym[1:],
        )
        J_fn = cd.Function("J", [Z_sym], [cd.jacobian(state_block, Z_sym)])
        state_vec_fn = cd.Function("sv", [Z_sym], [state_block])

        # Ensure QP solver object exists for 2N decision variables.
        if self._qp_solver is None or self._qp_N_built_for != N:
            qp_struct = {
                "h": cd.Sparsity.dense(2 * N, 2 * N),
                "a": cd.Sparsity.dense(5 * N, 2 * N),
            }
            self._qp_solver = cd.conic(
                "qp", "qrqp", qp_struct,
                {"print_iter": False, "print_header": False,
                 "error_on_fail": False},
            )
            self._qp_N_built_for = N

        def one_qp_step(u_bar: np.ndarray, v_bar: np.ndarray):
            """Linearize cost+constraints at (u_bar, v_bar) and take one QP
            step. Returns (u_opt, v_opt, true_cost) or (None, None, +inf)
            if the QP fails."""
            z_bar = np.concatenate([u_bar / self.rate_max, v_bar / v_max])

            g_val = np.asarray(grad_fn(z_bar)).flatten()
            H_val = np.asarray(H_fn(z_bar))
            H_val = 0.5 * (H_val + H_val.T)
            min_eig = float(np.linalg.eigvalsh(H_val)[0])
            ridge = max(1e-6, -min_eig + 1e-3) if min_eig < 0 else 1e-6
            H_val = H_val + ridge * np.eye(2 * N)

            J = np.asarray(J_fn(z_bar))
            s_bar = np.asarray(state_vec_fn(z_bar)).flatten()
            J_q = J[0 * N: 1 * N, :]
            J_R = J[1 * N: 2 * N, :]
            J_T = J[2 * N: 3 * N, :]
            q_bar = s_bar[0 * N: 1 * N]
            R_bar = s_bar[1 * N: 2 * N]
            T_bar = s_bar[2 * N: 3 * N]

            A = np.vstack([J_q, -J_q, J_R, J_T, -J_T])
            ubA = np.concatenate([
                self.q_max - q_bar, q_bar,
                4 * self.rtt_target - R_bar,
                1.0 - T_bar, T_bar,
            ])
            lbA = np.full(ubA.shape, -np.inf)

            # Both halves of Z are in [0, 1].
            z_lo = np.zeros(2 * N)
            z_hi = np.ones(2 * N)
            lbZ = z_lo - z_bar
            ubZ = z_hi - z_bar

            try:
                sol = self._qp_solver(
                    h=cd.DM(H_val), g=cd.DM(g_val),
                    a=cd.DM(A), lba=cd.DM(lbA), uba=cd.DM(ubA),
                    lbx=cd.DM(lbZ), ubx=cd.DM(ubZ),
                )
                if not self._qp_solver.stats().get("success", True):
                    return None, None, float("inf")
                dZ = np.asarray(sol["x"]).flatten()
            except Exception:
                return None, None, float("inf")

            z_opt = z_bar + dZ
            u_opt = np.clip(z_opt[:N] * self.rate_max, 0.0, self.rate_max)
            v_opt = np.clip(z_opt[N:] * v_max, 0.0, self.v_theta_max)

            # Score by the true nonlinear cost.
            z_eval = np.concatenate([u_opt / self.rate_max, v_opt / v_max])
            c = float(cost_fn(z_eval))
            return u_opt, v_opt, c

        best_u, best_v, best_cost = None, None, float("inf")
        for u_cand, v_cand in candidates:
            u_try, v_try, c_try = one_qp_step(u_cand, v_cand)
            if u_try is not None and c_try < best_cost:
                best_u, best_v, best_cost = u_try, v_try, c_try

        if best_u is None:
            return NetworkMPCCResult(
                success=False,
                send_rate=self._fallback_rate(bw_safe),
                solve_time_ms=(time.monotonic() - t0) * 1000,
            )

        u_opt = best_u
        v_opt = best_v

        theta_opt = np.empty(N + 1)
        theta_opt[0] = theta_0
        for k in range(N):
            theta_opt[k + 1] = min(1.0, max(0.0,
                                            theta_opt[k] + self.dt * v_opt[k]))

        self._last_u = u_opt
        self._last_v_theta = v_opt
        self._next_theta0 = float(theta_opt[1])

        traj = self._nominal_trajectory(x0, u_opt, bw_safe)
        return NetworkMPCCResult(
            success=True,
            send_rate=float(u_opt[0]),
            predicted_tput=traj[:, 0],
            predicted_rtt=traj[:, 1],
            predicted_queue=traj[:, 2],
            control_sequence=u_opt,
            theta_sequence=theta_opt,
            v_theta_sequence=v_opt,
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
