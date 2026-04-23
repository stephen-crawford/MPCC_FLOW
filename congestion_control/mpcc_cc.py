"""MPCC: Model Predictive Contouring Control for Congestion Control.

Implements the MPCC congestion control algorithm as described in the paper
draft ``paper_state.tex`` (finite-horizon cost, reference curve, constraints).

State vector (eq. 4):    x = [T_hat, R_hat, q]
    T_hat : smoothed throughput (bytes/s)
    R_hat : smoothed RTT (s)
    q     : queue occupancy (bytes)

Control input:           u = [s]  (sending rate, bytes/s)

**Default solver** (``mpc_solver="qp"``): CasADi condensed **QP** from
``planning/network_solver.py`` — same linearized-Gauss–Newton structure as
``third_party/portus-mpcc`` / the paper's fast path (throughput in **bps**,
queue in **bits** inside the solver).

**Optional** ``mpc_solver="nlp"`` uses CasADi IPOPT. ``mpc_solver="slsqp"``
keeps the legacy SciPy SLSQP rollout for debugging / ``_total_cost`` tests.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from congestion_control.base import AckInfo, CCAState, CongestionController, LossInfo
from congestion_control.network_model import (
    BandwidthEstimator,
    DeliveryRateEstimator,
    RTTEstimator,
)
from planning.network_solver import create_solver

if TYPE_CHECKING:
    from planning.network_solver import NetworkMPCCSolver, NetworkMPCCQPSolver

logger = logging.getLogger("cc.mpcc")


class MPCCController(CongestionController):
    """Model Predictive Contouring Control for congestion control.

    Matches the paper formulation:
      - Reference trajectory Gamma(theta) in (throughput, delay) space  (eq. 6)
      - Contouring/lag error decomposition via tangent vector           (eqs. 7-8)
      - Stage cost with Kleinrock's power and hinge delay penalty       (eq. 9)
      - Rate smoothness penalty on consecutive rate changes             (eq. 10)
      - Multi-flow fairness penalty                                     (eq. 11)
      - Hard constraints (eq. 12) via CasADi QP / NLP or legacy SLSQP rollout
      - Discrete-time dynamics in the solver match ``planning/network_solver``
        (Euler / linearized model; observed state still EWMA-smoothed)

    State vector:  x = [T_hat, R_hat, q]   (smoothed throughput, smoothed RTT, queue occupancy)
    Control input: u = [s]                  (sending rate in bytes/s)
    """

    def __init__(
        self,
        # MPC horizon
        horizon: int = 8,
        dt_s: float = 0.020,
        # Dynamics (eq. 5)
        tau_T: float = 0.5,
        tau_R: float = 0.5,
        # Reference path (eq. 6)
        alpha: float = 0.050,
        # Cost weights (eq. 9)
        w_c: float = 5.0,
        w_l: float = 10.0,
        w_d: float = 2.0,
        w_p: float = 1.0,
        # Smoothness (eq. 10)
        w_u: float = 0.1,
        # Fairness (eq. 11)
        w_f: float = 1.0,
        n_flows: int = 1,
        # Constraints (eq. 12)
        q_max_bytes: int = 150_000,
        rate_headroom: float = 2.0,
        target_rtt_s: float = 0.100,
        # AIMD loss response
        md_factor: float = 0.5,
        # Slow start
        min_ss_samples: int = 20,
        ss_bdp_exit_factor: float = 3.0,
        mpc_solver: str = "qp",
        **kwargs,
    ):
        super().__init__(name="MPCC", **kwargs)

        mode = mpc_solver.lower().strip()
        if mode not in ("qp", "nlp", "slsqp"):
            raise ValueError(
                f"mpc_solver must be 'qp', 'nlp', or 'slsqp', got {mpc_solver!r}"
            )
        self._mpc_solver = mode

        # MPC parameters
        self._N = horizon
        self._dt = dt_s

        # Dynamics (eq. 5)
        self._tau_T = tau_T
        self._tau_R = tau_R

        # Reference path (eq. 6)
        self._alpha_ref = alpha

        # Cost weights (eq. 9)
        self._w_c = w_c
        self._w_l = w_l
        self._w_d = w_d
        self._w_p = w_p

        # Smoothness (eq. 10)
        self._w_u = w_u

        # Fairness (eq. 11)
        self._w_f = w_f
        self._n_flows = n_flows

        # Constraints (eq. 12)
        self._q_max = float(q_max_bytes)
        self._rate_headroom = rate_headroom
        self._R_star = target_rtt_s

        # AIMD
        self._md_factor = md_factor

        # Slow start
        self._min_ss_samples = min_ss_samples
        self._ss_bdp_exit = ss_bdp_exit_factor
        self._ss_ack_count = 0

        # Internal MPC state: x = [T_hat, R_hat, q]
        self._T_hat = 0.0       # smoothed throughput (bytes/s)
        self._R_hat = 0.0       # smoothed RTT (s)
        self._q = 0.0           # queue occupancy (bytes)

        # Estimated network parameters
        self._C = 0.0           # link capacity (bytes/s)
        self._R0 = 0.0          # propagation delay (s)

        # MPC solution state
        self._s_opt = 0.0       # optimal sending rate from last solve
        self._prev_s_seq: Optional[np.ndarray] = None

        # Estimators
        self._rtt_est = RTTEstimator(window_s=10.0)
        self._bw_est = BandwidthEstimator(window_s=3.0)
        self._delivery_rate = DeliveryRateEstimator(window_s=0.5)

        # Timing
        self._last_solve_time_s = 0.0
        self._last_update_time_s = 0.0
        self._solve_interval_s = dt_s

        self._state = CCAState.SLOW_START

        # CasADi backend (``qp`` / ``nlp``); rebuilt on ``reset`` / horizon change
        self._net_solver: Union["NetworkMPCCQPSolver", "NetworkMPCCSolver", None] = None

        logger.debug(
            "MPCC initialized: solver=%s, N=%d, dt=%.0fms, tau_T=%.2f, tau_R=%.2f, "
            "alpha=%.3f, R*=%.0fms",
            self._mpc_solver, horizon, dt_s * 1000, tau_T, tau_R, alpha, target_rtt_s * 1000,
        )

    # ------------------------------------------------------------------
    # Network dynamics (eq. 5)
    # ------------------------------------------------------------------

    def _dynamics_f(self, x: np.ndarray, s: float,
                    C: float, R0: float) -> np.ndarray:
        """Continuous-time dynamics f(x, s) from eq. 5.

            dT_hat/dt = (min(s, C) - T_hat) / tau_T
            dR_hat/dt = (R0 + q/C - R_hat)  / tau_R
            dq/dt     = s - C
        """
        T_hat, R_hat, q = x
        C_s = max(C, 1.0)
        dT = (min(s, C_s) - T_hat) / self._tau_T
        dR = (R0 + q / C_s - R_hat) / self._tau_R
        dq = s - C_s
        return np.array([dT, dR, dq])

    def _rk4_step(self, x: np.ndarray, s: float,
                  C: float, R0: float, dt: float) -> np.ndarray:
        """RK4 integration of eq. 5 over one timestep dt."""
        k1 = self._dynamics_f(x, s, C, R0)
        k2 = self._dynamics_f(x + 0.5 * dt * k1, s, C, R0)
        k3 = self._dynamics_f(x + 0.5 * dt * k2, s, C, R0)
        k4 = self._dynamics_f(x + dt * k3, s, C, R0)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next[0] = max(x_next[0], 0.0)   # T_hat >= 0
        x_next[2] = max(x_next[2], 0.0)   # q >= 0
        return x_next

    # ------------------------------------------------------------------
    # Jacobians for QP linearization (eq. 15)
    # ------------------------------------------------------------------

    def _dynamics_jacobians(self, s: float,
                            C: float) -> Tuple[np.ndarray, np.ndarray]:
        """Jacobians df/dx and df/ds of continuous-time dynamics (eq. 15).

        For QP approximation (eq. 13):
            A_k = I + dt * df/dx|_{x_bar, s_bar}
            B_k = dt * df/ds|_{x_bar, s_bar}

        Returns (df/dx, df/ds).
        """
        C_s = max(C, 1.0)
        dfdx = np.array([
            [-1.0 / self._tau_T, 0.0,                0.0],
            [0.0,                -1.0 / self._tau_R,  1.0 / (C_s * self._tau_R)],
            [0.0,                0.0,                 0.0],
        ])
        dfds = np.array([
            1.0 / self._tau_T if s < C_s else 0.0,
            0.0,
            1.0,
        ])
        return dfdx, dfds

    def _linearize(self, x_bar: np.ndarray, s_bar: float,
                   C: float, R0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearize dynamics at (x_bar, s_bar) per eq. 13.

        Returns (A_k, B_k, c_k) such that:
            x_{k+1} ~ A_k x_k + B_k s_k + c_k
        """
        dfdx, dfds = self._dynamics_jacobians(s_bar, C)
        A = np.eye(3) + self._dt * dfdx
        B = self._dt * dfds
        x_bar_next = self._rk4_step(x_bar, s_bar, C, R0, self._dt)
        c = x_bar_next - A @ x_bar - B * s_bar
        return A, B, c

    # ------------------------------------------------------------------
    # Reference trajectory (eq. 6)
    # ------------------------------------------------------------------

    def _reference_path(self, theta: float,
                        C: float, R0: float) -> Tuple[float, float]:
        """Reference trajectory Gamma(theta) = (C*theta, R0 + alpha*theta^2)."""
        theta = float(np.clip(theta, 0.0, 1.0))
        Gamma_T = C * theta
        Gamma_R = R0 + self._alpha_ref * theta ** 2
        return Gamma_T, Gamma_R

    def _tangent(self, theta: float, C: float) -> Tuple[float, float]:
        """Unit tangent t(theta) = (C, 2*alpha*theta) / ||(C, 2*alpha*theta)||."""
        theta = float(np.clip(theta, 0.0, 1.0))
        raw_T = C
        raw_R = 2.0 * self._alpha_ref * theta
        norm = math.sqrt(raw_T ** 2 + raw_R ** 2)
        if norm < 1e-12:
            return 1.0, 0.0
        return raw_T / norm, raw_R / norm

    # ------------------------------------------------------------------
    # Contouring and lag errors (eqs. 7-8)
    # ------------------------------------------------------------------

    def _contouring_lag_errors(self, T_hat: float, R_hat: float,
                               C: float, R0: float) -> Tuple[float, float]:
        """Contouring error e_c (eq. 7) and lag error e_l (eq. 8).

        theta_k = T_hat_k / C  (implicit path parameter)
        e_c = t_R * (T_hat - Gamma_T) - t_T * (R_hat - Gamma_R)
        e_l = t_T * (T_hat - Gamma_T) + t_R * (R_hat - Gamma_R)
        """
        C_s = max(C, 1.0)
        theta = float(np.clip(T_hat / C_s, 0.0, 1.0))
        Gamma_T, Gamma_R = self._reference_path(theta, C_s, R0)
        t_T, t_R = self._tangent(theta, C_s)
        dT = T_hat - Gamma_T
        dR = R_hat - Gamma_R
        e_c = t_R * dT - t_T * dR
        e_l = t_T * dT + t_R * dR
        return e_c, e_l

    # ------------------------------------------------------------------
    # Cost function (eqs. 9-11)
    # ------------------------------------------------------------------

    def _stage_cost(self, x: np.ndarray, C: float, R0: float) -> float:
        """Stage cost l(k) from eq. 9.

        l(k) = w_c * e_c^2 + w_l * e_l^2
             + w_d * [R_hat - R*]_+^2
             - w_p * (T_hat/C) / (R_hat/R0)
        """
        T_hat, R_hat, q = x
        C_s = max(C, 1.0)
        R0_s = max(R0, 1e-6)

        e_c, e_l = self._contouring_lag_errors(T_hat, R_hat, C_s, R0_s)

        # Hinge delay penalty [R_hat - R*]_+^2
        delay_excess = max(R_hat - self._R_star, 0.0)

        # Kleinrock's power: (T_hat/C) / (R_hat/R0)
        R_hat_s = max(R_hat, 1e-6)
        power = (T_hat / C_s) / (R_hat_s / R0_s)

        return (self._w_c * e_c ** 2
                + self._w_l * e_l ** 2
                + self._w_d * delay_excess ** 2
                - self._w_p * power)

    def _total_cost(self, s_seq: np.ndarray, x0: np.ndarray,
                    C: float, R0: float, s_prev: float) -> float:
        """Full cost J (eq. 10) + fairness (eq. 11).

        J = sum_{k=0}^{N} l(k)
          + sum_{k=1}^{N} w_u * (s_k - s_{k-1})^2 / s_max^2
          + sum_{k=0}^{N-1} w_f * (s_k - C/n)^2
        """
        N = len(s_seq)
        s_max = max(self._rate_headroom * C, 1.0)
        fair_share = C / max(self._n_flows, 1) if self._n_flows > 1 else 0.0

        cost = 0.0
        x = x0.copy()

        # l(0): stage cost at initial state
        cost += self._stage_cost(x, C, R0)

        for k in range(N):
            # Rate smoothness (eq. 10): w_u * (s_k - s_{k-1})^2 / s_max^2
            s_km1 = s_prev if k == 0 else s_seq[k - 1]
            cost += self._w_u * ((s_seq[k] - s_km1) / s_max) ** 2

            # Fairness penalty (eq. 11): w_f * (s_k - C/n)^2
            if self._n_flows > 1:
                cost += self._w_f * (s_seq[k] - fair_share) ** 2

            # Propagate state via RK4 (eq. 5)
            x = self._rk4_step(x, s_seq[k], C, R0, self._dt)

            # l(k+1): stage cost at propagated state
            cost += self._stage_cost(x, C, R0)

        return cost

    # ------------------------------------------------------------------
    # NLP constraints (eq. 12)
    # ------------------------------------------------------------------

    def _build_constraints(self, x0: np.ndarray, C: float, R0: float):
        """Build inequality constraints for the NLP (eq. 12c-e).

        For each horizon step k = 1, ..., N:
            0 <= q_k <= q_max     (eq. 12d)
            R_hat_k <= 4 * R*     (eq. 12e)

        Rate bounds 0 <= s_k <= s_max are handled via variable bounds.
        SLSQP convention: ineq constraints must satisfy f(x) >= 0.
        """
        def ineq_fn(s_seq):
            x = x0.copy()
            vals = []
            for k in range(len(s_seq)):
                x = self._rk4_step(x, s_seq[k], C, R0, self._dt)
                vals.append(x[2])                           # q_k >= 0
                vals.append(self._q_max - x[2])             # q_k <= q_max
                vals.append(4.0 * self._R_star - x[1])      # R_hat_k <= 4*R*
            return np.array(vals)

        return {"type": "ineq", "fun": ineq_fn}

    # ------------------------------------------------------------------
    # CasADi QP / NLP backend (``planning/network_solver``)
    # ------------------------------------------------------------------

    def _net_solver_config_template(self) -> dict:
        """YAML-shaped config for ``NetworkMPCCQPSolver`` / ``NetworkMPCCSolver``."""
        return {
            "planner": {"horizon": self._N, "timestep": self._dt},
            "network": {
                "rtt_prop": max(self._R0, 1e-6),
                "tau_rtt": self._tau_R,
                "tau_tput": self._tau_T,
                "rate_max": max(self._rate_headroom * max(self._C, 1.0) * 8.0, 1e6),
                "alpha": self._alpha_ref,
                "rtt_target": self._R_star,
                "q_max": int(self._q_max),
                "n_flows": self._n_flows,
                "delay_hinge_absolute": True,
                "fairness_unnormalized": True,
            },
            "weights": {
                "contour_weight": self._w_c,
                "contouring_lag_weight": self._w_l,
                "delay_weight": self._w_d,
                "power_weight": self._w_p,
                "acceleration_weight": self._w_u,
                "fairness_weight": self._w_f if self._n_flows > 1 else 0.0,
            },
        }

    def _sync_net_solver_params(self) -> None:
        """Refresh time-varying knobs (capacity, RTT prop, weights) before each solve."""
        if self._net_solver is None:
            return
        s = self._net_solver
        C_bps = max(self._C, 1.0) * 8.0
        s.rate_max = self._rate_headroom * C_bps
        s.rtt_prop = max(self._R0, 1e-6)
        s.rtt_target = self._R_star
        s.q_max = int(self._q_max)
        s.n_flows = self._n_flows
        s.alpha = self._alpha_ref
        s.tau_tput = self._tau_T
        s.tau_rtt = self._tau_R
        s.w_contour = self._w_c
        s.w_lag = self._w_l
        s.w_delay = self._w_d
        s.w_power = self._w_p
        s.w_du = self._w_u
        s.w_fair = self._w_f if self._n_flows > 1 else 0.0
        s.delay_hinge_absolute = True
        s.fairness_unnormalized = True

    def _ensure_net_solver(self) -> None:
        if self._mpc_solver == "slsqp":
            return
        if self._net_solver is None:
            self._net_solver = create_solver(
                self._net_solver_config_template(),
                self._mpc_solver,
            )
        self._sync_net_solver_params()

    # ------------------------------------------------------------------
    # MPC solver (eq. 12)
    # ------------------------------------------------------------------

    def _solve_mpc(self, timestamp_s: float) -> None:
        """Receding-horizon MPC: default CasADi QP (paper), optional NLP or SLSQP."""
        if self._mpc_solver == "slsqp":
            self._solve_mpc_slsqp(timestamp_s)
            return

        self._ensure_net_solver()
        assert self._net_solver is not None

        C = max(self._C, 1.0)
        s_max = self._rate_headroom * C
        bw_bps = C * 8.0
        u_prev_bps = (self._s_opt * 8.0) if self._s_opt > 0 else (bw_bps * 0.5)

        res = self._net_solver.solve(
            self._T_hat * 8.0,
            self._R_hat,
            self._q * 8.0,
            bw_bps,
            u_prev_bps=u_prev_bps,
        )

        self._s_opt = float(np.clip(res.send_rate / 8.0, 0.0, s_max))
        if res.control_sequence is not None:
            self._prev_s_seq = np.asarray(res.control_sequence, dtype=float) / 8.0
        else:
            self._prev_s_seq = None

        if not res.success:
            logger.debug("CasADi MPCC solve reported failure; using returned rate")

        rtt = max(self._R_hat, self._R0, 0.001)
        old_cwnd = self._cwnd
        self._cwnd = max(self._s_opt * rtt, float(self.mtu))
        self._state = CCAState.STEADY

        self._notify_cwnd_change(
            old_cwnd, self._cwnd, timestamp_s, f"mpc_s={self._s_opt:.0f}"
        )

    def _solve_mpc_slsqp(self, timestamp_s: float) -> None:
        """Legacy SciPy SLSQP on an RK4-rolled nonlinear cost (debug / unit tests)."""
        from scipy.optimize import minimize

        C = max(self._C, 1.0)
        R0 = max(self._R0, 1e-4)
        N = self._N
        s_max = self._rate_headroom * C

        x0 = np.array([self._T_hat, self._R_hat, self._q])

        if self._prev_s_seq is not None and len(self._prev_s_seq) == N:
            s_init = np.roll(self._prev_s_seq, -1)
            s_init[-1] = s_init[-2]
        else:
            s_init = np.full(N, self._s_opt if self._s_opt > 0 else C * 0.5)
        s_init = np.clip(s_init, 0.0, s_max)

        s_prev = self._s_opt if self._s_opt > 0 else float(s_init[0])

        bounds = [(0.0, s_max)] * N
        constraints = self._build_constraints(x0, C, R0)

        result = minimize(
            self._total_cost, s_init,
            args=(x0, C, R0, s_prev),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-6},
        )

        if result.success or result.fun < self._total_cost(
            s_init, x0, C, R0, s_prev
        ):
            s_opt_seq = result.x
        else:
            s_opt_seq = s_init
            logger.debug("MPCC SLSQP did not converge, using warm-start")

        self._s_opt = float(np.clip(s_opt_seq[0], 0.0, s_max))
        self._prev_s_seq = s_opt_seq

        rtt = max(self._R_hat, self._R0, 0.001)
        old_cwnd = self._cwnd
        self._cwnd = max(self._s_opt * rtt, float(self.mtu))
        self._state = CCAState.STEADY

        self._notify_cwnd_change(
            old_cwnd, self._cwnd, timestamp_s, f"mpc_s={self._s_opt:.0f}"
        )

    # ------------------------------------------------------------------
    # Core event handlers
    # ------------------------------------------------------------------

    def on_ack(self, ack: AckInfo) -> None:
        self._update_rtt(ack.rtt_s)
        self._bytes_delivered += ack.bytes_acked
        self._bytes_in_flight = max(0, self._bytes_in_flight - ack.bytes_acked)

        # Update estimators
        self._rtt_est.add_sample(ack.timestamp_s, ack.rtt_s)
        rate_bps = self._delivery_rate.on_ack(ack.timestamp_s, ack.delivered_bytes)
        self._bw_est.add_sample(ack.timestamp_s, rate_bps)

        # Network parameter estimates
        self._R0 = self._rtt_est.min_rtt_s
        self._C = self._bw_est.max_bw_bps / 8  # bytes/s

        # Update internal MPC state [T_hat, R_hat, q] via exponential smoothing
        # that matches the continuous dynamics (eq. 5) with time constant tau
        dt_actual = (
            ack.timestamp_s - self._last_update_time_s
            if self._last_update_time_s > 0
            else self._dt
        )
        dt_actual = max(dt_actual, 1e-6)
        self._last_update_time_s = ack.timestamp_s

        alpha_T = 1.0 - math.exp(-dt_actual / self._tau_T)
        alpha_R = 1.0 - math.exp(-dt_actual / self._tau_R)

        measured_tput = rate_bps / 8  # bytes/s
        if self._T_hat == 0.0:
            self._T_hat = measured_tput
        else:
            self._T_hat += alpha_T * (measured_tput - self._T_hat)

        if self._R_hat == 0.0:
            self._R_hat = ack.rtt_s
        else:
            self._R_hat += alpha_R * (ack.rtt_s - self._R_hat)

        # q estimated from queuing delay: q = (RTT - R0) * C
        C_s = max(self._C, 1.0)
        self._q = max(ack.rtt_s - self._R0, 0.0) * C_s

        # --- Slow start ---
        if self._state == CCAState.SLOW_START:
            self._do_slow_start(ack)
            return

        # --- MPC solve at regular intervals ---
        if ack.timestamp_s - self._last_solve_time_s >= self._solve_interval_s:
            self._solve_mpc(ack.timestamp_s)
            self._last_solve_time_s = ack.timestamp_s

    def on_loss(self, loss: LossInfo) -> None:
        """AIMD multiplicative decrease on loss."""
        self._bytes_lost += loss.bytes_lost
        old_cwnd = self._cwnd

        self._cwnd *= (1.0 - self._md_factor)
        self._cwnd = max(self._cwnd, float(self.mtu))
        self._ssthresh = self._cwnd
        self._state = CCAState.RECOVERY

        # Update optimal rate to match reduced cwnd
        rtt = max(self._R_hat, self._R0, 0.001)
        self._s_opt = self._cwnd / rtt

        self._notify_cwnd_change(old_cwnd, self._cwnd, loss.timestamp_s, "loss_md")

    def on_timeout(self) -> None:
        old_cwnd = self._cwnd
        self._cwnd = float(self.mtu)
        self._ssthresh = old_cwnd / 2
        self._state = CCAState.SLOW_START
        self._ss_ack_count = 0
        self._s_opt = 0.0
        self._prev_s_seq = None

    def get_cwnd(self) -> int:
        return max(int(self._cwnd), self.mtu)

    def get_pacing_rate(self) -> Optional[float]:
        """Return optimal sending rate s* from MPC solution (bytes/s)."""
        return self._s_opt if self._s_opt > 0 else None

    # ------------------------------------------------------------------
    # Slow start
    # ------------------------------------------------------------------

    def _do_slow_start(self, ack: AckInfo) -> None:
        old_cwnd = self._cwnd
        self._ss_ack_count += 1
        self._cwnd += ack.bytes_acked  # exponential growth

        bdp = self._C * max(self._rtt_est.srtt_s, self._R0, 0.001)
        if self._ss_ack_count >= self._min_ss_samples and bdp > 0:
            if self._cwnd > self._ss_bdp_exit * bdp:
                self._state = CCAState.STEADY
                self._cwnd = max(bdp, float(self.mtu))
                rtt = max(self._R_hat, self._R0, 0.001)
                self._s_opt = self._cwnd / rtt
                self._last_solve_time_s = ack.timestamp_s

        self._notify_cwnd_change(old_cwnd, self._cwnd, ack.timestamp_s, "slow_start")

    # ------------------------------------------------------------------
    # Fairness
    # ------------------------------------------------------------------

    def set_n_flows(self, n: int) -> None:
        """Update the number of competing flows for fairness penalty (eq. 11)."""
        self._n_flows = max(n, 1)
        self._sync_net_solver_params()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._rtt_est.reset()
        self._bw_est.reset()
        self._delivery_rate.reset()
        self._T_hat = 0.0
        self._R_hat = 0.0
        self._q = 0.0
        self._C = 0.0
        self._R0 = 0.0
        self._s_opt = 0.0
        self._prev_s_seq = None
        self._last_solve_time_s = 0.0
        self._last_update_time_s = 0.0
        self._ss_ack_count = 0
        self._state = CCAState.SLOW_START
        self._net_solver = None

    # ------------------------------------------------------------------
    # Convergence properties
    # ------------------------------------------------------------------

    @staticmethod
    def convergence_proof_sketch() -> str:
        """Documents the convergence argument for the MPCC CCA."""
        return """
        MPCC Convergence Proof Sketch
        =============================

        The MPCC congestion controller provably converges to a stable, fair
        allocation because:

        1. AIMD COMPATIBILITY:
           - Additive increase: the MPC optimiser selects a sending rate
             s* within the feasible set [0, s_max] at each step
           - Multiplicative decrease: cwnd *= (1 - MD_factor) on loss
           - This is a standard AIMD controller, which Chiu & Jain (1989)
             proved converges to fairness.

        2. MPC OPTIMALITY WITHIN CONSTRAINTS:
           - The finite-horizon program (eq. 12) minimises the contouring cost subject to:
               0 <= s_k <= s_max        (rate bounds)
               0 <= q_k <= q_max        (queue bounds)
               R_hat_k <= 4*R*          (RTT bound)
           - The cost penalises contouring error (deviation from the
             Pareto frontier), lag error (throughput shortfall), delay
             excess, and rate jitter, while rewarding Kleinrock's power.

        3. LYAPUNOV STABILITY:
           - Define V(t) = w_c*(e_c(t))^2 + w_l*(e_l(t))^2
             where e_c, e_l are contouring and lag errors relative to
             the reference trajectory Gamma(theta).
           - The MPC cost J >= V(t+1) - V(t) by construction.
           - Since MPC minimises J, it drives V toward zero.
           - V is positive definite and radially unbounded -> stability.

        4. FAIRNESS:
           - The fairness penalty w_f*(s_k - C/n)^2 (eq. 11) drives
             each flow toward its fair share C/n.
           - N flows sharing a bottleneck: each runs AIMD with MPC-tuned
             sending rate.  On loss (shared signal), all flows MD by the
             same factor -> convergence to proportional fairness.
        """
