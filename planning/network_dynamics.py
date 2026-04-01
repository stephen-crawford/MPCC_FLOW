"""
Network dynamics model for MPCC-based congestion control.

State mapping to ContouringObjective interface:
    x      = throughput (bytes/s) — contouring "x-position"
    y      = RTT (seconds) — contouring "y-position"
    psi    = 0 (unused heading, required by ContouringObjective)
    v      = queue occupancy (bytes) — mapped to "velocity" slot
    spline = arc-length progress along reference trajectory

Control: [send_rate] in bytes/s

The state names (x, y, psi, v, spline) match what ContouringObjective expects
so the contouring/lag error math works without modification. The reference
trajectory is generated in throughput-RTT space (see network_reference.py).
"""

import casadi as cd
import numpy as np

from planning.dynamic_models import DynamicsModel
from utils.utils import LOG_INFO


class NetworkDynamicsModel(DynamicsModel):

    DEFAULT_RTT_PROP = 0.025
    DEFAULT_TAU_RTT = 0.1
    DEFAULT_TAU_TPUT = 0.1
    DEFAULT_RATE_MAX = 50e6
    DEFAULT_Q_MAX = 500_000
    DEFAULT_RTT_MAX = 0.200
    DEFAULT_BW_MAX = 100e6

    def __init__(self, config=None):
        super().__init__()

        cfg = (config or {}).get("network", {})
        self.rtt_prop = cfg.get("rtt_prop", self.DEFAULT_RTT_PROP)
        self.tau_rtt = cfg.get("tau_rtt", self.DEFAULT_TAU_RTT)
        self.tau_tput = cfg.get("tau_tput", self.DEFAULT_TAU_TPUT)
        self.rate_max = cfg.get("rate_max", self.DEFAULT_RATE_MAX)
        self.q_max = cfg.get("q_max", self.DEFAULT_Q_MAX)
        self.rtt_max = cfg.get("rtt_max", self.DEFAULT_RTT_MAX)
        self.bw_max = cfg.get("bw_max", self.DEFAULT_BW_MAX)

        self.nu = 1
        self.state_dimension = 5

        self.dependent_vars = ["x", "y", "psi", "v", "spline"]
        self.inputs = ["send_rate"]

        self.lower_bound = [
            0.0,
            0.0,
            self.rtt_prop,
            -np.pi,
            0.0,
            0.0,
        ]
        self.upper_bound = [
            self.rate_max,
            self.bw_max,
            self.rtt_max,
            np.pi,
            self.q_max,
            10000.0,
        ]

        self.state_dimension_integrate = 4
        self.do_not_use_integration_for_last_n_states(n=1)

        LOG_INFO(
            f"NetworkDynamicsModel: rtt_prop={self.rtt_prop:.3f}s, "
            f"rate_max={self.rate_max/1e6:.1f}Mbps, q_max={self.q_max/1e3:.0f}KB"
        )

    def continuous_model(self, x, u, p):
        send_rate = u[0]
        tput = x[0]
        rtt = x[1]
        q = x[3]

        bw_safe = cd.fmax(self._get_bw_est(), 1.0)

        dq_dt = send_rate - bw_safe
        rtt_target = self.rtt_prop + q / bw_safe
        drtt_dt = (rtt_target - rtt) / self.tau_rtt
        effective_rate = cd.fmin(send_rate, bw_safe)
        dtput_dt = (effective_rate - tput) / self.tau_tput
        dpsi_dt = 0.0

        return cd.vertcat(dtput_dt, drtt_dt, dpsi_dt, dq_dt)

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        x = self.get_x()
        s = x[4] if x.size1() > 4 else 0.0
        dt = kwargs.get("timestep", 0.05)

        tput_new = integrated_states[0]
        bw_safe = cd.fmax(self._get_bw_est(), 1.0)
        utilization = tput_new / bw_safe
        ds = utilization * dt
        new_s = cd.fmax(s + ds, 0.0)

        return cd.vertcat(integrated_states, new_s)

    def symbolic_dynamics(self, x, u, p, timestep):
        x_integrate = x[:4]
        s = x[4]

        k1 = self.continuous_model(x, u, p)
        k2 = self.continuous_model(cd.vertcat(x_integrate + timestep / 2 * k1, s), u, p)
        k3 = self.continuous_model(cd.vertcat(x_integrate + timestep / 2 * k2, s), u, p)
        k4 = self.continuous_model(cd.vertcat(x_integrate + timestep * k3, s), u, p)
        x_next = x_integrate + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        q_next = cd.fmax(x_next[3], 0.0)
        x_next = cd.vertcat(x_next[0], x_next[1], x_next[2], q_next)

        bw_safe = cd.fmax(self._get_bw_est(), 1.0)
        utilization = x_next[0] / bw_safe
        s_next = cd.fmax(s + utilization * timestep, 0.0)

        return cd.vertcat(x_next, s_next)

    def _get_bw_est(self):
        """Return the current bandwidth estimate from params or a default."""
        if self.params is not None and hasattr(self.params, '__len__') and len(self.params) > 0:
            return self.params[0]
        return 10e6
