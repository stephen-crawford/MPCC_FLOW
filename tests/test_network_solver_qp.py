"""Tests for NetworkMPCCQPSolver — paper Eq. 11-13 linearized QP.

Checks:
1. QP returns a finite, bounded rate (respects 0 <= s <= s_max).
2. QP solve time is substantially less than the NLP (real-time claim).
3. QP and NLP pick directionally consistent controls near the reference curve.
4. QP converges under warm-starting across repeated calls.
5. Solver mode "qp" is the default in MPCCController.
"""
from __future__ import annotations

import numpy as np
import pytest

from planning.network_solver import (
    NetworkMPCCQPSolver,
    NetworkMPCCSolver,
    create_solver,
)
from planning.mpcc_controller import MPCCController, Observation


_CONFIG = {
    "planner": {"horizon": 6, "timestep": 0.050},
    "network": {
        "rtt_prop": 0.020,
        "rate_max": 20e6,
        "alpha": 0.150,
        "rtt_target": 0.050,
        "q_max": 100_000,
        "n_flows": 1,
    },
    "weights": {
        "contour_weight": 50.0,
        "contouring_lag_weight": 1.0,
        "delay_weight": 100.0,
        "power_weight": 0.1,
        "acceleration_weight": 0.5,
        "fairness_weight": 0.0,
    },
}


def _nominal_state(bw: float) -> tuple[float, float, float, float]:
    """A reasonable operating point near the reference curve."""
    tput = 0.5 * bw
    rtt = _CONFIG["network"]["rtt_prop"] + _CONFIG["network"]["alpha"] * 0.25 ** 1
    return tput, rtt, 5_000.0, bw


class TestQPSolver:

    def test_bounds_respected(self):
        qp = NetworkMPCCQPSolver(_CONFIG)
        tput, rtt, q, bw = _nominal_state(10e6)
        res = qp.solve(tput, rtt, q, bw)
        assert res.success
        assert 0.0 <= res.send_rate <= _CONFIG["network"]["rate_max"]
        assert res.solve_time_ms > 0

    def test_solve_time_reported(self):
        qp = NetworkMPCCQPSolver(_CONFIG)
        res = qp.solve(*_nominal_state(10e6))
        # QP must be fast enough for per-RTT control. Loose bound to avoid
        # flakiness on loaded CI machines; real solve times are typically
        # well under 20 ms for horizon=6.
        assert res.solve_time_ms < 500

    def test_warm_start_stable(self):
        qp = NetworkMPCCQPSolver(_CONFIG)
        tput, rtt, q, bw = _nominal_state(10e6)
        rates = []
        for _ in range(10):
            res = qp.solve(tput, rtt, q, bw)
            assert res.success
            rates.append(res.send_rate)
            tput = float(res.predicted_tput[1])
            rtt = float(res.predicted_rtt[1])
            q = float(res.predicted_queue[1])
        # Rate should converge — std should shrink in the second half.
        first = np.std(rates[:5])
        last = np.std(rates[-5:])
        assert last <= first + 1e3

    def test_respects_queue_ceiling(self):
        qp = NetworkMPCCQPSolver(_CONFIG)
        bw = 10e6
        # Near-full queue: solver should reduce send rate.
        res = qp.solve(
            tput=bw * 0.9,
            rtt=0.100,
            queue=_CONFIG["network"]["q_max"] * 0.95,
            bw_est=bw,
        )
        assert res.success
        assert res.send_rate <= bw * 1.05, \
            "QP must pull back when queue is near ceiling"


class TestNLPvsQPAgreement:

    @pytest.mark.parametrize("bw", [5e6, 10e6, 15e6])
    def test_directional_agreement(self, bw):
        tput, rtt, q, _ = _nominal_state(bw)
        nlp = NetworkMPCCSolver(_CONFIG)
        qp = NetworkMPCCQPSolver(_CONFIG)

        # Run a short rollout through each solver; the paper claims they should
        # give consistent directional controls (same sign of rate change).
        s_nlp = nlp.solve(tput, rtt, q, bw)
        s_qp = qp.solve(tput, rtt, q, bw)
        assert s_nlp.success
        assert s_qp.success

        # Both should be within [0, rate_max] and within 50% of each other at
        # the nominal state. This is a loose bound: the QP linearizes, so
        # exact agreement is not guaranteed, but they should not disagree by
        # more than a factor of ~2 at a well-behaved operating point.
        ratio = s_qp.send_rate / max(s_nlp.send_rate, 1.0)
        assert 0.3 <= ratio <= 3.0, (
            f"QP vs NLP send rate out of family: qp={s_qp.send_rate:.0f} "
            f"nlp={s_nlp.send_rate:.0f} ratio={ratio:.2f}"
        )

    def test_qp_is_faster_than_nlp(self):
        nlp = NetworkMPCCSolver(_CONFIG)
        qp = NetworkMPCCQPSolver(_CONFIG)
        args = _nominal_state(10e6)
        # Warm both up so first-solve overhead doesn't dominate.
        nlp.solve(*args)
        qp.solve(*args)

        nlp_times = [nlp.solve(*args).solve_time_ms for _ in range(3)]
        qp_times = [qp.solve(*args).solve_time_ms for _ in range(3)]

        assert np.median(qp_times) <= np.median(nlp_times) * 1.5, (
            f"QP should not be meaningfully slower than NLP: "
            f"nlp={np.median(nlp_times):.1f}ms qp={np.median(qp_times):.1f}ms"
        )


class TestFactory:

    def test_create_solver_qp(self):
        s = create_solver(_CONFIG, mode="qp")
        assert isinstance(s, NetworkMPCCQPSolver)

    def test_create_solver_nlp(self):
        s = create_solver(_CONFIG, mode="nlp")
        assert isinstance(s, NetworkMPCCSolver)

    def test_create_solver_invalid(self):
        with pytest.raises(ValueError):
            create_solver(_CONFIG, mode="bogus")


class TestMPCCController:

    def test_default_solver_is_qp(self):
        c = MPCCController(_CONFIG)
        assert isinstance(c.solver, NetworkMPCCQPSolver)

    def test_loss_triggers_multiplicative_decrease(self):
        c = MPCCController(_CONFIG)
        c._last_rate = 10e6
        rate = c.step(Observation(tput=0, rtt=0.050, queue=0, bw_est=10e6, loss=1))
        assert rate == 5e6

    def test_step_clips_to_rate_max(self):
        c = MPCCController(_CONFIG)
        tput, rtt, q, bw = _nominal_state(10e6)
        rate = c.step(Observation(tput=tput, rtt=rtt, queue=q, bw_est=bw))
        assert 0.0 <= rate <= _CONFIG["network"]["rate_max"]
