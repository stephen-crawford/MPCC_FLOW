"""
Tests for NetworkDynamicsModel and bandwidth estimators.

State mapping: x=throughput, y=RTT, psi=heading(0), v=queue, spline=progress
"""

import numpy as np
import pytest
import casadi as cd

from planning.network_dynamics import NetworkDynamicsModel
from planning.network_estimator import (
    AckInfo,
    EMAEstimator,
    SproutEstimator,
    create_estimator,
)


class TestNetworkDynamicsModel:

    def test_init_defaults(self):
        model = NetworkDynamicsModel()
        assert model.nu == 1
        assert model.state_dimension == 5
        assert model.state_dimension_integrate == 4
        assert model.dependent_vars == ["x", "y", "psi", "v", "spline"]
        assert model.inputs == ["send_rate"]

    def test_init_custom_config(self):
        cfg = {"network": {"rtt_prop": 0.050, "rate_max": 100e6, "q_max": 1_000_000}}
        model = NetworkDynamicsModel(cfg)
        assert model.rtt_prop == 0.050
        assert model.rate_max == 100e6
        assert model.q_max == 1_000_000

    def test_bounds_shape(self):
        model = NetworkDynamicsModel()
        assert len(model.lower_bound) == model.nu + model.state_dimension
        assert len(model.upper_bound) == model.nu + model.state_dimension

    def test_continuous_model_queue_grows_when_overloaded(self):
        model = NetworkDynamicsModel()
        x = cd.DM([0.0, 0.025, 0.0, 0.0, 0.0])
        u = cd.DM([15e6])
        p = cd.DM([])

        dx = model.continuous_model(x, u, p)
        dq_dt = float(dx[3])
        assert dq_dt > 0, f"Queue should grow when overloaded, got dq/dt={dq_dt}"

    def test_continuous_model_queue_drains_when_underloaded(self):
        model = NetworkDynamicsModel()
        x = cd.DM([5e6, 0.050, 0.0, 100000.0, 0.0])
        u = cd.DM([5e6])
        p = cd.DM([])

        dx = model.continuous_model(x, u, p)
        dq_dt = float(dx[3])
        assert dq_dt < 0, f"Queue should drain when underloaded, got dq/dt={dq_dt}"

    def test_continuous_model_rtt_increases_with_queue(self):
        model = NetworkDynamicsModel()
        x = cd.DM([10e6, 0.025, 0.0, 500000.0, 0.0])
        u = cd.DM([10e6])
        p = cd.DM([])

        dx = model.continuous_model(x, u, p)
        drtt_dt = float(dx[1])
        assert drtt_dt > 0, f"RTT should increase with queue, got drtt/dt={drtt_dt}"

    def test_continuous_model_throughput_tracks_rate(self):
        model = NetworkDynamicsModel()
        x = cd.DM([0.0, 0.025, 0.0, 0.0, 0.0])
        u = cd.DM([8e6])
        p = cd.DM([])

        dx = model.continuous_model(x, u, p)
        dtput_dt = float(dx[0])
        assert dtput_dt > 0, f"Throughput should increase, got dtput/dt={dtput_dt}"

    def test_continuous_model_throughput_capped_by_bandwidth(self):
        model = NetworkDynamicsModel()
        x = cd.DM([0.0, 0.025, 0.0, 0.0, 0.0])
        u = cd.DM([20e6])
        p = cd.DM([])

        dx = model.continuous_model(x, u, p)
        dtput_dt = float(dx[0])
        expected_dtput_dt = 10e6 / model.tau_tput
        assert abs(dtput_dt - expected_dtput_dt) < 1e3

    def test_continuous_model_psi_constant(self):
        model = NetworkDynamicsModel()
        x = cd.DM([0.0, 0.025, 0.0, 0.0, 0.0])
        u = cd.DM([8e6])
        p = cd.DM([])

        dx = model.continuous_model(x, u, p)
        dpsi_dt = float(dx[2])
        assert dpsi_dt == 0.0

    def test_symbolic_dynamics_shape(self):
        model = NetworkDynamicsModel()
        x = cd.MX.sym("x", 5)
        u = cd.MX.sym("u", 1)
        p = cd.MX.sym("p", 0)

        x_next = model.symbolic_dynamics(x, u, p, 0.05)
        assert x_next.shape == (5, 1)

    def test_symbolic_dynamics_numeric(self):
        model = NetworkDynamicsModel()
        x = cd.DM([0.0, 0.025, 0.0, 0.0, 0.0])
        u = cd.DM([5e6])
        p = cd.DM([])

        x_next = model.symbolic_dynamics(x, u, p, 0.05)
        assert x_next.shape == (5, 1)

        tput_next = float(x_next[0])
        rtt_next = float(x_next[1])
        q_next = float(x_next[3])
        spline_next = float(x_next[4])

        assert q_next >= 0
        assert rtt_next > 0
        assert rtt_next < 1.0
        assert tput_next >= 0
        assert spline_next >= 0


class TestEMAEstimator:

    def test_initial_estimate(self):
        est = EMAEstimator(alpha=0.3, initial_bw=1e6)
        assert est.get_estimate() == 1e6

    def test_converges_to_measured_rate(self):
        est = EMAEstimator(alpha=0.5, initial_bw=1e6)
        measured_bw = 5e6
        pkt_size = 1400
        interval = pkt_size / measured_bw

        for i in range(100):
            t = i * interval + i * 0.011
            ack = AckInfo(
                timestamp=t,
                send_timestamp=t - 0.025,
                seq_num=i,
                bytes_acked=pkt_size,
            )
            est.update(ack)

        assert est.get_estimate() > 0

    def test_rtt_computation(self):
        est = EMAEstimator()
        ack = AckInfo(timestamp=1.025, send_timestamp=1.000, seq_num=0, bytes_acked=1400)
        rtt = est.get_rtt(ack)
        assert abs(rtt - 0.025) < 1e-6


class TestSproutEstimator:

    def test_initial_estimate(self):
        est = SproutEstimator(initial_bw=5e6)
        assert est.get_estimate() == 5e6

    def test_forecast_conservative(self):
        est = SproutEstimator(initial_bw=10e6)
        forecast = est.forecast(horizon_s=0.5)
        assert forecast <= est.get_estimate()

    def test_forecast_longer_horizon_more_conservative(self):
        est = SproutEstimator(initial_bw=10e6)
        short = est.forecast(horizon_s=0.1)
        long = est.forecast(horizon_s=1.0)
        assert long <= short


class TestEstimatorFactory:

    def test_creates_ema_by_default(self):
        est = create_estimator({})
        assert isinstance(est, EMAEstimator)

    def test_creates_ema_explicitly(self):
        est = create_estimator({"network": {"estimator": "ema"}})
        assert isinstance(est, EMAEstimator)

    def test_creates_sprout(self):
        est = create_estimator({"network": {"estimator": "sprout"}})
        assert isinstance(est, SproutEstimator)
