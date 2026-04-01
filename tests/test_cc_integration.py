"""
Integration tests for the MPCC congestion control pipeline.

State mapping: x=throughput, y=RTT, psi=heading(0), v=queue, spline=progress
"""

import numpy as np
import pytest

from planning.network_dynamics import NetworkDynamicsModel
from planning.network_reference import generate_network_reference
from planning.network_estimator import create_estimator


class TestNetworkPipelineIntegration:

    def test_model_and_reference_compatible(self):
        model = NetworkDynamicsModel()
        ref = generate_network_reference(bw_est=10e6)

        assert model.dependent_vars[0] == "x"
        assert model.dependent_vars[1] == "y"

        assert ref.x[0] == 0.0
        assert ref.x[-1] > 0
        assert ref.y[0] > 0

    def test_create_network_planner(self):
        from planning.network_problem import NetworkMPCProblem

        config = {
            "network": {"rtt_prop": 0.025, "rate_max": 50e6},
            "planner": {"horizon": 10, "timestep": 0.05},
            "contouring_objective": {"lag_weight": 1.0, "contour_weight": 10.0},
            "contouring_constraints": {"road_width": 0.100},
        }

        ref = generate_network_reference(bw_est=10e6, rtt_prop=0.025)

        problem = NetworkMPCProblem(config)
        problem.setup(
            initial_state={
                "x": 0.0,
                "y": 0.025,
                "psi": 0.0,
                "v": 0.0,
                "spline": 0.0,
            },
            reference_path=ref,
        )

        assert problem.get_model_type() is not None
        assert problem.get_data() is not None
        assert problem.get_x0() is not None
        assert len(problem.get_obstacles()) == 0

        x0 = problem.get_x0()
        assert x0.get("y") == pytest.approx(0.025)

    def test_estimator_feeds_reference(self):
        from planning.network_reference import AdaptiveNetworkReference

        estimator = create_estimator({"network": {"estimator": "ema", "ema_alpha": 0.5}})
        adaptive_ref = AdaptiveNetworkReference(update_threshold=0.20)

        ref1 = adaptive_ref.get_reference(bw_est=5e6)
        assert ref1.x[-1] == pytest.approx(5e6)

        ref2 = adaptive_ref.get_reference(bw_est=10e6)
        assert ref2.x[-1] == pytest.approx(10e6)
        assert ref2 is not ref1

    def test_dynamics_forward_simulation(self):
        import casadi as cd

        model = NetworkDynamicsModel()

        x = cd.DM([0.0, 0.025, 0.0, 0.0, 0.0])
        u = cd.DM([8e6])
        p = cd.DM([])
        dt = 0.05

        states = [np.array(x).flatten()]
        for _ in range(20):
            x = model.symbolic_dynamics(x, u, p, dt)
            states.append(np.array(x).flatten())

        states = np.array(states)

        final_tput = states[-1, 0]
        assert final_tput > 5e6, f"Throughput should converge toward 8 Mbps, got {final_tput/1e6:.1f}"

        assert states[-1, 3] == pytest.approx(0.0, abs=1.0)
        assert states[-1, 1] < 0.030
        assert states[-1, 4] > 0

    def test_dynamics_overload_scenario(self):
        import casadi as cd

        model = NetworkDynamicsModel()

        x = cd.DM([0.0, 0.025, 0.0, 0.0, 0.0])
        u = cd.DM([15e6])
        p = cd.DM([])
        dt = 0.05

        states = [np.array(x).flatten()]
        for _ in range(20):
            x = model.symbolic_dynamics(x, u, p, dt)
            states.append(np.array(x).flatten())

        states = np.array(states)

        assert states[-1, 3] > 0, "Queue should grow when overloaded"
        assert states[-1, 1] > 0.025, "RTT should increase with queue"
