"""
Network MPC problem definition and planner factory.

Self-contained: does not modify any existing MPCC_FLOW files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from planning.planner import Planner
from planning.types import State, Data, define_robot_area
from planning.network_dynamics import NetworkDynamicsModel
from modules.objectives.contouring_objective import ContouringObjective
from utils.utils import LOG_WARN


class NetworkMPCProblem:

    def __init__(self, config: Dict):
        self.config = config
        self.model_type = NetworkDynamicsModel(config)
        self.modules = []
        self.obstacles = []
        self.data = None
        self.x0 = None
        self._state = None

    def setup(self, initial_state: Dict[str, float], reference_path):
        self.data = Data()
        self.data.reference_path = reference_path
        self.data.dynamics_model = self.model_type

        planner_config = self.config.get("planner", {})
        self.data.horizon = planner_config.get("horizon", 15)
        self.data.timestep = planner_config.get("timestep", 0.05)

        self.data.goal = np.array([0.0, 0.0])
        self.data.goal_received = False
        self.data.parameters = {}
        self.data.robot_area = define_robot_area(length=0.0, width=0.0, n_discs=1)
        self.data.road_width = self.config.get("contouring_constraints", {}).get(
            "road_width", 0.100
        )
        self.data.dynamic_obstacles = []
        self.obstacles = []

        self.x0 = State(self.model_type)
        for key, val in initial_state.items():
            self.x0.set(key, val)
        self.data.state = self.x0
        self._state = self.x0

        self.modules = []
        try:
            self.modules.append(ContouringObjective())
        except Exception as e:
            LOG_WARN(f"Could not create ContouringObjective: {e}")

    def get_model_type(self):
        return self.model_type

    def get_modules(self):
        return self.modules

    def get_obstacles(self):
        return self.obstacles

    def get_data(self):
        return self.data

    def get_x0(self):
        return self.x0

    def get_state(self):
        return self._state if self._state is not None else self.x0

    def get_horizon(self):
        return self.data.horizon if self.data else 15

    def get_timestep(self):
        return self.data.timestep if self.data else 0.05


def create_network_planner(
    initial_state: Dict[str, float],
    reference_path,
    config: Optional[Dict] = None,
) -> Planner:
    if config is None:
        config = _default_network_config()

    problem = NetworkMPCProblem(config)
    problem.setup(initial_state, reference_path)
    return Planner(problem, config)


def _default_network_config() -> dict:
    return {
        "planner": {"horizon": 10, "timestep": 0.05},
        "solver": {"solver": "casadi", "shift_previous_solution_forward": True},
        "solver_iterations": 1,
        "max_obstacles": 0,
        "max_obstacle_distance": 0.0,
        "integrator_step": 0.05,
        "network": {"rtt_prop": 0.025, "rate_max": 50e6, "alpha": 0.150},
        "obstacle_constraint_type": "scenario",
        "obstacle_constraint": {
            "num_scenarios": 0,
            "ego_radius": 0.0,
            "obstacle_radius": 0.0,
            "safety_margin": 0.0,
        },
        "contouring_constraints": {"road_width": 0.100},
        "contouring_objective": {
            "lag_weight": 1.0,
            "contour_weight": 10.0,
            "progress_weight": 0.5,
        },
        "goal_objective": {"weight": 0.0},
        "weights": {
            "contour_weight": 10.0,
            "contouring_lag_weight": 1.0,
            "goal_weight": 0.0,
            "goal_angle_weight": 0.0,
            "goal_terminal_weight": 0.0,
            "goal_terminal_angle": 0.0,
            "acceleration_weight": 0.1,
            "angular_velocity": 0.0,
            "slack_weight": 5.0,
            "velocity_tracking_weight": 0.0,
            "reference_velocity": 0.0,
        },
    }
