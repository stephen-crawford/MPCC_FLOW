"""
Factory for creating MPC planners.

This module provides the MPCProblem class and create_planner function.
"""

import numpy as np
from typing import Dict, List, Optional

from planning.planner import Planner
from planning.types import State, Data, DynamicObstacle, define_robot_area
from planning.dynamic_models import ContouringSecondOrderUnicycleModel

from modules.constraints.base_constraint import BaseConstraint
from modules.constraints.contouring_constraints import ContouringConstraints
from modules.objectives.goal_objective import GoalObjective
from modules.objectives.contouring_objective import ContouringObjective

from pympc.registry import get_constraint_class, list_constraint_types
from pympc.config import create_default_config
from utils.utils import LOG_INFO, LOG_DEBUG, LOG_WARN


class MPCProblem:
    """
    Problem definition for the MPC planner.

    Encapsulates the dynamics model, modules, obstacles, and data.
    Supports configurable obstacle constraint types.
    """

    def __init__(self, config: Dict):
        """
        Initialize the problem.

        Args:
            config: Configuration dictionary. Should contain "obstacle_constraint_type"
                   to specify which constraint type to use.
        """
        self.config = config
        # Use ContouringSecondOrderUnicycleModel which has the 'spline' state
        # required by ContouringObjective and ContouringConstraints
        self.model_type = ContouringSecondOrderUnicycleModel()
        self.modules = []
        self.obstacles = []
        self.data = None
        self.x0 = None
        self._state = None

        # Get constraint type from config
        self.constraint_type = config.get("obstacle_constraint_type", "scenario")
        LOG_INFO(f"MPCProblem using obstacle constraint type: {self.constraint_type}")

    def setup(
        self,
        initial_state: Dict[str, float],
        reference_path,
        obstacles: List[DynamicObstacle],
        goal: np.ndarray
    ):
        """
        Setup the problem with initial conditions.

        Args:
            initial_state: Dictionary with initial state values (x, y, psi, v, spline)
            reference_path: Reference path for contouring
            obstacles: List of dynamic obstacles
            goal: Goal position [x, y]
        """
        # Create data container
        self.data = Data()

        # Set reference path
        self.data.reference_path = reference_path

        # Set goal
        self.data.goal = goal
        self.data.goal_received = True
        self.data.parameters = {
            'goal_x': float(goal[0]),
            'goal_y': float(goal[1]),
        }

        # Set dynamics model
        self.data.dynamics_model = self.model_type

        # Set horizon and timestep
        planner_config = self.config.get("planner", {})
        self.data.horizon = planner_config.get("horizon", 20)
        self.data.timestep = planner_config.get("timestep", 0.1)

        # Set robot area (collision discs)
        self.data.robot_area = define_robot_area(length=2.0, width=1.0, n_discs=1)

        # Set road width for contouring
        self.data.road_width = self.config.get("contouring_constraints", {}).get("road_width", 4.0)

        # Store obstacles
        self.obstacles = obstacles
        self.data.dynamic_obstacles = obstacles

        # Create initial state
        self.x0 = State(self.model_type)
        for key, val in initial_state.items():
            self.x0.set(key, val)
        self.data.state = self.x0
        self._state = self.x0

        # Create modules
        self._create_modules()

    def _create_modules(self):
        """Create constraint and objective modules based on configuration."""
        self.modules = []

        # Goal objective - drives towards goal
        goal_obj = GoalObjective()
        self.modules.append(goal_obj)

        # Contouring objective - follows reference path
        try:
            contouring_obj = ContouringObjective()
            self.modules.append(contouring_obj)
        except Exception as e:
            LOG_DEBUG(f"Could not create ContouringObjective: {e}")

        # Obstacle constraint - selected based on config
        obstacle_constraint = self._create_obstacle_constraint()
        if obstacle_constraint is not None:
            self.modules.append(obstacle_constraint)
        else:
            LOG_WARN(f"Could not create obstacle constraint of type: {self.constraint_type}")

        # Contouring constraints - keeps vehicle on road
        try:
            contouring_constraint = ContouringConstraints()
            self.modules.append(contouring_constraint)
        except Exception as e:
            LOG_DEBUG(f"Could not create ContouringConstraints: {e}")

    def _create_obstacle_constraint(self) -> Optional[BaseConstraint]:
        """
        Create the obstacle constraint based on configuration.

        Returns:
            Configured obstacle constraint module, or None if creation fails
        """
        constraint_class = get_constraint_class(self.constraint_type)

        if constraint_class is None:
            LOG_WARN(f"Unknown obstacle constraint type: {self.constraint_type}")
            LOG_WARN(f"Available types: {list_constraint_types()}")
            # Fall back to scenario if available
            constraint_class = get_constraint_class("scenario")
            if constraint_class is None:
                return None

        try:
            # Some constraint classes accept config, some don't
            # Try with config first, then without
            try:
                return constraint_class(self.config)
            except TypeError:
                return constraint_class()
        except Exception as e:
            LOG_WARN(f"Failed to create {self.constraint_type} constraint: {e}")
            return None

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
        return self.data.horizon if self.data else 20

    def get_timestep(self):
        return self.data.timestep if self.data else 0.1


def create_planner(
    initial_state: Dict[str, float],
    reference_path,
    obstacles: List[DynamicObstacle],
    goal: np.ndarray,
    config: Optional[Dict] = None,
    constraint_type: Optional[str] = None
) -> Planner:
    """
    Create a configured MPC planner.

    Args:
        initial_state: Dictionary with initial state values
        reference_path: Reference path for contouring
        obstacles: List of dynamic obstacles
        goal: Goal position [x, y]
        config: Optional configuration dictionary
        constraint_type: Optional override for obstacle constraint type ("scenario" or "linearized").

    Returns:
        Configured Planner instance
    """
    if config is None:
        config = create_default_config(constraint_type or "scenario")
    elif constraint_type is not None:
        config["obstacle_constraint_type"] = constraint_type

    problem = MPCProblem(config)
    problem.setup(initial_state, reference_path, obstacles, goal)

    planner = Planner(problem, config)
    return planner
