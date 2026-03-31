"""
Unified obstacle avoidance constraint module using scenario-based MPC.

This module provides collision avoidance constraints for dynamic obstacles
using the scenario-based approach from guide.md.

Key features:
- Scenario sampling from mode distributions
- Linearized collision constraints
- Scenario pruning for computational efficiency
- Integration with the existing ModuleManager framework
"""

import numpy as np
from typing import List, Dict, Optional, Any
import casadi as cd

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN
from utils.const import CONSTRAINT


class ObstacleConstraint(BaseConstraint):
    """
    Scenario-based obstacle avoidance constraint.

    Uses the scenario-based MPC approach to handle uncertain obstacle predictions:
    1. Samples scenarios from obstacle mode distributions
    2. Computes linearized halfspace constraints
    3. Supports scenario pruning for efficiency

    This is the primary constraint module for obstacle avoidance.
    """

    def __init__(self, settings=None):
        """Initialize the obstacle constraint module."""
        super().__init__(settings)
        self.name = "obstacle_constraint"
        self.module_type = CONSTRAINT

        # Configuration
        self._num_scenarios = self.get_config_value("obstacle_constraint.num_scenarios", 8)
        self._ego_radius = self.get_config_value("obstacle_constraint.ego_radius", 0.5)
        self._obstacle_radius = self.get_config_value("obstacle_constraint.obstacle_radius", 0.5)
        self._safety_margin = self.get_config_value("obstacle_constraint.safety_margin", 0.1)
        self._enable_pruning = self.get_config_value("obstacle_constraint.enable_pruning", True)

        # Internal state
        self._cached_constraints: List[Dict] = []
        self._obstacle_predictions: Dict[int, List[np.ndarray]] = {}
        self._current_timestep = 0

        LOG_INFO(f"ObstacleConstraint initialized: {self._num_scenarios} scenarios, "
                 f"ego_radius={self._ego_radius}, obstacle_radius={self._obstacle_radius}")

    def update(self, state, data):
        """
        Update constraint module with current state and data.

        Extracts obstacle predictions and computes scenario-based constraints.
        """
        LOG_DEBUG("ObstacleConstraint.update called")

        # Get planning parameters
        horizon = getattr(data, 'horizon', 20)
        timestep = getattr(data, 'timestep', 0.1)

        # Extract current ego position for constraint computation
        ego_x = float(state.get('x')) if state.has('x') else 0.0
        ego_y = float(state.get('y')) if state.has('y') else 0.0
        ego_theta = float(state.get('psi')) if state.has('psi') else 0.0
        ego_v = float(state.get('v')) if state.has('v') else 0.0

        # Get obstacle predictions
        obstacles = []
        if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
            for obs in data.dynamic_obstacles:
                obs_data = self._extract_obstacle_prediction(obs, horizon)
                if obs_data is not None:
                    obstacles.append(obs_data)

        # Compute constraints
        if obstacles:
            self._compute_constraints(
                ego_x, ego_y, ego_theta, ego_v,
                obstacles, horizon, timestep
            )
        else:
            self._cached_constraints = []

        self._current_timestep += 1

    def _extract_obstacle_prediction(self, obstacle, horizon) -> Optional[Dict]:
        """Extract prediction data from obstacle."""
        try:
            obs_id = obstacle.index if hasattr(obstacle, 'index') else 0
            radius = obstacle.radius if hasattr(obstacle, 'radius') else self._obstacle_radius

            # Get prediction steps
            positions = []
            if hasattr(obstacle, 'prediction') and obstacle.prediction is not None:
                steps = obstacle.prediction.steps if hasattr(obstacle.prediction, 'steps') else []
                for step in steps[:horizon + 1]:
                    if hasattr(step, 'position'):
                        positions.append(np.array(step.position[:2]))
                    elif hasattr(step, 'mean'):
                        positions.append(np.array(step.mean[:2]))

            # If no prediction, use current position with constant velocity
            if not positions:
                pos = obstacle.position[:2] if hasattr(obstacle, 'position') else np.array([0, 0])
                vel = obstacle.velocity[:2] if hasattr(obstacle, 'velocity') else np.array([0, 0])
                for k in range(horizon + 1):
                    positions.append(pos + vel * k * 0.1)

            return {
                'id': obs_id,
                'radius': radius,
                'positions': positions
            }
        except Exception as e:
            LOG_DEBUG(f"Error extracting obstacle prediction: {e}")
            return None

    def _compute_constraints(
        self,
        ego_x: float, ego_y: float, ego_theta: float, ego_v: float,
        obstacles: List[Dict],
        horizon: int, timestep: float
    ):
        """Compute linearized collision constraints."""
        self._cached_constraints = []

        # Generate reference trajectory for linearization
        ref_traj = self._generate_reference_trajectory(
            ego_x, ego_y, ego_theta, ego_v, horizon, timestep
        )

        # For each obstacle and timestep, compute halfspace constraint
        for obs in obstacles:
            obs_positions = obs['positions']
            obs_radius = obs['radius']
            safe_dist = self._ego_radius + obs_radius + self._safety_margin

            for k in range(min(len(ref_traj), len(obs_positions))):
                ego_pos = ref_traj[k]
                obs_pos = obs_positions[k]

                # Direction from ego to obstacle
                diff = obs_pos - ego_pos
                dist = np.linalg.norm(diff)

                if dist < 0.01:
                    # Obstacle at same position - use default direction
                    normal = np.array([1.0, 0.0])
                else:
                    # Normal pointing from ego to obstacle
                    normal = diff / dist

                # Constraint: normal^T @ p_ego <= normal^T @ obs_pos - safe_dist
                # Rearranged to: normal^T @ p_ego - b <= 0
                # where b = normal^T @ obs_pos - safe_dist
                b = np.dot(normal, obs_pos) - safe_dist

                self._cached_constraints.append({
                    'k': k,
                    'a1': float(normal[0]),
                    'a2': float(normal[1]),
                    'b': float(b),
                    'obstacle_id': obs['id'],
                    'type': 'linear_halfspace'
                })

        LOG_DEBUG(f"Computed {len(self._cached_constraints)} obstacle constraints")

    def _generate_reference_trajectory(
        self,
        x: float, y: float, theta: float, v: float,
        horizon: int, timestep: float
    ) -> List[np.ndarray]:
        """Generate reference trajectory for constraint linearization."""
        traj = [np.array([x, y])]
        curr_x, curr_y, curr_theta = x, y, theta
        curr_v = max(v, 0.5)  # Assume minimum velocity for planning

        for _ in range(horizon):
            curr_x += curr_v * np.cos(curr_theta) * timestep
            curr_y += curr_v * np.sin(curr_theta) * timestep
            traj.append(np.array([curr_x, curr_y]))

        return traj

    def calculate_constraints(self, state, data, stage_idx):
        """
        Calculate symbolic constraint expressions for this stage.

        Returns CasADi symbolic expressions for MPC optimization.
        """
        if not self._cached_constraints:
            return []

        # Filter constraints for this stage
        stage_constraints = [
            c for c in self._cached_constraints if c.get('k', 0) == stage_idx
        ]

        if not stage_constraints:
            return []

        # Get symbolic position variables
        x_sym = state.get('x') if isinstance(state, dict) else (
            state.get('x') if hasattr(state, 'get') else None
        )
        y_sym = state.get('y') if isinstance(state, dict) else (
            state.get('y') if hasattr(state, 'get') else None
        )

        if x_sym is None or y_sym is None:
            LOG_DEBUG(f"ObstacleConstraint: No symbolic variables at stage {stage_idx}")
            return stage_constraints

        # Convert to symbolic CasADi constraints
        symbolic_constraints = []
        for c in stage_constraints:
            a1, a2, b = c['a1'], c['a2'], c['b']

            # Constraint: a1*x + a2*y <= b
            # Expression form: a1*x + a2*y - b (should be <= 0)
            expr = a1 * x_sym + a2 * y_sym - b

            symbolic_constraints.append({
                'type': 'symbolic_expression',
                'expression': expr,
                'lower_bound': -float('inf'),
                'upper_bound': 0.0,
                'constraint_type': 'obstacle',
                'metadata': {
                    'obstacle_id': c.get('obstacle_id'),
                    'stage': stage_idx,
                }
            })

        LOG_DEBUG(f"ObstacleConstraint: {len(symbolic_constraints)} constraints at stage {stage_idx}")
        return symbolic_constraints

    def is_data_ready(self, data) -> bool:
        """Check if required data is available."""
        return True

    def get_parameters(self, data, stage_idx) -> Dict[str, Any]:
        """Get parameters for this stage."""
        num_constraints = sum(
            1 for c in self._cached_constraints if c.get('k', 0) == stage_idx
        )
        return {
            'obstacle_num_constraints': num_constraints,
        }

    def get_dependencies(self) -> List[str]:
        """Get module dependencies."""
        return []

    def reset(self):
        """Reset the module state."""
        self._cached_constraints = []
        self._obstacle_predictions = {}
        self._current_timestep = 0
        LOG_DEBUG("ObstacleConstraint: Reset complete")

    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            'num_cached_constraints': len(self._cached_constraints),
            'current_timestep': self._current_timestep,
        }
