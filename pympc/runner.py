"""
MPC execution runner.

This module provides functions for running the MPC planner.
"""

import numpy as np
from typing import Dict

from planning.planner import Planner
from planning.types import Prediction, PredictionStep, PredictionType
from utils.utils import LOG_INFO


def propagate_obstacle_predictions(data, dt: float, horizon: int):
    """Propagate obstacle predictions over the planning horizon.

    Args:
        data: Data object with dynamic_obstacles attribute.
        dt: Timestep.
        horizon: Planning horizon.
    """
    if not hasattr(data, 'dynamic_obstacles') or not data.dynamic_obstacles:
        return

    for obstacle in data.dynamic_obstacles:
        pred = obstacle.prediction
        if pred is None:
            pred = Prediction(PredictionType.GAUSSIAN)
            obstacle.prediction = pred

        pred.steps = []
        velocity = getattr(obstacle, 'velocity', np.array([0.0, 0.0]))
        current_pos = obstacle.position.copy()

        for k in range(int(horizon) + 1):
            future_pos = current_pos + velocity * k * dt
            angle = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0 else 0.0
            uncertainty = 0.1 + k * 0.02  # Growing uncertainty

            pred.steps.append(PredictionStep(
                position=future_pos,
                angle=angle,
                major_radius=obstacle.radius + uncertainty,
                minor_radius=obstacle.radius + uncertainty * 0.5
            ))


def run_mpc(
    planner: Planner,
    max_steps: int = 100,
    goal_threshold: float = 1.0
) -> Dict:
    """
    Run the MPC planner until goal is reached or max steps.

    Args:
        planner: Configured Planner instance
        max_steps: Maximum number of MPC iterations
        goal_threshold: Distance threshold for goal reaching

    Returns:
        Dictionary with trajectory and statistics
    """
    config = planner.config
    goal = planner.data.goal

    ego_trajectory = []
    control_history = []
    step = 0

    LOG_INFO("Starting MPC loop...")

    while step < max_steps:
        # Get current state
        current_state = planner.get_state()
        ego_x = current_state.get('x')
        ego_y = current_state.get('y')
        ego_trajectory.append((ego_x, ego_y))

        # Check if goal reached
        dist_to_goal = np.sqrt((ego_x - goal[0])**2 + (ego_y - goal[1])**2)
        if dist_to_goal < goal_threshold:
            LOG_INFO(f"Goal reached at step {step}!")
            break

        # Propagate obstacle predictions
        propagate_obstacle_predictions(
            planner.data,
            config["planner"]["timestep"],
            config["planner"]["horizon"]
        )

        # Solve MPC
        output = planner.solve_mpc(planner.data)

        if output.success:
            control = getattr(output, 'control', None)
            if control and isinstance(control, dict) and control:
                control_history.append(control)

                # Propagate state
                planner.state = planner.state.propagate(
                    control,
                    config["planner"]["timestep"],
                    dynamics_model=planner.model_type
                )
                planner.data.state = planner.state

                # Update obstacles
                for obs in planner.data.dynamic_obstacles:
                    if hasattr(obs, 'velocity'):
                        obs.position = obs.position + obs.velocity * config["planner"]["timestep"]

            if step % 20 == 0:
                LOG_INFO(f"Step {step}: position=({ego_x:.2f}, {ego_y:.2f}), "
                        f"dist_to_goal={dist_to_goal:.2f}")
        else:
            LOG_INFO(f"Step {step}: MPC solve failed!")
            # Apply braking
            planner.state.set('v', max(0, planner.state.get('v') - 0.1))

        step += 1

    LOG_INFO(f"MPC completed after {step} steps")

    return {
        'trajectory': ego_trajectory,
        'control_history': control_history,
        'steps': step,
        'goal_reached': dist_to_goal < goal_threshold if ego_trajectory else False,
    }
