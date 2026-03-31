"""
PyMPC - Model Predictive Control for Motion Planning

This is the main entry point for the MPC planner.
Supports multiple obstacle constraint types:
- scenario: Scenario-based collision avoidance (default)
- linearized: Linearized halfspace constraints
- gaussian: Gaussian/probabilistic constraints
- ellipsoid: Ellipsoid-based constraints

Usage:
    python main.py

    Or import and use programmatically:
        from pympc import create_planner, run_mpc
        # or
        from main import create_planner, run_mpc
"""

import numpy as np

# Import from pympc package for clean interface
from pympc import (
    create_default_config,
    create_planner,
    run_mpc,
    list_constraint_types,
)

from planning.types import DynamicObstacle, Prediction, PredictionType
from planning.types import generate_reference_path


def main():
    """Main entry point demonstrating the MPC planner."""
    print("=" * 60)
    print("PyMPC - Model Predictive Control for Motion Planning")
    print("=" * 60)

    # Show available constraint types
    print(f"\nAvailable constraint types: {list_constraint_types()}")

    # Select constraint type (default: scenario)
    constraint_type = "scenario"
    print(f"Using constraint type: {constraint_type}")

    # Create configuration
    config = create_default_config(constraint_type)

    # Create reference path (straight line for demo)
    start = [0.0, 0.0, 0.0]
    goal_pos = [20.0, 0.0, 0.0]
    ref_path = generate_reference_path(start, goal_pos, path_type="straight", num_points=50)
    print(f"\nReference path: {len(ref_path.x)} points")

    # Create obstacles
    obstacles = []

    # Obstacle coming towards ego
    obs1 = DynamicObstacle(
        index=0,
        position=np.array([10.0, 0.5]),
        angle=np.pi,
        radius=0.5
    )
    obs1.prediction = Prediction(PredictionType.GAUSSIAN)
    obs1.velocity = np.array([-0.3, 0.0])
    obstacles.append(obs1)

    # Obstacle moving perpendicular
    obs2 = DynamicObstacle(
        index=1,
        position=np.array([8.0, -2.0]),
        angle=np.pi / 2,
        radius=0.5
    )
    obs2.prediction = Prediction(PredictionType.GAUSSIAN)
    obs2.velocity = np.array([0.0, 0.2])
    obstacles.append(obs2)

    print(f"Created {len(obstacles)} obstacles")

    # Define goal
    goal = np.array([18.0, 0.0])
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")

    # Initial state
    initial_state = {
        'x': 0.0,
        'y': 0.0,
        'psi': 0.0,
        'v': 0.5,
        'spline': 0.0,
    }
    print(f"Initial: x={initial_state['x']}, y={initial_state['y']}, v={initial_state['v']}")

    # Create planner
    print("\nInitializing planner...")
    planner = create_planner(initial_state, ref_path, obstacles, goal, config)

    # Run MPC
    print("\nRunning MPC...")
    result = run_mpc(planner, max_steps=80)

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Steps: {result['steps']}")
    print(f"  Goal reached: {result['goal_reached']}")
    if result['trajectory']:
        final_pos = result['trajectory'][-1]
        print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
