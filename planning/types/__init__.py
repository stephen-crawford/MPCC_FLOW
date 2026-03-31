"""
Types package for PyMPC planning.

This package provides all the core types used in MPC planning.
For backward compatibility, all types are re-exported from this module.

Submodules:
- state: State and Disc classes
- trajectory: Trajectory and FixedSizeTrajectory classes
- data: Data, Problem, and Costmap classes
- obstacles: StaticObstacle, DynamicObstacle, Prediction types
- path: ReferencePath and Bound classes
- scenario: Scenario types for SMPC

Usage:
    from planning.types import State, Data, Trajectory, ReferencePath
    # or
    from planning.types.state import State
    from planning.types.data import Data
"""

# Import everything from the implementation file for full backward compatibility
# This ensures all existing code continues to work
from planning.types_impl import *

# Import ScenarioDisc from state submodule (it's a nested class in Disc)
from planning.types.state import ScenarioDisc

# Also re-export from submodules for those who prefer the new structure
# Note: These are the same classes, just accessible via different import paths

__all__ = [
    # State
    'State',
    'Disc',
    'ScenarioDisc',
    'define_robot_area',
    # Trajectory
    'Trajectory',
    'FixedSizeTrajectory',
    # Data
    'Data',
    'Costmap',
    'Problem',
    # Obstacles
    'PredictionType',
    'PredictionStep',
    'Prediction',
    'ObstacleType',
    'StaticObstacle',
    'DynamicObstacle',
    'generate_dynamic_obstacles',
    'generate_static_obstacles',
    'propagate_obstacles',
    'propagate_prediction_uncertainty',
    'propagate_prediction_uncertainty_for_obstacles',
    'get_constant_velocity_prediction',
    # Path
    'ReferencePath',
    'Bound',
    'generate_reference_path',
    'calculate_path_normals',
    'calculate_path_normals_improved',
    'calculate_spline_normals',
    'generate_road_boundaries_improved',
    'smooth_boundaries',
    'create_improved_boundaries',
    # Scenario
    'ScenarioStatus',
    'ScenarioSolveStatus',
    'ConstraintSide',
    'ScenarioBase',
    'Scenario',
    'ScenarioConstraint',
    'SupportSubsample',
    'SupportData',
    'Partition',
    # Type alias
    'trajectory_sample',
]
