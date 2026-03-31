"""
PyMPC Utils Package.

This package provides utility functions and classes used throughout the codebase.

Modules:
- const: Constants used across the codebase
- utils: Logging, configuration, and general utilities
- math_tools: Mathematical utilities (splines, geometry, decomposition)
- solver_utils: Solver access helpers
- visualizer: ROS-free visualization utilities

Subpackages:
- math: Basic math functions
- geometry: Geometric primitives
- splines: Spline classes
- decomposition: Decomposition algorithms

Usage:
    from utils import LOG_INFO, LOG_DEBUG, LOG_WARN, LOG_ERROR
    from utils import TimeTracker, PROFILE_SCOPE
    from utils.math_tools import TKSpline, Spline2D
    from utils.solver_utils import SolverAccessor
"""

# Core logging and utilities from utils.py
from utils.utils import (
    # Logging
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    PYMPC_ASSERT,
    logger,
    PROFILE_SCOPE,
    # Configuration
    CONFIG,
    SAVE_FOLDER,
    SAVE_FILE,
    load_yaml,
    read_config_file,
    write_to_config,
    get_config_dotted,
    # Path utilities
    get_base_path,
    save_config_path,
    model_map_path,
    write_to_yaml,
    # Display utilities
    bcolors,
    print_value,
    print_path,
    print_success,
    print_warning,
    print_header,
    # Time tracking
    TimeTracker,
    # Robot area (re-exported from planning.types)
    define_robot_area,
)

# Constants
from utils.const import (
    GAUSSIAN,
    PROBABILISTIC,
    DETERMINISTIC,
    PATH_FOLLOWING,
    LINEAR,
    OBJECTIVE,
    CONSTRAINT,
    DYNAMIC,
    WEIGHT_PARAMS,
    BISECTION_TOLERANCE,
    DEFAULT_BRAKING,
    # Groupings
    PREDICTION_TYPES,
    PLANNING_MODES,
    MODULE_TYPES,
)

__all__ = [
    # Logging
    'LOG_DEBUG',
    'LOG_INFO',
    'LOG_WARN',
    'LOG_ERROR',
    'PYMPC_ASSERT',
    'logger',
    'PROFILE_SCOPE',
    # Configuration
    'CONFIG',
    'SAVE_FOLDER',
    'SAVE_FILE',
    'load_yaml',
    'read_config_file',
    'write_to_config',
    'get_config_dotted',
    # Path utilities
    'get_base_path',
    'save_config_path',
    'model_map_path',
    'write_to_yaml',
    # Display utilities
    'bcolors',
    'print_value',
    'print_path',
    'print_success',
    'print_warning',
    'print_header',
    # Time tracking
    'TimeTracker',
    # Robot area
    'define_robot_area',
    # Constants
    'GAUSSIAN',
    'PROBABILISTIC',
    'DETERMINISTIC',
    'PATH_FOLLOWING',
    'LINEAR',
    'OBJECTIVE',
    'CONSTRAINT',
    'DYNAMIC',
    'WEIGHT_PARAMS',
    'BISECTION_TOLERANCE',
    'DEFAULT_BRAKING',
    # Constant groupings
    'PREDICTION_TYPES',
    'PLANNING_MODES',
    'MODULE_TYPES',
]
