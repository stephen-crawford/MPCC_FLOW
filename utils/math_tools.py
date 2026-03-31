"""
Math tools module for PyMPC.

This module provides mathematical utilities used throughout the codebase.
For backward compatibility, all functions and classes are re-exported from this module.

Submodules:
- utils.math: Basic math functions (distance, safe_norm, linspace, etc.)
- utils.geometry: Geometric primitives (Halfspace, Ellipsoid, Polyhedron)
- utils.splines: Spline classes (TKSpline, Spline2D, Spline3D)
- utils.decomposition: Decomposition algorithms (EllipsoidDecomp, SeedDecomp)

Usage:
    from utils.math_tools import TKSpline, Spline2D, distance
    # or from submodules:
    from utils.splines import TKSpline, Spline2D
    from utils.geometry import Halfspace, Ellipsoid
"""

# Import everything from the implementation file for full backward compatibility
from utils.math_tools_impl import *

__all__ = [
    # Basic math
    'distance',
    'safe_norm',
    'linspace',
    'sgn',
    'chi_square_quantile',
    'haar_difference_without_abs',
    'bisection',
    # Rotation
    'rotation_matrix',
    'casadi_rotation_matrix',
    # Angles
    'angular_difference',
    'angle_to_quaternion',
    'quaternion_to_angle',
    # Geometry
    'Halfspace',
    'Hyperplane',
    'LinearConstraint',
    'Ellipsoid',
    'Polyhedron',
    'Clothoid2D',
    'GeometricUtils',
    # Splines
    'TKSpline',
    'Spline',
    'SplineSegment',
    'Spline2D',
    'Spline3D',
    'BandMatrix',
    'create_numeric_spline2d',
    'create_numeric_spline3d',
    'evaluate_path_at',
    'evaluate_path_derivative_at',
    # Decomposition
    'DecompBase',
    'LineSegment',
    'SeedDecomp',
    'EllipsoidDecomp',
    'IterativeDecomp',
    # Utilities
    'Colors',
    'DataUtils',
    'DouglasRachford',
    'RandomGenerator',
    'define_robot_area',
]
