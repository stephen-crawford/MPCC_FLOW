"""
Spline utilities.

This module provides spline classes for path representation and evaluation.
- TKSpline: Numeric spline for evaluation (fast, but not symbolic)
- Spline2D: 2D symbolic spline for CasADi optimization
- Spline3D: 3D symbolic spline for CasADi optimization
"""

from utils.math_tools_impl import (
    TKSpline,
    Spline,
    SplineSegment,
    Spline2D,
    Spline3D,
    BandMatrix,
    create_numeric_spline2d,
    create_numeric_spline3d,
    evaluate_path_at,
    evaluate_path_derivative_at,
)

__all__ = [
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
]
