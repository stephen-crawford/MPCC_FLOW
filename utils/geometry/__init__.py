"""
Geometry utilities.

This module provides geometric primitives like halfspaces, ellipsoids, and polyhedra.
"""

from utils.math_tools_impl import (
    Halfspace,
    Hyperplane,
    LinearConstraint,
    Ellipsoid,
    Polyhedron,
    Clothoid2D,
    GeometricUtils,
)

__all__ = [
    'Halfspace',
    'Hyperplane',
    'LinearConstraint',
    'Ellipsoid',
    'Polyhedron',
    'Clothoid2D',
    'GeometricUtils',
]
