"""
Basic math utilities.

This module provides basic mathematical functions used throughout the codebase.
"""

from utils.math_tools_impl import (
    distance,
    safe_norm,
    linspace,
    sgn,
    chi_square_quantile,
    haar_difference_without_abs,
    bisection,
)

__all__ = [
    'distance',
    'safe_norm',
    'linspace',
    'sgn',
    'chi_square_quantile',
    'haar_difference_without_abs',
    'bisection',
]
