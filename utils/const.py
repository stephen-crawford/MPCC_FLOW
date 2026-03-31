"""
PyMPC Constants.

This module defines constants used throughout the codebase.
Constants are organized by category for clarity.
"""

from typing import Final, List

# ============================================================================
# Prediction Types
# ============================================================================
GAUSSIAN: Final[str] = "GAUSSIAN"
PROBABILISTIC: Final[str] = "PROBABILISTIC"
DETERMINISTIC: Final[str] = "DETERMINISTIC"

# ============================================================================
# Planning Modes
# ============================================================================
PATH_FOLLOWING: Final[str] = "PATH_FOLLOWING"
LINEAR: Final[str] = "LINEAR"

# ============================================================================
# Module Types
# ============================================================================
OBJECTIVE: Final[str] = "OBJECTIVE"
CONSTRAINT: Final[str] = "CONSTRAINT"

# ============================================================================
# Obstacle Types
# ============================================================================
DYNAMIC: Final[str] = "DYNAMIC"

# ============================================================================
# Solver Parameters
# ============================================================================
BISECTION_TOLERANCE: Final[float] = 1e-6

# ============================================================================
# Control Parameters
# ============================================================================
DEFAULT_BRAKING: Final[float] = -2.0

# ============================================================================
# Weight Parameter Names (for configuration)
# ============================================================================
WEIGHT_PARAMS: List[str] = []

# ============================================================================
# Module Types Enum-like Groupings
# ============================================================================
PREDICTION_TYPES = frozenset({GAUSSIAN, PROBABILISTIC, DETERMINISTIC})
PLANNING_MODES = frozenset({PATH_FOLLOWING, LINEAR})
MODULE_TYPES = frozenset({OBJECTIVE, CONSTRAINT})
