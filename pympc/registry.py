"""
Constraint type registry for the MPC planner.

This module manages the registration of different obstacle constraint types.
"""

from typing import Dict, List, Optional, Type

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG


# Available obstacle constraint types
CONSTRAINT_TYPES: Dict[str, Type[BaseConstraint]] = {}


def register_constraint_type(name: str, constraint_class: Type[BaseConstraint]):
    """Register an obstacle constraint type.

    Args:
        name: The name for this constraint type (e.g., "scenario", "linearized").
        constraint_class: The constraint class to register.
    """
    CONSTRAINT_TYPES[name] = constraint_class


def get_constraint_class(name: str) -> Optional[Type[BaseConstraint]]:
    """Get the constraint class for a given type name.

    Args:
        name: The constraint type name.

    Returns:
        The constraint class or None if not found.
    """
    return CONSTRAINT_TYPES.get(name)


def list_constraint_types() -> List[str]:
    """List all available constraint types.

    Returns:
        List of constraint type names.
    """
    return list(CONSTRAINT_TYPES.keys())


def _register_all_constraint_types():
    """Register all available constraint types."""
    # Scenario-based (default)
    try:
        from modules.constraints.obstacle_constraint import ObstacleConstraint
        register_constraint_type("scenario", ObstacleConstraint)
    except ImportError as e:
        LOG_DEBUG(f"Could not import ObstacleConstraint: {e}")

    # Linearized constraints
    try:
        from modules.constraints.linearized_constraints import LinearizedConstraints
        register_constraint_type("linearized", LinearizedConstraints)
    except ImportError as e:
        LOG_DEBUG(f"Could not import LinearizedConstraints: {e}")


_register_all_constraint_types()
