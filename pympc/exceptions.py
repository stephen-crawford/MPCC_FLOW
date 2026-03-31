"""
PyMPC Exception Hierarchy.

This module defines all custom exceptions used in the PyMPC package.
Using a structured exception hierarchy enables:
- Clear error categorization
- Targeted exception handling
- Better debugging and logging
- Clean API error contracts
"""

from typing import Any, Optional


class PyMPCError(Exception):
    """Base exception for all PyMPC errors.

    All custom exceptions in PyMPC should inherit from this class.
    This allows catching all PyMPC-related errors with a single except clause.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional context.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(PyMPCError):
    """Error in configuration loading or validation."""

    pass


class ConfigNotFoundError(ConfigurationError):
    """Configuration file not found."""

    def __init__(self, config_path: str):
        super().__init__(
            f"Configuration file not found: {config_path}",
            details={"path": config_path},
        )


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""

    def __init__(self, key: str, reason: str, value: Any = None):
        details = {"key": key, "reason": reason}
        if value is not None:
            details["value"] = str(value)
        super().__init__(
            f"Invalid configuration for '{key}': {reason}",
            details=details,
        )


# =============================================================================
# Solver Errors
# =============================================================================


class SolverError(PyMPCError):
    """Base class for solver-related errors."""

    pass


class SolverNotInitializedError(SolverError):
    """Solver has not been initialized."""

    def __init__(self):
        super().__init__("Solver has not been initialized. Call setup() first.")


class SolverFailedError(SolverError):
    """Solver failed to find a solution."""

    def __init__(self, reason: str = "Unknown", iterations: Optional[int] = None):
        details = {"reason": reason}
        if iterations is not None:
            details["iterations"] = iterations
        super().__init__(
            f"Solver failed to find a solution: {reason}",
            details=details,
        )


class InfeasibleProblemError(SolverError):
    """The optimization problem is infeasible."""

    def __init__(self, violated_constraints: Optional[list] = None):
        details = {}
        if violated_constraints:
            details["violated_constraints"] = violated_constraints
        super().__init__(
            "Optimization problem is infeasible",
            details=details,
        )


class NumericalError(SolverError):
    """Numerical issues during optimization."""

    def __init__(self, description: str):
        super().__init__(
            f"Numerical error during optimization: {description}",
            details={"description": description},
        )


# =============================================================================
# Planning Errors
# =============================================================================


class PlanningError(PyMPCError):
    """Base class for planning-related errors."""

    pass


class InvalidStateError(PlanningError):
    """Invalid state provided to planner."""

    def __init__(self, state_name: str, reason: str):
        super().__init__(
            f"Invalid state '{state_name}': {reason}",
            details={"state": state_name, "reason": reason},
        )


class NoReferencePathError(PlanningError):
    """No reference path available for planning."""

    def __init__(self):
        super().__init__("No reference path available for planning")


class InvalidReferencePathError(PlanningError):
    """Reference path is invalid or malformed."""

    def __init__(self, reason: str):
        super().__init__(
            f"Invalid reference path: {reason}",
            details={"reason": reason},
        )


class GoalNotSetError(PlanningError):
    """Goal position has not been set."""

    def __init__(self):
        super().__init__("Goal position has not been set")


# =============================================================================
# Module Errors
# =============================================================================


class ModuleError(PyMPCError):
    """Base class for module-related errors."""

    pass


class ModuleNotFoundError(ModuleError):
    """Requested module not found in registry."""

    def __init__(self, module_name: str, available: Optional[list] = None):
        details = {"module": module_name}
        if available:
            details["available"] = available
        super().__init__(
            f"Module '{module_name}' not found",
            details=details,
        )


class ModuleInitializationError(ModuleError):
    """Module failed to initialize."""

    def __init__(self, module_name: str, reason: str):
        super().__init__(
            f"Failed to initialize module '{module_name}': {reason}",
            details={"module": module_name, "reason": reason},
        )


class ConstraintError(ModuleError):
    """Error in constraint evaluation or setup."""

    def __init__(self, constraint_name: str, reason: str):
        super().__init__(
            f"Constraint error in '{constraint_name}': {reason}",
            details={"constraint": constraint_name, "reason": reason},
        )


class ObjectiveError(ModuleError):
    """Error in objective evaluation or setup."""

    def __init__(self, objective_name: str, reason: str):
        super().__init__(
            f"Objective error in '{objective_name}': {reason}",
            details={"objective": objective_name, "reason": reason},
        )


# =============================================================================
# Data Errors
# =============================================================================


class DataError(PyMPCError):
    """Base class for data-related errors."""

    pass


class InvalidObstacleError(DataError):
    """Invalid obstacle data."""

    def __init__(self, obstacle_id: Any, reason: str):
        super().__init__(
            f"Invalid obstacle (id={obstacle_id}): {reason}",
            details={"obstacle_id": obstacle_id, "reason": reason},
        )


class InvalidPredictionError(DataError):
    """Invalid prediction data."""

    def __init__(self, reason: str):
        super().__init__(
            f"Invalid prediction: {reason}",
            details={"reason": reason},
        )


# =============================================================================
# Runtime Errors
# =============================================================================


class RuntimeLimitError(PyMPCError):
    """Runtime limit exceeded."""

    def __init__(self, limit_type: str, limit_value: float, actual_value: float):
        super().__init__(
            f"{limit_type} limit exceeded: {actual_value} > {limit_value}",
            details={
                "limit_type": limit_type,
                "limit": limit_value,
                "actual": actual_value,
            },
        )


class TimeoutError(RuntimeLimitError):
    """Operation timed out."""

    def __init__(self, timeout_seconds: float, elapsed_seconds: float):
        super().__init__("Timeout", timeout_seconds, elapsed_seconds)


class IterationLimitError(RuntimeLimitError):
    """Maximum iterations exceeded."""

    def __init__(self, max_iterations: int, actual_iterations: int):
        super().__init__("Iteration", float(max_iterations), float(actual_iterations))
