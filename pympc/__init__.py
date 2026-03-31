"""
PyMPC — minimal Model Predictive Contouring Control (MPCC) stack in Python.

Obstacle handling: ``scenario`` (sampled half-spaces) or ``linearized`` (half-spaces).

Example::

    from pympc import create_planner, run_mpc, create_default_config

    config = create_default_config("scenario")
    planner = create_planner(initial_state, ref_path, obstacles, goal, config)
    result = run_mpc(planner)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Stephen"

# =============================================================================
# Core API
# =============================================================================

from pympc.registry import (
    register_constraint_type,
    get_constraint_class,
    list_constraint_types,
    CONSTRAINT_TYPES,
)

from pympc.config import (
    create_default_config,
    load_config,
    MPCConfig,
    ConfigManager,
    get_config,
    init_config,
)

from pympc.factory import (
    MPCProblem,
    create_planner,
)

from pympc.runner import (
    propagate_obstacle_predictions,
    run_mpc,
)

# =============================================================================
# Logging
# =============================================================================

from pympc.logging import (
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_WARNING,
    LOG_ERROR,
    LOG_CRITICAL,
    PYMPC_ASSERT,
    TimeTracker,
    PROFILE_SCOPE,
    profile_scope,
    get_logger,
    setup_logging,
    LogContext,
    timed,
)

# =============================================================================
# Exceptions
# =============================================================================

from pympc.exceptions import (
    PyMPCError,
    ConfigurationError,
    ConfigNotFoundError,
    ConfigValidationError,
    SolverError,
    SolverNotInitializedError,
    SolverFailedError,
    InfeasibleProblemError,
    NumericalError,
    PlanningError,
    InvalidStateError,
    NoReferencePathError,
    InvalidReferencePathError,
    GoalNotSetError,
    ModuleError,
    ModuleNotFoundError,
    ModuleInitializationError,
    ConstraintError,
    ObjectiveError,
    DataError,
    InvalidObstacleError,
    InvalidPredictionError,
    RuntimeLimitError,
    TimeoutError,
    IterationLimitError,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Registry
    "register_constraint_type",
    "get_constraint_class",
    "list_constraint_types",
    "CONSTRAINT_TYPES",
    # Config
    "create_default_config",
    "load_config",
    "MPCConfig",
    "ConfigManager",
    "get_config",
    "init_config",
    # Factory
    "MPCProblem",
    "create_planner",
    # Runner
    "propagate_obstacle_predictions",
    "run_mpc",
    # Logging
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARN",
    "LOG_WARNING",
    "LOG_ERROR",
    "LOG_CRITICAL",
    "PYMPC_ASSERT",
    "TimeTracker",
    "PROFILE_SCOPE",
    "profile_scope",
    "get_logger",
    "setup_logging",
    "LogContext",
    "timed",
    # Exceptions
    "PyMPCError",
    "ConfigurationError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "SolverError",
    "SolverNotInitializedError",
    "SolverFailedError",
    "InfeasibleProblemError",
    "NumericalError",
    "PlanningError",
    "InvalidStateError",
    "NoReferencePathError",
    "InvalidReferencePathError",
    "GoalNotSetError",
    "ModuleError",
    "ModuleNotFoundError",
    "ModuleInitializationError",
    "ConstraintError",
    "ObjectiveError",
    "DataError",
    "InvalidObstacleError",
    "InvalidPredictionError",
    "RuntimeLimitError",
    "TimeoutError",
    "IterationLimitError",
]
