"""
Solver module for PyMPC.

This module provides the CasADi-based solver for Model Predictive Control
and performance optimization utilities.

Usage:
    from solver import CasADiSolver, PerformanceConfig

    # Default balanced mode
    solver = CasADiSolver(config)

    # Fast mode for real-time applications
    solver = CasADiSolver(config, performance="fast")

    # Safe mode for maximum accuracy
    solver = CasADiSolver(config, performance="safe")

    # Custom configuration
    perf_config = PerformanceConfig.fast()
    perf_config.enable_jit = False  # Disable JIT
    solver = CasADiSolver(config, performance=perf_config)

    # Change mode at runtime
    solver.set_performance_mode("fast")

    # Get performance statistics
    stats = solver.get_performance_stats()
"""

from solver.casadi_solver import CasADiSolver
from solver.base_solver import BaseSolver

# Import performance optimization components
try:
    from solver.performance_optimizations import (
        PerformanceConfig,
        PerformanceOptimizer,
        PerformanceLevel,
        BoundsCache,
        SplineCache,
        create_optimized_solver_options,
        enable_fast_mode,
        enable_safe_mode,
        get_global_optimizer,
        set_global_optimizer,
        timed,
    )
    HAS_PERFORMANCE = True
except ImportError:
    HAS_PERFORMANCE = False
    PerformanceConfig = None
    PerformanceOptimizer = None
    PerformanceLevel = None

__all__ = [
    "CasADiSolver",
    "BaseSolver",
    "PerformanceConfig",
    "PerformanceOptimizer",
    "PerformanceLevel",
    "HAS_PERFORMANCE",
]

if HAS_PERFORMANCE:
    __all__.extend([
        "BoundsCache",
        "SplineCache",
        "create_optimized_solver_options",
        "enable_fast_mode",
        "enable_safe_mode",
        "get_global_optimizer",
        "set_global_optimizer",
        "timed",
    ])
