"""
Performance Optimization Module for PyMPC

This module provides configurable performance optimizations for the MPC solver:
1. JIT (Just-In-Time) code generation for CasADi
2. Caching for expensive operations (bounds, splines, constraints)
3. Logging level control for reduced overhead
4. Warmstart optimization strategies
5. Constraint pruning and simplification

Usage:
    from solver.performance_optimizations import PerformanceConfig, PerformanceOptimizer

    # Create optimized configuration
    perf_config = PerformanceConfig.fast()  # or .balanced() or .safe()

    # Apply to solver
    optimizer = PerformanceOptimizer(perf_config)
    optimizer.configure_solver(solver)

Reference: https://github.com/tud-amr/mpc_planner
"""

import logging
import time
import functools
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
import numpy as np

try:
    import casadi as cs
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


class PerformanceLevel(Enum):
    """Performance level presets."""
    FAST = "fast"           # Maximum speed, reduced accuracy
    BALANCED = "balanced"   # Good balance of speed and accuracy
    SAFE = "safe"           # Maximum accuracy, slower
    CUSTOM = "custom"       # User-defined settings


@dataclass
class PerformanceConfig:
    """
    Configuration for performance optimizations.

    Attributes:
        level: Performance level preset
        enable_jit: Enable JIT compilation for CasADi
        enable_caching: Enable caching for bounds, splines, etc.
        enable_logging_optimization: Reduce logging overhead
        cache_bounds: Cache dynamics model bounds lookups
        cache_splines: Cache spline evaluations
        cache_constraints: Cache constraint computations when possible
        jit_compiler: Compiler to use for JIT ('gcc', 'clang', 'ccache gcc')
        jit_flags: Compiler flags for JIT
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        ipopt_settings: IPOPT solver settings override
        warmstart_strategy: Warmstart strategy ('shift', 'interpolate', 'cache')
        constraint_pruning: Enable constraint pruning
        parallel_constraint_eval: Enable parallel constraint evaluation
    """
    level: PerformanceLevel = PerformanceLevel.BALANCED

    # JIT Compilation
    enable_jit: bool = False
    jit_compiler: str = "gcc"
    jit_flags: List[str] = field(default_factory=lambda: ["-O3", "-march=native"])

    # Caching
    enable_caching: bool = True
    cache_bounds: bool = True
    cache_splines: bool = True
    cache_constraints: bool = False  # Experimental
    cache_max_size: int = 1000

    # Logging
    enable_logging_optimization: bool = True
    log_level: str = "WARNING"

    # IPOPT Settings
    ipopt_settings: Dict[str, Any] = field(default_factory=dict)

    # Warmstart
    warmstart_strategy: str = "shift"
    warmstart_cache_size: int = 5

    # Constraint Optimization
    constraint_pruning: bool = False
    parallel_constraint_eval: bool = False
    max_constraints_per_stage: int = 20

    @classmethod
    def fast(cls) -> "PerformanceConfig":
        """Create fast performance configuration - maximum speed."""
        return cls(
            level=PerformanceLevel.FAST,
            enable_jit=True,
            jit_compiler="gcc",
            jit_flags=["-O3", "-march=native", "-ffast-math"],
            enable_caching=True,
            cache_bounds=True,
            cache_splines=True,
            cache_constraints=True,
            enable_logging_optimization=True,
            log_level="ERROR",
            ipopt_settings={
                'ipopt.max_iter': 200,
                'ipopt.tol': 5e-2,
                'ipopt.acceptable_tol': 1.0,
                'ipopt.acceptable_iter': 3,
                'ipopt.constr_viol_tol': 5e-2,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-3,
                'ipopt.warm_start_mult_bound_push': 1e-3,
                'ipopt.mu_init': 1e-2,
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
            },
            warmstart_strategy="shift",
            constraint_pruning=True,
            max_constraints_per_stage=10,
        )

    @classmethod
    def balanced(cls) -> "PerformanceConfig":
        """Create balanced performance configuration - good speed/accuracy trade-off."""
        return cls(
            level=PerformanceLevel.BALANCED,
            enable_jit=True,
            jit_compiler="gcc",
            jit_flags=["-O2"],
            enable_caching=True,
            cache_bounds=True,
            cache_splines=True,
            cache_constraints=False,
            enable_logging_optimization=True,
            log_level="WARNING",
            ipopt_settings={
                'ipopt.max_iter': 500,
                'ipopt.tol': 1e-2,
                'ipopt.acceptable_tol': 5e-1,
                'ipopt.acceptable_iter': 5,
                'ipopt.constr_viol_tol': 1e-2,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-4,
                'ipopt.warm_start_mult_bound_push': 1e-4,
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
            },
            warmstart_strategy="shift",
            constraint_pruning=False,
            max_constraints_per_stage=15,
        )

    @classmethod
    def safe(cls) -> "PerformanceConfig":
        """Create safe performance configuration - maximum accuracy."""
        return cls(
            level=PerformanceLevel.SAFE,
            enable_jit=False,
            enable_caching=True,
            cache_bounds=True,
            cache_splines=False,
            cache_constraints=False,
            enable_logging_optimization=False,
            log_level="INFO",
            ipopt_settings={
                'ipopt.max_iter': 2000,
                'ipopt.tol': 1e-4,
                'ipopt.acceptable_tol': 1e-2,
                'ipopt.acceptable_iter': 15,
                'ipopt.constr_viol_tol': 1e-4,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
            },
            warmstart_strategy="interpolate",
            constraint_pruning=False,
            max_constraints_per_stage=30,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PerformanceConfig":
        """Create from configuration dictionary."""
        perf_config = config.get("performance", {})

        # Get preset level
        level_str = perf_config.get("level", "balanced")
        if level_str == "fast":
            base = cls.fast()
        elif level_str == "safe":
            base = cls.safe()
        else:
            base = cls.balanced()

        # Override with specific settings
        for key, value in perf_config.items():
            if hasattr(base, key) and key != "level":
                setattr(base, key, value)

        return base


class BoundsCache:
    """Cache for dynamics model bounds lookups."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Tuple[float, float, Any]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, dynamics_model, var_name: str) -> Optional[Tuple[float, float, Any]]:
        """Get bounds from cache or compute and cache."""
        # Create cache key from model class and variable name
        model_id = id(dynamics_model.__class__)
        key = f"{model_id}:{var_name}"

        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1

        try:
            bounds = dynamics_model.get_bounds(var_name)

            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                # Remove first 10% of entries
                keys_to_remove = list(self._cache.keys())[:self._max_size // 10]
                for k in keys_to_remove:
                    del self._cache[k]

            self._cache[key] = bounds
            return bounds
        except Exception:
            return None

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }


class SplineCache:
    """Cache for spline evaluations."""

    def __init__(self, max_size: int = 500):
        self._cache: Dict[str, np.ndarray] = {}
        self._max_size = max_size

    def get_or_compute(self, spline, s_values: np.ndarray,
                       cache_key: str = None) -> np.ndarray:
        """Get spline evaluation from cache or compute."""
        if cache_key is None:
            # Create key from s_values hash
            cache_key = f"{id(spline)}:{hash(s_values.tobytes())}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute
        result = np.array([float(spline(s)) for s in s_values])

        # Cache if not full
        if len(self._cache) < self._max_size:
            self._cache[cache_key] = result

        return result

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


class PerformanceOptimizer:
    """
    Main optimizer class that applies performance configurations to the solver.

    Usage:
        optimizer = PerformanceOptimizer(PerformanceConfig.fast())
        optimizer.configure_solver(solver)

        # Or use context manager for scoped optimization
        with optimizer.optimized_context():
            solver.solve(...)
    """

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig.balanced()
        self._bounds_cache = BoundsCache(self.config.cache_max_size)
        self._spline_cache = SplineCache(self.config.cache_max_size // 2)
        self._original_log_levels: Dict[str, int] = {}
        self._timing_data: List[Dict[str, Any]] = []
        self._is_configured = False

    def configure_solver(self, solver) -> None:
        """Apply performance optimizations to a CasADi solver."""
        if not HAS_CASADI:
            return

        # Store reference for later use
        solver._perf_optimizer = self
        solver._bounds_cache = self._bounds_cache
        solver._spline_cache = self._spline_cache

        # Apply logging optimization
        if self.config.enable_logging_optimization:
            self._optimize_logging()

        self._is_configured = True

    def get_ipopt_options(self) -> Dict[str, Any]:
        """Get optimized IPOPT options based on configuration."""
        base_opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.nlp_scaling_method': 'gradient-based',
            'ipopt.linear_solver': 'mumps',
        }

        # Merge with config-specific settings
        base_opts.update(self.config.ipopt_settings)

        # Add JIT options if enabled
        if self.config.enable_jit and HAS_CASADI:
            base_opts['jit'] = True
            base_opts['jit_options'] = {
                'compiler': self.config.jit_compiler,
                'flags': self.config.jit_flags,
            }

        return base_opts

    def _optimize_logging(self) -> None:
        """Reduce logging overhead by adjusting log levels."""
        loggers_to_adjust = [
            "PyMPC",
            "CasADiSolver",
            "Planner",
            "integration_test",
            "modules.objectives",
            "modules.constraints",
        ]

        level = getattr(logging, self.config.log_level.upper(), logging.WARNING)

        for logger_name in loggers_to_adjust:
            logger = logging.getLogger(logger_name)
            self._original_log_levels[logger_name] = logger.level
            logger.setLevel(level)

    def restore_logging(self) -> None:
        """Restore original logging levels."""
        for logger_name, level in self._original_log_levels.items():
            logging.getLogger(logger_name).setLevel(level)
        self._original_log_levels.clear()

    def get_bounds_cached(self, dynamics_model, var_name: str) -> Tuple[float, float, Any]:
        """Get bounds with caching."""
        if self.config.cache_bounds:
            result = self._bounds_cache.get(dynamics_model, var_name)
            if result is not None:
                return result

        # Fallback to direct lookup
        return dynamics_model.get_bounds(var_name)

    def evaluate_spline_cached(self, spline, s_values: np.ndarray,
                                cache_key: str = None) -> np.ndarray:
        """Evaluate spline with caching."""
        if self.config.cache_splines:
            return self._spline_cache.get_or_compute(spline, s_values, cache_key)

        # Direct evaluation
        return np.array([float(spline(s)) for s in s_values])

    def start_timing(self, label: str) -> float:
        """Start timing a section."""
        return time.perf_counter()

    def end_timing(self, label: str, start_time: float) -> float:
        """End timing and record."""
        elapsed = time.perf_counter() - start_time
        self._timing_data.append({
            "label": label,
            "elapsed_ms": elapsed * 1000,
            "timestamp": time.time(),
        })
        return elapsed

    @property
    def timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        if not self._timing_data:
            return {}

        stats = {}
        for entry in self._timing_data:
            label = entry["label"]
            if label not in stats:
                stats[label] = {"count": 0, "total_ms": 0, "times": []}
            stats[label]["count"] += 1
            stats[label]["total_ms"] += entry["elapsed_ms"]
            stats[label]["times"].append(entry["elapsed_ms"])

        # Calculate averages
        for label, data in stats.items():
            data["avg_ms"] = data["total_ms"] / data["count"]
            data["min_ms"] = min(data["times"])
            data["max_ms"] = max(data["times"])
            del data["times"]  # Remove raw times to save memory

        return stats

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "bounds_cache": self._bounds_cache.stats,
            "spline_cache": {"size": len(self._spline_cache._cache)},
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._bounds_cache.clear()
        self._spline_cache.clear()

    def __enter__(self):
        """Context manager entry - apply optimizations."""
        if self.config.enable_logging_optimization:
            self._optimize_logging()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore state."""
        self.restore_logging()
        return False


def create_optimized_solver_options(config: Dict[str, Any] = None,
                                     level: str = "balanced") -> Dict[str, Any]:
    """
    Convenience function to create optimized IPOPT solver options.

    Args:
        config: Optional configuration dictionary
        level: Performance level ('fast', 'balanced', 'safe')

    Returns:
        Dictionary of IPOPT solver options
    """
    if level == "fast":
        perf_config = PerformanceConfig.fast()
    elif level == "safe":
        perf_config = PerformanceConfig.safe()
    else:
        perf_config = PerformanceConfig.balanced()

    if config:
        perf_config = PerformanceConfig.from_config(config)

    optimizer = PerformanceOptimizer(perf_config)
    return optimizer.get_ipopt_options()


# Decorator for timing functions
def timed(label: str = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal label
            if label is None:
                label = func.__name__

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if elapsed > 0.1:  # Only log if > 100ms
                    logging.debug(f"[PERF] {label}: {elapsed*1000:.1f}ms")
        return wrapper
    return decorator


# Global optimizer instance for convenience
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_global_optimizer() -> PerformanceOptimizer:
    """Get or create the global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(PerformanceConfig.balanced())
    return _global_optimizer


def set_global_optimizer(config: PerformanceConfig) -> PerformanceOptimizer:
    """Set the global performance optimizer configuration."""
    global _global_optimizer
    _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer


def enable_fast_mode() -> PerformanceOptimizer:
    """Enable fast performance mode globally."""
    return set_global_optimizer(PerformanceConfig.fast())


def enable_safe_mode() -> PerformanceOptimizer:
    """Enable safe performance mode globally."""
    return set_global_optimizer(PerformanceConfig.safe())
