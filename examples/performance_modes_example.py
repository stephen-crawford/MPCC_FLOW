#!/usr/bin/env python3
"""
Performance Modes Example for PyMPC

Demonstrates how to use the configurable performance optimization pipeline.

Usage:
    python examples/performance_modes_example.py

Performance Modes:
    - fast: Maximum speed, reduced accuracy (real-time applications)
    - balanced: Good trade-off between speed and accuracy (default)
    - safe: Maximum accuracy, slower (validation/testing)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demonstrate_performance_config():
    """Show how to create and customize performance configurations."""
    from solver.performance_optimizations import (
        PerformanceConfig,
        PerformanceOptimizer,
        PerformanceLevel,
    )

    print("=" * 60)
    print("Performance Configuration Examples")
    print("=" * 60)

    # 1. Using presets
    print("\n1. Using presets:")

    fast_config = PerformanceConfig.fast()
    print(f"   Fast mode:")
    print(f"     - Level: {fast_config.level.value}")
    print(f"     - JIT enabled: {fast_config.enable_jit}")
    print(f"     - IPOPT max_iter: {fast_config.ipopt_settings['ipopt.max_iter']}")
    print(f"     - IPOPT tolerance: {fast_config.ipopt_settings['ipopt.tol']}")

    balanced_config = PerformanceConfig.balanced()
    print(f"   Balanced mode:")
    print(f"     - Level: {balanced_config.level.value}")
    print(f"     - JIT enabled: {balanced_config.enable_jit}")
    print(f"     - IPOPT max_iter: {balanced_config.ipopt_settings['ipopt.max_iter']}")

    safe_config = PerformanceConfig.safe()
    print(f"   Safe mode:")
    print(f"     - Level: {safe_config.level.value}")
    print(f"     - JIT enabled: {safe_config.enable_jit}")
    print(f"     - IPOPT max_iter: {safe_config.ipopt_settings['ipopt.max_iter']}")

    # 2. Customizing presets
    print("\n2. Customizing a preset:")
    custom_config = PerformanceConfig.fast()
    custom_config.enable_jit = False  # Disable JIT for debugging
    custom_config.log_level = "DEBUG"  # Enable debug logging
    print(f"   Custom (fast + modifications):")
    print(f"     - JIT enabled: {custom_config.enable_jit}")
    print(f"     - Log level: {custom_config.log_level}")

    # 3. Creating from YAML config
    print("\n3. Creating from config dictionary:")
    yaml_style_config = {
        "performance": {
            "level": "balanced",
            "enable_jit": True,
            "cache_max_size": 2000,
        }
    }
    from_yaml = PerformanceConfig.from_config(yaml_style_config)
    print(f"   From config dict:")
    print(f"     - Level: {from_yaml.level.value}")
    print(f"     - Cache max size: {from_yaml.cache_max_size}")


def demonstrate_solver_integration():
    """Show how to use performance modes with CasADi solver."""
    print("\n" + "=" * 60)
    print("Solver Integration Examples")
    print("=" * 60)

    print("""
Example code for solver integration:

    from solver import CasADiSolver, PerformanceConfig

    config = {
        "planner": {
            "horizon": 10,
            "timestep": 0.1,
        }
    }

    # Option 1: Use string preset
    solver = CasADiSolver(config, performance="fast")

    # Option 2: Use default (balanced)
    solver = CasADiSolver(config)

    # Option 3: Use custom configuration
    perf_config = PerformanceConfig.fast()
    perf_config.enable_jit = False
    solver = CasADiSolver(config, performance=perf_config)

    # Option 4: Change mode at runtime
    solver.set_performance_mode("safe")

    # Option 5: Get performance statistics
    stats = solver.get_performance_stats()
    print(f"Timing: {stats['timing']}")
    print(f"Cache: {stats['cache']}")
""")


def demonstrate_global_optimizer():
    """Show how to use the global optimizer."""
    from solver.performance_optimizations import (
        enable_fast_mode,
        enable_safe_mode,
        get_global_optimizer,
    )

    print("\n" + "=" * 60)
    print("Global Optimizer Examples")
    print("=" * 60)

    # Enable fast mode globally
    enable_fast_mode()
    optimizer = get_global_optimizer()
    print(f"\nGlobal optimizer set to: {optimizer.config.level.value}")

    # Enable safe mode globally
    enable_safe_mode()
    optimizer = get_global_optimizer()
    print(f"Global optimizer changed to: {optimizer.config.level.value}")


def demonstrate_caching():
    """Show how caching works."""
    from solver.performance_optimizations import BoundsCache, SplineCache
    import numpy as np

    print("\n" + "=" * 60)
    print("Caching Examples")
    print("=" * 60)

    # Bounds cache
    print("\n1. Bounds Cache:")
    cache = BoundsCache(max_size=100)

    class MockDynamicsModel:
        def get_bounds(self, var_name):
            # Simulate expensive computation
            return (-10.0, 10.0, 1.0)

    model = MockDynamicsModel()

    # First call - cache miss
    result = cache.get(model, "velocity")
    print(f"   First call (miss): {result}")
    print(f"   Cache stats: {cache.stats}")

    # Second call - cache hit
    result = cache.get(model, "velocity")
    print(f"   Second call (hit): {result}")
    print(f"   Cache stats: {cache.stats}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("PyMPC Performance Optimization Module Demo")
    print("=" * 60)

    demonstrate_performance_config()
    demonstrate_solver_integration()
    demonstrate_global_optimizer()
    demonstrate_caching()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("""
Summary:
    - Use PerformanceConfig.fast() for real-time applications
    - Use PerformanceConfig.balanced() for general use (default)
    - Use PerformanceConfig.safe() for validation and testing

    Create solver with performance mode:
        solver = CasADiSolver(config, performance="fast")

    Or change at runtime:
        solver.set_performance_mode("safe")
""")


if __name__ == "__main__":
    main()
