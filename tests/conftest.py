"""
Pytest configuration and fixtures for PyMPC tests.

This module provides shared fixtures for testing:
- Configuration fixtures
- State and data fixtures
- Obstacle fixtures
- Reference path fixtures
- Planner fixtures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Create default MPC configuration."""
    from pympc import create_default_config

    return create_default_config("scenario")


@pytest.fixture
def scenario_config() -> Dict[str, Any]:
    """Create configuration for scenario constraints."""
    from pympc import create_default_config

    return create_default_config("scenario")


@pytest.fixture
def linearized_config() -> Dict[str, Any]:
    """Create configuration for linearized constraints."""
    from pympc import create_default_config

    return create_default_config("linearized")


@pytest.fixture
def mpc_config():
    """Create typed MPC configuration."""
    from pympc.config import MPCConfig

    return MPCConfig()


# =============================================================================
# State Fixtures
# =============================================================================


@pytest.fixture
def initial_state() -> Dict[str, float]:
    """Create standard initial state."""
    return {
        "x": 0.0,
        "y": 0.0,
        "psi": 0.0,
        "v": 0.5,
        "spline": 0.0,
    }


@pytest.fixture
def initial_state_dict() -> Dict[str, float]:
    """Alias for initial_state."""
    return {
        "x": 0.0,
        "y": 0.0,
        "psi": 0.0,
        "v": 0.5,
        "spline": 0.0,
    }


@pytest.fixture
def goal_position() -> np.ndarray:
    """Create standard goal position."""
    return np.array([18.0, 0.0])


@pytest.fixture
def start_position() -> List[float]:
    """Create standard start position."""
    return [0.0, 0.0, 0.0]


@pytest.fixture
def goal_pos() -> List[float]:
    """Create standard goal position as list."""
    return [20.0, 0.0, 0.0]


# =============================================================================
# Reference Path Fixtures
# =============================================================================


@pytest.fixture
def straight_path():
    """Create a straight reference path."""
    from planning.types import generate_reference_path

    start = [0.0, 0.0, 0.0]
    goal = [20.0, 0.0, 0.0]
    return generate_reference_path(start, goal, path_type="straight", num_points=50)


@pytest.fixture
def curved_path():
    """Create a curved reference path."""
    from planning.types import generate_reference_path

    start = [0.0, 0.0, 0.0]
    goal = [15.0, 10.0, np.pi / 4]
    return generate_reference_path(start, goal, path_type="curved", num_points=50)


@pytest.fixture
def reference_path(straight_path):
    """Default reference path fixture."""
    return straight_path


# =============================================================================
# Obstacle Fixtures
# =============================================================================


@pytest.fixture
def single_obstacle():
    """Create a single dynamic obstacle."""
    from planning.types import DynamicObstacle, Prediction, PredictionType

    obs = DynamicObstacle(
        index=0,
        position=np.array([10.0, 0.5]),
        angle=np.pi,
        radius=0.5,
    )
    obs.prediction = Prediction(PredictionType.GAUSSIAN)
    obs.velocity = np.array([-0.3, 0.0])
    return obs


@pytest.fixture
def two_obstacles():
    """Create two dynamic obstacles."""
    from planning.types import DynamicObstacle, Prediction, PredictionType

    obstacles = []

    # Obstacle coming towards ego
    obs1 = DynamicObstacle(
        index=0,
        position=np.array([10.0, 0.5]),
        angle=np.pi,
        radius=0.5,
    )
    obs1.prediction = Prediction(PredictionType.GAUSSIAN)
    obs1.velocity = np.array([-0.3, 0.0])
    obstacles.append(obs1)

    # Obstacle moving perpendicular
    obs2 = DynamicObstacle(
        index=1,
        position=np.array([8.0, -2.0]),
        angle=np.pi / 2,
        radius=0.5,
    )
    obs2.prediction = Prediction(PredictionType.GAUSSIAN)
    obs2.velocity = np.array([0.0, 0.2])
    obstacles.append(obs2)

    return obstacles


@pytest.fixture
def obstacles(two_obstacles):
    """Default obstacles fixture."""
    return two_obstacles


@pytest.fixture
def no_obstacles() -> List:
    """Create empty obstacle list."""
    return []


# =============================================================================
# Planner Fixtures
# =============================================================================


@pytest.fixture
def planner(initial_state, reference_path, obstacles, goal_position, default_config):
    """Create a fully configured planner."""
    from pympc import create_planner

    return create_planner(
        initial_state,
        reference_path,
        obstacles,
        goal_position,
        default_config,
    )


@pytest.fixture
def scenario_planner(initial_state, reference_path, obstacles, goal_position, scenario_config):
    """Create a planner with scenario constraints."""
    from pympc import create_planner

    return create_planner(
        initial_state,
        reference_path,
        obstacles,
        goal_position,
        scenario_config,
    )


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_trajectory() -> List[tuple]:
    """Create a sample trajectory for testing."""
    return [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.1),
        (3.0, 0.2),
        (4.0, 0.1),
        (5.0, 0.0),
    ]


@pytest.fixture
def sample_control() -> Dict[str, float]:
    """Create sample control input."""
    return {
        "a": 0.5,  # acceleration
        "delta": 0.1,  # steering angle
    }


# =============================================================================
# Temporary Files Fixtures
# =============================================================================


@pytest.fixture
def temp_config_file(tmp_path) -> Path:
    """Create a temporary configuration file."""
    import yaml

    config = {
        "planner": {
            "horizon": 15,
            "timestep": 0.2,
        },
        "obstacle_constraint_type": "scenario",
    }

    config_path = tmp_path / "test_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Marker Registrations
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "mahimahi: requires Mahimahi tools/traces"
    )
    config.addinivalue_line(
        "markers", "nimbus: requires Nimbus checkout or binary"
    )
    config.addinivalue_line(
        "markers", "optional_tooling: optional system tools (iperf3, ccp module)"
    )
    config.addinivalue_line(
        "markers", "requires_casadi: marks tests that require CasADi"
    )


# =============================================================================
# Skip Conditions
# =============================================================================


@pytest.fixture
def skip_without_casadi():
    """Skip test if CasADi is not available."""
    pytest.importorskip("casadi")
