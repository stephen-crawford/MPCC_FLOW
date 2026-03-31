"""
Configuration management for PyMPC.

This module provides:
- MPCConfig: Typed configuration dataclass
- ConfigManager: Central configuration management with environment support
- create_default_config: Factory function for default configurations
- load_config: Load configuration from YAML files
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from pympc.exceptions import ConfigNotFoundError, ConfigValidationError


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class PlannerConfig:
    """Planner-specific configuration."""

    horizon: int = 20
    timestep: float = 0.1

    def validate(self) -> None:
        """Validate planner configuration."""
        if self.horizon < 1:
            raise ConfigValidationError("planner.horizon", "must be >= 1", self.horizon)
        if self.timestep <= 0:
            raise ConfigValidationError("planner.timestep", "must be > 0", self.timestep)


@dataclass
class SolverConfig:
    """Solver-specific configuration."""

    solver: str = "casadi"
    shift_previous_solution_forward: bool = True
    max_iterations: int = 100
    tolerance: float = 1e-6
    print_level: int = 0

    def validate(self) -> None:
        """Validate solver configuration."""
        valid_solvers = {"casadi", "ipopt", "qpoases"}
        if self.solver not in valid_solvers:
            raise ConfigValidationError(
                "solver.solver", f"must be one of {valid_solvers}", self.solver
            )


@dataclass
class ObstacleConstraintConfig:
    """Obstacle constraint configuration."""

    num_scenarios: int = 8
    ego_radius: float = 0.5
    obstacle_radius: float = 0.5
    safety_margin: float = 0.1
    enable_pruning: bool = True
    confidence_level: float = 0.95

    def validate(self) -> None:
        """Validate obstacle constraint configuration."""
        if self.ego_radius <= 0:
            raise ConfigValidationError("ego_radius", "must be > 0", self.ego_radius)
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ConfigValidationError(
                "confidence_level", "must be in (0, 1)", self.confidence_level
            )


@dataclass
class ContouringConfig:
    """Contouring objective/constraint configuration."""

    road_width: float = 4.0
    lag_weight: float = 1.0
    contour_weight: float = 10.0
    progress_weight: float = 0.5

    def validate(self) -> None:
        """Validate contouring configuration."""
        if self.road_width <= 0:
            raise ConfigValidationError("road_width", "must be > 0", self.road_width)


@dataclass
class GoalConfig:
    """Goal objective configuration."""

    weight: float = 5.0
    threshold: float = 0.5

    def validate(self) -> None:
        """Validate goal configuration."""
        if self.weight < 0:
            raise ConfigValidationError("goal.weight", "must be >= 0", self.weight)


@dataclass
class MPCConfig:
    """Complete MPC configuration."""

    # Core settings
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    # Constraint type selection
    obstacle_constraint_type: str = "scenario"

    # Constraint configurations
    obstacle_constraint: ObstacleConstraintConfig = field(
        default_factory=ObstacleConstraintConfig
    )
    contouring: ContouringConfig = field(default_factory=ContouringConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)

    # General settings
    max_obstacles: int = 10
    max_obstacle_distance: float = 50.0
    integrator_step: float = 0.1

    def validate(self) -> None:
        """Validate all configuration settings."""
        self.planner.validate()
        self.solver.validate()
        self.obstacle_constraint.validate()
        self.contouring.validate()
        self.goal.validate()

        valid_constraint_types = {"scenario", "linearized"}
        if self.obstacle_constraint_type not in valid_constraint_types:
            raise ConfigValidationError(
                "obstacle_constraint_type",
                f"must be one of {valid_constraint_types}",
                self.obstacle_constraint_type,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format (for backward compatibility)."""
        return {
            "planner": {
                "horizon": self.planner.horizon,
                "timestep": self.planner.timestep,
            },
            "solver": {
                "solver": self.solver.solver,
                "shift_previous_solution_forward": self.solver.shift_previous_solution_forward,
            },
            "solver_iterations": self.solver.max_iterations,
            "max_obstacles": self.max_obstacles,
            "max_obstacle_distance": self.max_obstacle_distance,
            "integrator_step": self.integrator_step,
            "obstacle_constraint_type": self.obstacle_constraint_type,
            "obstacle_constraint": {
                "num_scenarios": self.obstacle_constraint.num_scenarios,
                "ego_radius": self.obstacle_constraint.ego_radius,
                "obstacle_radius": self.obstacle_constraint.obstacle_radius,
                "safety_margin": self.obstacle_constraint.safety_margin,
                "enable_pruning": self.obstacle_constraint.enable_pruning,
            },
            "linearized_constraints": {
                "ego_radius": self.obstacle_constraint.ego_radius,
                "obstacle_radius": self.obstacle_constraint.obstacle_radius,
                "safety_margin": self.obstacle_constraint.safety_margin,
            },
            "contouring_constraints": {
                "road_width": self.contouring.road_width,
            },
            "contouring_objective": {
                "lag_weight": self.contouring.lag_weight,
                "contour_weight": self.contouring.contour_weight,
                "progress_weight": self.contouring.progress_weight,
            },
            "goal_objective": {
                "weight": self.goal.weight,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MPCConfig":
        """Create MPCConfig from dictionary."""
        planner_data = data.get("planner", {})
        solver_data = data.get("solver", {})
        obstacle_data = data.get("obstacle_constraint", {})
        contouring_data = {
            **data.get("contouring_constraints", {}),
            **data.get("contouring_objective", {}),
        }
        goal_data = data.get("goal_objective", {})

        return cls(
            planner=PlannerConfig(
                horizon=planner_data.get("horizon", 20),
                timestep=planner_data.get("timestep", 0.1),
            ),
            solver=SolverConfig(
                solver=solver_data.get("solver", "casadi"),
                shift_previous_solution_forward=solver_data.get(
                    "shift_previous_solution_forward", True
                ),
                max_iterations=data.get("solver_iterations", 100),
            ),
            obstacle_constraint_type=data.get("obstacle_constraint_type", "scenario"),
            obstacle_constraint=ObstacleConstraintConfig(
                num_scenarios=obstacle_data.get("num_scenarios", 8),
                ego_radius=obstacle_data.get("ego_radius", 0.5),
                obstacle_radius=obstacle_data.get("obstacle_radius", 0.5),
                safety_margin=obstacle_data.get("safety_margin", 0.1),
                enable_pruning=obstacle_data.get("enable_pruning", True),
                confidence_level=obstacle_data.get("confidence_level", 0.95),
            ),
            contouring=ContouringConfig(
                road_width=contouring_data.get("road_width", 4.0),
                lag_weight=contouring_data.get("lag_weight", 1.0),
                contour_weight=contouring_data.get("contour_weight", 10.0),
                progress_weight=contouring_data.get("progress_weight", 0.5),
            ),
            goal=GoalConfig(
                weight=goal_data.get("weight", 5.0),
            ),
            max_obstacles=data.get("max_obstacles", 10),
            max_obstacle_distance=data.get("max_obstacle_distance", 50.0),
            integrator_step=data.get("integrator_step", 0.1),
        )


# =============================================================================
# Configuration Manager
# =============================================================================


class ConfigManager:
    """Central configuration management with environment variable support.

    Environment variables take precedence over config files.
    Config files take precedence over defaults.

    Environment variable format: PYMPC_<SECTION>_<KEY>
    Example: PYMPC_PLANNER_HORIZON=30
    """

    ENV_PREFIX = "PYMPC"

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Optional path to YAML configuration file.
        """
        self._config: Optional[MPCConfig] = None
        self._config_path = Path(config_path) if config_path else None
        self._raw_config: Dict[str, Any] = {}

    def load(self, validate: bool = True) -> MPCConfig:
        """Load and return configuration.

        Args:
            validate: Whether to validate configuration after loading.

        Returns:
            Loaded MPCConfig instance.
        """
        # Start with defaults
        self._raw_config = create_default_config()

        # Override with config file if provided
        if self._config_path:
            self._load_from_file(self._config_path)

        # Override with environment variables
        self._load_from_env()

        # Create typed config
        self._config = MPCConfig.from_dict(self._raw_config)

        if validate:
            self._config.validate()

        return self._config

    def _load_from_file(self, path: Path) -> None:
        """Load configuration from YAML file."""
        if not path.exists():
            raise ConfigNotFoundError(str(path))

        with open(path, "r") as f:
            file_config = yaml.safe_load(f)

        if file_config:
            self._deep_update(self._raw_config, file_config)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(f"{self.ENV_PREFIX}_"):
                config_key = key[len(self.ENV_PREFIX) + 1 :].lower()
                self._set_nested_value(config_key, value)

    def _set_nested_value(self, key: str, value: str) -> None:
        """Set a nested configuration value from environment variable."""
        parts = key.split("_")
        target = self._raw_config

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Try to parse value as appropriate type
        final_key = parts[-1]
        target[final_key] = self._parse_value(value)

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse string value to appropriate Python type."""
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    @staticmethod
    def _deep_update(base: dict, update: dict) -> dict:
        """Deep merge update into base dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    @property
    def config(self) -> MPCConfig:
        """Get current configuration (loads if not already loaded)."""
        if self._config is None:
            self.load()
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dotted key path.

        Args:
            key: Dotted key path (e.g., "planner.horizon").
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        parts = key.split(".")
        value = self._raw_config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value


# =============================================================================
# Factory Functions (Backward Compatibility)
# =============================================================================


def create_default_config(constraint_type: str = "scenario") -> Dict[str, Any]:
    """Create default configuration dictionary.

    This function maintains backward compatibility with existing code.

    Args:
        constraint_type: Type of obstacle constraint to use: "scenario" or "linearized".

    Returns:
        Configuration dictionary.
    """
    return {
        "planner": {
            "horizon": 20,
            "timestep": 0.1,
        },
        "solver": {
            "solver": "casadi",
            "shift_previous_solution_forward": True,
        },
        "solver_iterations": 100,
        "max_obstacles": 10,
        "max_obstacle_distance": 50.0,
        "integrator_step": 0.1,
        "obstacle_constraint_type": constraint_type,
        "obstacle_constraint": {
            "num_scenarios": 8,
            "ego_radius": 0.5,
            "obstacle_radius": 0.5,
            "safety_margin": 0.1,
            "enable_pruning": True,
            "confidence_level": 0.95,
        },
        "linearized_constraints": {
            "ego_radius": 0.5,
            "obstacle_radius": 0.5,
            "safety_margin": 0.1,
        },
        "contouring_constraints": {
            "road_width": 4.0,
        },
        "contouring_objective": {
            "lag_weight": 1.0,
            "contour_weight": 10.0,
            "progress_weight": 0.5,
        },
        "goal_objective": {
            "weight": 5.0,
        },
    }


def load_config(path: Union[str, Path], validate: bool = True) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.
        validate: Whether to validate configuration.

    Returns:
        Configuration dictionary.
    """
    manager = ConfigManager(path)
    config = manager.load(validate=validate)
    return config.to_dict()


# Global configuration instance
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config


def init_config(path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Initialize global configuration from file.

    Args:
        path: Optional path to configuration file.

    Returns:
        Initialized ConfigManager instance.
    """
    global _global_config
    _global_config = ConfigManager(path)
    _global_config.load()
    return _global_config
