"""
Configuration helper utilities for module configuration loading.

This module provides helper classes and functions to reduce boilerplate
when loading configuration values in constraint and objective modules.
"""

from typing import Any, Dict, List, Optional, Tuple
from utils.utils import get_config_dotted


class ConfigHelper:
    """Helper class for common configuration loading patterns.

    This class provides static methods for loading common configuration
    groups like weights, obstacle parameters, and road parameters.
    """

    @staticmethod
    def load_obstacle_params(module, prefix: str = "") -> Dict[str, Any]:
        """Load common obstacle-related configuration parameters.

        Args:
            module: Module instance with get_config_value method.
            prefix: Optional prefix for config keys (e.g., "gaussian_constraints.").

        Returns:
            Dict with keys: num_discs, max_obstacles, robot_radius, disc_radius
        """
        def _get(key: str, default):
            full_key = f"{prefix}{key}" if prefix else key
            return module.get_config_value(full_key, default)

        return {
            'num_discs': int(_get("num_discs", 1)),
            'max_obstacles': int(_get("max_obstacles", 10)),
            'robot_radius': float(_get("robot.radius", 0.5)),
            'disc_radius': float(_get("disc_radius", 0.5)),
        }

    @staticmethod
    def load_weights(module, keys: List[str], defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Load weight configuration values.

        Args:
            module: Module instance with get_config_value method.
            keys: List of weight keys to load (e.g., ["goal_weight", "contour_weight"]).
            defaults: Optional dict of default values for each key.

        Returns:
            Dict mapping weight names to their float values.
        """
        if defaults is None:
            defaults = {}

        weights = {}
        for key in keys:
            full_key = f"weights.{key}"
            default_val = defaults.get(key, 0.0)
            weights[key] = float(module.get_config_value(full_key, default_val))
        return weights

    @staticmethod
    def load_road_params(module) -> Dict[str, Any]:
        """Load road-related configuration parameters.

        Args:
            module: Module instance with get_config_value method.

        Returns:
            Dict with keys: width, width_half, two_way
        """
        width = float(module.get_config_value("road.width", 7.0))
        return {
            'width': width,
            'width_half': width / 2.0,
            'two_way': bool(module.get_config_value("road.two_way", False)),
        }

    @staticmethod
    def load_contouring_params(module) -> Dict[str, Any]:
        """Load contouring-related configuration parameters.

        Args:
            module: Module instance with get_config_value method.

        Returns:
            Dict with contouring configuration values.
        """
        return {
            'num_segments': int(module.get_config_value("contouring.num_segments", 10)),
            'add_road_constraints': bool(module.get_config_value("contouring.add_road_constraints", True)),
            'dynamic_velocity_reference': bool(module.get_config_value("contouring.dynamic_velocity_reference", False)),
            'goal_reaching_contouring': bool(module.get_config_value("contouring.goal_reaching_contouring", True)),
            'three_dimensional_contouring': bool(module.get_config_value("contouring.three_dimensional_contouring", False)),
            'terminal_contouring': float(module.get_config_value("contouring.terminal_contouring", 10.0)),
            'terminal_angle': float(module.get_config_value("contouring.terminal_angle", 1.0)),
        }

    @staticmethod
    def load_gaussian_params(module) -> Dict[str, Any]:
        """Load Gaussian constraint configuration parameters.

        Args:
            module: Module instance with get_config_value method.

        Returns:
            Dict with Gaussian constraint configuration values.
        """
        return {
            'risk_level': float(module.get_config_value("gaussian_constraints.risk_level", 0.05)),
            'min_distance': float(module.get_config_value("gaussian_constraints.min_distance", 0.1)),
            'use_douglas_rachford': bool(module.get_config_value("gaussian_constraints.use_douglas_rachford", True)),
        }


class WeightLoader:
    """Helper class for loading and managing objective weights.

    This class provides a cleaner interface for loading multiple weights
    at once and accessing them by name.

    Usage:
        weights = WeightLoader(module)
        weights.load([
            ("goal_weight", 10.0),
            ("contour_weight", 1.0),
            ("lag_weight", 0.1),
        ])
        # Access weights
        goal_w = weights["goal_weight"]
        contour_w = weights.get("contour_weight", 1.0)
    """

    def __init__(self, module):
        """Initialize with module reference.

        Args:
            module: Module instance with get_config_value method.
        """
        self._module = module
        self._weights: Dict[str, float] = {}

    def load(self, weight_specs: List[Tuple[str, float]]) -> 'WeightLoader':
        """Load multiple weights from config.

        Args:
            weight_specs: List of (key, default) tuples.
                         Key is the weight name (will be prefixed with "weights.").

        Returns:
            self for method chaining.
        """
        for key, default in weight_specs:
            full_key = f"weights.{key}"
            self._weights[key] = float(self._module.get_config_value(full_key, default))
        return self

    def __getitem__(self, key: str) -> float:
        """Get weight by name."""
        return self._weights[key]

    def get(self, key: str, default: float = 0.0) -> float:
        """Get weight with default fallback."""
        return self._weights.get(key, default)

    @property
    def all(self) -> Dict[str, float]:
        """Get all loaded weights as dict."""
        return self._weights.copy()


def load_module_config(module, config_spec: Dict[str, Tuple[str, Any, type]]) -> Dict[str, Any]:
    """Load configuration values based on a specification.

    This function provides a declarative way to load configuration values
    with type conversion and defaults.

    Args:
        module: Module instance with get_config_value method.
        config_spec: Dict mapping output key to (config_key, default, type).
                    Example: {
                        "horizon": ("planner.horizon", 10, int),
                        "timestep": ("planner.timestep", 0.1, float),
                        "enabled": ("feature.enabled", True, bool),
                    }

    Returns:
        Dict with loaded and type-converted configuration values.

    Example:
        config = load_module_config(self, {
            "num_discs": ("num_discs", 1, int),
            "robot_radius": ("robot.radius", 0.5, float),
            "use_warmstart": ("solver.use_warmstart", True, bool),
        })
        self.num_discs = config["num_discs"]
        self.robot_radius = config["robot_radius"]
    """
    result = {}
    for output_key, (config_key, default, type_fn) in config_spec.items():
        raw_val = module.get_config_value(config_key, default)
        try:
            result[output_key] = type_fn(raw_val)
        except (TypeError, ValueError):
            result[output_key] = default
    return result
