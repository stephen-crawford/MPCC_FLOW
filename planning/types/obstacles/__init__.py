"""
Obstacle types - re-exported from types_impl for backward compatibility.
"""

from planning.types_impl import (
    StaticObstacle,
    DynamicObstacle,
    generate_dynamic_obstacles,
    generate_static_obstacles,
    propagate_obstacles,
    propagate_prediction_uncertainty,
    propagate_prediction_uncertainty_for_obstacles,
    get_constant_velocity_prediction,
    PredictionType,
    PredictionStep,
    Prediction,
    ObstacleType,
)

__all__ = [
    'StaticObstacle',
    'DynamicObstacle',
    'generate_dynamic_obstacles',
    'generate_static_obstacles',
    'propagate_obstacles',
    'propagate_prediction_uncertainty',
    'propagate_prediction_uncertainty_for_obstacles',
    'get_constant_velocity_prediction',
    'PredictionType',
    'PredictionStep',
    'Prediction',
    'ObstacleType',
]
