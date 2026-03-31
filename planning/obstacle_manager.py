
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from dataclasses import dataclass

from planning.types import DynamicObstacle, PredictionType, PredictionStep, ObstacleType
from planning.dynamic_models import DynamicsModel, SecondOrderUnicycleModel, SecondOrderBicycleModel, PointMassModel

@dataclass
class ObstacleConfig:
    """Configuration for obstacle creation."""
    obstacle_id: int
    initial_position: np.ndarray
    initial_velocity: np.ndarray = None
    initial_angle: float = 0.0
    radius: float = 0.35
    dynamics_type: str = "unicycle"
    prediction_type: PredictionType = PredictionType.GAUSSIAN
    control_inputs: Optional[np.ndarray] = None
    uncertainty_params: Optional[Dict[str, float]] = None
    behavior: str = "plot_wander"  # Options: "collide", "path_wander", "plot_wander", "goal", "path_intersect"

class ObstacleManager:
    """Manages obstacles for integration tests with state integration."""
    
    def __init__(self, config: Dict[str, Any], plot_bounds: Optional[Tuple[float, float, float, float]] = None, 
                 temperature: float = 0.5):
        """Initialize obstacle manager.
        
        Args:
            config: Configuration dictionary
            plot_bounds: Optional (x_min, x_max, y_min, y_max) for bouncing behavior
            temperature: Temperature setting (0.0-1.0) controlling direction change frequency.
                        Higher temperature = more frequent direction changes. Default: 0.5
        """
        self.config = config
        self.obstacles: List[DynamicObstacle] = []
        self.obstacle_states: List[List[np.ndarray]] = []
        self.obstacle_dynamics: List[DynamicsModel] = []
        self.logger = logging.getLogger("obstacle_manager")
        
        # Plot bounds for bouncing behavior (x_min, x_max, y_min, y_max)
        self.plot_bounds = plot_bounds
        
        # Temperature setting (0.0-1.0) for direction change frequency
        # Higher temperature = more frequent direction changes
        self.temperature = float(np.clip(temperature, 0.0, 1.0))
        
        # Default uncertainty parameters
        # CRITICAL: Smaller uncertainty values allow vehicle to navigate more easily
        # while still providing probabilistic safety guarantees
        self.default_uncertainty = {
            "position_std": 0.05,      # Reduced from 0.1 to allow tighter maneuvering
            "velocity_std": 0.025,     # Reduced from 0.05
            "angle_std": 0.05,         # Reduced from 0.1
            "uncertainty_growth": 0.01  # Reduced from 0.02 to limit horizon growth
        }
        
        # Track heading change timing for each obstacle (for arbitrary heading changes)
        self.obstacle_heading_change_times = []  # Time steps until next heading change
        self.obstacle_heading_change_periods = []  # Period between heading changes
        
        # Store behavior for each obstacle
        self.obstacle_behaviors = []  # List of behavior strings per obstacle
        
        # Store behavior-specific state (e.g., path_intersect lateral offset, goal target)
        self.obstacle_behavior_state = []  # List of dicts with behavior-specific data
        
        # Store reference path and goal for behaviors that need them
        self.reference_path = None
        self.goal = None
        self.vehicle_state = None
        # Store path boundaries for path_intersect behavior
        self.left_bound = None
        self.right_bound = None
        self.left_boundary_splines = None  # (x_spline, y_spline) tuple
        self.right_boundary_splines = None  # (x_spline, y_spline) tuple
        
    def create_obstacle(self, obstacle_config: 'ObstacleConfig') -> DynamicObstacle:
        """Create a new obstacle with specified dynamics."""
        self.logger.info(f"Creating obstacle {obstacle_config.obstacle_id} with {obstacle_config.dynamics_type} dynamics")
        
        # Create obstacle
        obstacle = DynamicObstacle(
            index=obstacle_config.obstacle_id,
            position=obstacle_config.initial_position.copy(),
            angle=obstacle_config.initial_angle,
            radius=obstacle_config.radius
        )
        
        # Set prediction type
        obstacle.prediction.type = obstacle_config.prediction_type
        obstacle.prediction.steps = []
        
        # Create dynamics model
        dynamics_model = self._create_dynamics_model(obstacle_config.dynamics_type)
        self.obstacle_dynamics.append(dynamics_model)
        
        # Initialize state
        initial_state = self._create_initial_state(obstacle_config, dynamics_model)
        self.obstacle_states.append([initial_state.copy()])
        
        # Store behavior for this obstacle
        behavior = getattr(obstacle_config, 'behavior', 'plot_wander')
        self.obstacle_behaviors.append(behavior)
        
        # Store uncertainty_params on obstacle for propagate_obstacles to use
        if hasattr(obstacle_config, 'uncertainty_params') and obstacle_config.uncertainty_params:
            obstacle.uncertainty_params = obstacle_config.uncertainty_params

        # Transfer mode attributes for adaptive mode sampling
        if hasattr(obstacle_config, 'available_modes'):
            obstacle.available_modes = obstacle_config.available_modes
        if hasattr(obstacle_config, 'initial_mode'):
            obstacle.initial_mode = obstacle_config.initial_mode
        if hasattr(obstacle_config, 'current_mode'):
            obstacle.current_mode = obstacle_config.current_mode
        if hasattr(obstacle_config, 'mode_switch_probability'):
            obstacle.mode_switch_probability = obstacle_config.mode_switch_probability
        if hasattr(obstacle_config, 'mode_name'):
            obstacle.mode_name = obstacle_config.mode_name

        # Initialize heading change tracking for this obstacle
        # Temperature affects period: higher temperature = shorter period (more frequent changes)
        # Period range: [5, 15] at temp=0.0, [2, 8] at temp=1.0
        # For path_intersect behavior, use temperature=0 (longest periods, least frequent changes)
        import random
        effective_temperature = 0.0 if behavior == "path_intersect" else self.temperature
        min_period = int(5 - 3 * effective_temperature)  # 5 at temp=0, 2 at temp=1
        max_period = int(15 - 7 * effective_temperature)  # 15 at temp=0, 8 at temp=1
        heading_change_period = random.randint(max(2, min_period), max(2, max_period))
        self.obstacle_heading_change_times.append(heading_change_period)
        self.obstacle_heading_change_periods.append(heading_change_period)
        
        # Initialize behavior-specific state
        behavior_state = {}
        if behavior == "path_intersect":
            # Determine initial side based on obstacle's actual position relative to path
            # This ensures the obstacle moves toward the centerline first, then crosses
            if self.reference_path is not None and hasattr(self.reference_path, 'x_spline'):
                try:
                    # Get obstacle's initial position
                    obs_x = float(obstacle_config.initial_position[0])
                    obs_y = float(obstacle_config.initial_position[1])
                    
                    # Find closest point on path
                    s_arr = np.asarray(self.reference_path.s, dtype=float)
                    min_dist = float('inf')
                    closest_s = s_arr[0] if len(s_arr) > 0 else 0.0
                    closest_point = np.array([obs_x, obs_y])
                    
                    for s in np.linspace(s_arr[0], s_arr[-1], 50):
                        try:
                            path_x = float(self.reference_path.x_spline(s))
                            path_y = float(self.reference_path.y_spline(s))
                            dist = np.sqrt((obs_x - path_x)**2 + (obs_y - path_y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_s = s
                                closest_point = np.array([path_x, path_y])
                        except:
                            continue
                    
                    # Get path normal
                    try:
                        dx = float(self.reference_path.x_spline.derivative()(closest_s))
                        dy = float(self.reference_path.y_spline.derivative()(closest_s))
                        norm = np.sqrt(dx**2 + dy**2)
                        if norm > 1e-6:
                            normal = np.array([-dy/norm, dx/norm])  # Left normal
                        else:
                            normal = np.array([1.0, 0.0])
                    except:
                        normal = np.array([1.0, 0.0])
                    
                    # Determine which side obstacle is on: positive = left, negative = right
                    to_obstacle = np.array([obs_x, obs_y]) - closest_point
                    signed_dist = np.dot(to_obstacle, normal)
                    # Set side to opposite of current side so obstacle moves toward centerline
                    # If on left (positive), set side=1 (right) to move toward right
                    # If on right (negative), set side=-1 (left) to move toward left
                    behavior_state['side'] = 1 if signed_dist > 0 else -1
                    behavior_state['lateral_offset'] = abs(signed_dist)
                    self.logger.debug(f"Obstacle {obstacle_config.obstacle_id} path_intersect: initialized on {'LEFT' if signed_dist > 0 else 'RIGHT'} side "
                                    f"(signed_dist={signed_dist:.3f}), targeting {'RIGHT' if behavior_state['side'] == 1 else 'LEFT'} side")
                except Exception as e:
                    # Fallback to random if path calculation fails
                    self.logger.warning(f"Could not determine initial side for obstacle {obstacle_config.obstacle_id}: {e}, using random")
                    behavior_state['side'] = random.choice([-1, 1])
                    behavior_state['lateral_offset'] = random.uniform(0.5, 2.0)
            else:
                # No reference path available, use random
                behavior_state['side'] = random.choice([-1, 1])
                behavior_state['lateral_offset'] = random.uniform(0.5, 2.0)
            behavior_state['intersection_count'] = 0
        elif behavior == "goal":
            behavior_state['target_reached'] = False
        elif behavior == "path_wander":
            behavior_state['target_s'] = None  # Will be set dynamically
            behavior_state['wander_radius'] = random.uniform(1.0, 3.0)
        self.obstacle_behavior_state.append(behavior_state)
        
        # Generate prediction steps
        self._generate_prediction_steps(obstacle, dynamics_model, obstacle_config)
        
        self.obstacles.append(obstacle)
        return obstacle
        
    def _create_dynamics_model(self, dynamics_type: str) -> DynamicsModel:
        """Create dynamics model for obstacle."""
        if dynamics_type == "unicycle":
            return SecondOrderUnicycleModel()
        elif dynamics_type == "bicycle":
            return SecondOrderBicycleModel()
        elif dynamics_type == "point_mass":
            return PointMassModel()
        else:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")
            
    def _create_initial_state(self, obstacle_config: 'ObstacleConfig', dynamics_model: DynamicsModel) -> np.ndarray:
        """Create initial state vector for obstacle."""
        if obstacle_config.dynamics_type == "unicycle":
            # State: [x, y, psi, v]
            initial_velocity = obstacle_config.initial_velocity if obstacle_config.initial_velocity is not None else np.array([1.0, 0.0])
            speed = np.linalg.norm(initial_velocity)
            return np.array([
                obstacle_config.initial_position[0],
                obstacle_config.initial_position[1],
                obstacle_config.initial_angle,
                speed
            ])
        elif obstacle_config.dynamics_type == "bicycle":
            # State: [x, y, psi, v, delta, spline]
            initial_velocity = obstacle_config.initial_velocity if obstacle_config.initial_velocity is not None else np.array([1.0, 0.0])
            speed = np.linalg.norm(initial_velocity)
            return np.array([
                obstacle_config.initial_position[0],
                obstacle_config.initial_position[1],
                obstacle_config.initial_angle,
                speed,
                0.0,  # delta
                0.0   # spline
            ])
        elif obstacle_config.dynamics_type == "point_mass":
            # State: [x, y, vx, vy]
            initial_velocity = obstacle_config.initial_velocity if obstacle_config.initial_velocity is not None else np.array([1.0, 0.0])
            return np.array([
                obstacle_config.initial_position[0],
                obstacle_config.initial_position[1],
                initial_velocity[0],
                initial_velocity[1]
            ])
        else:
            raise ValueError(f"Unknown dynamics type: {obstacle_config.dynamics_type}")
            
    def _generate_prediction_steps(self, obstacle: DynamicObstacle, dynamics_model: DynamicsModel, 
                                  obstacle_config: 'ObstacleConfig'):
        """Generate prediction steps for obstacle."""
        horizon_length = self.config.get("horizon", 10)
        timestep = self.config.get("timestep", 0.1)
        
        # Get uncertainty parameters
        uncertainty_params = obstacle_config.uncertainty_params or self.default_uncertainty
        
        # Check if obstacle is static
        initial_state = self._create_initial_state(obstacle_config, dynamics_model)
        is_static = False
        if hasattr(obstacle, 'type') and obstacle.type == ObstacleType.STATIC:
            is_static = True
        elif dynamics_model.__class__.__name__ == "PointMassModel":
            if len(initial_state) >= 4:
                vx = initial_state[2]
                vy = initial_state[3]
                is_static = abs(vx) < 1e-6 and abs(vy) < 1e-6
        else:
            if len(initial_state) > 3:
                speed = initial_state[3]
                is_static = abs(speed) < 1e-6
        
        # Generate prediction steps
        for step in range(horizon_length):
            # Integrate obstacle state
            current_state = self.obstacle_states[-1][-1] if self.obstacle_states[-1] else initial_state
            
            # For static obstacles, keep position fixed
            if is_static:
                next_state = current_state.copy()
                control_input = np.zeros(dynamics_model.nu)
            else:
                # Create control input (can be specified or generated)
                if obstacle_config.control_inputs is not None:
                    control_input = obstacle_config.control_inputs[step] if step < len(obstacle_config.control_inputs) else np.zeros(dynamics_model.nu)
                else:
                    # Get obstacle index (last obstacle in list)
                    obstacle_idx = len(self.obstacles) - 1 if len(self.obstacles) > 0 else 0
                    control_input = self._generate_control_input(dynamics_model, current_state, step, obstacle_idx)
                
                # Integrate state
                next_state = self._integrate_state(dynamics_model, current_state, control_input, timestep)
            
            # Store state
            if hasattr(next_state, 'copy'):
                self.obstacle_states[-1].append(next_state.copy())
            else:
                # Handle CasADi objects
                self.obstacle_states[-1].append(next_state)
            
            # Create prediction step
            position = next_state[:2]  # x, y
            angle = next_state[2] if len(next_state) > 2 else 0.0
            
            # Add uncertainty depending on prediction type
            if obstacle.prediction.type == PredictionType.GAUSSIAN:
                uncertainty_std = uncertainty_params["position_std"] + step * uncertainty_params["uncertainty_growth"]
                major_r = obstacle.radius + uncertainty_std
                minor_r = obstacle.radius + uncertainty_std
            else:
                # Deterministic (and any non-gaussian default): no growth
                uncertainty_std = 0.0
                major_r = obstacle.radius
                minor_r = obstacle.radius
            
            prediction_step = PredictionStep(
                position=position,
                angle=angle,
                major_radius=major_r,
                minor_radius=minor_r
            )
            
            obstacle.prediction.steps.append(prediction_step)
            
    def _generate_control_input(self, dynamics_model: DynamicsModel, current_state: np.ndarray, step: int, obstacle_idx: int = None) -> np.ndarray:
        """Generate control input for obstacle based on behavior configuration.
        
        Args:
            dynamics_model: Dynamics model for the obstacle
            current_state: Current state vector [x, y, psi, v, ...]
            step: Current time step
            obstacle_idx: Obstacle index for behavior lookup
        """
        import random
        
        if obstacle_idx is None or obstacle_idx >= len(self.obstacle_behaviors):
            # Fallback to plot_wander if index invalid
            behavior = "plot_wander"
        else:
            behavior = self.obstacle_behaviors[obstacle_idx]
        
        x, y = current_state[0], current_state[1]
        psi = current_state[2] if len(current_state) > 2 else 0.0
        v = current_state[3] if len(current_state) > 3 else 1.0
        
        # Get behavior state
        behavior_state = self.obstacle_behavior_state[obstacle_idx] if obstacle_idx < len(self.obstacle_behavior_state) else {}
        
        # Generate control based on behavior
        if behavior == "collide":
            return self._generate_collide_control(dynamics_model, current_state, obstacle_idx)
        elif behavior == "path_wander":
            return self._generate_path_wander_control(dynamics_model, current_state, obstacle_idx, behavior_state)
        elif behavior == "plot_wander":
            return self._generate_plot_wander_control(dynamics_model, current_state, step, obstacle_idx)
        elif behavior == "goal":
            return self._generate_goal_control(dynamics_model, current_state, obstacle_idx, behavior_state)
        elif behavior == "path_intersect":
            return self._generate_path_intersect_control(dynamics_model, current_state, obstacle_idx, behavior_state)
        else:
            # Default to plot_wander
            return self._generate_plot_wander_control(dynamics_model, current_state, step, obstacle_idx)
    
    def _generate_collide_control(self, dynamics_model: DynamicsModel, current_state: np.ndarray, obstacle_idx: int) -> np.ndarray:
        """Generate control to try to collide with vehicle."""
        if self.vehicle_state is None:
            # Fallback to random if no vehicle state
            import random
            if dynamics_model.nu == 2:
                return np.array([0.0, random.uniform(-0.5, 0.5)])
            elif dynamics_model.nu == 3:
                return np.array([0.0, random.uniform(-0.5, 0.5), 0.0])
            return np.zeros(dynamics_model.nu)
        
        # Get vehicle position
        vehicle_pos = np.array([self.vehicle_state[0], self.vehicle_state[1]])
        obstacle_pos = np.array([current_state[0], current_state[1]])
        
        # Calculate direction to vehicle
        direction_to_vehicle = vehicle_pos - obstacle_pos
        dist = np.linalg.norm(direction_to_vehicle)
        
        if dist < 1e-6:
            # Already at vehicle, random turn
            import random
            if dynamics_model.nu == 2:
                return np.array([0.0, random.uniform(-0.5, 0.5)])
            elif dynamics_model.nu == 3:
                return np.array([0.0, random.uniform(-0.5, 0.5), 0.0])
            return np.zeros(dynamics_model.nu)
        
        # Desired heading to vehicle
        desired_heading = np.arctan2(direction_to_vehicle[1], direction_to_vehicle[0])
        current_heading = current_state[2] if len(current_state) > 2 else 0.0
        
        # Calculate heading error
        heading_error = desired_heading - current_heading
        # Normalize to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Proportional control for heading
        kp = 2.0  # Proportional gain
        angular_vel = kp * heading_error
        angular_vel = np.clip(angular_vel, -1.5, 1.5)
        
        # Accelerate towards vehicle
        accel = 0.5 if dist > 2.0 else 0.0
        
        if dynamics_model.nu == 2:
            return np.array([accel, angular_vel])
        elif dynamics_model.nu == 3:
            return np.array([accel, angular_vel, 0.0])
        return np.zeros(dynamics_model.nu)
    
    def _generate_path_wander_control(self, dynamics_model: DynamicsModel, current_state: np.ndarray, 
                                     obstacle_idx: int, behavior_state: dict) -> np.ndarray:
        """Generate control to wander around the reference path."""
        if self.reference_path is None or not hasattr(self.reference_path, 'x_spline'):
            # Fallback to plot_wander if no reference path
            return self._generate_plot_wander_control(dynamics_model, current_state, 0, obstacle_idx)
        
        import random
        x, y = current_state[0], current_state[1]
        
        # Find closest point on path
        s_arr = np.asarray(self.reference_path.s, dtype=float)
        min_dist = float('inf')
        closest_s = s_arr[0] if len(s_arr) > 0 else 0.0
        
        for s in np.linspace(s_arr[0], s_arr[-1], 50):
            try:
                path_x = float(self.reference_path.x_spline(s))
                path_y = float(self.reference_path.y_spline(s))
                dist = np.sqrt((x - path_x)**2 + (y - path_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_s = s
            except:
                continue
        
        # Get target s (wander along path)
        # Temperature affects how often we change targets (higher temp = more frequent changes)
        target_change_threshold = 0.5 + (1.0 - self.temperature) * 1.5  # 0.5-2.0m based on temperature
        if behavior_state.get('target_s') is None or min_dist < target_change_threshold:
            # Set new target
            wander_radius = behavior_state.get('wander_radius', 2.0)
            # Higher temperature = larger wander radius (more exploration)
            wander_radius = wander_radius * (0.5 + self.temperature * 0.5)  # Scale by 0.5-1.0
            target_s = closest_s + random.uniform(-wander_radius, wander_radius)
            target_s = np.clip(target_s, s_arr[0], s_arr[-1])
            behavior_state['target_s'] = target_s
        
        target_s = behavior_state['target_s']
        
        # Get path point at target_s
        try:
            target_x = float(self.reference_path.x_spline(target_s))
            target_y = float(self.reference_path.y_spline(target_s))
        except:
            # Fallback
            target_x, target_y = x, y
        
        # Calculate direction to target
        direction = np.array([target_x - x, target_y - y])
        dist = np.linalg.norm(direction)
        
        if dist < 0.5:
            # Reached target, set new one
            behavior_state['target_s'] = None
            return self._generate_path_wander_control(dynamics_model, current_state, obstacle_idx, behavior_state)
        
        desired_heading = np.arctan2(direction[1], direction[0])
        current_heading = current_state[2] if len(current_state) > 2 else 0.0
        
        heading_error = desired_heading - current_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        kp = 1.5
        angular_vel = np.clip(kp * heading_error, -1.0, 1.0)
        accel = 0.3
        
        if dynamics_model.nu == 2:
            return np.array([accel, angular_vel])
        elif dynamics_model.nu == 3:
            return np.array([accel, angular_vel, 0.0])
        return np.zeros(dynamics_model.nu)
    
    def _generate_plot_wander_control(self, dynamics_model: DynamicsModel, current_state: np.ndarray, 
                                     step: int, obstacle_idx: int) -> np.ndarray:
        """Generate control to wander randomly about the plot."""
        import random
        
        # Periodically apply stronger heading changes
        # Temperature affects frequency: higher temp = more frequent changes
        if obstacle_idx is not None and obstacle_idx < len(self.obstacle_heading_change_periods):
            period = self.obstacle_heading_change_periods[obstacle_idx]
            # Adjust period based on temperature (higher temp = shorter effective period)
            effective_period = int(period * (1.0 - self.temperature * 0.5))  # Reduce period by up to 50% at high temp
            effective_period = max(1, effective_period)  # Ensure at least 1
            apply_strong_turn = (step % effective_period == 0) if step > 0 and effective_period > 0 else False
        else:
            # Fallback: use temperature to determine period
            min_period = max(2, int(5 - 3 * self.temperature))
            max_period = max(2, int(15 - 7 * self.temperature))
            apply_strong_turn = (step % random.randint(min_period, max_period) == 0) if step > 0 else False
        
        if dynamics_model.nu == 2:
            if apply_strong_turn:
                angular_vel = random.uniform(-1.5, 1.5)
            else:
                angular_vel = random.uniform(-0.5, 0.5)
            return np.array([0.0, angular_vel])
        elif dynamics_model.nu == 3:
            if apply_strong_turn:
                angular_vel = random.uniform(-1.5, 1.5)
            else:
                angular_vel = random.uniform(-0.5, 0.5)
            return np.array([0.0, angular_vel, 0.0])
        else:
            if hasattr(dynamics_model, 'inputs') and isinstance(dynamics_model.inputs, list):
                control = np.zeros(dynamics_model.nu)
                for i, name in enumerate(dynamics_model.inputs):
                    if 'w' in name.lower() or 'omega' in name.lower() or 'angular' in name.lower():
                        if apply_strong_turn:
                            control[i] = random.uniform(-1.5, 1.5)
                        else:
                            control[i] = random.uniform(-0.5, 0.5)
                        break
                return control
            return np.zeros(dynamics_model.nu)
    
    def _generate_goal_control(self, dynamics_model: DynamicsModel, current_state: np.ndarray, 
                              obstacle_idx: int, behavior_state: dict) -> np.ndarray:
        """Generate control to reach goal or path end."""
        # Determine target
        target = None
        if self.goal is not None:
            target = np.array(self.goal[:2])  # Use goal if available
        elif self.reference_path is not None and hasattr(self.reference_path, 's'):
            # Use path end
            s_arr = np.asarray(self.reference_path.s, dtype=float)
            if len(s_arr) > 0:
                try:
                    end_s = s_arr[-1]
                    target = np.array([
                        float(self.reference_path.x_spline(end_s)),
                        float(self.reference_path.y_spline(end_s))
                    ])
                except:
                    pass
        
        if target is None:
            # Fallback to plot_wander
            return self._generate_plot_wander_control(dynamics_model, current_state, 0, obstacle_idx)
        
        obstacle_pos = np.array([current_state[0], current_state[1]])
        direction = target - obstacle_pos
        dist = np.linalg.norm(direction)
        
        if dist < 1.0:
            behavior_state['target_reached'] = True
            # Stay near goal
            import random
            if dynamics_model.nu == 2:
                return np.array([0.0, random.uniform(-0.3, 0.3)])
            elif dynamics_model.nu == 3:
                return np.array([0.0, random.uniform(-0.3, 0.3), 0.0])
            return np.zeros(dynamics_model.nu)
        
        desired_heading = np.arctan2(direction[1], direction[0])
        current_heading = current_state[2] if len(current_state) > 2 else 0.0
        
        heading_error = desired_heading - current_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        kp = 2.0
        angular_vel = np.clip(kp * heading_error, -1.5, 1.5)
        accel = 0.5 if dist > 3.0 else 0.2
        
        if dynamics_model.nu == 2:
            return np.array([accel, angular_vel])
        elif dynamics_model.nu == 3:
            return np.array([accel, angular_vel, 0.0])
        return np.zeros(dynamics_model.nu)
    
    def _generate_path_intersect_control(self, dynamics_model: DynamicsModel, current_state: np.ndarray, 
                                        obstacle_idx: int, behavior_state: dict) -> np.ndarray:
        """Generate control to move back and forth across reference path, going outside path bounds before turning."""
        if self.reference_path is None or not hasattr(self.reference_path, 'x_spline'):
            return self._generate_plot_wander_control(dynamics_model, current_state, 0, obstacle_idx)
        
        x, y = current_state[0], current_state[1]
        current_heading = current_state[2] if len(current_state) > 2 else 0.0
        
        # Find closest point on path
        s_arr = np.asarray(self.reference_path.s, dtype=float)
        min_dist = float('inf')
        closest_s = s_arr[0] if len(s_arr) > 0 else 0.0
        closest_point = np.array([x, y])
        
        for s in np.linspace(s_arr[0], s_arr[-1], 50):
            try:
                path_x = float(self.reference_path.x_spline(s))
                path_y = float(self.reference_path.y_spline(s))
                dist = np.sqrt((x - path_x)**2 + (y - path_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_s = s
                    closest_point = np.array([path_x, path_y])
            except:
                continue
        
        # Get path tangent and normal
        try:
            dx = float(self.reference_path.x_spline.derivative()(closest_s))
            dy = float(self.reference_path.y_spline.derivative()(closest_s))
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-6:
                tangent = np.array([dx/norm, dy/norm])
                normal = np.array([-dy/norm, dx/norm])  # Left normal
            else:
                normal = np.array([1.0, 0.0])
                tangent = np.array([0.0, 1.0])
        except:
            normal = np.array([1.0, 0.0])
            tangent = np.array([0.0, 1.0])
        
        side = behavior_state.get('side', 1)  # -1 for left, 1 for right
        
        # Get actual boundary points if available
        boundary_margin = 1.0  # Additional margin beyond boundary (increased from 0.5)
        target_point = None
        
        if self.left_boundary_splines is not None and self.right_boundary_splines is not None:
            # Use actual boundary splines
            try:
                left_spline_x, left_spline_y = self.left_boundary_splines
                right_spline_x, right_spline_y = self.right_boundary_splines
                
                # Clamp s to valid range
                closest_s = max(s_arr[0], min(s_arr[-1], closest_s))
                
                if side == 1:  # Right side - go to right boundary
                    boundary_x = float(right_spline_x(closest_s))
                    boundary_y = float(right_spline_y(closest_s))
                    boundary_point = np.array([boundary_x, boundary_y])
                    # Target is boundary point + margin in normal direction (further right)
                    # normal points left, so -normal points right
                    target_point = boundary_point - normal * boundary_margin
                    self.logger.debug(f"Obstacle {obstacle_idx} right side: target at boundary + {boundary_margin}m margin")
                else:  # Left side - go to left boundary
                    boundary_x = float(left_spline_x(closest_s))
                    boundary_y = float(left_spline_y(closest_s))
                    boundary_point = np.array([boundary_x, boundary_y])
                    # Target is boundary point + margin in normal direction (further left)
                    # normal points left, so +normal is further left
                    target_point = boundary_point + normal * boundary_margin
                    self.logger.debug(f"Obstacle {obstacle_idx} left side: target at boundary + {boundary_margin}m margin")
            except Exception as e:
                self.logger.warning(f"Error evaluating boundary splines for obstacle {obstacle_idx}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
        
        # Fallback: use fixed offset if boundaries not available
        if target_point is None:
            lateral_offset = behavior_state.get('lateral_offset', 3.0)  # Increased default
            target_point = closest_point + normal * side * lateral_offset
            self.logger.debug(f"Obstacle {obstacle_idx} path_intersect: Using fallback offset {lateral_offset}m (no boundary splines)")
        
        # Check if obstacle has left the path bounds (crossed the path)
        obstacle_to_path = closest_point - np.array([x, y])
        side_of_path = np.sign(np.dot(obstacle_to_path, normal))
        
        # Check if we're outside the bounds at the spline segment corresponding to obstacle's position
        # Use the spline segment that corresponds to closest_s (the segment the obstacle is at)
        is_outside_bounds = False
        if self.left_boundary_splines is not None and self.right_boundary_splines is not None:
            try:
                left_spline_x, left_spline_y = self.left_boundary_splines
                right_spline_x, right_spline_y = self.right_boundary_splines
                
                # Clamp s to valid range - this is the spline segment the obstacle corresponds to
                segment_s = max(s_arr[0], min(s_arr[-1], closest_s))
                
                # Get boundary points at this specific spline segment
                left_boundary = np.array([float(left_spline_x(segment_s)), float(left_spline_y(segment_s))])
                right_boundary = np.array([float(right_spline_x(segment_s)), float(right_spline_y(segment_s))])
                
                # Get path point at this segment
                path_point_at_segment = np.array([float(self.reference_path.x_spline(segment_s)), 
                                                   float(self.reference_path.y_spline(segment_s))])
                
                # Get normal at this segment (recompute to ensure accuracy)
                try:
                    dx_seg = float(self.reference_path.x_spline.derivative()(segment_s))
                    dy_seg = float(self.reference_path.y_spline.derivative()(segment_s))
                    norm_seg = np.sqrt(dx_seg**2 + dy_seg**2)
                    if norm_seg > 1e-6:
                        normal_at_segment = np.array([-dy_seg/norm_seg, dx_seg/norm_seg])  # Left normal
                    else:
                        normal_at_segment = normal  # Fallback to previously computed normal
                except:
                    normal_at_segment = normal  # Fallback to previously computed normal
                
                # Compute signed distances from path centerline to boundaries and obstacle
                # along the normal vector at this segment
                to_left_boundary = left_boundary - path_point_at_segment
                to_right_boundary = right_boundary - path_point_at_segment
                to_obstacle = np.array([x, y]) - path_point_at_segment
                
                # Signed distances: positive = left of path, negative = right of path
                left_bound_dist_signed = np.dot(to_left_boundary, normal_at_segment)  # Should be positive
                right_bound_dist_signed = np.dot(to_right_boundary, normal_at_segment)  # Should be negative
                obstacle_dist_signed = np.dot(to_obstacle, normal_at_segment)
                
                # Check if obstacle has fully exited the bounds at this segment
                # Use a stricter margin to ensure obstacle has clearly exited
                exit_margin = 0.5  # Obstacle must be at least 0.5m outside the boundary
                
                if side == 1:  # Right side (targeting right boundary)
                    # Obstacle must be further right than right boundary (more negative)
                    # right_boundary is at right_bound_dist_signed, obstacle needs to be clearly < right_bound_dist_signed
                    is_outside_bounds = obstacle_dist_signed < (right_bound_dist_signed - exit_margin)
                    self.logger.debug(f"Obstacle {obstacle_idx} right side at s={segment_s:.3f}: "
                                    f"obstacle_dist={obstacle_dist_signed:.3f}, right_bound={right_bound_dist_signed:.3f}, "
                                    f"threshold={right_bound_dist_signed - exit_margin:.3f}, outside={is_outside_bounds}")
                else:  # Left side (targeting left boundary)
                    # Obstacle must be further left than left boundary (more positive)
                    # left_boundary is at left_bound_dist_signed, obstacle needs to be clearly > left_bound_dist_signed
                    is_outside_bounds = obstacle_dist_signed > (left_bound_dist_signed + exit_margin)
                    self.logger.debug(f"Obstacle {obstacle_idx} left side at s={segment_s:.3f}: "
                                    f"obstacle_dist={obstacle_dist_signed:.3f}, left_bound={left_bound_dist_signed:.3f}, "
                                    f"threshold={left_bound_dist_signed + exit_margin:.3f}, outside={is_outside_bounds}")
            except Exception as e:
                self.logger.warning(f"Error checking bounds for obstacle {obstacle_idx}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                # Fallback: use simple side check
                is_outside_bounds = (side_of_path * side < 0)
        else:
            # Fallback: use simple side check if no boundaries
            if self.left_boundary_splines is None or self.right_boundary_splines is None:
                self.logger.debug(f"Obstacle {obstacle_idx} path_intersect: No boundary splines available, using fallback")
            is_outside_bounds = (side_of_path * side < 0)
        
        # Calculate distance from obstacle to path centerline for probabilistic turning
        # Use signed distance: positive = left of path, negative = right of path
        distance_from_center_signed = np.dot(obstacle_to_path, normal)
        distance_from_center = abs(distance_from_center_signed)
        
        # Get accurate distance using segment-based calculation if available
        if self.left_boundary_splines is not None and self.right_boundary_splines is not None:
            try:
                segment_s = max(s_arr[0], min(s_arr[-1], closest_s))
                path_point_at_segment = np.array([float(self.reference_path.x_spline(segment_s)), 
                                                   float(self.reference_path.y_spline(segment_s))])
                try:
                    dx_seg = float(self.reference_path.x_spline.derivative()(segment_s))
                    dy_seg = float(self.reference_path.y_spline.derivative()(segment_s))
                    norm_seg = np.sqrt(dx_seg**2 + dy_seg**2)
                    if norm_seg > 1e-6:
                        normal_at_segment = np.array([-dy_seg/norm_seg, dx_seg/norm_seg])
                        to_obstacle = np.array([x, y]) - path_point_at_segment
                        distance_from_center_signed = np.dot(to_obstacle, normal_at_segment)
                        distance_from_center = abs(distance_from_center_signed)
                except:
                    pass
            except:
                pass
        
        # Calculate turn probability based on distance from path centerline
        # The further the obstacle is from the centerline (past it), the higher the probability to turn back
        # This allows the obstacle to continue past the centerline before turning
        import random
        
        # Parameters for distance-based turn probability
        min_distance_for_turn = 2.0  # Minimum distance past centerline before considering turn (allows continuation further past)
        max_distance = 6.0  # Distance at which probability reaches maximum
        min_probability = 0.1  # Minimum probability when just past min_distance
        max_probability = 0.9  # Maximum probability at max_distance
        
        # Only consider turning if obstacle has crossed the path centerline (is on opposite side)
        has_crossed_path = (side_of_path * side < 0)
        
        # Calculate turn probability based on how far past the centerline the obstacle is
        if has_crossed_path and distance_from_center >= min_distance_for_turn:
            # Normalize distance: 0 at min_distance_for_turn, 1 at max_distance
            if distance_from_center >= max_distance:
                turn_probability = max_probability
            else:
                # Use distance beyond min_distance_for_turn
                effective_distance = distance_from_center - min_distance_for_turn
                effective_max = max_distance - min_distance_for_turn
                normalized_distance = effective_distance / effective_max if effective_max > 0 else 0.0
                # Use exponential curve for smoother transition (more gradual near min_distance, faster increase later)
                # Exponent 2.0 gives quadratic growth
                normalized_distance = min(1.0, normalized_distance)
                turn_probability = min_probability + (max_probability - min_probability) * (normalized_distance ** 2.0)
            
            # Roll dice based on probability
            turn_roll = random.random()
            should_turn = (turn_roll < turn_probability)
            
            self.logger.debug(f"Obstacle {obstacle_idx} path_intersect: distance_from_center={distance_from_center:.3f}m "
                            f"(signed={distance_from_center_signed:.3f}), min_dist={min_distance_for_turn:.3f}m, "
                            f"prob={turn_probability:.3f}, roll={turn_roll:.3f}, turn={should_turn}")
        else:
            # Not crossed yet or too close to centerline - don't turn
            should_turn = False
            if has_crossed_path:
                self.logger.debug(f"Obstacle {obstacle_idx} path_intersect: crossed but too close to centerline "
                                f"(distance={distance_from_center:.3f}m < min={min_distance_for_turn:.3f}m), continuing")
        
        # Turn around if probability check passes and obstacle has crossed the path
        if should_turn:
            # Update behavior state - switch sides
            behavior_state['side'] = -side
            behavior_state['intersection_count'] = behavior_state.get('intersection_count', 0) + 1
            
            # Reset velocity when changing direction
            if len(current_state) > 3:
                import random
                new_speed = random.uniform(0.8, 1.2)
                current_state[3] = new_speed
                self.logger.debug(f"Obstacle {obstacle_idx} path_intersect: reset velocity to {new_speed:.2f} m/s after crossing path")
            
            # Update target point for new side
            side = behavior_state['side']
            if self.left_boundary_splines is not None and self.right_boundary_splines is not None:
                try:
                    left_spline_x, left_spline_y = self.left_boundary_splines
                    right_spline_x, right_spline_y = self.right_boundary_splines
                    closest_s = max(s_arr[0], min(s_arr[-1], closest_s))
                    
                    if side == 1:  # Right side
                        boundary_x = float(right_spline_x(closest_s))
                        boundary_y = float(right_spline_y(closest_s))
                        boundary_point = np.array([boundary_x, boundary_y])
                        target_point = boundary_point - normal * boundary_margin
                    else:  # Left side
                        boundary_x = float(left_spline_x(closest_s))
                        boundary_y = float(left_spline_y(closest_s))
                        boundary_point = np.array([boundary_x, boundary_y])
                        target_point = boundary_point + normal * boundary_margin
                except:
                    lateral_offset = behavior_state.get('lateral_offset', 2.0)
                    target_point = closest_point + normal * side * lateral_offset
            else:
                lateral_offset = behavior_state.get('lateral_offset', 2.0)
                target_point = closest_point + normal * side * lateral_offset
        
        # Calculate direction to target
        direction = target_point - np.array([x, y])
        dist = np.linalg.norm(direction)
        
        if dist < 0.2:
            # Very close to target, extend target further
            if self.left_boundary_splines is not None and self.right_boundary_splines is not None:
                try:
                    left_spline_x, left_spline_y = self.left_boundary_splines
                    right_spline_x, right_spline_y = self.right_boundary_splines
                    closest_s = max(s_arr[0], min(s_arr[-1], closest_s))
                    
                    if side == 1:  # Right side
                        boundary_x = float(right_spline_x(closest_s))
                        boundary_y = float(right_spline_y(closest_s))
                        boundary_point = np.array([boundary_x, boundary_y])
                        target_point = boundary_point - normal * (boundary_margin + 1.0)
                    else:  # Left side
                        boundary_x = float(left_spline_x(closest_s))
                        boundary_y = float(left_spline_y(closest_s))
                        boundary_point = np.array([boundary_x, boundary_y])
                        target_point = boundary_point + normal * (boundary_margin + 1.0)
                except:
                    target_point = closest_point + normal * side * 3.0
            else:
                target_point = closest_point + normal * side * 3.0
            direction = target_point - np.array([x, y])
        
        desired_heading = np.arctan2(direction[1], direction[0])
        heading_error = desired_heading - current_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        kp = 2.0
        angular_vel = np.clip(kp * heading_error, -1.5, 1.5)
        accel = 0.4
        
        if dynamics_model.nu == 2:
            return np.array([accel, angular_vel])
        elif dynamics_model.nu == 3:
            return np.array([accel, angular_vel, 0.0])
        return np.zeros(dynamics_model.nu)
            
    def _integrate_state(self, dynamics_model: DynamicsModel, current_state: np.ndarray, 
                        control_input: np.ndarray, timestep: float) -> np.ndarray:
        """Integrate obstacle state using simple numeric integration."""
        try:
            # Use simple Euler integration for obstacles
            # This avoids CasADi symbolic issues
            
            if dynamics_model.__class__.__name__ == "SecondOrderUnicycleModel":
                # State: [x, y, psi, v]
                x, y, psi, v = current_state
                a, w = control_input
                
                # Simple Euler integration
                x_new = x + v * np.cos(psi) * timestep
                y_new = y + v * np.sin(psi) * timestep
                psi_new = psi + w * timestep
                v_new = v + a * timestep
                
                return np.array([x_new, y_new, psi_new, v_new])
                
            elif dynamics_model.__class__.__name__ == "SecondOrderBicycleModel":
                # State: [x, y, psi, v, delta, spline]
                x, y, psi, v, delta, spline = current_state
                a, w, slack = control_input
                
                # Simple Euler integration
                x_new = x + v * np.cos(psi) * timestep
                y_new = y + v * np.sin(psi) * timestep
                psi_new = psi + w * timestep
                v_new = v + a * timestep
                delta_new = delta + w * timestep  # Simplified
                spline_new = spline + v * timestep
                
                return np.array([x_new, y_new, psi_new, v_new, delta_new, spline_new])
                
            elif dynamics_model.__class__.__name__ == "PointMassModel":
                # State: [x, y, vx, vy]
                x, y, vx, vy = current_state
                ax, ay = control_input
                
                # Simple Euler integration
                x_new = x + vx * timestep
                y_new = y + vy * timestep
                vx_new = vx + ax * timestep
                vy_new = vy + ay * timestep
                
                return np.array([x_new, y_new, vx_new, vy_new])
                
            else:
                # Fallback: simple constant velocity
                return current_state + np.concatenate([control_input, np.zeros(len(current_state) - len(control_input))]) * timestep
                
        except Exception as e:
            self.logger.warning(f"Error integrating obstacle state: {e}")
            # Return current state as fallback
            return current_state.copy()
            
    def update_obstacle_states(self, timestep: float, vehicle_state: Optional[np.ndarray] = None, 
                              reference_path: Optional[Any] = None, goal: Optional[np.ndarray] = None,
                              data: Optional[Any] = None):
        """Update all obstacle states for one timestep with bouncing and behavior-based control.
        
        Args:
            timestep: Time step duration
            vehicle_state: Optional vehicle state vector [x, y, psi, v, ...] for behaviors that need it
            reference_path: Optional reference path for path-based behaviors
            goal: Optional goal position [x, y] or [x, y, z] for goal behavior
            data: Optional data object containing boundary information
        """
        # Update stored references for behaviors
        if vehicle_state is not None:
            self.vehicle_state = vehicle_state
        if reference_path is not None:
            self.reference_path = reference_path
        if goal is not None:
            self.goal = goal
        
        # Update boundary information if available in data
        if data is not None:
            # Try to get boundary splines first (most efficient)
            if hasattr(data, 'left_spline_x') and hasattr(data, 'left_spline_y') and data.left_spline_x is not None and data.left_spline_y is not None:
                self.left_boundary_splines = (data.left_spline_x, data.left_spline_y)
                self.logger.debug(f"ObstacleManager: Stored left boundary splines")
            if hasattr(data, 'right_spline_x') and hasattr(data, 'right_spline_y') and data.right_spline_x is not None and data.right_spline_y is not None:
                self.right_boundary_splines = (data.right_spline_x, data.right_spline_y)
                self.logger.debug(f"ObstacleManager: Stored right boundary splines")
            
            # Also store Bound objects if available
            if hasattr(data, 'left_bound') and data.left_bound is not None:
                self.left_bound = data.left_bound
            if hasattr(data, 'right_bound') and data.right_bound is not None:
                self.right_bound = data.right_bound
        
        # Debug: log plot bounds status
        if self.plot_bounds is None:
            self.logger.debug("Plot bounds not set - bouncing disabled")
        else:
            self.logger.debug(f"Plot bounds: x=[{self.plot_bounds[0]:.2f}, {self.plot_bounds[1]:.2f}], y=[{self.plot_bounds[2]:.2f}, {self.plot_bounds[3]:.2f}]")
        
        for i, (obstacle, dynamics_model) in enumerate(zip(self.obstacles, self.obstacle_dynamics)):
            if i < len(self.obstacle_states) and len(self.obstacle_states[i]) > 0:
                current_state = self.obstacle_states[i][-1]
                
                # Skip static obstacles - they should not move
                # Check if obstacle type is STATIC
                is_static_by_type = hasattr(obstacle, 'type') and obstacle.type == ObstacleType.STATIC
                
                # Check if obstacle has zero velocity (for obstacles created as DynamicObstacle with zero velocity)
                is_static_by_velocity = False
                if dynamics_model.__class__.__name__ == "PointMassModel":
                    # For point mass: state is [x, y, vx, vy]
                    if len(current_state) >= 4:
                        vx = current_state[2]
                        vy = current_state[3]
                        is_static_by_velocity = abs(vx) < 1e-6 and abs(vy) < 1e-6
                else:
                    # For unicycle/bicycle: state is [x, y, psi, v, ...]
                    if len(current_state) > 3:
                        speed = current_state[3]
                        is_static_by_velocity = abs(speed) < 1e-6
                
                # If obstacle is static, keep it in place and skip all movement logic
                if is_static_by_type or is_static_by_velocity:
                    # Keep obstacle at current position - no movement
                    # Just append the same state to maintain history
                    self.obstacle_states[i].append(current_state.copy())
                    # Ensure obstacle position stays fixed
                    obstacle.position = current_state[:2].copy()
                    if len(current_state) > 2:
                        obstacle.angle = current_state[2]
                    continue  # Skip all movement logic for static obstacles
                
                # Generate control input (with behavior-based control)
                control_input = self._generate_control_input(dynamics_model, current_state, len(self.obstacle_states[i]), i)
                
                # Limit velocity to prevent obstacles from moving faster than solver can handle
                # Maximum reasonable velocity: 3.0 m/s (conservative limit for solver stability)
                max_velocity = 3.0
                if len(current_state) > 3:
                    if current_state[3] > max_velocity:
                        current_state[3] = max_velocity
                        self.logger.debug(f"Obstacle {i}: clamped velocity to {max_velocity:.2f} m/s (solver limit)")
                
                # For path_intersect behavior, use lower limit
                if i < len(self.obstacle_behaviors) and self.obstacle_behaviors[i] == "path_intersect":
                    path_intersect_max_velocity = 1.5  # Lower limit for path_intersect
                    if len(current_state) > 3 and current_state[3] > path_intersect_max_velocity:
                        current_state[3] = path_intersect_max_velocity
                        self.logger.debug(f"Obstacle {i} path_intersect: clamped velocity to {path_intersect_max_velocity:.2f} m/s")
                
                # Integrate state
                next_state = self._integrate_state(dynamics_model, current_state, control_input, timestep)
                
                # Post-integration velocity check (in case integration increased velocity)
                if len(next_state) > 3:
                    if next_state[3] > max_velocity:
                        next_state[3] = max_velocity
                        self.logger.debug(f"Obstacle {i}: post-integration clamped velocity to {max_velocity:.2f} m/s")
                    # Also check path_intersect limit
                    if i < len(self.obstacle_behaviors) and self.obstacle_behaviors[i] == "path_intersect":
                        if next_state[3] > path_intersect_max_velocity:
                            next_state[3] = path_intersect_max_velocity
                            self.logger.debug(f"Obstacle {i} path_intersect: post-integration clamped velocity to {path_intersect_max_velocity:.2f} m/s")
                
                # Initialize bounced flag (used for periodic heading changes)
                bounced = False
                
                # Apply bouncing behavior if plot bounds are set
                if self.plot_bounds is not None:
                    x_min, x_max, y_min, y_max = self.plot_bounds
                    x, y = next_state[0], next_state[1]
                    
                    # Check x bounds and bounce
                    if x < x_min:
                        next_state[0] = x_min
                        # Reverse x velocity component
                        if dynamics_model.__class__.__name__ == "PointMassModel" and len(next_state) >= 4:
                            next_state[2] = abs(next_state[2])  # Ensure positive vx (moving right)
                        elif len(next_state) >= 3:
                            # For unicycle/bicycle: reflect heading about vertical axis
                            # If heading is pointing left, reflect it to point right
                            psi = next_state[2]
                            # Normalize psi to [0, 2*pi)
                            psi = psi % (2 * np.pi)
                            # Reflect: if moving left (cos(psi) < 0), flip to right
                            if np.cos(psi) < 0:  # Moving left
                                next_state[2] = np.pi - psi  # Reflect about vertical
                            else:
                                # Already moving right, but hit left boundary - reverse
                                next_state[2] = (np.pi - psi) % (2 * np.pi)
                        bounced = True
                        self.logger.info(f"Obstacle {i} bounced off left boundary (x={x_min:.2f}), new heading={np.degrees(next_state[2]):.1f}°")
                    elif x > x_max:
                        next_state[0] = x_max
                        # Reverse x velocity component
                        if dynamics_model.__class__.__name__ == "PointMassModel" and len(next_state) >= 4:
                            next_state[2] = -abs(next_state[2])  # Ensure negative vx (moving left)
                        elif len(next_state) >= 3:
                            # For unicycle/bicycle: reflect heading about vertical axis
                            psi = next_state[2]
                            # Normalize psi to [0, 2*pi)
                            psi = psi % (2 * np.pi)
                            # Reflect: if moving right (cos(psi) > 0), flip to left
                            if np.cos(psi) > 0:  # Moving right
                                next_state[2] = np.pi - psi  # Reflect about vertical
                            else:
                                # Already moving left, but hit right boundary - reverse
                                next_state[2] = (np.pi - psi) % (2 * np.pi)
                        bounced = True
                        self.logger.info(f"Obstacle {i} bounced off right boundary (x={x_max:.2f}), new heading={np.degrees(next_state[2]):.1f}°")
                    
                    # Check y bounds and bounce
                    if y < y_min:
                        next_state[1] = y_min
                        # Reverse y velocity component
                        if dynamics_model.__class__.__name__ == "PointMassModel" and len(next_state) >= 4:
                            next_state[3] = abs(next_state[3])  # Ensure positive vy (moving up)
                        elif len(next_state) >= 3:
                            # For unicycle/bicycle: reflect heading about horizontal axis
                            psi = next_state[2]
                            # Normalize psi to [0, 2*pi)
                            psi = psi % (2 * np.pi)
                            # Reflect: if moving down (sin(psi) < 0), flip to up
                            if np.sin(psi) < 0:  # Moving down
                                next_state[2] = -psi % (2 * np.pi)  # Reflect about horizontal
                            else:
                                # Already moving up, but hit bottom boundary - reverse
                                next_state[2] = (-psi) % (2 * np.pi)
                        bounced = True
                        self.logger.info(f"Obstacle {i} bounced off bottom boundary (y={y_min:.2f}), new heading={np.degrees(next_state[2]):.1f}°")
                    elif y > y_max:
                        next_state[1] = y_max
                        # Reverse y velocity component
                        if dynamics_model.__class__.__name__ == "PointMassModel" and len(next_state) >= 4:
                            next_state[3] = -abs(next_state[3])  # Ensure negative vy (moving down)
                        elif len(next_state) >= 3:
                            # For unicycle/bicycle: reflect heading about horizontal axis
                            psi = next_state[2]
                            # Normalize psi to [0, 2*pi)
                            psi = psi % (2 * np.pi)
                            # Reflect: if moving up (sin(psi) > 0), flip to down
                            if np.sin(psi) > 0:  # Moving up
                                next_state[2] = (-psi) % (2 * np.pi)  # Reflect about horizontal
                            else:
                                # Already moving down, but hit top boundary - reverse
                                next_state[2] = (-psi) % (2 * np.pi)
                        bounced = True
                        self.logger.info(f"Obstacle {i} bounced off top boundary (y={y_max:.2f}), new heading={np.degrees(next_state[2]):.1f}°")
                
                # Apply arbitrary heading changes periodically (less frequently to avoid overriding bouncing)
                if i < len(self.obstacle_heading_change_times):
                    self.obstacle_heading_change_times[i] -= 1
                    if self.obstacle_heading_change_times[i] <= 0 and not bounced:
                        # Time for a heading change (but don't override bouncing)
                        import random
                        # Random new heading (0 to 2*pi)
                        if len(next_state) >= 3:
                            new_heading = random.uniform(0, 2 * np.pi)
                            next_state[2] = new_heading
                            self.logger.info(f"Obstacle {i} changed heading to {np.degrees(new_heading):.1f}°")
                        # Reset timer with new random period
                        self.obstacle_heading_change_times[i] = random.randint(10, 20)  # Longer period
                
                # Store new state
                self.obstacle_states[i].append(next_state.copy())

                # Update obstacle position
                obstacle.position = next_state[:2]
                if len(next_state) > 2:
                    obstacle.angle = next_state[2]

                # Detect and update current mode based on observed behavior
                # This enables adaptive mode sampling to track mode changes
                if hasattr(obstacle, 'available_modes') and obstacle.available_modes:
                    detected_mode = self._detect_obstacle_mode(i, obstacle, dynamics_model)
                    current_mode = getattr(obstacle, 'current_mode', None)
                    # Log mode detection periodically (every 10 steps)
                    if len(self.obstacle_states[i]) % 10 == 0:
                        self.logger.info(f"Obstacle {i} mode detection: detected={detected_mode}, current={current_mode}")
                    if detected_mode and detected_mode != current_mode:
                        obstacle.current_mode = detected_mode
                        self.logger.info(f"Obstacle {i} mode changed: {current_mode} -> {detected_mode}")

    def _detect_obstacle_mode(self, obstacle_idx: int, obstacle: DynamicObstacle,
                              dynamics_model: DynamicsModel) -> Optional[str]:
        """
        Detect the current mode of an obstacle based on its recent state history.

        Modes are detected by analyzing:
        - Acceleration (speed change): constant_velocity, accelerating, decelerating
        - Angular velocity (heading change): turn_left, turn_right
        - Lateral drift: lane_change_left, lane_change_right

        Args:
            obstacle_idx: Index of the obstacle
            obstacle: The obstacle object
            dynamics_model: The dynamics model for this obstacle

        Returns:
            Detected mode string or None if detection fails
        """
        states = self.obstacle_states[obstacle_idx]
        if len(states) < 3:
            # Not enough history to detect mode
            return getattr(obstacle, 'current_mode', None)

        # Get recent states for analysis
        recent_states = states[-5:] if len(states) >= 5 else states

        # Extract velocities and headings
        if dynamics_model.__class__.__name__ == "PointMassModel":
            # State is [x, y, vx, vy]
            velocities = [np.array([s[2], s[3]]) for s in recent_states if len(s) >= 4]
            speeds = [np.linalg.norm(v) for v in velocities]
            # Compute heading from velocity
            headings = [np.arctan2(v[1], v[0]) for v in velocities if np.linalg.norm(v) > 0.01]
        else:
            # Unicycle/bicycle: state is [x, y, psi, v, ...]
            speeds = [s[3] if len(s) > 3 else 0.0 for s in recent_states]
            headings = [s[2] if len(s) > 2 else 0.0 for s in recent_states]

        if len(speeds) < 2 or len(headings) < 2:
            return getattr(obstacle, 'current_mode', None)

        # Compute acceleration (speed change rate)
        speed_changes = [speeds[i+1] - speeds[i] for i in range(len(speeds)-1)]
        avg_acceleration = np.mean(speed_changes) if speed_changes else 0.0

        # Compute angular velocity (heading change rate)
        def angle_diff(a1, a2):
            """Compute smallest angle difference."""
            diff = a2 - a1
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            return diff

        heading_changes = [angle_diff(headings[i], headings[i+1]) for i in range(len(headings)-1)]
        avg_angular_velocity = np.mean(heading_changes) if heading_changes else 0.0

        # Compute lateral drift (perpendicular to heading)
        positions = [s[:2] for s in recent_states]
        if len(positions) >= 2 and len(headings) >= 1:
            avg_heading = np.mean(headings)
            # Normal vector to heading (perpendicular)
            normal = np.array([-np.sin(avg_heading), np.cos(avg_heading)])
            # Project position change onto normal
            pos_change = positions[-1] - positions[0]
            lateral_drift = np.dot(pos_change, normal)
        else:
            lateral_drift = 0.0

        # Mode detection thresholds
        accel_threshold = 0.05  # m/s per timestep
        angular_threshold = 0.05  # rad per timestep
        lateral_threshold = 0.1  # m total drift

        available_modes = obstacle.available_modes if hasattr(obstacle, 'available_modes') else []

        # Priority-based mode detection
        # 1. Check for lane change (lateral drift dominates)
        if abs(lateral_drift) > lateral_threshold:
            if lateral_drift > 0 and "lane_change_left" in available_modes:
                return "lane_change_left"
            elif lateral_drift < 0 and "lane_change_right" in available_modes:
                return "lane_change_right"

        # 2. Check for turning (angular velocity dominates)
        if abs(avg_angular_velocity) > angular_threshold:
            if avg_angular_velocity > 0 and "turn_left" in available_modes:
                return "turn_left"
            elif avg_angular_velocity < 0 and "turn_right" in available_modes:
                return "turn_right"

        # 3. Check for acceleration/deceleration
        if abs(avg_acceleration) > accel_threshold:
            if avg_acceleration > 0 and "accelerating" in available_modes:
                return "accelerating"
            elif avg_acceleration < 0 and "decelerating" in available_modes:
                return "decelerating"

        # 4. Default to constant velocity if available
        if "constant_velocity" in available_modes:
            return "constant_velocity"

        # Fallback to current mode
        return getattr(obstacle, 'current_mode', None)

    def get_obstacle_states(self, obstacle_id: int) -> List[np.ndarray]:
        """Get state history for specific obstacle."""
        if 0 <= obstacle_id < len(self.obstacle_states):
            return self.obstacle_states[obstacle_id]
        return []
        
    def get_all_obstacle_states(self) -> List[List[np.ndarray]]:
        """Get state history for all obstacles."""
        return self.obstacle_states
        
    def get_obstacle_at_time(self, obstacle_id: int, time_step: int) -> Optional[np.ndarray]:
        """Get obstacle state at specific time step."""
        if 0 <= obstacle_id < len(self.obstacle_states):
            states = self.obstacle_states[obstacle_id]
            if 0 <= time_step < len(states):
                return states[time_step]
        return None
        
    def create_obstacles_from_config(self, obstacle_configs: List['ObstacleConfig']) -> List[DynamicObstacle]:
        """Create multiple obstacles from configuration list."""
        obstacles = []
        for config in obstacle_configs:
            obstacle = self.create_obstacle(config)
            obstacles.append(obstacle)
        return obstacles
        
    def create_random_obstacles(self, num_obstacles: int, dynamics_types: List[str], 
                               bounds: Tuple[float, float, float, float] = (0.0, 20.0, -5.0, 5.0)) -> List[DynamicObstacle]:
        """Create random obstacles for testing."""
        obstacles = []
        
        for i in range(num_obstacles):
            dynamics_type = dynamics_types[i % len(dynamics_types)]
            
            # Random initial position
            x = np.random.uniform(bounds[0], bounds[1])
            y = np.random.uniform(bounds[2], bounds[3])
            
            # Random initial velocity
            speed = np.random.uniform(0.5, 2.0)
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            
            config = ObstacleConfig(
                obstacle_id=i,
                initial_position=np.array([x, y]),
                initial_velocity=velocity,
                initial_angle=angle,
                radius=self.config.get("obstacle_radius", 0.35),
                dynamics_type=dynamics_type,
                prediction_type=PredictionType.GAUSSIAN
            )
            
            obstacle = self.create_obstacle(config)
            obstacles.append(obstacle)
            
        return obstacles
        
    def reset_obstacles(self):
        """Reset all obstacles to initial states."""
        for i, obstacle in enumerate(self.obstacles):
            if i < len(self.obstacle_states) and len(self.obstacle_states[i]) > 0:
                initial_state = self.obstacle_states[i][0]
                obstacle.position = initial_state[:2]
                if len(initial_state) > 2:
                    obstacle.angle = initial_state[2]
                    
        # Reset state histories to initial states only
        for i in range(len(self.obstacle_states)):
            if len(self.obstacle_states[i]) > 0:
                self.obstacle_states[i] = [self.obstacle_states[i][0].copy()]
                
    def get_obstacle_info(self) -> Dict[str, Any]:
        """Get information about all obstacles."""
        info = {
            "num_obstacles": len(self.obstacles),
            "obstacle_details": []
        }
        
        for i, obstacle in enumerate(self.obstacles):
            obstacle_info = {
                "id": obstacle.index,
                "position": obstacle.position.tolist(),
                "angle": obstacle.angle,
                "radius": obstacle.radius,
                "prediction_type": obstacle.prediction.type.name,
                "num_prediction_steps": len(obstacle.prediction.steps),
                "dynamics_type": self.obstacle_dynamics[i].__class__.__name__ if i < len(self.obstacle_dynamics) else "Unknown",
                "state_history_length": len(self.obstacle_states[i]) if i < len(self.obstacle_states) else 0
            }
            info["obstacle_details"].append(obstacle_info)
            
        return info


# Convenience functions for creating obstacle configurations
def create_unicycle_obstacle(obstacle_id: int, position: np.ndarray, velocity: np.ndarray = None, 
                           angle: float = 0.0, radius: float = 0.35, behavior: str = "plot_wander") -> ObstacleConfig:
    """Create unicycle obstacle configuration.
    
    Args:
        behavior: Behavior type - "collide", "path_wander", "plot_wander", "goal", or "path_intersect"
    """
    if velocity is None:
        velocity = np.array([1.0, 0.0])
    return ObstacleConfig(
        obstacle_id=obstacle_id,
        initial_position=position,
        initial_velocity=velocity,
        initial_angle=angle,
        radius=radius,
        dynamics_type="unicycle",
        prediction_type=PredictionType.DETERMINISTIC,
        behavior=behavior
    )


def create_bicycle_obstacle(obstacle_id: int, position: np.ndarray, velocity: np.ndarray = None, 
                          angle: float = 0.0, radius: float = 0.35, behavior: str = "plot_wander") -> ObstacleConfig:
    """Create bicycle obstacle configuration.
    
    Args:
        behavior: Behavior type - "collide", "path_wander", "plot_wander", "goal", or "path_intersect"
    """
    if velocity is None:
        velocity = np.array([1.0, 0.0])
    return ObstacleConfig(
        obstacle_id=obstacle_id,
        initial_position=position,
        initial_velocity=velocity,
        initial_angle=angle,
        radius=radius,
        dynamics_type="bicycle",
        prediction_type=PredictionType.DETERMINISTIC,
        behavior=behavior
    )


def create_point_mass_obstacle(obstacle_id: int, position: np.ndarray, velocity: np.ndarray = None, 
                             radius: float = 0.35, behavior: str = "plot_wander") -> ObstacleConfig:
    """Create point mass obstacle configuration.
    
    Args:
        behavior: Behavior type - "collide", "path_wander", "plot_wander", "goal", or "path_intersect"
    """
    if velocity is None:
        velocity = np.array([1.0, 0.0])
    return ObstacleConfig(
        obstacle_id=obstacle_id,
        initial_position=position,
        initial_velocity=velocity,
        initial_angle=0.0,
        radius=radius,
        dynamics_type="point_mass",
        prediction_type=PredictionType.DETERMINISTIC,
        behavior=behavior
    )

