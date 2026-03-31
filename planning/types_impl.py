import os
import random
import traceback
from dataclasses import dataclass
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from utils.math_tools import TKSpline
from typing import List, Optional

from planning.dynamic_models import DynamicsModel
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_tools import Halfspace
from utils.utils import LOG_DEBUG, LOG_WARN, CONFIG


class State:
    """
    State class that manages the vehicle state and integrates with the SMPC constraint system.
    Supports different vehicle models and provides a clean interface for the solver.
    """

    def __init__(self, model_type=None):
        # Internal state storage
        self._state_dict = {}  # Dictionary for name-based access--this includes both state values for the model and its control inputs
        self._state_vector = []  # Vector for numeric operations

        # Model configuration
        self._model_type = model_type
        self._config = {}

        self.initialize(model_type)

    def initialize(self, model_type: DynamicsModel):
        LOG_DEBUG("Initializing state")
        if model_type:
            self._model_type = model_type
        else:
            return
        self._state_dict = {}
        all_vars = self._model_type.get_all_vars()
        self._state_vector = np.zeros(len(all_vars))

        for i, var in enumerate(all_vars):
            self._state_dict[var] = 0.0

        LOG_DEBUG(f"Initialized state with variables: {all_vars}")

    def get(self, var_name):
        """Get a state variable by name."""
        # First check if it's in the state dictionary
        if var_name in self._state_dict:
            return self._state_dict[var_name]
        # Not found
        LOG_DEBUG(f"Variable {var_name} not found in State, returning 0.0")
        return 0.0

    def set(self, var_name, value):
        self._state_dict[var_name] = value

        # Only try to update the vector if value is numeric
        if isinstance(value, (float, int)):
            var_idx = self._get_index(var_name)
            if var_idx is not None and 0 <= var_idx < len(self._state_vector):
                self._state_vector[var_idx] = value

    def _get_index(self, var_name):
        """Get the index of a variable in the state vector."""
        # First check if it's in the state dictionary (ordered)
        if var_name in self._state_dict:
            # Find its position in the ordered keys
            keys = list(self._state_dict.keys())
            if var_name in keys:
                return keys.index(var_name)

        # Not found
        return None

    def has(self, key):
        """Check if the state has a specific variable."""
        return key in self._state_dict

    def get_position(self):
        """Get the position as a tuple (x, y)."""
        return self.get("x"), self.get("y")

    def get_full_state(self):
        """Return the full state vector."""
        return self._state_vector.copy()

    def set_full_state(self, state_vector):
        """Set the full state vector."""
        if len(state_vector) == len(self._state_vector):
            self._state_vector = np.array(state_vector)

            # Also update the dictionary
            state_vars = list(self._state_dict.keys())
            for i, var in enumerate(state_vars):
                if i < len(state_vector):
                    self._state_dict[var] = state_vector[i]
        else:
            LOG_DEBUG(
                f"Error: State vector length mismatch. Expected {len(self._state_vector)}, got {len(state_vector)}")

    def update_from_dict(self, state_dict):
        """Update state from a dictionary."""
        for var_name, value in state_dict.items():
            self.set(var_name, value)

    def to_dict(self):
        """Convert state to dictionary for debugging."""
        return self._state_dict.copy()

    def reset(self):
        """Reset the state to zeros."""
        for key in self._state_dict:
            self._state_dict[key] = 0.0
        self._state_vector = np.zeros_like(self._state_vector)

    def get_state_dict(self):
        return self._state_dict.copy()

    def get_all_vars(self):
        """Get list of all variable names in this state."""
        return list(self._state_dict.keys())

    def get_model_type(self):
        return self._model_type

    def __str__(self):
        """Return a readable string representation of the current state."""
        lines = [f"State: ({self._model_type}):"]
        for var, value in self._state_dict.items():
            lines.append(f" {var}: {value}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        state = State()
        state.initialize(self._model_type)
        state._state_dict = self._state_dict.copy()
        state._config = self._config.copy()
        state._state_vector = self._state_vector.copy()
        return state

    def propagate(self, control_dict, timestep, dynamics_model=None, parameter_manager=None):
        """
        Propagate state forward using the dynamics model's discrete_dynamics method.
        
        CRITICAL: This method MUST use the dynamics model's discrete_dynamics to properly
        handle RK4 integration and model-specific updates (e.g., spline progression for contouring).
        
        Reference: https://github.com/tud-amr/mpc_planner - state propagation uses discrete_dynamics.
        
        Args:
            control_dict: Dictionary of control inputs (e.g., {'a': 0.5, 'w': 0.1})
            timestep: Time step for integration
            dynamics_model: Optional dynamics model instance. If None, uses self._model_type.
            parameter_manager: Optional parameter manager for path parameters (needed for contouring).
        
        Returns:
            New State object with propagated values
        """
        if self._model_type is None and dynamics_model is None:
            LOG_WARN("State.propagate: No dynamics model available, returning copy of current state")
            return self.copy()
        
        # Use provided dynamics_model or try to get from _model_type
        model = dynamics_model if dynamics_model is not None else self._model_type
        
        if model is None:
            LOG_WARN("State.propagate: No dynamics model available, returning copy of current state")
            return self.copy()
        
        # Create new state for result
        new_state = self.copy()
        
        # Get current state values as a vector
        state_vars = model.get_dependent_vars()
        x_current = []
        for var in state_vars:
            x_current.append(self.get(var))
        x_current = np.array(x_current, dtype=float)
        
        # Get control inputs in order
        input_vars = model.get_inputs()
        u_current = []
        for var in input_vars:
            u_current.append(control_dict.get(var, 0.0))
        u_current = np.array(u_current, dtype=float)
        
        # CRITICAL: Use dynamics model's discrete_dynamics method for proper state propagation
        # This ensures RK4 integration and model-specific updates (e.g., spline progression)
        # Reference: test files use vehicle.discrete_dynamics() and numeric_rk4()
        try:
            import casadi as cd
            from planning.dynamic_models import numeric_rk4
            
            # Construct z vector: [u, x] = [control inputs, state variables]
            z_k = np.concatenate([u_current, x_current])
            # CRITICAL: For numeric evaluation, use CasADi DM (not MX)
            # This ensures discrete_dynamics uses numeric RK4 path instead of symbolic
            z_k = cd.DM(z_k)  # Use DM for numeric evaluation
            
            # Load state into model
            model.load(z_k)
            
            # Get parameter getter for discrete_dynamics
            # If parameter_manager is provided, use it; otherwise create a simple getter
            if parameter_manager is not None:
                def param_getter(key):
                    try:
                        if hasattr(parameter_manager, "get_all"):
                            params_stage = parameter_manager.get_all(0)
                            if key in params_stage:
                                return params_stage[key]
                    except Exception:
                        pass
                    try:
                        # Some parameter managers may expose direct get(stage, key)
                        return parameter_manager.get(0, key)  # type: ignore[attr-defined]
                    except Exception:
                        return None
            else:
                def param_getter(key):
                    return None
            
            # Create parameter wrapper for discrete_dynamics
            class ParamWrapper:
                def __init__(self, p_getter):
                    self.p_getter = p_getter
                def get(self, key, default=None):
                    val = self.p_getter(key)
                    return val if val is not None else default
                def get_p(self):
                    # Return self for compatibility with discrete_dynamics signature
                    return self
            
            params = ParamWrapper(param_getter)
            
            # Call discrete_dynamics to get next state (symbolic)
            # discrete_dynamics performs RK4 integration and calls model_discrete_dynamics for spline updates
            LOG_DEBUG(f"[PROPAGATE] Calling discrete_dynamics: z_k shape={z_k.shape if hasattr(z_k, 'shape') else len(z_k)}, timestep={timestep}")
            next_state_symbolic = model.discrete_dynamics(z_k, params, timestep)
            LOG_DEBUG(f"[PROPAGATE] discrete_dynamics returned: type={type(next_state_symbolic)}, shape={next_state_symbolic.shape if hasattr(next_state_symbolic, 'shape') else 'N/A'}")
            
            # Evaluate numerically using numeric_rk4
            LOG_DEBUG(f"[PROPAGATE] Calling numeric_rk4 to evaluate symbolic result")
            next_state = numeric_rk4(next_state_symbolic, model, params, timestep)
            LOG_DEBUG(f"[PROPAGATE] numeric_rk4 returned: type={type(next_state)}, shape={next_state.shape if hasattr(next_state, 'shape') else len(next_state) if isinstance(next_state, (list, np.ndarray)) else 'N/A'}")
            
            # Extract numeric values from CasADi DM or numpy array
            if isinstance(next_state, cd.DM):
                next_state = np.array(next_state).flatten()
            elif isinstance(next_state, np.ndarray):
                next_state = next_state.flatten()
            else:
                # Fallback: try to convert to numpy
                try:
                    next_state = np.array([float(v) for v in next_state]).flatten()
                except:
                    LOG_WARN(f"State.propagate: Could not convert next_state to numpy array, using Euler fallback")
                    # Fallback to Euler integration
                    v = x_current[3] if len(x_current) > 3 else 0.0
                    psi = x_current[2] if len(x_current) > 2 else 0.0
                    a = u_current[0] if len(u_current) > 0 else 0.0
                    w = u_current[1] if len(u_current) > 1 else 0.0
                    
                    x_next = x_current.copy()
                    x_next[0] += v * np.cos(psi) * timestep
                    x_next[1] += v * np.sin(psi) * timestep
                    x_next[2] += w * timestep
                    x_next[3] += a * timestep
                    if len(x_next) > 4:
                        x_next[4] += v * timestep
                    next_state = x_next
            
            # Update new_state with propagated values
            for i, var in enumerate(state_vars):
                if i < len(next_state):
                    new_state.set(var, float(next_state[i]))
                else:
                    LOG_WARN(f"State.propagate: next_state has {len(next_state)} elements, but need {len(state_vars)}. Missing {var}.")
                    
        except Exception as e:
            LOG_WARN(f"State.propagate: Error using discrete_dynamics ({e}), falling back to Euler integration")
            import traceback
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to Euler integration
            try:
                v = x_current[3] if len(x_current) > 3 else 0.0
                psi = x_current[2] if len(x_current) > 2 else 0.0
                a = u_current[0] if len(u_current) > 0 else 0.0
                w = u_current[1] if len(u_current) > 1 else 0.0
                
                x_next = x_current.copy()
                x_next[0] += v * np.cos(psi) * timestep
                x_next[1] += v * np.sin(psi) * timestep
                x_next[2] += w * timestep
                x_next[3] += a * timestep
                if len(x_next) > 4:
                    x_next[4] += v * timestep
                
                for i, var in enumerate(state_vars):
                    if i < len(x_next):
                        new_state.set(var, float(x_next[i]))
            except Exception as e2:
                LOG_WARN(f"State.propagate: Error in Euler fallback ({e2}), returning copy of current state")
                return self.copy()
        
        return new_state


class Disc:
    def __init__(self, offset: float, radius: float):
        self.offset = offset
        self.radius = radius

    def get_position(self, robot_position: np.ndarray, angle: float) -> np.ndarray:
        """Returns the disc's position relative to the robot's position and orientation angle."""
        return robot_position + np.array([
            self.offset * np.cos(angle),
            self.offset * np.sin(angle)
        ])

    def to_robot_center(self, disc_position: np.ndarray, angle: float) -> np.ndarray:
        """Returns the robot's center position from the disc's position and orientation angle."""
        return disc_position - np.array([
            self.offset * np.cos(angle),
            self.offset * np.sin(angle)
        ])

    class ScenarioDisc:
        def __init__(self, offset: float, radius: float):
            self.offset = offset
            self.radius = radius

        def get_position(self, robot_position: np.ndarray, angle: float) -> np.ndarray:
            """Returns the disc's position relative to the robot's position and orientation angle."""
            return robot_position + np.array([
                self.offset * np.cos(angle),
                self.offset * np.sin(angle)
            ])

        def to_robot_center(self, disc_position: np.ndarray, angle: float) -> np.ndarray:
            """Returns the robot's center position from the disc's position and orientation angle."""
            return disc_position - np.array([
                self.offset * np.cos(angle),
                self.offset * np.sin(angle)
            ])


class PredictionType(Enum):
    DETERMINISTIC = 0
    GAUSSIAN = 1
    NONGAUSSIAN = 2
    NONE = 3


class ScenarioStatus(Enum):
    SUCCESS = 0
    PROJECTED_SUCCESS = 1
    BACKUP_PLAN = 2
    INFEASIBLE = 3,
    DATA_MISSING = 4,
    RESET = 5,
    NONE = 6

class ScenarioSolveStatus(Enum):
    SUCCESS = 0,
    INFEASIBLE = 1,
    SUPPORT_EXCEEDED = 2,
    NONZERO_SLACK = 3

import numpy as np
from enum import Enum
from typing import List


# ==== trajectory_sample ====
# C++: std::vector<std::vector<Eigen::VectorXd>>
# In Python: list[list[np.ndarray]], where each np.ndarray is 1D
trajectory_sample = List[List[np.ndarray]]


# ==== Enums ====

class ObstacleType(Enum):
    STATIC = 0
    DYNAMIC = 1
    RANGE = 2

class ConstraintSide(Enum):
    BOTTOM = 0
    TOP = 1
    UNDEFINED = 2


class ScenarioBase:
    def __init__(self, sampler=None):
        self.support_subsample = None
        self.sampler = None
        if sampler is not None:
            self.sampler = sampler
        self.status = ScenarioStatus.NONE
        self.polygon_failed = False

    def set_parameters(self, data, step):
        return

    def set_sampler(self, sampler):
        return

    def update(self, data):
        return

    def compute_active_constraints(self, active_constraints_aggregate, infeasible_scenarios):
        return

# ==== Scenario Struct ====

class Scenario:
    def __init__(self, idx: int, obstacle_idx: int):
        self.idx_ = idx
        self.obstacle_idx_ = obstacle_idx



@dataclass
class ScenarioConstraint:
    """Meta-data of constructed constraints"""
    a1: float
    a2: float
    b: float
    scenario: Scenario
    k: int  # time step


# ==== ScenarioConstraint Struct ====

class ScenarioConstraint:
    def __init__(self, scenario: Scenario,
                 type_: ObstacleType,
                 side_: ConstraintSide):
        self.scenario_ = scenario
        self.type_ = type_
        self.side_ = side_

    def get_halfspace_index(self, sample_size: int) -> int:
        if self.type_ == ObstacleType.DYNAMIC:
            return sample_size * self.scenario_.obstacle_idx_ + self.scenario_.idx_
        else:
            return self.scenario_.idx_


# ==== SupportSubsample Struct ====

class SupportSubsample:
    def __init__(self, initial_size: int = 150):
        self.support_indices_: List[int] = []
        self.scenarios_: List[Scenario] = []
        self.size_ = 0

    def add(self, scenario: Scenario):
        if self.contains_scenario(scenario):
            return
        self.support_indices_.append(scenario.idx_)
        self.scenarios_.append(scenario)
        self.size_ += 1

    def reset(self):
        self.size_ = 0
        self.support_indices_.clear()
        self.scenarios_.clear()

    def contains_scenario(self, scenario: Scenario) -> bool:
        return scenario.idx_ in self.support_indices_[:self.size_]

    def merge_with(self, other: "SupportSubsample"):
        for sc in other.scenarios_:
            if not self.contains_scenario(sc):
                self.add(sc)

    def print_(self):
        print("=" * 40)
        print("Support Subsample")
        for sc in self.scenarios_:
            print(f"Scenario {sc.idx_}, Obstacle {sc.obstacle_idx_}")
        print("=" * 40)

    def print_update(self, solver_id: int, bound: int, iterations: int):
        print(f"[Solver {solver_id}] SQP ({iterations}): Support = {self.size_}/{bound}")


@dataclass
class SupportData:
    """Class for managing the Convex SP setting the support maximum"""
    name: str
    n_collected: int = 0
    max_support: int = 0

    def __post_init__(self):
        # DataSaver removed - not needed for new framework
        pass

    def add(self, support: int):
        # DataSaver removed - not needed for new framework
        self.n_collected += 1
        self.max_support = max(self.max_support, support)

    def save(self):
        # DataSaver removed - not needed for new framework
        pass

    def load(self) -> bool:
        return self.support_saver.load()

    def get_filename(self) -> str:
        return f"{self.name}_support-data"


# ==== Partition Struct ====

class Partition:
    def __init__(self, id_: int, velocity: float):
        self.id = id_
        self.velocity = velocity

####### Obstacle Logic ######

class PredictionStep:
    def __init__(self, position: np.ndarray, angle: float, major_radius: float, minor_radius: float):
        # Mean
        self.position = position
        self.angle = angle
        # Covariance
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def __str__(self):

        return f"PredictionStep({self.position}, {self.angle}, {self.major_radius}, {self.minor_radius})"

class Prediction:
    def __init__(self, type_=None):
        self.type = type_
        self.steps = []
        self.probabilities = []

    def empty(self) -> bool:
        return len(self.steps) == 0


class StaticObstacle:
    def __init__(self, position=None, angle=None, radius=None, _type=None):
        self.position = position
        self.angle = angle
        self.radius = radius
        self.type = _type if _type is not None else ObstacleType.STATIC
        self.prediction = Prediction()
        self.halfspaces = []  # List to store halfspace constraints

    def add_halfspace(self, A, b):
        """Add a halfspace constraint defined by Ax <= b"""
        self.halfspaces.append(Halfspace(A, b))

    def set(self, attribute, value):
        self.__setattr__(attribute, value)

class DynamicObstacle:
    def __init__(self, index: int, position: np.ndarray, angle: float, radius: float,
                 _type: 'ObstacleType' = ObstacleType.DYNAMIC):
        self.index = index
        self.position = position
        self.angle = angle
        self.radius = radius
        self.type = _type
        self.s = 0.0
        self.prediction = Prediction()


def generate_dynamic_obstacles(
    number: int,
    prediction_type: str,
    size: float,
    distribution: str = "random_paths",
    area: tuple = ((0, 20), (0, 20), (0, 0)),  # x, y, z ranges
    path_types=("straight", "curved", "s-turn", "circle"),
    num_points=11,
    dim: int = 2
) -> List[DynamicObstacle]:
    """
    Generate multiple dynamic obstacles, optionally in 3D, each with its own independent path.

    Args:
        number: Number of obstacles.
        prediction_type: Behavior of prediction, DETERMINISTIC, GAUSSIAN, NONGAUSSIAN
        size: Radius of each obstacle.
        distribution: 'random_paths' or 'grid'.
        area: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) for randomization.
        path_types: Allowed path types (randomly chosen per obstacle).
        num_points: Points per reference path.
        dim: 2 or 3 (controls whether paths and positions include z).

    Returns:
        List[DynamicObstacle]
    """
    obstacles = []

    for i in range(number):
        # Generate start and goal positions
        if distribution == "random_paths":
            start = [
                np.random.uniform(area[0][0], area[0][1]),
                np.random.uniform(area[1][0], area[1][1]),
            ]
            goal = [
                np.random.uniform(area[0][0], area[0][1]),
                np.random.uniform(area[1][0], area[1][1]),
            ]
            if dim == 3:
                start.append(np.random.uniform(area[2][0], area[2][1]))
                goal.append(np.random.uniform(area[2][0], area[2][1]))
            else:
                start.append(0.0)
                goal.append(0.0)

        elif distribution == "grid":
            # Simple grid for 2D or 3D
            grid_size = int(np.ceil(np.sqrt(number)))
            row, col = divmod(i, grid_size)
            step_x = (area[0][1] - area[0][0]) / grid_size
            step_y = (area[1][1] - area[1][0]) / grid_size
            start = [area[0][0] + col * step_x, area[1][0] + row * step_y]
            goal = [
                np.random.uniform(area[0][0], area[0][1]),
                np.random.uniform(area[1][0], area[1][1]),
            ]
            if dim == 3:
                start.append(np.random.uniform(area[2][0], area[2][1]))
                goal.append(np.random.uniform(area[2][0], area[2][1]))
            else:
                start.append(0.0)
                goal.append(0.0)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Generate unique path for this obstacle
        path_type = np.random.choice(path_types)
        ref_path = generate_reference_path(start, goal, path_type, num_points=num_points)

        # Initial position (3D if dim == 3)
        pos = np.array([ref_path.x[0], ref_path.y[0], ref_path.z[0]]) if dim == 3 else np.array([ref_path.x[0], ref_path.y[0]])
        # Compute heading angle in XY plane
        angle = np.arctan2(ref_path.y[1] - ref_path.y[0], ref_path.x[1] - ref_path.x[0])

        # Create obstacle
        obst = DynamicObstacle(index=i, position=pos, angle=angle, radius=size)

        # Prediction behavior
        if prediction_type == DETERMINISTIC:
            obst.prediction.path = ref_path
            obst.prediction.type = PredictionType.DETERMINISTIC
            obst.prediction.steps = []
        elif prediction_type == GAUSSIAN:
            obst.prediction.path = ref_path
            obst.prediction.type = PredictionType.GAUSSIAN
            obst.prediction.steps = []

        obstacles.append(obst)

    return obstacles


class ReferencePath:
    def __init__(self):
        self.x = [] # list of x coordinates
        self.y = [] # list of y coordinates
        self.z = [] # list of z coordinates
        self.psi = [] # list of headings
        self.v = [] # list of velocities
        self.s = [] # list of arc length progress for aligning with other attributes

        self.x_spline = None # TKSpline for numeric evaluation (x as dependent variable)
        self.y_spline = None # TKSpline for numeric evaluation (y as dependent variable)
        self.z_spline = None # TKSpline for numeric evaluation (z as dependent variable)
        self.v_spline = None # TKSpline for numeric evaluation (v as dependent variable)

    def clear(self):
        self.x = []
        self.y = []
        self.psi = []
        self.v = []
        self.s = []
        self.x_spline = None # TKSpline for numeric evaluation (x as dependent variable)
        self.y_spline = None # TKSpline for numeric evaluation (y as dependent variable)
        self.z_spline = None # TKSpline for numeric evaluation (z as dependent variable)
        self.v_spline = None # TKSpline for numeric evaluation (v as dependent variable)

    def empty(self) -> bool:
        return len(self.x) == 0

    def get_velocity(self):
        return self.v

    def has_velocity(self) -> bool:
        return len(self.v) > 0

    def has_distance(self) -> bool:
        return len(self.s) > 0

    def set(self, attribute, value):
        self.__setattr__(attribute, value)

    def get_arc_length(self):
        return self.s[-1] - self.s[0]

    def get(self, attribute):
        return self.__getattribute__(attribute)

    def format_val(self, val):
        return f"{val:.3f}" if isinstance(val, (float, int)) else "   ---"

    def __str__(self):
        if self.empty():
            return "ReferencePath: [empty]"

        header = f"{'Index':>5} | {'x':>8} | {'y':>8} | {'psi':>8} | {'v':>8} | {'s':>8}"
        lines = [header, "-" * len(header)]

        length = len(self.x)
        for i in range(length):
            x = self.x[i] if i < len(self.x) else None
            y = self.y[i] if i < len(self.y) else None
            psi = self.psi[i] if i < len(self.psi) else None
            v = self.v[i] if i < len(self.v) else None
            s = self.s[i] if i < len(self.s) else None
            lines.append(
                f"{i:5d} | {self.format_val(x):>8} | {self.format_val(y):>8} | "
                f"{self.format_val(psi):>8} | {self.format_val(v):>8} | {self.format_val(s):>8}"
            )

        return "\n".join(lines)

def generate_reference_path(start, goal, path_type="curved", num_points=11) -> ReferencePath:
    """
    Generate a reference path for MPC tracking.
    
    Reference: C++ mpc_planner (tud-amr/mpc_planner) - uses arc length parameterization
    with continuous curvature to prevent boundary self-intersection.
    
    Key design decisions:
    - Use sufficient point density for smooth boundaries (minimum 50 points)
    - Limit curvature to prevent boundary self-intersection with typical road widths (3-4m half-width)
    - Maximum curvature κ should satisfy: κ < 1/road_half_width to prevent inner boundary crossing
    
    Args:
        start: Start position [x, y, z] or [x, y]
        goal: Goal position [x, y, z] or [x, y]
        path_type: "straight", "curved", "s-turn", or "circle"
        num_points: Number of path points (minimum 50 for smooth boundaries)
    
    Returns:
        ReferencePath object with x, y, z, s arrays and splines
    """
    # Ensure sufficient point density for smooth boundaries
    # Reference: C++ mpc_planner uses dense sampling for smooth constraint evaluation
    num_points = max(num_points, 50)
    t = np.linspace(0, 1, num_points)

    if path_type == "straight":
        x = np.linspace(start[0], goal[0], num_points)
        y = np.linspace(start[1], goal[1], num_points)
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)

    elif path_type == "curved":
        start_np = np.array(start[:2])
        goal_np = np.array(goal[:2])
        direction = goal_np - start_np
        length_sg = np.linalg.norm(direction)
        
        if length_sg > 1e-6:
            # Perpendicular vector (90 degrees counterclockwise)
            perp = np.array([-direction[1], direction[0]]) / length_sg
            
            # CRITICAL: Limit curve offset to prevent boundary self-intersection
            # For road half-width w, max curvature κ = 1/w means offset ≈ κ * L² / 8
            # With typical road half-width 3.5m, keep offset < 10% of path length
            # This ensures inner boundary radius > 0 (no self-intersection)
            max_offset_ratio = 0.1  # 10% of path length
            offset_ratio = min(0.15, max_offset_ratio)  # Slight curve
            
            mid = 0.5 * (start_np + goal_np) + offset_ratio * length_sg * perp
        else:
            mid = 0.5 * (start_np + goal_np)
        
        # Use cubic Bezier curve for smoother curvature distribution
        # Control points: P0=start, P1=control1, P2=control2, P3=goal
        # This gives C1 continuity and bounded curvature
        control1 = start_np + 0.33 * direction + 0.5 * (mid - 0.5 * (start_np + goal_np))
        control2 = start_np + 0.67 * direction + 0.5 * (mid - 0.5 * (start_np + goal_np))
        
        # Cubic Bezier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        x = ((1-t)**3 * start_np[0] + 
             3*(1-t)**2*t * control1[0] + 
             3*(1-t)*t**2 * control2[0] + 
             t**3 * goal_np[0])
        y = ((1-t)**3 * start_np[1] + 
             3*(1-t)**2*t * control1[1] + 
             3*(1-t)*t**2 * control2[1] + 
             t**3 * goal_np[1])
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)

    elif path_type == "s-turn":
        # S-turn with controlled amplitude to prevent boundary self-intersection
        # Reference: C++ mpc_planner - curvature-aware paths
        x = np.linspace(start[0], goal[0], num_points)
        path_length = abs(goal[0] - start[0])
        
        # CRITICAL: Limit amplitude to prevent boundary self-intersection
        # For sinusoidal path: curvature κ ≈ amplitude * (2π/wavelength)²
        # With wavelength = path_length and road half-width 3.5m:
        # amplitude < road_half_width * wavelength² / (4π²) ≈ 0.09 * path_length
        # Use 1.5m fixed amplitude for typical road widths
        amplitude = min(1.5, path_length * 0.08)
        
        # Single period sine for smooth S-curve (one complete oscillation)
        y = np.linspace(start[1], goal[1], num_points) + amplitude * np.sin(np.pi * t)
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)

    elif path_type == "circle":
        radius = 5.0
        theta = np.linspace(0, np.pi, num_points)
        x = start[0] + radius * np.cos(theta)
        y = start[1] + radius * np.sin(theta)
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)

    else:
        raise ValueError(f"Unknown path type: {path_type}")

    # Compute arc length
    s = np.zeros(len(x))
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dz = z[i] - z[i - 1]
        s[i] = s[i - 1] + np.sqrt(dx**2 + dy**2 + dz**2)

    # Build numeric splines using TKSpline (for post-optimization evaluation)
    # Note: For symbolic optimization, use Spline2D with parameter dictionaries
    spline_x = TKSpline(s, x)
    spline_y = TKSpline(s, y)
    spline_z = TKSpline(s, z)

    # Store in ReferencePath object
    path = ReferencePath()
    path.x = x
    path.y = y
    path.z = z
    path.s = s
    path.x_spline = spline_x
    path.y_spline = spline_y
    path.z_spline = spline_z
    path.length = float(s[-1])  # ✅ Set total path length here

    return path


def calculate_path_normals_improved(reference_path):
    """
    Calculate improved normal vectors for the reference path.
    This version handles curvature better and ensures consistent orientation.
    """
    normals = []

    for i in range(len(reference_path.x)):
        if i == 0:
            # First point: use forward difference
            dx = reference_path.x[i + 1] - reference_path.x[i]
            dy = reference_path.y[i + 1] - reference_path.y[i]
        elif i == len(reference_path.x) - 1:
            # Last point: use backward difference
            dx = reference_path.x[i] - reference_path.x[i - 1]
            dy = reference_path.y[i] - reference_path.y[i - 1]
        else:
            # Middle points: use central difference for smoother results
            dx = reference_path.x[i + 1] - reference_path.x[i - 1]
            dy = reference_path.y[i + 1] - reference_path.y[i - 1]

        # Normalize the tangent vector
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm > 1e-6:
            dx_norm = dx / norm
            dy_norm = dy / norm
        else:
            dx_norm = 1.0
            dy_norm = 0.0

        # Normal vector (perpendicular to tangent, pointing left)
        # Rotate tangent 90 degrees counterclockwise
        normal_x = -dy_norm
        normal_y = dx_norm

        normals.append((normal_x, normal_y))

    return normals


def generate_road_boundaries_improved(reference_path, road_width):
    """
    Generate road boundaries with curvature-aware offset to prevent self-intersection.
    
    Reference: C++ mpc_planner (tud-amr/mpc_planner) - road boundaries are computed
    by offsetting perpendicular to the path, with curvature compensation to prevent
    inner boundary from crossing itself.
    
    Key insight: For a curve with curvature κ, the inner boundary crosses itself
    when offset distance d > 1/κ (radius of curvature). We reduce offset in
    high-curvature regions to prevent this.
    
    Args:
        reference_path: ReferencePath object with x, y, s arrays
        road_width: Total road width (left + right boundary width)
    
    Returns:
        left_x, left_y, right_x, right_y: Lists of boundary coordinates
    """
    # Calculate improved normals using spline derivatives for accuracy
    normals = calculate_path_normals_improved(reference_path)
    
    half_width = road_width / 2.0
    n_points = len(reference_path.x)
    
    # Compute curvature at each point for adaptive boundary offset
    # Reference: Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    curvatures = np.zeros(n_points)
    
    for i in range(n_points):
        if i > 0 and i < n_points - 1:
            # Use three-point finite difference for first derivatives
            p_prev = np.array([reference_path.x[i-1], reference_path.y[i-1]])
            p_curr = np.array([reference_path.x[i], reference_path.y[i]])
            p_next = np.array([reference_path.x[i+1], reference_path.y[i+1]])
            
            # Arc length segments
            ds_prev = np.linalg.norm(p_curr - p_prev)
            ds_next = np.linalg.norm(p_next - p_curr)
            
            if ds_prev > 1e-6 and ds_next > 1e-6:
                # First derivatives (tangent)
                dx = (p_next[0] - p_prev[0]) / (ds_prev + ds_next)
                dy = (p_next[1] - p_prev[1]) / (ds_prev + ds_next)
                
                # Second derivatives (curvature direction)
                ddx = (p_next[0] - 2*p_curr[0] + p_prev[0]) / (0.5*(ds_prev + ds_next))**2
                ddy = (p_next[1] - 2*p_curr[1] + p_prev[1]) / (0.5*(ds_prev + ds_next))**2
                
                # Curvature magnitude
                denom = (dx**2 + dy**2)**(1.5)
                if denom > 1e-10:
                    curvatures[i] = abs(dx * ddy - dy * ddx) / denom
    
    # Smooth curvature estimate to reduce noise
    if n_points >= 5:
        from scipy.ndimage import gaussian_filter1d
        curvatures = gaussian_filter1d(curvatures, sigma=2.0)
    
    # Compute adaptive half-width based on curvature
    # CRITICAL: Reduce width where curvature * half_width > 0.8 to prevent self-intersection
    # The critical condition is: half_width < 1/curvature (radius of curvature)
    adjusted_half_width = np.zeros(n_points)
    
    for i in range(n_points):
        κ = curvatures[i]
        if κ > 1e-6:
            # Radius of curvature
            R = 1.0 / κ
            # Limit offset to 80% of radius to prevent self-intersection
            max_offset = 0.8 * R
            adjusted_half_width[i] = min(half_width, max_offset)
        else:
            adjusted_half_width[i] = half_width
    
    # Generate boundaries with adjusted widths
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    for i in range(n_points):
        nx, ny = normals[i]
        width = adjusted_half_width[i]

        # Left boundary (offset in positive normal direction)
        left_x.append(float(reference_path.x[i]) + nx * width)
        left_y.append(float(reference_path.y[i]) + ny * width)

        # Right boundary (offset in negative normal direction)
        right_x.append(float(reference_path.x[i]) - nx * width)
        right_y.append(float(reference_path.y[i]) - ny * width)

    return left_x, left_y, right_x, right_y


def smooth_boundaries(x_coords, y_coords, s_coords, smoothing_factor=0.1):
    """
    Apply smoothing to boundary coordinates to reduce sharp edges.
    """
    from scipy.ndimage import gaussian_filter1d

    # Apply Gaussian smoothing
    x_smooth = gaussian_filter1d(x_coords, sigma=smoothing_factor * len(x_coords))
    y_smooth = gaussian_filter1d(y_coords, sigma=smoothing_factor * len(y_coords))

    return x_smooth, y_smooth


# Modified test code section
def create_improved_boundaries(data, reference_path):
    """
    Replace the boundary generation section in your test code with this.
    """
    # Road width
    road_width = data.road_width if hasattr(data, 'road_width') and data.road_width is not None else 4.0

    # Generate improved boundaries
    left_x, left_y, right_x, right_y = generate_road_boundaries_improved(reference_path, road_width)

    # Optional: Apply smoothing
    left_x, left_y = smooth_boundaries(left_x, left_y, reference_path.s)
    right_x, right_y = smooth_boundaries(right_x, right_y, reference_path.s)

    # Create numeric splines for the boundaries using TKSpline
    left_boundary_spline_x = TKSpline(reference_path.s, np.array(left_x))
    left_boundary_spline_y = TKSpline(reference_path.s, np.array(left_y))
    right_boundary_spline_x = TKSpline(reference_path.s, np.array(right_x))
    right_boundary_spline_y = TKSpline(reference_path.s, np.array(right_y))

    # Store boundary data
    data.left_boundary_x = left_x
    data.left_boundary_y = left_y
    data.right_boundary_x = right_x
    data.right_boundary_y = right_y

    # Store the spline functions
    data.left_spline_x = left_boundary_spline_x
    data.left_spline_y = left_boundary_spline_y
    data.right_spline_x = right_boundary_spline_x
    data.right_spline_y = right_boundary_spline_y

    # Create Bound objects
    data.left_bound = Bound(left_x, left_y, reference_path.s)
    data.right_bound = Bound(right_x, right_y, reference_path.s)

    return data


# Alternative: Use spline-based normal calculation
def calculate_spline_normals(reference_path):
    """
    Calculate normals using spline derivatives for smoother results.
    """
    if not hasattr(reference_path, 'x_spline') or not hasattr(reference_path, 'y_spline'):
        # Create numeric splines if they don't exist using TKSpline
        reference_path.x_spline = TKSpline(reference_path.s, reference_path.x)
        reference_path.y_spline = TKSpline(reference_path.s, reference_path.y)

    normals = []

    for i, s_val in enumerate(reference_path.s):
        # Get derivatives from splines
        dx = reference_path.x_spline.derivative()(s_val)
        dy = reference_path.y_spline.derivative()(s_val)

        # Normalize
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm > 1e-6:
            dx_norm = dx / norm
            dy_norm = dy / norm
        else:
            dx_norm = 1.0
            dy_norm = 0.0

        # Normal vector (perpendicular to tangent, pointing left)
        normal_x = -dy_norm
        normal_y = dx_norm

        normals.append((normal_x, normal_y))

    return normals

def calculate_path_normals(reference_path: ReferencePath):
    normals = []
    for i in range(len(reference_path.x)):
        if i == 0:
            # For the first point, use the direction to the next point
            dx = reference_path.x[1] - reference_path.x[0]
            dy = reference_path.y[1] - reference_path.y[0]
        elif i == len(reference_path.x) - 1:
            # For the last point, use the direction from the previous point
            dx = reference_path.x[-1] - reference_path.x[-2]
            dy = reference_path.y[-1] - reference_path.y[-2]
        else:
            # For middle points, use the average of adjacent segments
            dx1 = reference_path.x[i] - reference_path.x[i - 1]
            dy1 = reference_path.y[i] - reference_path.y[i - 1]
            dx2 = reference_path.x[i + 1] - reference_path.x[i]
            dy2 = reference_path.y[i + 1] - reference_path.y[i]
            dx = dx1 + dx2
            dy = dy1 + dy2

        # Calculate the normal vector (perpendicular to the path direction)
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            nx = -dy / length
            ny = dx / length
            normals.append((nx, ny))
        else:
            # Avoid division by zero
            normals.append((0, 0))

    return normals


class Bound:
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s


def generate_static_obstacles(
    number: int,
    size: float,
    reference_path: Optional[ReferencePath] = None,
    area: tuple = ((0, 20), (0, 20), (0, 0)),  # (x_range, y_range, z_range)
    lateral_offset_range=(-2.0, 2.0),
    along_path_ratio=1.0
) -> List[StaticObstacle]:
    """
    Generate static obstacles either randomly or along a provided reference path.

    Args:
        number: Number of obstacles to generate.
        size: Radius of each obstacle.
        reference_path: If provided, obstacles will be placed near this path.
        area: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) for randomization.
        lateral_offset_range: Range for lateral deviation from the path.
        along_path_ratio: Fraction of obstacles to place along the path (0.0 - 1.0).

    Returns:
        List[StaticObstacle]
    """
    obstacles = []

    for i in range(number):
        if reference_path and np.random.rand() < along_path_ratio:
            # Place obstacle near reference path
            s_offset = np.random.uniform(0, reference_path.s[-1])  # choose random point along path
            x_center = reference_path.x_spline(s_offset)
            y_center = reference_path.y_spline(s_offset)

            # Compute tangent and normal for lateral offset
            ds = 0.1
            s_next = min(s_offset + ds, reference_path.s[-1])
            dx = reference_path.x_spline(s_next) - x_center
            dy = reference_path.y_spline(s_next) - y_center
            tangent = np.array([dx, dy]) / (np.linalg.norm([dx, dy]) + 1e-6)
            normal = np.array([-tangent[1], tangent[0]])

            lateral_offset = np.random.uniform(*lateral_offset_range)
            position = np.array([
                x_center + lateral_offset * normal[0],
                y_center + lateral_offset * normal[1],
                0.0  # z fixed to 0 for 2D case
            ])
        else:
            # Place randomly in the given area
            x = np.random.uniform(area[0][0], area[0][1])
            y = np.random.uniform(area[1][0], area[1][1])
            z = np.random.uniform(area[2][0], area[2][1])
            position = np.array([x, y, z])

        # Orientation for static obstacles can be random or zero
        angle = np.random.uniform(-np.pi, np.pi)

        # Create the obstacle
        obst = StaticObstacle(position=position, angle=angle, radius=size)
        obstacles.append(obst)

    return obstacles


def define_robot_area(length: float, width: float, n_discs: int) -> list[Disc]:
    """Define the robot area using discs."""
    # Where is the center w.r.t. the back of the vehicle
    center_offset = length / 2.  # Could become a parameter
    radius = width / 2.

    robot_area = []
    assert n_discs > 0, "Trying to create a collision region with less than a disc"

    if n_discs == 1:
        robot_area.append(Disc(0., radius))
    else:
        for i in range(n_discs):
            if i == 0:
                # First disc at the back of the car
                robot_area.append(Disc(-center_offset + radius, radius))
            elif i == n_discs - 1:
                # Last disc at the front of the car
                robot_area.append(Disc(-center_offset + length - radius, radius))
            else:
                # Other discs in between
                offset = -center_offset + radius + (i * (length - 2. * radius) / (n_discs - 1.))
                robot_area.append(Disc(offset, radius))

            logging.debug(f"Disc {i}: offset {robot_area[-1].offset}, radius {robot_area[-1].radius}")

    return robot_area


def propagate_obstacles(data, dt=0.1, horizon=10, speed=0, sigma_pos=0.2):
  """Propagate obstacle predictions over the horizon.

  Prefers obstacle-provided dynamics models when available; otherwise falls back
  to existing behavior (path-based or constant-velocity prediction).
  """
  # Guard: no obstacles to propagate
  if not hasattr(data, 'dynamic_obstacles') or data.dynamic_obstacles is None or len(data.dynamic_obstacles) == 0:
      return

  for obstacle in data.dynamic_obstacles:
      pred = obstacle.prediction
      path = getattr(pred, "path", None)

      # Prefer dynamics model if obstacle provides one
      dyn_model = getattr(obstacle, 'dynamics_model', None)
      # If a dynamics_type string is provided but no model, instantiate from known vehicle models
      if dyn_model is None and hasattr(obstacle, 'dynamics_type') and isinstance(obstacle.dynamics_type, str):
          try:
              from planning.dynamic_models import SecondOrderUnicycleModel, SecondOrderBicycleModel
              dt_str = obstacle.dynamics_type.lower().strip()
              if dt_str in ("unicycle", "second_order_unicycle", "secondorderunicycle"):
                  dyn_model = SecondOrderUnicycleModel()
              elif dt_str in ("bicycle", "second_order_bicycle", "secondorderbicycle"):
                  dyn_model = SecondOrderBicycleModel()
              if dyn_model is not None:
                  setattr(obstacle, 'dynamics_model', dyn_model)
          except Exception:
              dyn_model = None
      try:
          if dyn_model is not None:
              # Use the same numeric integrator as vehicle models if available
              try:
                  from planning.dynamic_models import numeric_rk4 as _nrk4
              except Exception:
                  _nrk4 = None

              # Initialize steps list
              steps = []
              # Build initial state vector [x, y, ...] based on model variables
              if hasattr(obstacle, 'state') and isinstance(obstacle.state, (list, tuple, np.ndarray)):
                  # If obstacle provides a numeric state vector directly
                  state_vec = np.array(obstacle.state, dtype=float)
              else:
                  # Minimal state from position/angle/velocity if present
                  x0 = float(obstacle.position[0]) if hasattr(obstacle, 'position') else 0.0
                  y0 = float(obstacle.position[1]) if hasattr(obstacle, 'position') else 0.0
                  psi0 = float(getattr(obstacle, 'angle', 0.0))
                  v0 = float(getattr(obstacle, 'speed', speed))
                  state_vec = np.array([x0, y0, psi0, v0], dtype=float)

              # Controls assumed zero if not specified
              if hasattr(dyn_model, 'inputs') and isinstance(dyn_model.inputs, list):
                  u = [0.0 for _ in dyn_model.inputs]
              else:
                  u = []

              # Simulate forward over horizon
              pred_steps = []
              # Get base uncertainty parameters for Gaussian predictions
              base_std = sigma_pos  # Default uncertainty
              growth_rate = 0.0
              if pred.type == PredictionType.GAUSSIAN:
                  # Try to get uncertainty params from obstacle if available
                  if hasattr(obstacle, 'uncertainty_params') and obstacle.uncertainty_params:
                      base_std = obstacle.uncertainty_params.get('position_std', sigma_pos)
                      growth_rate = obstacle.uncertainty_params.get('uncertainty_growth', 0.0)
              
              for k in range(int(horizon) + 1):
                  pos = state_vec[:2] if state_vec.shape[0] >= 2 else np.array([0.0, 0.0])
                  angle = float(state_vec[2]) if state_vec.shape[0] >= 3 else 0.0
                  
                  # Set uncertainty radii based on prediction type
                  if pred.type == PredictionType.GAUSSIAN:
                      # For Gaussian predictions, use uncertainty_std (which may grow over time)
                      # Add obstacle radius to uncertainty radius for visualization
                      
                      # IMPROVEMENT: Enhanced uncertainty growth for moving obstacles
                      # Use quadratic growth for moving obstacles to account for:
                      # - Prediction errors accumulating over time
                      # - Obstacle acceleration/deceleration uncertainty
                      # - Obstacle behavior changes
                      obstacle_speed = 0.0
                      if hasattr(obstacle, 'velocity') and obstacle.velocity is not None:
                          obstacle_speed = np.linalg.norm(obstacle.velocity)
                      elif hasattr(obstacle, 'speed') and obstacle.speed is not None:
                          obstacle_speed = float(obstacle.speed)
                      
                      # Linear growth for static/slow obstacles
                      uncertainty_std = base_std + k * growth_rate
                      
                      # Add quadratic term for moving obstacles (faster growth for further predictions)
                      if obstacle_speed > 0.5:  # Moving faster than 0.5 m/s
                          quadratic_growth = (k * k) * 0.01  # Quadratic growth term
                          uncertainty_std = uncertainty_std + quadratic_growth
                      
                      obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
                      major_r = obstacle_radius + uncertainty_std * 2
                      minor_r = obstacle_radius + uncertainty_std
                  elif pred.type == PredictionType.NONGAUSSIAN:
                      obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
                      major_r = obstacle_radius + sigma_pos * 3
                      minor_r = obstacle_radius + sigma_pos * 1.5
                  else:
                      # Deterministic: just obstacle radius
                      obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
                      major_r = obstacle_radius
                      minor_r = obstacle_radius
                  
                  pred_steps.append(PredictionStep(pos, angle, major_r, minor_r))
                  if k == horizon:
                      break
                  # Advance one step with model if integrator available
                  if _nrk4 is not None and hasattr(dyn_model, 'discrete_dynamics'):
                      import casadi as _cs
                      z_k = _cs.vertcat(*u, *_cs.vertcat(*state_vec)) if u else _cs.vertcat(*state_vec)
                      dyn_model.load(z_k)
                      next_symbolic = dyn_model.discrete_dynamics(z_k, None, dt)
                      next_numeric = _nrk4(next_symbolic, dyn_model, None, dt)
                      state_vec = np.array([float(v) for v in next_numeric]).reshape(-1)
                  else:
                      # Fallback: simple kinematic step if dynamics missing
                      vx = (state_vec[3] * np.cos(state_vec[2])) if state_vec.shape[0] >= 4 else speed
                      vy = (state_vec[3] * np.sin(state_vec[2])) if state_vec.shape[0] >= 4 else 0.0
                      state_vec = state_vec.copy()
                      state_vec[0] += vx * dt
                      if state_vec.shape[0] >= 2:
                          state_vec[1] += vy * dt
              obstacle.prediction.steps = pred_steps
              continue
      except Exception as e:
          # If any issue arises, fall back to legacy behavior
          # But preserve Gaussian prediction type and regenerate steps with correct radii
          import logging
          logging.getLogger("integration_test").debug(f"Exception in propagate_obstacles for obstacle {getattr(obstacle, 'index', 'unknown')}: {e}")
          # If this was a Gaussian prediction, we need to regenerate steps with correct radii
          if pred.type == PredictionType.GAUSSIAN:
              # Get uncertainty params
              base_std = sigma_pos
              growth_rate = 0.0
              if hasattr(obstacle, 'uncertainty_params') and obstacle.uncertainty_params:
                  base_std = obstacle.uncertainty_params.get('position_std', sigma_pos)
                  growth_rate = obstacle.uncertainty_params.get('uncertainty_growth', 0.0)
              
              # Regenerate prediction steps with Gaussian radii
              obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
              pos = np.array([float(obstacle.position[0]), float(obstacle.position[1])]) if hasattr(obstacle, 'position') else np.array([0.0, 0.0])
              angle = float(getattr(obstacle, 'angle', 0.0))
              velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
              
              pred_steps = []
              for k in range(int(horizon) + 1):
                  uncertainty_std = base_std + k * growth_rate
                  major_r = obstacle_radius + uncertainty_std * 2
                  minor_r = obstacle_radius + uncertainty_std
                  current_pos = pos + velocity * k * dt
                  pred_steps.append(PredictionStep(current_pos, angle, major_r, minor_r))
              
              obstacle.prediction.steps = pred_steps
              continue
          pass

      # Fallback: constant velocity if no path and no usable dynamics model
      if path is None:
          # Preserve prediction type before creating new prediction
          original_pred_type = pred.type if hasattr(pred, 'type') else None
          original_uncertainty_params = getattr(obstacle, 'uncertainty_params', None)
          velocity = np.array([np.cos(getattr(obstacle, 'angle', 0.0)), np.sin(getattr(obstacle, 'angle', 0.0))]) * speed
          obstacle.prediction = get_constant_velocity_prediction(getattr(obstacle, 'position', np.array([0.0, 0.0])), velocity, dt, horizon)
          # Restore original prediction type if it was Gaussian and regenerate steps with correct radii
          if original_pred_type == PredictionType.GAUSSIAN and hasattr(obstacle.prediction, 'type'):
              obstacle.prediction.type = original_pred_type
              # Regenerate steps with Gaussian radii
              if original_uncertainty_params:
                  base_std = original_uncertainty_params.get('position_std', sigma_pos)
                  growth_rate = original_uncertainty_params.get('uncertainty_growth', 0.0)
              else:
                  base_std = sigma_pos
                  growth_rate = 0.0
              
              obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
              pos = np.array([float(obstacle.position[0]), float(obstacle.position[1])]) if hasattr(obstacle, 'position') else np.array([0.0, 0.0])
              angle = float(getattr(obstacle, 'angle', 0.0))
              
              pred_steps = []
              for k in range(int(horizon) + 1):
                  uncertainty_std = base_std + k * growth_rate
                  major_r = obstacle_radius + uncertainty_std * 2
                  minor_r = obstacle_radius + uncertainty_std
                  current_pos = pos + velocity * k * dt
                  pred_steps.append(PredictionStep(current_pos, angle, major_r, minor_r))
              
              obstacle.prediction.steps = pred_steps
          continue

      total_length = path.s[-1]

      # CRITICAL: For path-based obstacles, update obstacle.s based on CURRENT position
      # This ensures predictions start from the obstacle's actual current position, not stale arc length
      # Find closest point on path to current obstacle position
      if hasattr(obstacle, 'position') and obstacle.position is not None:
          try:
              # Sample path to find closest point
              s_samples = np.linspace(0, total_length, min(100, len(path.s)))
              path_points = np.array([[path.x_spline(s), path.y_spline(s)] for s in s_samples])
              obstacle_pos_2d = obstacle.position[:2]
              distances = np.linalg.norm(path_points - obstacle_pos_2d, axis=1)
              closest_idx = np.argmin(distances)
              obstacle.s = s_samples[closest_idx]
          except Exception:
              # Fallback: use existing s or initialize to 0
              if not hasattr(obstacle, "s"):
                  obstacle.s = 0.0
      else:
          # Initialize progress in arc length if no position available
          if not hasattr(obstacle, "s"):
              obstacle.s = 0.0
      
      # Advance arc length based on speed
      obstacle.s += speed * dt

      # If reached end of current path → generate new path
      if obstacle.s >= total_length:
          start = [path.x[-1], path.y[-1], path.z[-1] if hasattr(path, "z") else 0.0]

          # Choose new goal
          if hasattr(obstacle, "road_to_follow"):
              road = obstacle.road_to_follow
              s_offset = np.random.uniform(0, road.length)
              x_center = road.x_spline(s_offset)
              y_center = road.y_spline(s_offset)

              # Tangent & lateral offset to simulate lane variation
              dx = road.x_spline(min(s_offset + 0.1, road.length)) - x_center
              dy = road.y_spline(min(s_offset + 0.1, road.length)) - y_center
              tangent = np.array([dx, dy]) / np.linalg.norm([dx, dy])
              normal = np.array([-tangent[1], tangent[0]])
              lateral_offset = np.random.uniform(-3.0, 3.0)
              goal = [x_center + lateral_offset * normal[0], y_center + lateral_offset * normal[1], 0.0]
          else:
              goal = [
                  np.random.uniform(start[0] - 20, start[0] + 20),
                  np.random.uniform(start[1] - 20, start[1] + 20),
                  0.0
              ]

          # Generate new reference path
          new_path = generate_reference_path(
              start, goal, path_type=np.random.choice(["straight", "curved", "s-turn", "circle"]),
              num_points=10
          )

          obstacle.prediction.path = new_path
          obstacle.s = 0.0
          path = new_path
          total_length = path.s[-1]

      # ✅ Compute position using arc-length splines
      # CRITICAL: If obstacle.position was already updated (e.g., by obstacle_manager),
      # sync obstacle.s to match the current position instead of overwriting it
      if hasattr(obstacle, 'position') and obstacle.position is not None:
          try:
              # Check if position matches what we would compute from current s
              s_now = min(obstacle.s, total_length)
              x_from_s = path.x_spline(s_now)
              y_from_s = path.y_spline(s_now)
              pos_from_s = np.array([x_from_s, y_from_s])
              current_pos_2d = obstacle.position[:2]
              dist_diff = np.linalg.norm(pos_from_s - current_pos_2d)
              
              # If position differs significantly, sync s to match current position
              if dist_diff > 0.1:  # More than 10cm difference
                  # Find closest point on path to current position
                  s_samples = np.linspace(0, total_length, min(200, len(path.s)))
                  path_points = np.array([[path.x_spline(s), path.y_spline(s)] for s in s_samples])
                  distances = np.linalg.norm(path_points - current_pos_2d, axis=1)
                  closest_idx = np.argmin(distances)
                  obstacle.s = s_samples[closest_idx]
                  s_now = obstacle.s
          except Exception:
              # Fallback: use existing s
              s_now = min(obstacle.s, total_length)
      else:
          s_now = min(obstacle.s, total_length)
      
      x = path.x_spline(s_now)
      y = path.y_spline(s_now)
      z = path.z_spline(s_now) if path.z_spline else 0.0
      obstacle.position = np.array([x, y, z])

      # ✅ Compute heading
      ds = 0.1
      s_next = min(s_now + ds, total_length)
      dx = path.x_spline(s_next) - x
      dy = path.y_spline(s_next) - y
      obstacle.angle = np.arctan2(dy, dx)

      # ✅ Build prediction horizon
      pred_steps = []
      s_future = s_now
      for _ in range(horizon):
          s_future = min(s_future + speed * dt, total_length)
          px = path.x_spline(s_future)
          py = path.y_spline(s_future)
          pz = path.z_spline(s_future) if path.z_spline else 0.0

          s_next = min(s_future + ds, total_length)
          dx_f = path.x_spline(s_next) - px
          dy_f = path.y_spline(s_next) - py
          angle = np.arctan2(dy_f, dx_f)

          pos = np.array([px, py, pz])

          # Add noise & uncertainty
          if pred.type == PredictionType.GAUSSIAN:
              pos += np.random.normal(0, sigma_pos, size=pos.shape)
              major_r, minor_r = sigma_pos * 2, sigma_pos
          elif pred.type == PredictionType.NONGAUSSIAN:
              pos += np.random.standard_t(df=3, size=pos.shape) * sigma_pos
              major_r, minor_r = sigma_pos * 3, sigma_pos * 1.5
          else:
              major_r, minor_r = 0.1, 0.1

          pred_steps.append(PredictionStep(pos, angle, major_r, minor_r))

      pred.steps = pred_steps


def get_constant_velocity_prediction(position: np.ndarray, velocity: np.ndarray, dt: float, horizon: int) -> Prediction:
    """Create a constant velocity prediction for an obstacle.
    
    Args:
        position: Initial position [x, y]
        velocity: Constant velocity [vx, vy]
        dt: Time step
        horizon: Number of prediction steps
        
    Returns:
        Prediction object with constant velocity steps
    """
    pred = Prediction(PredictionType.DETERMINISTIC)
    pred_steps = []
    
    current_pos = np.array(position[:2])  # Ensure 2D
    vel = np.array(velocity[:2]) if len(velocity) >= 2 else np.array([velocity[0], 0.0])
    
    # Ensure horizon is not None
    horizon_val = horizon if horizon is not None else 10
    for i in range(horizon_val + 1):
        # Compute position at time t = i * dt
        pos = current_pos + vel * (i * dt)
        angle = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 1e-6 else 0.0
        # Use small uncertainty for deterministic prediction
        major_r = 0.1
        minor_r = 0.1
        pred_steps.append(PredictionStep(pos, angle, major_r, minor_r))
    
    pred.steps = pred_steps
    return pred


def remove_distant_obstacles(obstacles: list[DynamicObstacle], state: 'State') -> None:
    """Remove obstacles that are far from the current state."""
    nearby_obstacles = []

    pos = state.get_position()
    for obstacle in obstacles:
        if distance(pos, obstacle.position) < CONFIG["max_obstacle_distance"]:
            nearby_obstacles.append(obstacle)

    obstacles.clear()
    obstacles.extend(nearby_obstacles)

def filter_distant_obstacles(obstacles: list[DynamicObstacle], state: 'State', distance_limit = None):
    """Remove obstacles that are far from the current state."""
    nearby_obstacles = []

    dist = 0
    if distance_limit is not None:
        dist = distance_limit
    else:
        dist = CONFIG["max_obstacle_distance"]
    pos = state.get_position()
    for obstacle in obstacles:
        if distance(pos, obstacle.position) < dist:
            nearby_obstacles.append(obstacle)

    return nearby_obstacles


def ensure_obstacle_size(obstacles: list[DynamicObstacle], state: 'State') -> None:
    """Ensure that the number of obstacles matches the configured maximum."""
    import logging
    try:
        max_obstacles = CONFIG["max_obstacles"]
    except KeyError:
        # Fallback to default if max_obstacles not in CONFIG
        max_obstacles = 10
        logging.warning(f"max_obstacles not found in CONFIG, using default: {max_obstacles}")
    # Create an index list
    indices = list(range(len(obstacles)))

    # If more, we sort and retrieve the closest obstacles
    if len(obstacles) > max_obstacles:
        distances = []
        logging.debug(f"Received {len(obstacles)} > {max_obstacles} obstacles. Keeping the closest.")

        for obstacle in obstacles:
            min_dist = 1e5
            direction = np.array([np.cos(state.get("psi")), np.sin(state.get("psi"))])

            for k in range(CONFIG["horizon"]):
                # Linearly scaled
                dist = (k + 1) * 0.6 * distance(
                    obstacle.prediction.steps[k].position,
                    state.get_position() + state.get("v") * k * direction
                )

                if dist < min_dist:
                    min_dist = dist

            distances.append(min_dist)

        # Sort obstacles on distance
        indices.sort(key=lambda j: distances[j])

        # Keep the closest obstacles
        processed_obstacles = []

        for v in range(max_obstacles):
            processed_obstacles.append(obstacles[indices[v]])

        # Assign sequential IDs
        for i in range(len(processed_obstacles)):
            processed_obstacles[i].index = i

        obstacles.clear()
        obstacles.extend(processed_obstacles)

    elif len(obstacles) < max_obstacles:
        logging.debug(f"Received {len(obstacles)} < {max_obstacles} obstacles. Adding dummies.")

        for cur_size in range(len(obstacles), max_obstacles):
            dummy = get_dummy_obstacle(state)
            dummy.prediction = get_constant_velocity_prediction(
                dummy.position,
                np.array([0., 0.]),
                CONFIG["integrator_step"],
                CONFIG["horizon"]
            )
            obstacles.append(dummy)

    logging.debug(f"Obstacle size (after processing) is: {len(obstacles)}")


def propagate_prediction_uncertainty(prediction: Prediction) -> None:
    """Propagate uncertainty through the prediction steps."""
    if prediction.type != PredictionType.GAUSSIAN:
        return

    dt = CONFIG["integrator_step"]
    major = 0.
    minor = 0.

    for k in range(10):
        major = np.sqrt(major ** 2 + (prediction.steps[k].major_radius * dt) ** 2)
        minor = np.sqrt(minor ** 2 + (prediction.steps[k].minor_radius * dt) ** 2)
        prediction.steps[k].major_radius += major # This was originally straight assignment not addition
        prediction.steps[k].minor_radius += minor

def propagate_prediction_uncertainty_for_obstacles(obstacles: list[DynamicObstacle]) -> None:
    """Propagate uncertainty for all obstacles."""
    for obstacle in obstacles:
        propagate_prediction_uncertainty(obstacle.prediction)

class Trajectory:
    def __init__(self, timestep: float = 0.1, length: int = 30): # defaults match timestep and horizon for solver
        self.timestep = timestep
        self.states = []

    def add_state(self, state: State):
        if state is None:
            return
        else:
            self.states.append(state)

    def get_states(self):
        return self.states

    def get_element_history_by_name(self, name):
        history = []
        for state in self.states:
            history.append(state.get(name))
        return history

    def reset(self):
        self.states = []

    def __str__(self):
        """Return a readable string representation of the current state."""
        lines = [f"Trajectory object contents: ({self.states}) --> "]
        k = 0
        for state in self.states:
            lines.append(f"  {state}: at {k:.3f}")
            k += 1
        return "\n".join(lines)


class FixedSizeTrajectory:
    def __init__(self, size: int = 50):
        self._size = size
        self.positions = []

    def add(self, p: np.ndarray):
        self.positions.append(p.copy())
        if len(self.positions) > self._size:
            self.positions.pop(0)

class Data:
    def __init__(self):
        self._store = {}

    def empty(self):
        return len(self._store) == 0

    def add(self, key, value):
        if key in self._store:
            self._store[key].append(value)
        else:
            self._store[key] = [value]

    def has(self, key):
        return key in self._store

    def set(self, key, value):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key, None)

    def remove(self, key):
        if key in self._store:
            del self._store[key]

    def reset(self):
        self._store.clear()

    def __getattr__(self, key):
        if key in self._store:
            return self._store[key]
        else:
            self.__setattr__(key, None)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __str__(self):
        """Return a readable string representation of the current state."""
        lines = [f"Data object contents: ({self._store}) --> "]
        for var, value in self._store.items():
            if isinstance(value, (float, int)):
                lines.append(f"  {var}: {value:.3f}")
            elif isinstance(value, np.ndarray):
                lines.append(f"  {var}: {np.array2string(value, precision=3, separator=', ')}")
            else:
                lines.append(f"  {var}: {value}")
        return "\n".join(lines)

class Costmap:
    def __init__(self, width, height, resolution, origin, obstacle_threshold=100):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin)
        self.obstacle_threshold = obstacle_threshold
        self.data = np.zeros((height, width), dtype=np.uint8)

    def set_obstacles(self, obstacle_coords):
        for x, y in obstacle_coords:
            i, j = self.world_to_map(x, y)
            LOG_DEBUG(f"For obstacle with coordinates x: {x}, y: {y}, calculated i: {i}, j: {j}")
            if 0 <= i < self.width and 0 <= j < self.height:
                self.data[j, i] = 255  # Mark as obstacle

    def inflate_obstacles(self, inflation_radius):
        inflation_cells = int(inflation_radius / self.resolution)
        inflated = self.data.copy()
        obstacle_indices = np.argwhere(self.data >= self.obstacle_threshold)
        for j, i in obstacle_indices:
            min_i = max(0, i - inflation_cells)
            max_i = min(self.width, i + inflation_cells)
            min_j = max(0, j - inflation_cells)
            max_j = min(self.height, j + inflation_cells)
            inflated[min_j:max_j, min_i:max_i] = 255
        self.data = inflated

    def get_size_in_cells_x(self):
        return self.width

    def get_size_in_cells_y(self):
        return self.height

    def getCost(self, i, j):
        return self.data[j, i]

    def map_to_world(self, i, j):
        x = self.origin[0] + i * self.resolution
        y = self.origin[1] + j * self.resolution
        return x, y

    def world_to_map(self, x, y):
        i = int((x - self.origin[0]) / self.resolution)
        j = int((y - self.origin[1]) / self.resolution)
        return i, j

class Problem:
    def __init__(self):
        self.model_type = None
        self.modules = []
        self.obstacles = []
        self.data = None
        self.x0 = None
        self._state = None  # Current state for get_state()

    def get_model_type(self):
        return self.model_type
    
    def get_modules(self):
        return self.modules
    
    def get_obstacles(self):
        return self.obstacles
    
    def get_data(self):
        return self.data
    
    def get_x0(self):
        return self.x0
    
    def get_state(self):
        """Get the current state. Returns x0 if state not explicitly set."""
        if self._state is not None:
            return self._state
        elif self.x0 is not None:
            return self.x0
        elif self.model_type is not None:
            # Create default state from model_type
            return State(self.model_type)
        else:
            raise ValueError("Problem has no state, x0, or model_type set")
    
    def set_state(self, state):
        """Set the current state."""
        self._state = state
