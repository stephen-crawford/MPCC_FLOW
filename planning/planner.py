import time
import numpy as np

from planning.types import *
from planning.types import propagate_obstacles
from planning.obstacle_manager import ObstacleManager
from planning.modules_manager import ModuleManager
from planning.parameter_manager import ParameterManager
from solver.casadi_solver import CasADiSolver
from utils.const import OBJECTIVE
from utils.utils import CONFIG, LOG_INFO, LOG_WARN, LOG_DEBUG


class Planner:
    def __init__(self, problem, config=None):
        LOG_INFO("=== Planner Initialization ===")
        
        if config is None:
            self.config = CONFIG
            LOG_DEBUG("Using default CONFIG")

        else:
            self.config = config
            LOG_DEBUG("Using provided config")

        self.problem = problem
        self.module_manager = ModuleManager()
        self.parameter_manager = ParameterManager()
        self.obstacle_manager = ObstacleManager(self.config)
        self.output = PlannerOutput()
        self.state = None
        LOG_INFO("Starting problem setup...")
        self.problem_setup()
        LOG_INFO("Planner initialization complete")


    def problem_setup(self):
        LOG_INFO("Setting up problem configuration...")
        
        self.model_type = self.problem.get_model_type()
        LOG_INFO(f"Model type: {self.model_type.__class__.__name__ if self.model_type else 'None'}")
        
        self.modules = self.problem.get_modules()
        LOG_INFO(f"Number of modules: {len(self.modules) if self.modules else 0}")
        for i, module in enumerate(self.modules):
            module_name = getattr(module, 'name', 'Unknown')
            module_type = getattr(module, 'module_type', 'Unknown')
            LOG_DEBUG(f"  Module {i+1}: {module_name} (type: {module_type})")
        
        self.obstacles = self.problem.get_obstacles()
        LOG_INFO(f"Number of obstacles: {len(self.obstacles) if self.obstacles else 0}")
        
        self.data = self.problem.get_data()
        if self.data:
            LOG_INFO("Data object provided")
            # Log data fields
            data_fields = [attr for attr in dir(self.data) if not attr.startswith('_')]
            LOG_DEBUG(f"Data has {len(data_fields)} attributes")
            key_fields = ['horizon', 'timestep', 'dynamics_model', 'reference_path', 'robot_area', 'dynamic_obstacles']
            for field in key_fields:
                has_field = hasattr(self.data, field)
                field_val = getattr(self.data, field, None)
                if field == 'robot_area' and field_val is not None:
                    # Log detailed robot_area information
                    disc_info = []
                    for i, disc in enumerate(field_val):
                        if hasattr(disc, 'offset') and hasattr(disc, 'radius'):
                            disc_info.append(f"disc[{i}]: offset={disc.offset:.3f}, radius={disc.radius:.3f}")
                    LOG_INFO(f"  data.robot_area: {len(field_val)} disc(s) - {', '.join(disc_info)}")
                else:
                    LOG_DEBUG(f"  data.{field}: {'present' if has_field else 'missing'}, value={'set' if field_val is not None else 'None'}")
        else:
            LOG_WARN("No data object provided")
        
        self.x0 = self.problem.get_x0()
        if self.x0 is not None:
            LOG_INFO("Initial state (x0) provided")
            state_vars = {var: self.x0.get(var) for var in self.x0.get_all_vars() if self.x0.has(var)}
            LOG_DEBUG(f"Initial state variables: {list(state_vars.keys())}")
        else:
            LOG_DEBUG("No initial state provided, will create default")
        
        # Initialize state from x0 if provided
        if self.x0 is not None:
            self.state = self.x0
            LOG_DEBUG("Using provided x0 as initial state")
        else:
            # Create default state if not provided
            self.state = State(self.model_type)
            LOG_DEBUG("Created default state from model_type")
        
        solver_type = self.config.get("solver", {}).get("solver", "casadi") if isinstance(self.config.get("solver"), dict) else self.config.get("solver", "casadi")
        LOG_INFO(f"Creating solver: {solver_type}")
        if solver_type == "casadi":
            self.solver = CasADiSolver(self.config)
            LOG_DEBUG("CasADiSolver created successfully")
        else:
            raise ValueError("Solver not supported")
        
        # Add modules to solver's module manager
        LOG_INFO("Adding modules to solver's module manager...")
        for module in self.modules:
            module_name = getattr(module, 'name', 'Unknown')
            LOG_DEBUG(f"Adding module '{module_name}' to solver module manager")
            self.solver.module_manager.add_module(module)
        
        # Also add to planner's module manager for compatibility
        LOG_INFO("Adding modules to planner's module manager...")
        self.module_manager.add_modules(self.modules)
        
        objective_modules = self.module_manager.get_objective_modules()
        if objective_modules is None:
            LOG_WARN("No objective modules found!")
            raise ValueError("No objective modules found")
        else:
            LOG_INFO(f"Found {len(objective_modules)} objective module(s)")
            for obj_mod in objective_modules:
                LOG_DEBUG(f"  Objective: {getattr(obj_mod, 'name', 'Unknown')}")
        
        if self.module_manager.check_dependencies() is False:
            LOG_WARN("Module dependencies are not satisfied!")
            raise ValueError("Dependencies are not satisfied")
        else:
            LOG_DEBUG("All module dependencies satisfied")

        LOG_INFO("Adding obstacles to obstacle manager...")
        if self.data is not None:
                if self.obstacles:
                        # Use obstacle manager's create_obstacles_from_config if available
                        if hasattr(self.obstacle_manager, 'create_obstacles_from_config'):
                                # Convert obstacles to config format if needed
                                obstacle_configs = []
                                for i, obs in enumerate(self.obstacles):
                                        from planning.obstacle_manager import ObstacleConfig, PredictionType
                                        obs_radius = getattr(obs, 'radius', 0.5)  # Default radius
                                        LOG_DEBUG(f"  Obstacle {i}: position={getattr(obs, 'position', 'unknown')}, radius={obs_radius:.3f}")
                                        obstacle_configs.append(ObstacleConfig(
                                                obstacle_id=i,
                                                initial_position=obs.position if hasattr(obs, 'position') else np.array([0.0, 0.0]),
                                                initial_velocity=obs.velocity if hasattr(obs, 'velocity') else np.array([1.0, 0.0]),
                                                dynamics_type="unicycle"  # Default, may need to detect from obstacle type
                                        ))
                                self.data.dynamic_obstacles = self.obstacle_manager.create_obstacles_from_config(obstacle_configs)
                                # Log obstacle radii after creation
                                for i, obs in enumerate(self.data.dynamic_obstacles):
                                        obs_radius = getattr(obs, 'radius', 'unknown')
                                        LOG_DEBUG(f"  Created obstacle {i}: radius={obs_radius}")
                        else:
                                # Fallback: just use obstacles directly
                                self.data.dynamic_obstacles = self.obstacles if self.obstacles else []
                                # Log obstacle radii
                                for i, obs in enumerate(self.data.dynamic_obstacles):
                                        obs_radius = getattr(obs, 'radius', 'unknown')
                                        LOG_DEBUG(f"  Obstacle {i}: radius={obs_radius}")
                else:
                        self.data.dynamic_obstacles = []
                LOG_INFO(f"Obstacle manager now has {len(self.data.dynamic_obstacles) if self.data.dynamic_obstacles else 0} obstacles")
        else:
                LOG_WARN("Cannot add obstacles: data is None")
        
        # Store data reference in solver and set required attributes
        self.solver.data = self.data
        LOG_DEBUG("Data reference stored in solver")
        
        # Set dynamics model on solver
        if hasattr(self.solver, 'set_dynamics_model'):
            self.solver.set_dynamics_model(self.model_type)
            LOG_DEBUG("Dynamics model set via set_dynamics_model method (should be in data)")
        else:
            # DO NOT set solver.dynamics_model directly - it should only come from data
            # Ensure data has dynamics_model set instead
            if self.data and hasattr(self.data, 'dynamics_model'):
                self.data.dynamics_model = self.model_type
                LOG_DEBUG("Dynamics model set in data.dynamics_model")
            else:
                LOG_WARN("Cannot set dynamics_model: data is None or has no dynamics_model attribute")
        
        # Ensure data has horizon and dynamics_model for solver compatibility
        if self.data:
            self.data.horizon = self.problem.get_horizon()
            self.data.timestep = self.problem.get_timestep()
            self.data.dynamics_model = self.model_type
            LOG_INFO(f"Data configured: horizon={self.data.horizon}, timestep={self.data.timestep}")
        else:
            LOG_WARN("Cannot set data attributes: data is None")
        
        LOG_INFO("Problem setup complete")

    
    def solve(self):
        LOG_INFO("=== Starting Planner.solve() ===")
        solver_iterations = 0
        max_iterations = self.config.get("solver_iterations", 1)
        LOG_INFO(f"Maximum solver iterations: {max_iterations}")

        while solver_iterations < max_iterations:
            LOG_INFO(f"--- Solver iteration {solver_iterations + 1}/{max_iterations} ---")
            
            # Log current state before solve
            current_x = self.state.get('x') if self.state.has('x') else None
            current_y = self.state.get('y') if self.state.has('y') else None
            current_psi = self.state.get('psi') if self.state.has('psi') else None
            current_v = self.state.get('v') if self.state.has('v') else None
            current_spline = self.state.get('spline') if self.state.has('spline') else None
            LOG_INFO(f"Current state before solve: x={current_x:.2f}, y={current_y:.2f}, psi={current_psi:.2f}, v={current_v:.2f}" + 
                            (f", spline={current_spline:.3f}" if current_spline is not None else ""))
            
            mpc_output = self.solve_mpc()
            
            if mpc_output.success:
                LOG_DEBUG(f"Iteration {solver_iterations + 1}: MPC solve successful")
                if hasattr(mpc_output, 'control') and mpc_output.control:
                    LOG_INFO(f"Control output: {mpc_output.control}")
                    # Log angular velocity specifically
                    if 'w' in mpc_output.control:
                        LOG_INFO(f"  Angular velocity w = {mpc_output.control['w']:.4f} rad/s")
                    if 'a' in mpc_output.control:
                        LOG_INFO(f"  Acceleration a = {mpc_output.control['a']:.4f} m/s²")
                else:
                    LOG_WARN(f"Iteration {solver_iterations + 1}: MPC solved but no control output!")
                
                if mpc_output.control:
                    self.output.control_history.append(mpc_output.control)
                    # Log state before propagation
                    before_x = self.state.get('x') if self.state.has('x') else None
                    before_y = self.state.get('y') if self.state.has('y') else None
                    
                    # Propagate state using dynamics model (ensuring spline/state updates match solver)
                    dynamics_model = None
                    if hasattr(self.solver, 'dynamics_model') and self.solver.dynamics_model is not None:
                        dynamics_model = self.solver.dynamics_model
                    else:
                        dynamics_model = self.model_type
                    parameter_manager = getattr(self.solver, 'parameter_manager', None)
                    
                    self.state = self.problem.get_state().propagate(
                        mpc_output.control,
                        self.solver.timestep,
                        dynamics_model=dynamics_model,
                        parameter_manager=parameter_manager
                    )
                    
                    # Log state after propagation
                    after_x = self.state.get('x') if self.state.has('x') else None
                    after_y = self.state.get('y') if self.state.has('y') else None
                    after_psi = self.state.get('psi') if self.state.has('psi') else None
                    after_v = self.state.get('v') if self.state.has('v') else None
                    after_spline = self.state.get('spline') if self.state.has('spline') else None
                    
                    LOG_INFO(f"State propagated: ({before_x:.2f}, {before_y:.2f}) -> ({after_x:.2f}, {after_y:.2f})")
                    LOG_DEBUG(f"  After: x={after_x:.2f}, y={after_y:.2f}, psi={after_psi:.2f}, v={after_v:.2f}" + 
                                    (f", spline={after_spline:.3f}" if after_spline is not None else ""))
                    
                    self.data.update(self.state)
                    self.output.realized_trajectory.add_state(self.state)
                else:
                    LOG_WARN(f"Iteration {solver_iterations + 1}: No control to apply, skipping propagation")
                
                solver_iterations += 1
                
                if self.state.is_objective_reached(self.data):
                    LOG_INFO(f"Objective reached at iteration {solver_iterations}!")
                    break
            else:
                LOG_WARN(f"Iteration {solver_iterations + 1}: MPC solve failed")
                break
        
        LOG_INFO(f"Planner.solve() completed after {solver_iterations} iterations")
        return self.output
    
    def solve_mpc(self, data=None):
        """Solve MPC optimization problem.
        
        Args:
            data: Optional Data object. If provided, updates self.data.
        """
        LOG_INFO("=== Starting Planner.solve_mpc() ===")
        
        # Update data if provided
        if data is not None:
            LOG_DEBUG("Updating data from provided argument")
            self.data = data
            # Log goal if available
            if hasattr(data, 'goal') and data.goal is not None:
                LOG_INFO(f"Planner.solve_mpc: data.goal = ({data.goal[0]:.3f}, {data.goal[1]:.3f})")
            if hasattr(data, 'parameters') and data.parameters is not None:
                goal_x = data.parameters.get("goal_x")
                goal_y = data.parameters.get("goal_y")
                if goal_x is not None and goal_y is not None:
                    LOG_INFO(f"Planner.solve_mpc: parameters.goal = ({goal_x:.3f}, {goal_y:.3f})")
        else:
            LOG_DEBUG("Using existing self.data")
        
        # Ensure state is set in data - data is the single source of truth
        if self.data is not None:
            if not hasattr(self.data, 'state') or self.data.state is None:
                if self.state is not None:
                    self.data.state = self.state
                    LOG_DEBUG("Set data.state from planner.state")
            else:
                # Update planner.state to match data.state (data is authoritative)
                self.state = self.data.state
                LOG_DEBUG("Updated planner.state from data.state")
            
            # Log current state
            if self.state is not None:
                state_x = self.state.get('x') if self.state.has('x') else None
                state_y = self.state.get('y') if self.state.has('y') else None
                LOG_INFO(f"Planner.solve_mpc: current state = ({state_x:.3f}, {state_y:.3f})")
        
        # Ensure solver horizon and timestep are set
        if not hasattr(self.solver, 'horizon') or self.solver.horizon is None:
            planner_config = self.config.get("planner", {})
            self.solver.horizon = planner_config.get("horizon", 10)
            self.solver.timestep = planner_config.get("timestep", 0.1)
            LOG_DEBUG(f"Solver horizon/timestep set from config: {self.solver.horizon}/{self.solver.timestep}")
        else:
            LOG_DEBUG(f"Solver horizon/timestep already set: {self.solver.horizon}/{self.solver.timestep}")

        LOG_INFO("Checking data readiness for all modules...")
        modules = self.solver.module_manager.get_modules()
        data_ready_status = {}
        for module in modules:
            module_name = getattr(module, 'name', 'Unknown')
            is_ready = module.is_data_ready(self.data)
            data_ready_status[module_name] = is_ready
            LOG_DEBUG(f"  Module '{module_name}': data_ready={is_ready}")
        
        is_data_ready = all(data_ready_status.values())
        if not is_data_ready:
            not_ready = [name for name, ready in data_ready_status.items() if not ready]
            LOG_WARN(f"Data not ready for modules: {not_ready}")
            return self.output
        else:
            LOG_INFO("All modules report data is ready")

        # Ensure state is in data before passing to solver
        if self.data is not None and (not hasattr(self.data, 'state') or self.data.state is None):
            self.data.state = self.state
            LOG_DEBUG("Set data.state from planner.state before solver initialization")

        # Ensure solver is initialized before calling initialize_rollout
        if not hasattr(self.solver, 'opti') or self.solver.opti is None:
            LOG_DEBUG("Initializing solver (opti is None)")
            if self.data and hasattr(self.data, 'dynamics_model') and self.data.dynamics_model:
                # DO NOT set solver.dynamics_model - it should only come from data
                self.solver.intialize_solver(self.data)
                LOG_DEBUG("Solver initialized with dynamics model from data")
            else:
                LOG_WARN("Cannot initialize solver: data or dynamics_model missing")
        else:
            LOG_DEBUG("Solver already initialized")
        
        LOG_INFO("Initializing rollout...")
        # Pass state via data - solver will get it from data.state
        # Still pass state parameter for backward compatibility, but data.state is authoritative
        self.solver.initialize_rollout(self.state, self.data)

        # Ensure horizon is set for propagate_obstacles
        horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
        timestep_val = self.solver.timestep if self.solver.timestep is not None else 0.1
        LOG_DEBUG(f"Propagating obstacles: horizon={horizon_val}, timestep={timestep_val}")
        # Call with Data first per function signature: propagate_obstacles(data, dt, horizon, ...)
        propagate_obstacles(self.data, dt=timestep_val, horizon=horizon_val)
        LOG_DEBUG("Obstacles propagated")

        LOG_INFO("Updating all modules...")
        # CRITICAL: Update modules in order, but ensure contouring constraints are updated LAST
        # This ensures contouring warmstart projection happens after obstacle avoidance projections
        # Reference: C++ mpc_planner - contouring constraints must be satisfied after all other projections
        all_modules = self.solver.module_manager.get_modules()
        contouring_modules = []
        other_modules = []

        # CRITICAL: Set solver reference on all modules BEFORE calling update()
        # Some modules (e.g., LinearizedConstraints) need solver.get_reference_trajectory() in their update()
        for module in all_modules:
            if not hasattr(module, 'solver') or module.solver is None:
                module.solver = self.solver

        for module in all_modules:
            module_name = getattr(module, 'name', 'Unknown')
            if module_name == 'contouring_constraints':
                contouring_modules.append(module)
            else:
                other_modules.append(module)

        # Update non-contouring modules first
        for module in other_modules:
            module_name = getattr(module, 'name', 'Unknown')
            LOG_DEBUG(f"Updating module '{module_name}'")
            module.update(self.state, self.data)
        
        # Update contouring modules last (ensures final warmstart respects road boundaries)
        for module in contouring_modules:
            module_name = getattr(module, 'name', 'Unknown')
            LOG_DEBUG(f"Updating module '{module_name}' (last, to ensure road boundary compliance)")
            module.update(self.state, self.data)

        # Set data for all stages including final stage (horizon + 1 stages: 0 to horizon)
        LOG_INFO(f"Setting parameters for {horizon_val + 1} stages...")
        # CRITICAL FIX: Use solver's parameter_manager, not planner's separate one
        # The solver's parameter_manager is what modules access during get_value()
        for k in range(horizon_val + 1):
            for module in self.solver.module_manager.get_modules():
                self.solver.parameter_manager.set_parameters(module, self.data, k)
                # Log goal parameters for goal objective at stage 0
                if k == 0 and hasattr(module, 'name') and module.name == 'goal_objective':
                    params_0 = self.solver.parameter_manager.get_all(0)
                    goal_x = params_0.get('goal_x')
                    goal_y = params_0.get('goal_y')
                    if goal_x is not None and goal_y is not None:
                        LOG_INFO(f"Planner: After set_parameters, goal_x={goal_x:.3f}, goal_y={goal_y:.3f} at stage 0")
        
        # CRITICAL: Constraints and objectives are now computed symbolically in the solver
        # The solver creates symbolic states for each stage and passes them to modules.
        # We no longer pre-collect constraints/objectives here since they must use symbolic states.
        # Reference: https://github.com/tud-amr/mpc_planner - all computation is symbolic in the solver.
        LOG_INFO(f"Parameters set for {horizon_val + 1} stages. Constraints and objectives will be computed symbolically in solver.")
        
        # Set data.parameters for backward compatibility (used by some modules)
        if not hasattr(self.data, 'parameters') or self.data.parameters is None:
            self.data.parameters = {}
        params_0 = self.solver.parameter_manager.get_all(0)
        self.data.parameters.update(params_0)
        # Log goal parameters
        if 'goal_x' in params_0 and 'goal_y' in params_0:
            LOG_INFO(f"Planner: Set data.parameters with goal_x={params_0['goal_x']:.3f}, goal_y={params_0['goal_y']:.3f}")

        LOG_INFO("Calling solver.solve()...")
        LOG_DEBUG(f"  Solving MPC over horizon={horizon_val}, timestep={timestep_val}")
        LOG_DEBUG(f"  Current state: x={self.state.get('x') if self.state.has('x') else 'N/A'}, y={self.state.get('y') if self.state.has('y') else 'N/A'}, v={self.state.get('v') if self.state.has('v') else 'N/A'}")
        exit_flag = self.solver.solve(self.state, self.data)

        if exit_flag != 1:
            self.output.success = False
            error_msg = self.solver.explain_exit_flag(exit_flag)
            LOG_WARN(f"MPC solve failed: {error_msg}")
            LOG_WARN("  No control extracted - solver did not find a solution")
            
            # CRITICAL: Diagnose solver failure to identify root cause
            LOG_WARN("=== SOLVER FAILURE DIAGNOSTICS ===")
            self._diagnose_solver_failure()
            
            # CRITICAL: Apply braking fallback when solver fails
            # Reference: C++ mpc_planner - generates deceleration trajectory on failure
            # This slows the vehicle safely while steering toward reference path
            if hasattr(self.solver, 'apply_braking_fallback'):
                safe_control = self.solver.apply_braking_fallback(self.state)
                self.output.control = safe_control
                LOG_WARN(f"  Applied braking fallback control: {safe_control}")
            elif hasattr(self.solver, 'dynamics_model') and self.solver.dynamics_model is not None:
                # Fallback if braking method not available: use zero control
                input_vars = self.solver.dynamics_model.get_inputs()
                safe_control = {}
                for u_name in input_vars:
                    safe_control[u_name] = 0.0
                self.output.control = safe_control
                LOG_WARN(f"  Set zero fallback control: {safe_control}")
            else:
                # Clear control if we can't determine inputs
                self.output.control = {}
                LOG_WARN("  Cleared control dict (no safe fallback available)")
            
            return self.output

        self.output.success = True
        LOG_INFO("=== MPC SOLVE SUCCESSFUL ===")
        LOG_INFO("  Solver found optimal trajectory over the horizon")
        
        # Get reference trajectory (optimal trajectory over horizon)
        reference_trajectory = self.solver.get_reference_trajectory()
        LOG_INFO(f"  Reference trajectory length: {len(reference_trajectory.get_states()) if hasattr(reference_trajectory, 'get_states') else 'unknown'}")
        
        # Log trajectory details for diagnosis
        if hasattr(reference_trajectory, 'get_states'):
            traj_states = reference_trajectory.get_states()
            if traj_states and len(traj_states) > 0:
                first_state = traj_states[0]
                last_state = traj_states[-1] if len(traj_states) > 1 else first_state
                first_x = first_state.get('x') if first_state.has('x') else None
                first_y = first_state.get('y') if first_state.has('y') else None
                last_x = last_state.get('x') if last_state.has('x') else None
                last_y = last_state.get('y') if last_state.has('y') else None
                LOG_INFO(f"  Trajectory: start=({first_x:.2f}, {first_y:.2f})" + 
                                (f", end=({last_x:.2f}, {last_y:.2f})" if last_x is not None and last_y is not None else ""))
                LOG_DEBUG(f"  Trajectory has {len(traj_states)} states (horizon + 1)")
                # Log first few trajectory points
                for i in range(min(3, len(traj_states))):
                    state_i = traj_states[i]
                    x_i = state_i.get('x') if state_i.has('x') else None
                    y_i = state_i.get('y') if state_i.has('y') else None
                    v_i = state_i.get('v') if state_i.has('v') else None
                    spline_i = state_i.get('spline') if state_i.has('spline') else None
                    # Log control inputs for first state (k=0)
                    if i == 0:
                        u_a = state_i.get('a') if state_i.has('a') else None
                        u_w = state_i.get('w') if state_i.has('w') else None
                        LOG_DEBUG(f"  Traj[{i}]: x={x_i:.2f}, y={y_i:.2f}, v={v_i:.2f}" + 
                                        (f", spline={spline_i:.3f}" if spline_i is not None else "") +
                                        (f", u=(a={u_a:.3f}, w={u_w:.3f})" if u_a is not None and u_w is not None else ""))
                    else:
                        LOG_DEBUG(f"  Traj[{i}]: x={x_i:.2f}, y={y_i:.2f}, v={v_i:.2f}" + 
                                        (f", spline={spline_i:.3f}" if spline_i is not None else ""))
        
        self.output.trajectory_history.append(reference_trajectory)

        if self.output.success and self.config.get("planner", {}).get("debug_limits", False):
            LOG_DEBUG("Checking variable bounds...")
            self.solver.print_if_bound_limited()

        # Extract first-step control from solver so caller can apply it
        # CRITICAL: This is the control input at k=0 (first time step) that should be applied
        LOG_INFO("Extracting first control input (k=0) from optimal trajectory...")
        try:
            control_out = {}
            # Get dynamics model from data (authoritative source)
            dynamics_model = None
            if hasattr(self.data, 'dynamics_model') and self.data.dynamics_model is not None:
                dynamics_model = self.data.dynamics_model
            elif hasattr(self.solver, 'dynamics_model') and self.solver.dynamics_model is not None:
                dynamics_model = self.solver.dynamics_model
            elif hasattr(self, 'model_type') and self.model_type is not None:
                dynamics_model = self.model_type

            if dynamics_model is not None:
                input_vars = dynamics_model.get_inputs()
                LOG_DEBUG(f"  Looking for control inputs: {input_vars}")
                LOG_DEBUG(f"  Extracting control at k=0 (first time step of horizon)")
                for u_name in input_vars:
                    # Get control input at k=0 (first time step)
                    # Input variables have shape [horizon], so k=0 is the first control
                    val0 = self.solver.get_output(0, u_name)
                    if val0 is not None:
                        control_out[u_name] = float(val0)
                        LOG_DEBUG(f"  Extracted u[{u_name}][0] = {float(val0):.4f}")
                        # Log w specifically for debugging turning
                        if u_name == 'w':
                            LOG_INFO(f"  ⚠️  Angular velocity w[0] = {float(val0):.4f} rad/s (vehicle should turn if non-zero)")
                    else:
                        LOG_WARN(f"  Could not extract {u_name} from solver (returned None)")
                        # Optional fallback: extract from trajectory if enabled in config
                        if self.config.get("planner", {}).get("fallback_control_enabled", False):
                            if hasattr(reference_trajectory, 'get_states') and len(reference_trajectory.get_states()) > 0:
                                first_state = reference_trajectory.get_states()[0]
                                if first_state.has(u_name):
                                    fallback_val = first_state.get(u_name)
                                    control_out[u_name] = float(fallback_val)
                                    LOG_DEBUG(f"  Fallback: extracted {u_name} from trajectory state: {float(fallback_val):.4f}")
            # Store on output if anything was found
            if control_out:
                self.output.control = control_out
                LOG_INFO(f"  ✓ First control input extracted: {control_out}")
                LOG_INFO(f"  This control will be applied to transition from current state to next state")
                
                # Log detailed control values for debugging vehicle movement
                if 'a' in control_out:
                    LOG_INFO(f"    Acceleration a[0] = {control_out['a']:.4f} m/s²")
                if 'w' in control_out:
                    LOG_INFO(f"    Angular velocity w[0] = {control_out['w']:.4f} rad/s")
                    if abs(control_out['w']) < 1e-6:
                        LOG_WARN(f"    ⚠️  Angular velocity is near zero - vehicle may not turn")
                if 'a' in control_out and abs(control_out['a']) < 1e-6:
                    LOG_WARN(f"    ⚠️  Acceleration is near zero - vehicle may not move forward")
            else:
                if dynamics_model is None:
                    LOG_WARN("  ⚠️  No control extracted - dynamics_model is None!")
                    LOG_WARN("  ⚠️  Check that data.dynamics_model is set correctly")
                else:
                    LOG_WARN("  ⚠️  No control extracted from solver - all inputs returned None!")
                    LOG_WARN("  ⚠️  Vehicle will NOT move - solver may have failed or solution is invalid")
        except Exception as e:
            LOG_WARN(f"Error extracting control: {e}")
            import traceback
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

        LOG_INFO("Planner.solve_mpc() completed successfully")
        return self.output

    def get_symbolic_dynamics(self, dynamics_model, x, u, timestep, data=None):
        """
        Provide symbolic next-state for the solver.
        - If the model exposes symbolic_dynamics(x,u,p,timestep), use it.
        - Otherwise, default to Euler: x_next = x + dt * f(x,u,p), where f comes from continuous_model.
        The parameter accessor p is provided as a callable if data has parameters.
        """
        # Parameter getter callable (modules expect callable get)
        def _param_getter(key):
            try:
                if data is not None and hasattr(data, 'parameters'):
                    return data.parameters.get(key)
            except Exception:
                pass
            return 0.0

        # Prefer model-provided symbolic dynamics if available
        if hasattr(dynamics_model, 'symbolic_dynamics') and callable(getattr(dynamics_model, 'symbolic_dynamics')):
            try:
                return dynamics_model.symbolic_dynamics(x, u, _param_getter, timestep)
            except Exception:
                # Fall back to Euler if model symbolic fails
                pass

        # Default Euler step using model's continuous dynamics
        f = dynamics_model.continuous_model(x, u, _param_getter)
        try:
            # x and f are casadi symbols; ensure same shape
            return x + timestep * f
        except Exception:
            # As last resort, return x (no motion) to avoid crash
            return x

    def get_numeric_dynamics(self, dynamics_model, x, u, timestep, data=None):
        """Numeric next-state using Euler: x_next = x + dt * f(x,u,p)."""
        def _param_getter(key):
            try:
                if data is not None and hasattr(data, 'parameters'):
                    return data.parameters.get(key)
            except Exception:
                pass
            return 0.0
        try:
            f = dynamics_model.continuous_model(x, u, _param_getter)
            return x + timestep * f
        except Exception as e:
            LOG_DEBUG(f"get_numeric_dynamics failed: {e}")
            return x

    def reset(self):

        self.solver.reset()
        self.state.reset()
        self.data.reset()

    def is_objective_reached(self, data):
        # Only check modules that have is_objective_reached method (e.g., GoalObjective, ContouringObjective)
        # Other objectives like ControlEffortObjective don't have this method
        objective_modules = [module for module in self.solver.module_manager.modules if module.module_type == OBJECTIVE]
        reached_results = []
        for module in objective_modules:
            if hasattr(module, 'is_objective_reached'):
                try:
                    reached = module.is_objective_reached(self.state, data)
                    reached_results.append(reached)
                except Exception as e:
                    LOG_DEBUG(f"Error checking objective reached for {module.name}: {e}")
                    # If a module can't check, don't block goal reaching
                    continue
        # Goal is reached if at least one objective module says it's reached
        # (typically only GoalObjective or ContouringObjective will have this method)
        return any(reached_results) if reached_results else False

    def visualize(self, stage_idx=1):
        LOG_DEBUG("Planner::visualize")
        for module in self.solver.module_manager.modules:
            if hasattr(module, 'get_visualizer'):
                visualizer = module.get_visualizer()
                if visualizer and hasattr(visualizer, 'visualize'):
                    visualizer.visualize(self.state, self.data, stage_idx)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state
        LOG_DEBUG("Planner::set_state")

    def _diagnose_solver_failure(self):
        """Diagnose solver failure to identify root cause.

        Delegates to planning.diagnostics module for detailed analysis.

        Checks:
        - Constraint violations at current state
        - Warmstart feasibility
        - Which constraints are violated (Gaussian vs contouring)
        - Current vehicle position vs obstacles
        - Current vehicle position vs path boundaries
        """
        from planning.diagnostics import diagnose_solver_failure
        diagnose_solver_failure(self)
        return  # The code below is kept for reference but not executed

        # Legacy code kept for reference - now handled by planning.diagnostics
        LOG_WARN("=== SOLVER FAILURE DIAGNOSTICS ===")
        
        # Check current vehicle state
        try:
            vehicle_x = float(self.state.get('x')) if self.state.has('x') else None
            vehicle_y = float(self.state.get('y')) if self.state.has('y') else None
            vehicle_psi = float(self.state.get('psi')) if self.state.has('psi') else None
            vehicle_v = float(self.state.get('v')) if self.state.has('v') else None
            
            if vehicle_x is not None and vehicle_y is not None:
                psi_str = f"{vehicle_psi:.3f}" if vehicle_psi is not None else "N/A"
                v_str = f"{vehicle_v:.3f}" if vehicle_v is not None else "N/A"
                LOG_WARN(f"Current vehicle state: x={vehicle_x:.3f}, y={vehicle_y:.3f}, psi={psi_str}, v={v_str}")
            else:
                LOG_WARN("Cannot get vehicle position for diagnostics")
                return
        except Exception as e:
            LOG_WARN(f"Error getting vehicle state: {e}")
            return
        
        # Check Gaussian constraint violations
        try:
            LOG_WARN("--- Checking Gaussian Constraints ---")
            if hasattr(self.data, 'dynamic_obstacles') and self.data.dynamic_obstacles:
                from planning.types import PredictionType
                from scipy.stats import chi2
                
                gaussian_module = None
                for module in self.solver.module_manager.modules:
                    if hasattr(module, 'name') and module.name == 'gaussian_constraints':
                        gaussian_module = module
                        break
                
                if gaussian_module:
                    risk_level = float(gaussian_module.get_config_value("gaussian_constraints.risk_level", 0.05))
                    chi_squared_threshold = chi2.ppf(1.0 - risk_level, df=2)
                    robot_radius = float(gaussian_module.robot_radius) if gaussian_module.robot_radius else 0.5
                    
                    violations = []
                    for obs_id, obstacle in enumerate(self.data.dynamic_obstacles):
                        if (hasattr(obstacle, 'prediction') and obstacle.prediction is not None and
                                obstacle.prediction.type == PredictionType.GAUSSIAN and
                                hasattr(obstacle.prediction, 'steps') and len(obstacle.prediction.steps) > 0):
                            
                            pred_step = obstacle.prediction.steps[0]  # Current step
                            if hasattr(pred_step, 'position') and pred_step.position is not None:
                                mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
                                major_radius = float(getattr(pred_step, 'major_radius', 0.1))
                                minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
                                obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
                                safe_distance = robot_radius + obstacle_radius
                                
                                # Use effective covariance (matching constraint calculation)
                                sigma_x_eff = major_radius + safe_distance
                                sigma_y_eff = minor_radius + safe_distance
                                
                                # Compute Mahalanobis distance
                                diff = np.array([vehicle_x, vehicle_y]) - mean_pos
                                mahalanobis_dist_sq = (diff[0]**2 / sigma_x_eff**2) + (diff[1]**2 / sigma_y_eff**2)
                                
                                # Check violation
                                if mahalanobis_dist_sq < chi_squared_threshold:
                                    violation = chi_squared_threshold - mahalanobis_dist_sq
                                    violations.append((obs_id, violation, mean_pos, mahalanobis_dist_sq, chi_squared_threshold))
                    
                    if violations:
                        LOG_WARN(f"  ⚠️  Found {len(violations)} Gaussian constraint violation(s):")
                        for obs_id, violation, mean_pos, mah_dist_sq, threshold in violations:
                            euclidean_dist = np.linalg.norm(np.array([vehicle_x, vehicle_y]) - mean_pos)
                            LOG_WARN(f"    Obstacle {obs_id}: violation={violation:.6f}, "
                                            f"mahalanobis_dist_sq={mah_dist_sq:.3f} < threshold={threshold:.3f}, "
                                            f"euclidean_dist={euclidean_dist:.3f}m, obstacle_pos=({mean_pos[0]:.3f}, {mean_pos[1]:.3f})")
                    else:
                        LOG_WARN(f"  ✓ No Gaussian constraint violations at current position")
        except Exception as e:
            LOG_WARN(f"Error checking Gaussian constraints: {e}")
            import traceback
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
        
        # Check contouring constraint violations
        try:
            LOG_WARN("--- Checking Contouring Constraints ---")
            if hasattr(self.data, 'reference_path') and self.data.reference_path is not None:
                contouring_module = None
                for module in self.solver.module_manager.modules:
                    if hasattr(module, 'name') and module.name == 'contouring_constraints':
                        contouring_module = module
                        break
                
                if contouring_module and hasattr(contouring_module, '_reference_path') and contouring_module._reference_path is not None:
                    ref_path = contouring_module._reference_path
                    
                    # Get current spline value
                    current_spline = self.state.get('spline') if self.state.has('spline') else None
                    if current_spline is None:
                        # Estimate from closest point
                        if hasattr(ref_path, 's') and ref_path.s is not None:
                            s_arr = np.asarray(ref_path.s, dtype=float)
                            if s_arr.size > 0:
                                s_sample = np.linspace(s_arr[0], s_arr[-1], min(200, len(s_arr)))
                                x_sample = []
                                y_sample = []
                                for s_val in s_sample:
                                    try:
                                        if hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None:
                                            x_val = float(ref_path.x_spline(s_val))
                                            y_val = float(ref_path.y_spline(s_val))
                                            x_sample.append(x_val)
                                            y_sample.append(y_val)
                                    except:
                                        continue
                                if len(x_sample) > 0:
                                    distances = np.sqrt((np.array(x_sample) - vehicle_x)**2 + (np.array(y_sample) - vehicle_y)**2)
                                    closest_idx = np.argmin(distances)
                                    current_spline = float(s_sample[closest_idx])
                    
                    if current_spline is not None:
                        # Evaluate path at current spline
                        try:
                            if hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None:
                                path_x = float(ref_path.x_spline(current_spline))
                                path_y = float(ref_path.y_spline(current_spline))
                                path_dx = float(ref_path.x_spline.derivative()(current_spline))
                                path_dy = float(ref_path.y_spline.derivative()(current_spline))
                                
                                # Normalize tangent
                                norm = np.hypot(path_dx, path_dy)
                                if norm > 1e-9:
                                    path_dx_norm = path_dx / norm
                                    path_dy_norm = path_dy / norm
                                    
                                    # Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
                                    A = np.array([path_dy_norm, -path_dx_norm])
                                    path_point = np.array([path_x, path_y])
                                    vehicle_pos = np.array([vehicle_x, vehicle_y])
                                    
                                    # Contour error
                                    contour_error = np.dot(A, vehicle_pos - path_point)
                                    
                                    # Get road width
                                    width_half = contouring_module._road_width_half if contouring_module._road_width_half is not None else 3.5
                                    robot_radius = 0.5  # Default
                                    if hasattr(self.data, 'robot_area') and self.data.robot_area and len(self.data.robot_area) > 0:
                                        robot_radius = float(self.data.robot_area[0].radius)
                                    
                                    w_cur = robot_radius
                                    width_left = width_half
                                    width_right = width_half
                                    
                                    # Check violations
                                    # RIGHT boundary: contour_error >= -width_right + w_cur
                                    right_violation = (-width_right + w_cur) - contour_error if contour_error < (-width_right + w_cur) else 0.0
                                    # LEFT boundary: contour_error <= width_left - w_cur
                                    left_violation = contour_error - (width_left - w_cur) if contour_error > (width_left - w_cur) else 0.0
                                    
                                    if right_violation > 1e-6 or left_violation > 1e-6:
                                        LOG_WARN(f"  ⚠️  Contouring constraint violation detected:")
                                        LOG_WARN(f"    Contour error: {contour_error:.3f}m")
                                        LOG_WARN(f"    Allowed range: [{(-width_right + w_cur):.3f}, {(width_left - w_cur):.3f}]")
                                        if right_violation > 1e-6:
                                            LOG_WARN(f"    RIGHT boundary violation: {right_violation:.3f}m (vehicle too far RIGHT)")
                                        if left_violation > 1e-6:
                                            LOG_WARN(f"    LEFT boundary violation: {left_violation:.3f}m (vehicle too far LEFT)")
                                        LOG_WARN(f"    Path point: ({path_x:.3f}, {path_y:.3f}), spline={current_spline:.3f}")
                                    else:
                                        LOG_WARN(f"  ✓ No contouring constraint violations at current position")
                        except Exception as e:
                            LOG_WARN(f"Error evaluating contouring constraints: {e}")
        except Exception as e:
            LOG_WARN(f"Error checking contouring constraints: {e}")
            import traceback
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
        
        # Check warmstart feasibility at future stages
        try:
            LOG_WARN("--- Checking Warmstart Feasibility (Future Stages) ---")
            if hasattr(self.solver, 'warmstart_values') and self.solver.warmstart_values:
                ws_vals = self.solver.warmstart_values
                if 'x' in ws_vals and 'y' in ws_vals and len(ws_vals['x']) > 0:
                    ws_x = float(ws_vals['x'][0])
                    ws_y = float(ws_vals['y'][0])
                    ws_dist = np.hypot(ws_x - vehicle_x, ws_y - vehicle_y)
                    LOG_WARN(f"  Warmstart stage 0: ({ws_x:.3f}, {ws_y:.3f})")
                    LOG_WARN(f"  Current position: ({vehicle_x:.3f}, {vehicle_y:.3f})")
                    LOG_WARN(f"  Distance: {ws_dist:.3f}m")
                    if ws_dist > 1.0:
                        LOG_WARN(f"  ⚠️  Warmstart is far from current position - may cause infeasibility")
                    
                    # Check future stages for constraint violations
                    horizon = self.solver.horizon if hasattr(self.solver, 'horizon') and self.solver.horizon is not None else 10
                    max_check_stages = min(5, horizon + 1, len(ws_vals['x']))
                    
                    LOG_WARN(f"  Checking warmstart at future stages (0-{max_check_stages-1}):")
                    for stage_idx in range(1, max_check_stages):
                        if stage_idx < len(ws_vals['x']) and stage_idx < len(ws_vals['y']):
                            ws_x_future = float(ws_vals['x'][stage_idx])
                            ws_y_future = float(ws_vals['y'][stage_idx])
                            
                            # Check Gaussian constraints at this future stage
                            if hasattr(self.data, 'dynamic_obstacles') and self.data.dynamic_obstacles:
                                from planning.types import PredictionType
                                from scipy.stats import chi2
                                
                                gaussian_module = None
                                for module in self.solver.module_manager.modules:
                                    if hasattr(module, 'name') and module.name == 'gaussian_constraints':
                                        gaussian_module = module
                                        break
                                
                                if gaussian_module:
                                    risk_level = float(gaussian_module.get_config_value("gaussian_constraints.risk_level", 0.05))
                                    chi_squared_threshold = chi2.ppf(1.0 - risk_level, df=2)
                                    robot_radius = float(gaussian_module.robot_radius) if gaussian_module.robot_radius else 0.5
                                    
                                    violations_future = []
                                    for obs_id, obstacle in enumerate(self.data.dynamic_obstacles):
                                        if (hasattr(obstacle, 'prediction') and obstacle.prediction is not None and
                                                obstacle.prediction.type == PredictionType.GAUSSIAN and
                                                hasattr(obstacle.prediction, 'steps') and len(obstacle.prediction.steps) > stage_idx):
                                            
                                            pred_step = obstacle.prediction.steps[stage_idx]
                                            if hasattr(pred_step, 'position') and pred_step.position is not None:
                                                mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
                                                major_radius = float(getattr(pred_step, 'major_radius', 0.1))
                                                minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
                                                obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
                                                safe_distance = robot_radius + obstacle_radius
                                                
                                                sigma_x_eff = major_radius + safe_distance
                                                sigma_y_eff = minor_radius + safe_distance
                                                
                                                diff = np.array([ws_x_future, ws_y_future]) - mean_pos
                                                mahalanobis_dist_sq = (diff[0]**2 / sigma_x_eff**2) + (diff[1]**2 / sigma_y_eff**2)
                                                
                                                if mahalanobis_dist_sq < chi_squared_threshold:
                                                    violation = chi_squared_threshold - mahalanobis_dist_sq
                                                    violations_future.append((obs_id, violation, mean_pos))
                                    
                                    if violations_future:
                                        LOG_WARN(f"    Stage {stage_idx}: ⚠️  {len(violations_future)} Gaussian violation(s) at warmstart position ({ws_x_future:.3f}, {ws_y_future:.3f})")
                                        for obs_id, violation, mean_pos in violations_future:
                                            LOG_WARN(f"      Obstacle {obs_id}: violation={violation:.6f}, obstacle_pos=({mean_pos[0]:.3f}, {mean_pos[1]:.3f})")
                            
                            # Check contouring constraints at this future stage
                            if hasattr(self.data, 'reference_path') and self.data.reference_path is not None:
                                contouring_module = None
                                for module in self.solver.module_manager.modules:
                                    if hasattr(module, 'name') and module.name == 'contouring_constraints':
                                        contouring_module = module
                                        break
                                
                                if contouring_module and hasattr(contouring_module, '_reference_path') and contouring_module._reference_path is not None:
                                    ref_path = contouring_module._reference_path
                                    
                                    # Estimate spline value for this stage
                                    if 'spline' in ws_vals and stage_idx < len(ws_vals['spline']):
                                        ws_spline = float(ws_vals['spline'][stage_idx])
                                        
                                        try:
                                            if hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None:
                                                path_x = float(ref_path.x_spline(ws_spline))
                                                path_y = float(ref_path.y_spline(ws_spline))
                                                path_dx = float(ref_path.x_spline.derivative()(ws_spline))
                                                path_dy = float(ref_path.y_spline.derivative()(ws_spline))
                                                
                                                norm = np.hypot(path_dx, path_dy)
                                                if norm > 1e-9:
                                                    path_dx_norm = path_dx / norm
                                                    path_dy_norm = path_dy / norm
                                                    
                                                    A = np.array([path_dy_norm, -path_dx_norm])
                                                    path_point = np.array([path_x, path_y])
                                                    ws_pos = np.array([ws_x_future, ws_y_future])
                                                    
                                                    contour_error = np.dot(A, ws_pos - path_point)
                                                    
                                                    width_half = contouring_module._road_width_half if contouring_module._road_width_half is not None else 3.5
                                                    robot_radius = 0.5
                                                    if hasattr(self.data, 'robot_area') and self.data.robot_area and len(self.data.robot_area) > 0:
                                                        robot_radius = float(self.data.robot_area[0].radius)
                                                    
                                                    w_cur = robot_radius
                                                    width_left = width_half
                                                    width_right = width_half
                                                    
                                                    right_violation = (-width_right + w_cur) - contour_error if contour_error < (-width_right + w_cur) else 0.0
                                                    left_violation = contour_error - (width_left - w_cur) if contour_error > (width_left - w_cur) else 0.0
                                                    
                                                    if right_violation > 1e-6 or left_violation > 1e-6:
                                                        LOG_WARN(f"    Stage {stage_idx}: ⚠️  Contouring violation at warmstart ({ws_x_future:.3f}, {ws_y_future:.3f}), "
                                                                        f"contour_error={contour_error:.3f}, allowed=[{(-width_right + w_cur):.3f}, {(width_left - w_cur):.3f}]")
                                        except Exception:
                                            pass
        except Exception as e:
            LOG_WARN(f"Error checking warmstart future stages: {e}")
            import traceback
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
        
        # Check solver-reported constraint violations and identify which constraints
        try:
            LOG_WARN("--- Checking Solver-Reported Constraint Violations ---")
            if hasattr(self.solver, 'opti') and self.solver.opti is not None:
                if hasattr(self.solver.opti, 'debug') and hasattr(self.solver.opti.debug, 'g'):
                    try:
                        constraint_values = self.solver.opti.debug.value(self.solver.opti.debug.g)
                        if constraint_values is not None:
                            constraint_values_arr = np.array(constraint_values)
                            positive_violations = constraint_values_arr[constraint_values_arr > 1e-6]
                            if len(positive_violations) > 0:
                                max_violation = np.max(positive_violations)
                                max_idx = np.argmax(constraint_values_arr)
                                LOG_WARN(f"  ⚠️  Found {len(positive_violations)} constraint(s) with positive values (VIOLATIONS)")
                                LOG_WARN(f"  ⚠️  Maximum violation: {max_violation:.6f} at constraint index {max_idx}")
                                
                                # Identify constraint types
                                dynamics_model = self.solver._get_dynamics_model() if hasattr(self.solver, '_get_dynamics_model') else None
                                horizon = self.solver.horizon if hasattr(self.solver, 'horizon') and self.solver.horizon is not None else 10
                                
                                if dynamics_model:
                                    num_state_vars = len(dynamics_model.get_dependent_vars())
                                    num_dynamics = num_state_vars * horizon  # 5 × 10 = 50
                                    num_initial_state = num_state_vars  # 5
                                    
                                    LOG_WARN(f"  Constraint structure: dynamics={num_dynamics}, initial_state={num_initial_state}, module=varies")
                                    
                                    # Categorize violations
                                    dynamics_violations = []
                                    initial_state_violations = []
                                    module_violations = []
                                    
                                    positive_indices = np.where(constraint_values_arr > 1e-6)[0]
                                    for idx in positive_indices:
                                        val = float(constraint_values_arr[idx])
                                        if idx < num_dynamics:
                                            transition_idx = idx // num_state_vars
                                            state_idx = idx % num_state_vars
                                            state_vars = dynamics_model.get_dependent_vars()
                                            var_name = state_vars[state_idx] if state_idx < len(state_vars) else f"state_{state_idx}"
                                            dynamics_violations.append((idx, val, transition_idx, var_name))
                                        elif idx < num_dynamics + num_initial_state:
                                            initial_idx = idx - num_dynamics
                                            state_vars = dynamics_model.get_dependent_vars()
                                            var_name = state_vars[initial_idx] if initial_idx < len(state_vars) else f"state_{initial_idx}"
                                            initial_state_violations.append((idx, val, var_name))
                                        else:
                                            module_violations.append((idx, val))
                                    
                                    if dynamics_violations:
                                        LOG_WARN(f"  ⚠️  DYNAMICS constraint violations: {len(dynamics_violations)}")
                                        for idx, val, trans_idx, var_name in dynamics_violations[:5]:
                                            LOG_WARN(f"    Constraint {idx}: {val:.6f} [Dynamics: {var_name} at transition {trans_idx}->{trans_idx+1}]")
                                    
                                    if initial_state_violations:
                                        LOG_WARN(f"  ⚠️  INITIAL STATE constraint violations: {len(initial_state_violations)}")
                                        for idx, val, var_name in initial_state_violations[:5]:
                                            LOG_WARN(f"    Constraint {idx}: {val:.6f} [Initial state: {var_name}[0]]")
                                    
                                    if module_violations:
                                        LOG_WARN(f"  ⚠️  MODULE constraint violations: {len(module_violations)}")
                                        LOG_WARN(f"    First 5 module violations:")
                                        for idx, val in module_violations[:5]:
                                            LOG_WARN(f"      Constraint {idx}: {val:.6f} [Module constraint - need to identify type]")
                                    
                                    # Identify the maximum violation
                                    if max_idx < num_dynamics:
                                        transition_idx = max_idx // num_state_vars
                                        state_idx = max_idx % num_state_vars
                                        state_vars = dynamics_model.get_dependent_vars()
                                        var_name = state_vars[state_idx] if state_idx < len(state_vars) else f"state_{state_idx}"
                                        LOG_WARN(f"  ⚠️  MAX VIOLATION: Dynamics constraint for {var_name} at transition {transition_idx}->{transition_idx+1}")
                                        LOG_WARN(f"      This means warmstart values don't satisfy dynamics: {var_name}[{transition_idx+1}] != RK4({var_name}[{transition_idx}], u[{transition_idx}], dt)")
                                        
                                        # Check warmstart values for this transition
                                        if hasattr(self.solver, 'warmstart_values') and var_name in self.solver.warmstart_values:
                                            ws_vals = self.solver.warmstart_values[var_name]
                                            if transition_idx < len(ws_vals) and transition_idx + 1 < len(ws_vals):
                                                LOG_WARN(f"      Warmstart: {var_name}[{transition_idx}]={ws_vals[transition_idx]:.6f}, {var_name}[{transition_idx+1}]={ws_vals[transition_idx+1]:.6f}")
                                    elif max_idx < num_dynamics + num_initial_state:
                                        initial_idx = max_idx - num_dynamics
                                        state_vars = dynamics_model.get_dependent_vars()
                                        var_name = state_vars[initial_idx] if initial_idx < len(state_vars) else f"state_{initial_idx}"
                                        LOG_WARN(f"  ⚠️  MAX VIOLATION: Initial state constraint for {var_name}[0]")
                                        LOG_WARN(f"      This means warmstart doesn't match current state")
                                        
                                        # Check actual values
                                        current_val = self.state.get(var_name) if self.state.has(var_name) else None
                                        if hasattr(self.solver, 'warmstart_values') and var_name in self.solver.warmstart_values:
                                            ws_val = self.solver.warmstart_values[var_name][0] if len(self.solver.warmstart_values[var_name]) > 0 else None
                                            LOG_WARN(f"      Current state {var_name}: {current_val}")
                                            LOG_WARN(f"      Warmstart {var_name}[0]: {ws_val}")
                                            if current_val is not None and ws_val is not None:
                                                diff = abs(float(current_val) - float(ws_val))
                                                LOG_WARN(f"      Difference: {diff:.6f} (violation: {max_violation:.6f})")
                                                if diff > 1e-6:
                                                    LOG_WARN(f"      ⚠️  WARMSTART MISMATCH: {var_name}[0] should be {current_val} but is {ws_val}")
                                    else:
                                        LOG_WARN(f"  ⚠️  MAX VIOLATION: Module constraint at index {max_idx}")
                                        LOG_WARN(f"      This is likely a Gaussian or Contouring constraint violation")
                            else:
                                LOG_WARN(f"  ✓ No positive constraint violations in solver debug.g")
                    except Exception as e:
                        LOG_WARN(f"  Could not get constraint values from solver: {e}")
                        import traceback
                        LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
        except Exception as e:
            LOG_WARN(f"Error checking solver constraint violations: {e}")
            import traceback
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
        
        LOG_WARN("=== END DIAGNOSTICS ===")


class PlannerOutput:
    def __init__(self):
        self.success = False
        self.control_history = [] # The control history is a list of the actual control inputs used during the execution
        self.trajectory_history = [] # This is a list of horizon length trajectories which is added to each time the solver finds a solution
        self.realized_trajectory = Trajectory() # this is the trajectory executed by the physical robot when integrating based on the control history
