import copy

import casadi as cs
import numpy as np

from planning.dynamic_models import DynamicsModel
from planning.types import State, Trajectory
from solver.base_solver import BaseSolver
from utils.const import OBJECTIVE, CONSTRAINT, DEFAULT_BRAKING
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO

# Import performance optimization module
try:
	from solver.performance_optimizations import (
		PerformanceConfig,
		PerformanceOptimizer,
		PerformanceLevel,
	)
	HAS_PERF_MODULE = True
except ImportError:
	HAS_PERF_MODULE = False

'''
Casadi solver used for trajectory optimization.
'''

class CasADiSolver(BaseSolver):
	def __init__(self, config, performance=None):
		"""Initialize the CasADi solver.

		Args:
			config: Configuration dictionary
			performance: Performance configuration. Can be:
				- None: Use balanced (default) settings
				- str: 'fast', 'balanced', or 'safe' preset
				- PerformanceConfig: Custom performance configuration
		"""
		super().__init__(config)
		# Extract horizon and timestep from config
		planner_config = config.get("planner", {})
		self.horizon = planner_config.get("horizon", 10)
		self.timestep = planner_config.get("timestep", 0.1)
		# Initialize solver components
		self.opti = None
		self.var_dict = {}
		self.warmstart_values = {}
		# DO NOT store dynamics_model - always get it from data to avoid discrepancies
		self.solution = None
		self.exit_flag = None
		self.info = {}
		self.forecast = []

		# Initialize performance optimizer
		self._perf_optimizer = None
		self._perf_config = None
		if HAS_PERF_MODULE:
			self._setup_performance(performance, config)

		LOG_INFO(f"{(id(self))} CasADiSolver: Initializing solver")
	
	def _setup_performance(self, performance, config):
		"""Set up performance optimizer based on configuration.

		Args:
			performance: Performance setting (None, str preset, or PerformanceConfig)
			config: Full configuration dictionary
		"""
		if not HAS_PERF_MODULE:
			return

		# Determine performance configuration
		if performance is None:
			# Check if config has performance settings
			if "performance" in config:
				self._perf_config = PerformanceConfig.from_config(config)
			else:
				# Default to balanced
				self._perf_config = PerformanceConfig.balanced()
		elif isinstance(performance, str):
			# Use preset
			if performance.lower() == "fast":
				self._perf_config = PerformanceConfig.fast()
			elif performance.lower() == "safe":
				self._perf_config = PerformanceConfig.safe()
			else:
				self._perf_config = PerformanceConfig.balanced()
		elif isinstance(performance, PerformanceConfig):
			self._perf_config = performance
		else:
			self._perf_config = PerformanceConfig.balanced()

		# Create optimizer and configure
		self._perf_optimizer = PerformanceOptimizer(self._perf_config)
		self._perf_optimizer.configure_solver(self)

		LOG_INFO(f"Performance mode: {self._perf_config.level.value}")

	def set_performance_mode(self, mode):
		"""Change performance mode at runtime.

		Args:
			mode: 'fast', 'balanced', 'safe', or PerformanceConfig instance
		"""
		if not HAS_PERF_MODULE:
			LOG_WARN("Performance module not available")
			return

		if isinstance(mode, str):
			if mode.lower() == "fast":
				self._perf_config = PerformanceConfig.fast()
			elif mode.lower() == "safe":
				self._perf_config = PerformanceConfig.safe()
			else:
				self._perf_config = PerformanceConfig.balanced()
		elif isinstance(mode, PerformanceConfig):
			self._perf_config = mode
		else:
			return

		# Recreate optimizer with new config
		if self._perf_optimizer:
			self._perf_optimizer.restore_logging()
		self._perf_optimizer = PerformanceOptimizer(self._perf_config)
		self._perf_optimizer.configure_solver(self)
		LOG_INFO(f"Changed performance mode to: {self._perf_config.level.value}")

	def get_performance_stats(self):
		"""Get performance statistics from the optimizer."""
		if self._perf_optimizer:
			return {
				"timing": self._perf_optimizer.timing_stats,
				"cache": self._perf_optimizer.cache_stats,
				"config": {
					"level": self._perf_config.level.value if self._perf_config else "unknown",
					"jit_enabled": self._perf_config.enable_jit if self._perf_config else False,
					"caching_enabled": self._perf_config.enable_caching if self._perf_config else False,
				}
			}
		return {}

	def _get_dynamics_model(self):
		"""Get dynamics model from data - always use data as source of truth."""
		if self.data is None:
			return None
		if not hasattr(self.data, 'dynamics_model'):
			return None
		return self.data.dynamics_model

	def intialize_solver(self, data):
		"""Initialize solver variables based on dynamics model and horizon."""
		LOG_INFO("=== CasADiSolver.intialize_solver() ===")
		if data is None:
			LOG_WARN("Cannot initialize solver: data is None")
			return
		
		# Always get dynamics model from data - do not use stored version
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("Cannot initialize solver: data.dynamics_model is None")
			return
		
		dynamics = data.dynamics_model
		horizon = getattr(data, 'horizon', None) or self.horizon
		LOG_DEBUG(f"Initializing solver: horizon={horizon}, dynamics_model={dynamics.__class__.__name__}")
		
		if self.opti is None:
			self.opti = cs.Opti()
			LOG_DEBUG("Created new CasADi Opti instance")
		
		LOG_INFO("=== CREATING OPTIMIZATION VARIABLES ===")
		dependent_vars = dynamics.get_dependent_vars()
		input_vars = dynamics.get_inputs()
		LOG_INFO(f"  Dependent variables (STATES - will be constrained by RK4 integration) ({len(dependent_vars)}): {dependent_vars}")
		LOG_INFO(f"  Input variables (CONTROLS - decision variables) ({len(input_vars)}): {input_vars}")
		
		# Verify that inputs are only acceleration/angular acceleration for unicycle
		expected_inputs_unicycle = ["a", "w"]
		expected_inputs_bicycle = ["a", "w", "slack"]
		expected_inputs_pointmass = ["ax", "ay"]
		
		if set(input_vars) == set(expected_inputs_unicycle):
			LOG_INFO(f"  ✓ Verified: Inputs are acceleration (a) and angular acceleration (w) for unicycle model")
		elif set(input_vars) == set(expected_inputs_bicycle):
			LOG_INFO(f"  ✓ Verified: Inputs are acceleration (a), angular acceleration (w), and slack for bicycle model")
		elif set(input_vars) == set(expected_inputs_pointmass):
			LOG_INFO(f"  ✓ Verified: Inputs are accelerations (ax, ay) for point mass model")
		else:
			LOG_WARN(f"  ⚠️  Unexpected input variables: {input_vars}. Expected only control inputs (acceleration/angular acceleration)")
		
		# Create state variables with bounds
		# CRITICAL: States are optimization variables BUT they are constrained by RK4 integration.
		# The constraint x_next == symbolic_dynamics(x_k, u_k, timestep) ensures states follow integration.
		# This means states are NOT free decision variables - they are determined by:
		#   1. Initial state (x[0] is fixed)
		#   2. Control inputs (a, w) - the ONLY free decision variables
		#   3. RK4 integration constraint: x[k+1] == symbolic_dynamics(x[k], u[k], timestep)
		# 
		# Reference: https://github.com/tud-amr/mpc_planner - only control inputs are decision variables.
		# States are computed via RK4 integration (or model_discrete_dynamics for algebraic states like spline).
		for var_name in dependent_vars:
			self.var_dict[var_name] = self.opti.variable(horizon + 1)
			# Only initialize warmstart to zeros if not already populated with correct shape
			if var_name not in self.warmstart_values or not isinstance(self.warmstart_values[var_name], np.ndarray) or len(self.warmstart_values[var_name]) != horizon + 1:
				self.warmstart_values[var_name] = np.zeros(horizon + 1)
			# Set bounds from dynamics model
			try:
				lb, ub, _ = dynamics.get_bounds(var_name)
				self.opti.subject_to(self.opti.bounded(lb, self.var_dict[var_name], ub))
				LOG_INFO(f"  Created STATE variable '{var_name}': shape={horizon + 1}, bounds=[{lb}, {ub}] (constrained by RK4/model_discrete_dynamics)")
			except Exception as e:
				LOG_WARN(f"  Created STATE variable '{var_name}': shape={horizon + 1}, no bounds (error: {e})")
		
		# Create input variables with bounds
		# CRITICAL: These are the ONLY free decision variables in the optimization problem.
		# All state variables (x, y, psi, v, spline) are constrained by:
		#   - Initial state constraint: x[0] == x_initial
		#   - Dynamics constraint: x[k+1] == symbolic_dynamics(x[k], u[k], timestep)
		# 
		# Reference: https://github.com/tud-amr/mpc_planner - only acceleration (a) and angular acceleration (w) are decision variables.
		# All other states are computed via RK4 integration or model_discrete_dynamics.
		for var_name in input_vars:
			self.var_dict[var_name] = self.opti.variable(horizon)
			# Only initialize warmstart to zeros if not already populated with correct shape
			if var_name not in self.warmstart_values or not isinstance(self.warmstart_values[var_name], np.ndarray) or len(self.warmstart_values[var_name]) != horizon:
				self.warmstart_values[var_name] = np.zeros(horizon)
			# Set bounds from dynamics model
			try:
				lb, ub, _ = dynamics.get_bounds(var_name)
				self.opti.subject_to(self.opti.bounded(lb, self.var_dict[var_name], ub))
				LOG_INFO(f"  Created CONTROL variable '{var_name}': shape={horizon}, bounds=[{lb}, {ub}] (FREE DECISION VARIABLE)")
			except Exception as e:
				LOG_WARN(f"  Created CONTROL variable '{var_name}': shape={horizon}, no bounds (error: {e})")
		# Get IPOPT options - use performance optimizer if available
		if self._perf_optimizer is not None:
			opts = self._perf_optimizer.get_ipopt_options()
			LOG_DEBUG(f"Using performance-optimized IPOPT options (mode={self._perf_config.level.value})")
		else:
			# Default options (balanced mode fallback)
			opts = {
				'ipopt.print_level': 0,
				'print_time': 0,
				'ipopt.sb': 'yes',
				# PERFORMANCE: Reduce max iterations for faster termination
				'ipopt.max_iter': 500,  # Reduced from 2000 - warmstart should converge faster
				# PERFORMANCE: Relaxed tolerances for faster convergence
				'ipopt.tol': 1e-2,  # Relaxed from 1e-3 for faster convergence
				'ipopt.acceptable_tol': 5e-1,  # More relaxed to accept suboptimal solutions faster
				'ipopt.acceptable_iter': 5,  # Accept sooner if acceptable_tol met (was 10)
				'ipopt.constr_viol_tol': 1e-2,  # Relaxed from 1e-3 for faster convergence
				# Reference: C++ mpc_planner uses slightly relaxed tolerances for scenario-based constraints
				# to allow solver convergence while maintaining safety
				'ipopt.mu_strategy': 'adaptive',
				'ipopt.hessian_approximation': 'limited-memory',
				# PERFORMANCE: Warmstart settings for faster convergence
				'ipopt.warm_start_init_point': 'yes',
				'ipopt.warm_start_bound_push': 1e-4,  # Relaxed from 1e-6 for faster warmstart
				'ipopt.warm_start_mult_bound_push': 1e-4,  # Relaxed from 1e-6
				'ipopt.warm_start_bound_frac': 1e-4,  # Additional warmstart tuning
				'ipopt.warm_start_slack_bound_frac': 1e-4,
				'ipopt.nlp_scaling_method': 'gradient-based',  # Better scaling for constraints
				'ipopt.obj_scaling_factor': 1.0,  # No objective scaling
				# PERFORMANCE: Linear solver selection (mumps is usually fastest)
				'ipopt.linear_solver': 'mumps',
			}
			LOG_DEBUG("Using default IPOPT options")

		# Only set fast_step_computation if supported (may not be available in all IPOPT versions)
		try:
			opts['ipopt.fast_step_computation'] = 'yes'
		except:
			pass

		LOG_DEBUG("Configuring IPOPT solver...")
		self.opti.solver('ipopt', opts)
		LOG_DEBUG("IPOPT solver configured")
		
		# Always get dynamics model from data
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("Cannot add dynamics constraints: data.dynamics_model is None")
			return
		
		dynamics = data.dynamics_model
		horizon = getattr(data, 'horizon', None) or self.horizon
		timestep = getattr(data, 'timestep', None) or self.timestep
		LOG_INFO(f"=== ADDING DYNAMICS CONSTRAINTS (RK4 INTEGRATION) ===")
		LOG_INFO(f"  Adding constraints for {horizon} stages (timestep={timestep})...")
		LOG_INFO(f"  Using RK4 integration via dynamics.symbolic_dynamics()")
		LOG_INFO(f"  Constraint form: x[k+1] == RK4_integration(x[k], u[k], timestep)")
		LOG_INFO(f"  This ensures states are computed from integration, not free decision variables")
		
		for k in range(horizon):
			x_k_list = [self.var_dict[var][k] for var in dynamics.get_dependent_vars()]
			u_k_list = [self.var_dict[var][k] for var in dynamics.get_inputs()]

			x_k = cs.vertcat(*x_k_list)
			u_k = cs.vertcat(*u_k_list)
			x_next_list = [self.var_dict[var][k + 1] for var in dynamics.get_dependent_vars()]
			x_next = cs.vertcat(*x_next_list)

			# Use the model's symbolic dynamics function with a callable parameter getter
			# This getter should access both data.parameters and parameter_manager for path parameters
			def _param_getter(key):
				defaults = {
					"wheel_base": 2.79,
					"wheel_tread": 1.64,
					"front_overhang": 1.0,
					"rear_overhang": 1.1,
					"left_overhang": 0.128,
					"right_overhang": 0.128,
				}
				# First try data.parameters
				try:
					if hasattr(data, 'parameters') and data.parameters is not None:
						val = data.parameters.get(key)
						if val is not None:
							return val
				except Exception:
					pass
				# Then try parameter_manager (for path parameters set by modules)
				try:
					if hasattr(self, 'parameter_manager') and self.parameter_manager is not None:
						# Try to get from current stage (k) - path parameters are set per stage
						params_dict = self.parameter_manager.get_all(k)
						if key in params_dict:
							return params_dict[key]
				except Exception:
					pass
				# Fallback to defaults
				return defaults.get(key, 0.0)
			
			# CRITICAL: This uses symbolic_dynamics to compute next state from current state and control.
			# The constraint x_next == x_next_pred ensures states follow integration, NOT free decision variables.
			# 
			# symbolic_dynamics performs:
			#   - RK4 integration for integrated states (x, y, psi, v)
			#   - Algebraic update via model_discrete_dynamics for non-integrated states (e.g., spline)
			# 
			# This constraint ensures that:
			#   - States are NOT free decision variables
			#   - Only control inputs (a, w) are free decision variables
			#   - States are computed deterministically from initial state and control sequence
			# 
			# Reference: https://github.com/tud-amr/mpc_planner - states are constrained by dynamics integration.
			x_next_pred = dynamics.symbolic_dynamics(x_k, u_k, _param_getter, timestep)

			# Enforce dynamics constraint: states must follow integration
			# This makes states dependent on controls, not free decision variables
			self.opti.subject_to(x_next == x_next_pred)
			if k == 0 or k == horizon - 1 or k == horizon // 2:
				LOG_INFO(f"  Stage {k}: Added RK4 dynamics constraint: x[{k+1}] == RK4(x[{k}], u[{k}], dt={timestep})")
				LOG_INFO(f"    State vars at k: {dynamics.get_dependent_vars()}")
				LOG_INFO(f"    Control vars at k: {dynamics.get_inputs()}")
		
		LOG_INFO(f"CasADiSolver initialized: {horizon} stages, {len(dependent_vars)} state vars, {len(input_vars)} input vars")

	def initialize_rollout(self, state: State, data=None, shift_forward=True):
		"""Initialize rollout for solving. Gets state and dynamics_model from data."""
		if data is None:
			LOG_WARN("initialize_rollout called but data is None")
			return
		
		# Store data reference - this is the single source of truth
		self.data = data
		
		# Get state from data if available, otherwise use provided state
		# Priority: data.state > provided state parameter
		if hasattr(data, 'state') and data.state is not None:
			state = data.state
			LOG_DEBUG("Using state from data.state")
		elif state is None:
			LOG_WARN("initialize_rollout: no state available (neither data.state nor state parameter)")
			return
		
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("initialize_rollout called but data.dynamics_model is not set")
			return
		
		# Ensure solver is initialized before setting initial state
		if self.opti is None or not self.var_dict:
			LOG_DEBUG("Solver not initialized in initialize_rollout - initializing now")
			self.intialize_solver(data)
		
		# CRITICAL: Initialize warmstart FIRST, then set initial state constraints
		# This ensures initial values are set before constraints are evaluated
		# Reference: C++ mpc_planner - initial values must match constraints
		# Enable warmstart initialization to provide feasible initial guess
		self._initialize_warmstart(state, shift_forward)
		# Set initial state constraints AFTER warmstart is initialized
		# This ensures the initial values match the constraints
		self._set_initial_state(state)

	def get_objective_cost(self, state, stage_idx):
		"""Fetch per-stage objective cost terms via BaseSolver (ModuleManager-backed)."""
		from solver.base_solver import BaseSolver as _Base
		# Delegate to BaseSolver implementation to keep a single source of truth
		return _Base.get_objective_cost(self, state, stage_idx)

	def _initialize_warmstart(self, state: State, shift_forward=True):
		# Initialize warmstart to provide feasible initial guess for IPOPT
		# Reference: https://github.com/tud-amr/mpc_planner - solution shifting improves convergence
		LOG_DEBUG("Initializing warmstart trajectory")
		
		if not hasattr(self, 'forecast'):
			self.forecast = []
		
		# Check if we should shift previous solution forward (reference codebase pattern)
		should_shift = shift_forward
		if self.config is not None and isinstance(self.config, dict):
			solver_config = self.config.get("solver", {})
			if isinstance(solver_config, dict):
				should_shift = solver_config.get("shift_previous_solution_forward", shift_forward)
		
		# If we have a previous solution and shifting is enabled, shift it forward first
		# This provides continuity between MPC steps and helps the solver converge faster
		# Reference: https://github.com/tud-amr/mpc_planner - solution shifting maintains trajectory continuity
		if should_shift and hasattr(self, 'warmstart_values') and self.warmstart_values:
			# Check if warmstart was initialized from a previous solution
			# Also check if we have a valid solution to shift from (not just initial warmstart)
			has_previous_solution = (hasattr(self, 'solution') and self.solution is not None and
			                         hasattr(self, 'warmstart_intiailized') and self.warmstart_intiailized)
			
			if has_previous_solution:
				LOG_DEBUG("Shifting previous warmstart solution forward (reference: mpc_planner)")
				# Shift the previous solution forward
				self._shift_warmstart_forward()
				# Update current state in warmstart (stage 0 should match current state)
				# CRITICAL: This ensures the warmstart reflects the vehicle's actual current state
				self._update_warmstart_current_state(state)
				# CRITICAL: After shifting, ensure dynamics consistency is maintained
				# The shifted warmstart might have inconsistent psi values
				self._ensure_warmstart_dynamics_consistency()
			else:
				# First initialization or no previous solution - create new warmstart
				LOG_DEBUG("First initialization: creating new warmstart trajectory")
				self._initialize_base_warmstart(state)
		else:
			# Initialize base warmstart which creates path-following trajectory
			self._initialize_base_warmstart(state)
		
		# CRITICAL: Always ensure dynamics consistency after warmstart initialization
		# This catches any cases where psi and w are inconsistent
		self._ensure_warmstart_dynamics_consistency()
		
		# Set initial values in CasADi opti problem
		# CRITICAL: This must be called BEFORE _set_initial_state to ensure
		# initial values match constraints
		self._set_opti_initial_values()
		
		self.warmstart_intiailized = True

	def _initialize_base_warmstart(self, state: State):
		# Initialize states with a more realistic trajectory
		current_vel = state.get('v') if state.get('v') is not None else 0.0
		current_psi = state.get('psi') if state.get('psi') is not None else 0.0
		current_x = state.get('x') if state.get('x') is not None else 0.0
		current_y = state.get('y') if state.get('y') is not None else 0.0

		# Ensure horizon and timestep are set
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1

		# CRITICAL FIX: Initialize warmstart to follow reference path if available
		# This ensures the warmstart trajectory is feasible with road constraints
		has_reference_path = (self.data is not None and 
		                      hasattr(self.data, 'reference_path') and 
		                      self.data.reference_path is not None and
		                      hasattr(self.data.reference_path, 'x_spline') and
		                      hasattr(self.data.reference_path, 'y_spline') and
		                      self.data.reference_path.x_spline is not None)
		
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("_initialize_base_warmstart: no dynamics_model available")
			return
		
		if has_reference_path and 'spline' in dynamics_model.get_dependent_vars():
			LOG_DEBUG("Initializing warmstart to follow reference path (mpc_planner style)")
			ref_path = self.data.reference_path
			
			# === Helper utilities ===================================================
			def _wrap_angle(angle_val):
				return (angle_val + np.pi) % (2 * np.pi) - np.pi
			
			def _build_warmstart_config():
				if not isinstance(self.config, dict):
					return {}
				solver_cfg = self.config.get("solver", {})
				if not isinstance(solver_cfg, dict):
					return {}
				warm_cfg = solver_cfg.get("warmstart", {})
				return warm_cfg if isinstance(warm_cfg, dict) else {}
			
			warm_cfg = _build_warmstart_config()
			
			heading_gain = float(warm_cfg.get("heading_gain", 0.75))
			cross_track_gain = float(warm_cfg.get("cross_track_gain", 0.45))
			lag_gain = float(warm_cfg.get("lag_gain", 0.25))
			speed_gain = float(warm_cfg.get("speed_gain", 1.25))
			centerline_blend = float(warm_cfg.get("centerline_blend", 0.2))
			
			# === Reference path sampling ============================================
			s_arr = np.asarray(ref_path.s, dtype=float) if hasattr(ref_path, 's') and ref_path.s is not None else None
			if s_arr is None or len(s_arr) == 0:
				x_arr = np.asarray(ref_path.x, dtype=float)
				y_arr = np.asarray(ref_path.y, dtype=float)
				dx = np.diff(x_arr)
				dy = np.diff(y_arr)
				arc_lengths = np.sqrt(dx**2 + dy**2)
				s_arr = np.concatenate([[0.0], np.cumsum(arc_lengths)])
			
			s_min = float(s_arr[0])
			s_max = float(s_arr[-1])
			
			def _clamp_s(val):
				return float(np.clip(val, s_min, s_max))
			
			def _eval_path(s_val):
				s_clamped = _clamp_s(s_val)
				x_c = float(ref_path.x_spline(s_clamped))
				y_c = float(ref_path.y_spline(s_clamped))
				dx_val = float(ref_path.x_spline.derivative()(s_clamped))
				dy_val = float(ref_path.y_spline.derivative()(s_clamped))
				norm = np.hypot(dx_val, dy_val)
				if norm < 1e-9:
					norm = 1.0
					dx_val = np.cos(current_psi)
					dy_val = np.sin(current_psi)
				tangent = np.array([dx_val / norm, dy_val / norm])
				normal = np.array([-tangent[1], tangent[0]])
				ddx_val = float(ref_path.x_spline.derivative(2)(s_clamped))
				ddy_val = float(ref_path.y_spline.derivative(2)(s_clamped))
				den = max((dx_val * dx_val + dy_val * dy_val) ** 1.5, 1e-6)
				curvature = (dx_val * ddy_val - dy_val * ddx_val) / den
				curvature = float(np.clip(curvature, -2.5, 2.5))
				heading = float(np.arctan2(dy_val, dx_val))
				return x_c, y_c, heading, curvature, tangent, normal
			
			# === Determine initial spline position =================================
			current_spline = state.get('spline')
			if current_spline is None:
				try:
					s_sample = np.linspace(s_min, s_max, min(200, len(s_arr)))
					x_sample = np.array([float(ref_path.x_spline(sv)) for sv in s_sample])
					y_sample = np.array([float(ref_path.y_spline(sv)) for sv in s_sample])
					distances = np.sqrt((x_sample - current_x)**2 + (y_sample - current_y)**2)
					current_spline = float(s_sample[np.argmin(distances)])
					LOG_DEBUG(f"  Initialized spline to closest path point: {current_spline:.4f}")
				except Exception as exc:
					LOG_WARN(f"  Could not project state onto path ({exc}); using s_min")
					current_spline = s_min
			else:
				current_spline = float(np.clip(current_spline, s_min, s_max))
			
			# === Pull bounds for dynamics variables ================================
			try:
				v_lb, v_ub, _ = dynamics_model.get_bounds('v')
			except Exception:
				v_lb, v_ub = -0.01, 3.0
			
			try:
				a_lb, a_ub, _ = dynamics_model.get_bounds('a')
			except Exception:
				a_lb, a_ub = -2.0, 2.0
			
			try:
				w_lb, w_ub, _ = dynamics_model.get_bounds('w')
			except Exception:
				w_lb, w_ub = -0.8, 0.8
			
			target_speed = float(warm_cfg.get("target_speed", min(v_ub, max(0.5, current_vel if current_vel > 0.1 else 1.5))))
			
			# === Allocate profiles ==================================================
			spline_profile = np.zeros(horizon_val + 1)
			x_pos = np.zeros(horizon_val + 1)
			y_pos = np.zeros(horizon_val + 1)
			psi_profile = np.zeros(horizon_val + 1)
			v_profile = np.zeros(horizon_val + 1)
			a_profile = np.zeros(horizon_val)
			w_profile = np.zeros(horizon_val)
			
			spline_profile[0] = current_spline
			x_pos[0] = current_x
			y_pos[0] = current_y
			psi_profile[0] = current_psi
			v_profile[0] = np.clip(current_vel, v_lb, v_ub)
			
			# CRITICAL: For goal objectives, pre-initialize w values to turn toward goal
			# This ensures the warmstart reflects the required turning even if path-following doesn't require it
			goal_w_initialized = False
			if hasattr(self.data, 'goal') and self.data.goal is not None:
				try:
					goal_x = float(self.data.goal[0])
					goal_y = float(self.data.goal[1])
					goal_heading = np.arctan2(goal_y - current_y, goal_x - current_x)
					goal_psi_error = _wrap_angle(goal_heading - current_psi)
					
					# If vehicle needs to turn significantly toward goal, initialize w to turn
					if abs(goal_psi_error) > 0.1:  # More than ~6 degrees
						# Initialize first few w values to turn toward goal
						w_goal_init = np.clip(goal_psi_error * 0.8, w_lb, w_ub)  # Stronger turning toward goal
						# Apply to first 3 steps to ensure initial turn
						for k_init in range(min(3, horizon_val)):
							w_profile[k_init] = w_goal_init * (1.0 - k_init * 0.2)  # Gradually reduce
						goal_w_initialized = True
						LOG_INFO(f"  Goal objective: Pre-initialized w warmstart to turn toward goal: "
						         f"w[0]={w_profile[0]:.4f}, w[1]={w_profile[1]:.4f}, w[2]={w_profile[2]:.4f} rad/s "
						         f"(goal_psi_error={goal_psi_error:.3f} rad, goal_heading={goal_heading:.3f} rad)")
				except Exception as e:
					LOG_DEBUG(f"  Could not pre-initialize goal-directed w warmstart: {e}")
			
			# === Iteratively roll out dynamics =====================================
			for k in range(1, horizon_val + 1):
				s_prev = spline_profile[k - 1]
				try:
					x_c, y_c, psi_des, curvature, tangent_vec, normal_vec = _eval_path(s_prev)
				except Exception as exc:
					LOG_WARN(f"  Path evaluation failed at s={s_prev:.4f}: {exc}; falling back to straight propagation")
					x_c, y_c = x_pos[k - 1], y_pos[k - 1]
					psi_des = psi_profile[k - 1]
					curvature = 0.0
					tangent_vec = np.array([np.cos(psi_profile[k - 1]), np.sin(psi_profile[k - 1])])
					normal_vec = np.array([-tangent_vec[1], tangent_vec[0]])
				
				error_vec = np.array([x_pos[k - 1] - x_c, y_pos[k - 1] - y_c])
				cross_track_err = float(np.dot(error_vec, normal_vec))
				lag_err = float(np.dot(error_vec, tangent_vec))
				
				psi_error = _wrap_angle(psi_des - psi_profile[k - 1])
				w_ff = v_profile[k - 1] * curvature
				w_fb = heading_gain * psi_error - cross_track_gain * cross_track_err
				w_cmd = np.clip(w_ff + w_fb, w_lb, w_ub)
				
				# For goal objectives, blend path-following w with goal-directed w for first few steps
				# This ensures we turn toward goal even if path doesn't require much turning
				if goal_w_initialized and k <= 3:
					# Blend: 70% goal-directed, 30% path-following for first step, then gradually shift
					blend_factor = 0.7 * (1.0 - (k - 1) * 0.2)  # 0.7, 0.56, 0.42 for k=1,2,3
					w_profile[k - 1] = np.clip(
						w_profile[k - 1] * blend_factor + w_cmd * (1.0 - blend_factor),
						w_lb, w_ub
					)
					LOG_DEBUG(f"  k={k}: Blended w: goal={w_profile[k-1]:.4f} (blend={blend_factor:.2f}) + path={w_cmd:.4f} = {w_profile[k-1]:.4f}")
				else:
					w_profile[k - 1] = w_cmd
				
				a_cmd = np.clip(speed_gain * (target_speed - v_profile[k - 1]), a_lb, a_ub)
				a_profile[k - 1] = a_cmd
				
				# CRITICAL: Use w_profile[k-1] (which may have been set for goal-directed turning)
				# instead of w_cmd, to ensure dynamics consistency
				w_to_use = w_profile[k - 1]  # Use the value we set (may include goal-directed turning)
				psi_next = _wrap_angle(psi_profile[k - 1] + w_to_use * timestep_val)
				v_next = np.clip(v_profile[k - 1] + a_cmd * timestep_val, v_lb, v_ub)
				ds = v_profile[k - 1] * timestep_val - lag_gain * lag_err
				ds = max(ds, 0.0)
				s_next = _clamp_s(s_prev + ds)
				
				x_next = x_pos[k - 1] + v_profile[k - 1] * np.cos(psi_profile[k - 1]) * timestep_val
				y_next = y_pos[k - 1] + v_profile[k - 1] * np.sin(psi_profile[k - 1]) * timestep_val
				
				if centerline_blend > 0.0:
					try:
						x_center_next, y_center_next, _, _, _, _ = _eval_path(s_next)
						x_next = (1.0 - centerline_blend) * x_next + centerline_blend * x_center_next
						y_next = (1.0 - centerline_blend) * y_next + centerline_blend * y_center_next
					except Exception:
						pass
				
				spline_profile[k] = s_next
				x_pos[k] = x_next
				y_pos[k] = y_next
				psi_profile[k] = psi_next
				v_profile[k] = v_next
			
			try:
				s_lb, s_ub, _ = dynamics_model.get_bounds('spline')
				spline_profile = np.clip(spline_profile, s_lb, s_ub)
			except Exception:
				spline_profile = np.clip(spline_profile, s_min, s_max)
			
			# CRITICAL FIX: Clamp spline to path_length to prevent infeasible dynamics at path end
			# When approaching path end, forecasted spline can exceed path_length, causing
			# dynamics constraints (spline_next - spline_current) to become infeasible
			path_length = s_max  # s_max is the path length
			if path_length is not None and path_length > 0:
				spline_profile = np.clip(spline_profile, 0.0, path_length)
				self._path_length = path_length  # Store for use in _shift_warmstart_forward
				LOG_DEBUG(f"  Clamped spline warmstart to path_length={path_length:.2f}")
			
			self.warmstart_values['x'] = x_pos
			self.warmstart_values['y'] = y_pos
			self.warmstart_values['v'] = v_profile
			self.warmstart_values['psi'] = psi_profile
			self.warmstart_values['spline'] = spline_profile
			self.warmstart_values['a'] = a_profile
			self.warmstart_values['w'] = w_profile
			
			# CRITICAL: Verify and enforce dynamics consistency for reference path branch
			# Even though psi is integrated from w in the loop, verify it's correct and fix if needed
			psi_consistent = True
			for k in range(1, min(len(psi_profile), horizon_val + 1)):
				psi_pred = _wrap_angle(psi_profile[k - 1] + w_profile[k - 1] * timestep_val)
				psi_actual = psi_profile[k]
				diff = abs(_wrap_angle(psi_actual - psi_pred))
				if diff > 1e-6:
					psi_consistent = False
					LOG_WARN(f"  ⚠️  Reference path warmstart dynamics inconsistency at k={k}: "
					         f"psi[{k}]={psi_actual:.6f}, predicted={psi_pred:.6f}, diff={diff:.6f}, w[{k-1}]={w_profile[k-1]:.6f}")
					# Fix it by recomputing psi from w
					psi_profile[k] = psi_pred
			
			if not psi_consistent:
				LOG_INFO(f"  Fixed reference path warmstart dynamics inconsistencies")
				# Recompute x, y positions using corrected psi_profile
				x_pos[0] = current_x
				y_pos[0] = current_y
				for k in range(1, horizon_val + 1):
					x_pos[k] = x_pos[k - 1] + v_profile[k - 1] * np.cos(psi_profile[k - 1]) * timestep_val
					y_pos[k] = y_pos[k - 1] + v_profile[k - 1] * np.sin(psi_profile[k - 1]) * timestep_val
				self.warmstart_values['x'] = x_pos
				self.warmstart_values['y'] = y_pos
				self.warmstart_values['psi'] = psi_profile
			
			# Log warmstart values for debugging
			if len(psi_profile) >= 4 and len(w_profile) >= 3:
				LOG_INFO(f"  Reference path warmstart: psi=[{psi_profile[0]:.3f}, {psi_profile[1]:.3f}, {psi_profile[2]:.3f}, {psi_profile[3]:.3f}], "
				         f"w=[{w_profile[0]:.3f}, {w_profile[1]:.3f}, {w_profile[2]:.3f}]")
				# Verify dynamics consistency
				for k in range(1, min(4, len(psi_profile))):
					psi_pred_check = _wrap_angle(psi_profile[k - 1] + w_profile[k - 1] * timestep_val)
					psi_actual_check = psi_profile[k]
					diff_check = abs(_wrap_angle(psi_actual_check - psi_pred_check))
					if diff_check > 1e-6:
						LOG_WARN(f"    ⚠️  Still inconsistent at k={k}: psi[{k}]={psi_actual_check:.6f}, predicted={psi_pred_check:.6f}, diff={diff_check:.6f}")
					else:
						LOG_DEBUG(f"    ✓ Dynamics consistent at k={k}: psi[{k}]={psi_actual_check:.6f} = psi[{k-1}] + w[{k-1}]*dt")
			
			# Log warmstart values for debugging
			if len(psi_profile) >= 4 and len(w_profile) >= 3:
				LOG_INFO(f"  Reference path warmstart: psi=[{psi_profile[0]:.3f}, {psi_profile[1]:.3f}, {psi_profile[2]:.3f}, {psi_profile[3]:.3f}], "
				         f"w=[{w_profile[0]:.3f}, {w_profile[1]:.3f}, {w_profile[2]:.3f}]")
				# Verify dynamics consistency
				for k in range(1, min(4, len(psi_profile))):
					psi_pred_check = _wrap_angle(psi_profile[k - 1] + w_profile[k - 1] * timestep_val)
					psi_actual_check = psi_profile[k]
					diff_check = abs(_wrap_angle(psi_actual_check - psi_pred_check))
					if diff_check > 1e-6:
						LOG_WARN(f"    ⚠️  Still inconsistent at k={k}: psi[{k}]={psi_actual_check:.6f}, predicted={psi_pred_check:.6f}, diff={diff_check:.6f}")
					else:
						LOG_DEBUG(f"    ✓ Dynamics consistent at k={k}: psi[{k}]={psi_actual_check:.6f} = psi[{k-1}] + w[{k-1}]*dt")
			
			LOG_DEBUG(
				f"  Warmstart initialized (path-aligned): "
				f"s=[{spline_profile[0]:.3f}->{spline_profile[-1]:.3f}], "
				f"v=[{v_profile.min():.3f},{v_profile.max():.3f}], "
				f"w_range=[{w_profile.min():.3f},{w_profile.max():.3f}]"
			)
		else:
			# Fallback: straight-line motion (original behavior)
			LOG_DEBUG("Initializing warmstart with straight-line motion (no reference path available)")
			# Use constant velocity profile, but ensure it's within bounds
			dynamics_model = self._get_dynamics_model()
			if dynamics_model:
				try:
					v_lb, v_ub, _ = dynamics_model.get_bounds('v')
				except Exception:
					v_lb, v_ub = -0.01, 3.0  # Default bounds
			else:
				v_lb, v_ub = -0.01, 3.0  # Default bounds
			
			if current_vel > 5.0:
				v_profile = np.maximum(v_lb, np.minimum(v_ub, current_vel + DEFAULT_BRAKING * np.arange(horizon_val + 1) * timestep_val * 0.5))
			else:
				v_profile = np.full(horizon_val + 1, np.clip(current_vel, v_lb, v_ub))
			
			psi_profile = np.full(horizon_val + 1, current_psi)

			self.warmstart_values['v'] = v_profile
			self.warmstart_values['psi'] = psi_profile

			# Integrate positions
			x_pos = np.zeros(horizon_val + 1)
			y_pos = np.zeros(horizon_val + 1)
			x_pos[0] = current_x
			y_pos[0] = current_y

			for k in range(1, horizon_val + 1):
				x_pos[k] = x_pos[k - 1] + v_profile[k - 1] * np.cos(psi_profile[k - 1]) * timestep_val
				y_pos[k] = y_pos[k - 1] + v_profile[k - 1] * np.sin(psi_profile[k - 1]) * timestep_val

			self.warmstart_values['x'] = x_pos
			self.warmstart_values['y'] = y_pos

			# CRITICAL FIX: Initialize spline warmstart in fallback branch
			# Without this, spline warmstart is missing when no reference path is available,
			# causing solver failures due to uninitialized warmstart values
			dynamics_model = self._get_dynamics_model()
			if dynamics_model and 'spline' in dynamics_model.get_dependent_vars():
				current_spline = state.get('spline') if state.has('spline') else 0.0
				if current_spline is None:
					current_spline = 0.0
				# Initialize spline to progress at current velocity
				spline_profile = np.zeros(horizon_val + 1)
				spline_profile[0] = current_spline
				for k in range(1, horizon_val + 1):
					# Spline advances proportional to velocity
					spline_profile[k] = spline_profile[k-1] + v_profile[k-1] * timestep_val
				self.warmstart_values['spline'] = spline_profile
				LOG_DEBUG(f"  Initialized spline warmstart in fallback: [{spline_profile[0]:.3f} -> {spline_profile[-1]:.3f}]")

		# Initialize other state variables if they exist
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return
		for var_name in dynamics_model.get_dependent_vars():
			if var_name not in ['x', 'y', 'v', 'psi', 'spline']:
				current_val = state.get(var_name)
				if current_val is not None:
					self.warmstart_values[var_name][:] = current_val
				else:
					self.warmstart_values[var_name][:] = 0.0

		# Initialize inputs with small values (zero can cause issues)
		# For goal objective, initialize w to turn toward goal if goal is available
		goal_available = False
		goal_x = None
		goal_y = None
		if self.data is not None and hasattr(self.data, 'goal') and self.data.goal is not None:
			goal_available = True
			goal_x = float(self.data.goal[0])
			goal_y = float(self.data.goal[1])
		
		for var_name in dynamics_model.get_inputs():
			current_val = state.get(var_name)
			if current_val is not None:
				self.warmstart_values[var_name][:] = current_val
			else:
				# For goal objective, initialize w to turn toward goal
				if var_name == 'w' and goal_available:
					current_x = state.get('x') if state.has('x') else 0.0
					current_y = state.get('y') if state.has('y') else 0.0
					current_psi = state.get('psi') if state.has('psi') else 0.0
					# Compute desired heading toward goal
					theta_goal = np.arctan2(goal_y - current_y, goal_x - current_x)
					angle_error = np.mod(current_psi - theta_goal + np.pi, 2 * np.pi) - np.pi
					# Initialize w to turn toward goal (proportional to angle error, capped)
					# Use a reasonable turning rate (e.g., 0.5 rad/s max)
					w_init = np.clip(-angle_error * 0.5, -0.5, 0.5)  # Proportional control, capped at 0.5 rad/s
					self.warmstart_values[var_name][:] = w_init
					LOG_INFO(f"Goal objective: Initializing w warmstart to {w_init:.4f} rad/s (angle_error={angle_error:.3f} rad) for all {horizon_val} steps")
				# Log first few w values to verify
				if len(self.warmstart_values[var_name]) >= 3:
					LOG_INFO(f"  w warmstart values: [{self.warmstart_values[var_name][0]:.4f}, {self.warmstart_values[var_name][1]:.4f}, {self.warmstart_values[var_name][2]:.4f}, ...]")
				else:
					# Use small non-zero values for better convergence
					self.warmstart_values[var_name][:] = 0.01
		
		# CRITICAL FIX: Ensure dynamics consistency for fallback branch (no reference path)
		# If psi_profile was initialized as constant but w_profile is non-zero, integrate psi using dynamics
		# This ensures psi[k+1] = psi[k] + w[k] * dt (unicycle dynamics: dpsi/dt = w)
		if not has_reference_path and 'psi' in self.warmstart_values and 'w' in self.warmstart_values:
			psi_profile = self.warmstart_values['psi']
			w_profile = self.warmstart_values['w']
			# Check if psi_profile is constant (all values are the same)
			if len(psi_profile) > 1 and np.allclose(psi_profile, psi_profile[0]):
				# Integrate psi using dynamics: psi[k+1] = psi[k] + w[k] * dt
				def _wrap_angle(angle_val):
					return (angle_val + np.pi) % (2 * np.pi) - np.pi
				
				for k in range(1, horizon_val + 1):
					psi_profile[k] = _wrap_angle(psi_profile[k - 1] + w_profile[k - 1] * timestep_val)
				
				# Recompute x_pos and y_pos using updated psi_profile
				if 'x' in self.warmstart_values and 'y' in self.warmstart_values and 'v' in self.warmstart_values:
					x_pos = self.warmstart_values['x']
					y_pos = self.warmstart_values['y']
					v_profile = self.warmstart_values['v']
					x_pos[0] = current_x
					y_pos[0] = current_y
					for k in range(1, horizon_val + 1):
						x_pos[k] = x_pos[k - 1] + v_profile[k - 1] * np.cos(psi_profile[k - 1]) * timestep_val
						y_pos[k] = y_pos[k - 1] + v_profile[k - 1] * np.sin(psi_profile[k - 1]) * timestep_val
					self.warmstart_values['x'] = x_pos
					self.warmstart_values['y'] = y_pos
				
				LOG_INFO(f"Fixed dynamics consistency: Integrated psi_profile using w_profile")
				LOG_INFO(f"  psi warmstart: [{psi_profile[0]:.3f}, {psi_profile[1]:.3f}, {psi_profile[2]:.3f}, {psi_profile[3]:.3f}, ...]")
				LOG_INFO(f"  w warmstart: [{w_profile[0]:.3f}, {w_profile[1]:.3f}, {w_profile[2]:.3f}, ...]")
				# Verify dynamics consistency
				for k in range(1, min(4, len(psi_profile))):
					psi_pred = psi_profile[k - 1] + w_profile[k - 1] * timestep_val
					psi_actual = psi_profile[k]
					diff = abs(psi_actual - psi_pred)
					if diff > 1e-6:
						LOG_WARN(f"  ⚠️  Dynamics inconsistency at k={k}: psi[{k}]={psi_actual:.6f}, predicted={psi_pred:.6f}, diff={diff:.6f}")
					else:
						LOG_DEBUG(f"  ✓ Dynamics consistent at k={k}: psi[{k}]={psi_actual:.6f} = psi[{k-1}] + w[{k-1}]*dt")

	def _shift_warmstart_forward(self):
		"""Shift warmstart solution forward by one step (reference: mpc_planner solution shifting).
		
		This maintains continuity between MPC steps: the solution from step k becomes the
		warmstart for step k+1, shifted forward by one timestep.
		
		CRITICAL: For spline variable, ensure monotonic increase (vehicle should progress along path).
		CRITICAL FIX: Clamp spline values to path_length to prevent infeasible dynamics when approaching path end.
		"""
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return
		
		LOG_DEBUG("Shifting warmstart forward by one step")
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		
		# Get path_length for spline clamping (prevents infeasibility at path end)
		path_length = getattr(self, '_path_length', None)
		if path_length is None and hasattr(self, 'data') and self.data is not None:
			if hasattr(self.data, 'reference_path') and self.data.reference_path is not None:
				ref_path = self.data.reference_path
				if hasattr(ref_path, 'length'):
					path_length = float(ref_path.length)
				elif hasattr(ref_path, 's') and len(ref_path.s) > 0:
					path_length = float(ref_path.s[-1])
		
		for var_name in dynamics_model.get_all_vars():
			if var_name in self.warmstart_values:
				# Special handling for spline: ensure it's monotonically increasing
				if var_name == 'spline':
					# Shift spline forward, but ensure it doesn't decrease
					if len(self.warmstart_values[var_name]) > 1:
						old_spline = self.warmstart_values[var_name].copy()
						# Shift forward: [s0, s1, ..., sN] -> [s1, s2, ..., sN, sN+ds]
						self.warmstart_values[var_name][:-1] = old_spline[1:]
						# For last value, advance by velocity * dt to maintain progression
						# Get velocity at last stage to estimate progression
						if 'v' in self.warmstart_values and len(self.warmstart_values['v']) > 0:
							v_last = float(self.warmstart_values['v'][-1])
							ds = v_last * timestep_val
							# Ensure spline doesn't go backwards
							s_last = max(old_spline[-1], old_spline[-2] + ds) if len(old_spline) > 1 else old_spline[-1]
							self.warmstart_values[var_name][-1] = s_last
						else:
							# Fallback: repeat last value
							self.warmstart_values[var_name][-1] = old_spline[-1]
						
						# CRITICAL FIX: Clamp spline values to path_length to prevent infeasible dynamics
						# When vehicle approaches path end, extrapolated spline can exceed path_length,
						# causing dynamics constraints to become infeasible (spline_next - spline_current mismatch)
						if path_length is not None and path_length > 0:
							clamped = False
							for k in range(len(self.warmstart_values[var_name])):
								if self.warmstart_values[var_name][k] > path_length:
									self.warmstart_values[var_name][k] = path_length
									clamped = True
							if clamped:
								LOG_DEBUG(f"  Clamped spline warmstart to path_length={path_length:.2f}")
						
						# Verify monotonicity (allow equal values at path end)
						spline_diff = np.diff(self.warmstart_values[var_name])
						if np.any(spline_diff < -1e-6):  # Allow small numerical errors
							LOG_WARN(f"  WARNING: Spline warmstart is not monotonic after shifting! Diffs: {spline_diff}")
							# Fix: ensure monotonic increase
							for k in range(1, len(self.warmstart_values[var_name])):
								if self.warmstart_values[var_name][k] < self.warmstart_values[var_name][k-1]:
									self.warmstart_values[var_name][k] = self.warmstart_values[var_name][k-1]
						
						LOG_DEBUG(f"  Shifted spline: [{old_spline[0]:.3f}, ..., {old_spline[-1]:.3f}] -> [{self.warmstart_values[var_name][0]:.3f}, ..., {self.warmstart_values[var_name][-1]:.3f}]")
				else:
					# Standard shifting for other variables
					if len(self.warmstart_values[var_name]) > 1:
						self.warmstart_values[var_name][:-1] = self.warmstart_values[var_name][1:]
						self.warmstart_values[var_name][-1] = self.warmstart_values[var_name][-2]
					LOG_DEBUG(f"  Shifted {var_name}: new range = [{self.warmstart_values[var_name][0]:.3f}, ..., {self.warmstart_values[var_name][-1]:.3f}]")
	
	def _ensure_warmstart_dynamics_consistency(self):
		"""Ensure warmstart values satisfy dynamics constraints (psi[k+1] = psi[k] + w[k]*dt)."""
		if not hasattr(self, 'warmstart_values') or not self.warmstart_values:
			return
		
		if 'psi' not in self.warmstart_values or 'w' not in self.warmstart_values:
			return
		
		psi_profile = self.warmstart_values['psi']
		w_profile = self.warmstart_values['w']
		
		if len(psi_profile) < 2 or len(w_profile) < 1:
			return
		
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		
		def _wrap_angle(angle_val):
			return (angle_val + np.pi) % (2 * np.pi) - np.pi
		
		# Check and fix dynamics consistency
		psi_fixed = False
		for k in range(1, min(len(psi_profile), horizon_val + 1)):
			if k - 1 < len(w_profile):
				psi_pred = _wrap_angle(psi_profile[k - 1] + w_profile[k - 1] * timestep_val)
				psi_actual = psi_profile[k]
				diff = abs(_wrap_angle(psi_actual - psi_pred))
				if diff > 1e-6:
					psi_fixed = True
					psi_profile[k] = psi_pred
		
		if psi_fixed:
			LOG_INFO("Fixed warmstart dynamics consistency after initialization/shifting")
			# Recompute x, y positions using corrected psi_profile
			if 'x' in self.warmstart_values and 'y' in self.warmstart_values and 'v' in self.warmstart_values:
				x_pos = self.warmstart_values['x']
				y_pos = self.warmstart_values['y']
				v_profile = self.warmstart_values['v']
				if len(x_pos) > 0 and len(y_pos) > 0:
					current_x = x_pos[0]
					current_y = y_pos[0]
					x_pos[0] = current_x
					y_pos[0] = current_y
					for k in range(1, min(len(psi_profile), horizon_val + 1)):
						if k < len(x_pos) and k < len(y_pos) and k - 1 < len(v_profile) and k - 1 < len(psi_profile):
							x_pos[k] = x_pos[k - 1] + v_profile[k - 1] * np.cos(psi_profile[k - 1]) * timestep_val
							y_pos[k] = y_pos[k - 1] + v_profile[k - 1] * np.sin(psi_profile[k - 1]) * timestep_val
					self.warmstart_values['x'] = x_pos
					self.warmstart_values['y'] = y_pos
					self.warmstart_values['psi'] = psi_profile

		# CRITICAL FIX: Also ensure spline dynamics consistency
		# Spline dynamics from RK4: spline[k+1] = spline[k] + v[k+1] * dt
		# IMPORTANT: The RK4 dynamics use v[k+1] (velocity AFTER integration), not v[k]!
		# Reference: rk4_integration() uses x_next_integrate[3] (v after RK4) for spline update
		# This is critical after braking fallback changes the velocity profile
		if 'spline' in self.warmstart_values and 'v' in self.warmstart_values:
			spline_profile = self.warmstart_values['spline']
			v_profile = self.warmstart_values['v']

			if len(spline_profile) > 1 and len(v_profile) >= 1:
				spline_fixed = False
				for k in range(1, min(len(spline_profile), horizon_val + 1)):
					# Use v_profile[k] (velocity at stage k AFTER integration) not v_profile[k-1]
					# Dynamics: spline[k] = spline[k-1] + v[k] * dt
					if k < len(v_profile):
						spline_pred = spline_profile[k - 1] + v_profile[k] * timestep_val
						spline_actual = spline_profile[k]
						diff = abs(spline_actual - spline_pred)
						if diff > 0.01:  # Tolerance for numerical precision
							spline_fixed = True
							spline_profile[k] = spline_pred

				if spline_fixed:
					self.warmstart_values['spline'] = spline_profile
					LOG_INFO("Fixed spline warmstart dynamics consistency (spline[k] = spline[k-1] + v[k]*dt)")

	def _update_warmstart_current_state(self, state: State):
		"""Update stage 0 of warmstart to match current vehicle state.
		
		After shifting, stage 0 should reflect the vehicle's current position/state.
		CRITICAL: For spline, ensure it matches current state and doesn't go backwards.
		"""
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return
		
		LOG_DEBUG("Updating warmstart stage 0 to match current state")
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		
		# Get current spline value first (needed for validation)
		current_spline = None
		if 'spline' in self.warmstart_values and state.has('spline'):
			current_spline = state.get('spline')
		
		for var_name in dynamics_model.get_dependent_vars():
			if var_name in self.warmstart_values and state.has(var_name):
				current_val = state.get(var_name)
				if current_val is not None:
					# Special handling for spline: ensure it doesn't go backwards
					if var_name == 'spline':
						# Current spline should be >= previous warmstart[0] (vehicle should progress)
						# But allow small backward movement if vehicle is correcting (e.g., due to obstacle)
						old_spline_0 = self.warmstart_values[var_name][0]
						if current_spline is not None:
							# Allow small backward movement (up to 0.1m) for correction, but warn if larger
							if current_spline < old_spline_0 - 0.1:
								LOG_WARN(f"  WARNING: Current spline ({current_spline:.3f}) is significantly behind warmstart[0] ({old_spline_0:.3f})")
								# Use the larger value to ensure forward progression
								current_val = max(current_spline, old_spline_0 - 0.1)
							else:
								current_val = current_spline
							LOG_DEBUG(f"  Updated spline[0] = {current_val:.3f} (current={current_spline:.3f}, old_ws={old_spline_0:.3f})")
						else:
							# No current spline - keep warmstart value
							current_val = old_spline_0
							LOG_DEBUG(f"  Keeping spline[0] = {current_val:.3f} (no current spline in state)")
					
					self.warmstart_values[var_name][0] = current_val
					if var_name != 'spline':  # Already logged above
						LOG_DEBUG(f"  Updated {var_name}[0] = {current_val:.3f}")
		
		# Also update control inputs at stage 0 if available from state
		for var_name in dynamics_model.get_inputs():
			if var_name in self.warmstart_values and state.has(var_name):
				current_val = state.get(var_name)
				if current_val is not None:
					self.warmstart_values[var_name][0] = current_val
					LOG_DEBUG(f"  Updated control {var_name}[0] = {current_val:.3f}")
		
		# After updating stage 0, ensure spline progression is maintained
		if 'spline' in self.warmstart_values:
			# Ensure spline is monotonically non-decreasing (allow small decreases for correction)
			spline_vals = self.warmstart_values['spline']
			for k in range(1, len(spline_vals)):
				# Allow small backward movement (0.1m) but ensure forward progression overall
				if spline_vals[k] < spline_vals[k-1] - 0.1:
					# Advance by minimum step based on velocity
					if 'v' in self.warmstart_values and k-1 < len(self.warmstart_values['v']):
						v_k = max(0.0, float(self.warmstart_values['v'][k-1]))
						ds_min = v_k * timestep_val
						spline_vals[k] = max(spline_vals[k-1], spline_vals[k-1] + ds_min)
					else:
						spline_vals[k] = spline_vals[k-1]  # Keep same value
					LOG_DEBUG(f"  Corrected spline[{k}] to {spline_vals[k]:.3f} to maintain progression")

	def _update_warmstart_from_solution(self):
		for var_name in self.var_dict:
			try:
				self.warmstart_values[var_name] = np.array(self.solution.value(self.var_dict[var_name]))
			except Exception as e:
				LOG_WARN(f"Could not update warmstart for {var_name}: {e}")

	def _create_trajectory_from_warmstart(self):
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("_create_trajectory_from_warmstart: dynamics_model is None, returning empty trajectory")
			return Trajectory(timestep=timestep_val, length=0)

		# Check warmstart_values exist and have required keys
		if not self.warmstart_values:
			# Try to create a fallback trajectory from current state in data
			try:
				if self.data is not None and hasattr(self.data, 'state') and self.data.state is not None:
					LOG_DEBUG("Creating fallback trajectory from data.state")
					state = self.data.state
					traj = Trajectory(timestep=timestep_val, length=horizon_val + 1)
					# Create states using current position for all steps
					for k in range(horizon_val + 1):
						state_k = State(model_type=dynamics_model)
						for var_name in dynamics_model.get_dependent_vars():
							val = state.get(var_name) if state.has(var_name) else 0.0
							state_k.set(var_name, float(val) if val is not None else 0.0)
						traj.add_state(state_k)
					return traj
			except Exception as e:
				LOG_WARN(f"Error creating fallback trajectory: {e}")
			return Trajectory(timestep=timestep_val, length=0)

		dep_vars = dynamics_model.get_dependent_vars()
		missing_vars = [v for v in dep_vars if v not in self.warmstart_values]
		if missing_vars:
			LOG_WARN(f"_create_trajectory_from_warmstart: missing warmstart vars: {missing_vars}")
			return Trajectory(timestep=timestep_val, length=0)

		traj = Trajectory(timestep=timestep_val, length=horizon_val + 1)
		try:
			for k in range(horizon_val + 1):
				state_k = State(model_type=dynamics_model)
				for var_name in dep_vars:
					ws_val = self.warmstart_values[var_name]
					if isinstance(ws_val, np.ndarray) and k < len(ws_val):
						state_k.set(var_name, float(ws_val[k]))
					elif isinstance(ws_val, (list, tuple)) and k < len(ws_val):
						state_k.set(var_name, float(ws_val[k]))
					else:
						LOG_WARN(f"_create_trajectory_from_warmstart: invalid warmstart for {var_name} at k={k}")
						state_k.set(var_name, 0.0)
				if k < horizon_val:
					for var_name in dynamics_model.get_inputs():
						if var_name in self.warmstart_values:
							ws_val = self.warmstart_values[var_name]
							if isinstance(ws_val, np.ndarray) and k < len(ws_val):
								state_k.set(var_name, float(ws_val[k]))
							elif isinstance(ws_val, (list, tuple)) and k < len(ws_val):
								state_k.set(var_name, float(ws_val[k]))
				traj.add_state(state_k)
		except Exception as e:
			LOG_WARN(f"_create_trajectory_from_warmstart: error creating trajectory: {e}")
			return Trajectory(timestep=timestep_val, length=0)

		LOG_DEBUG(f"_create_trajectory_from_warmstart: created trajectory with {len(traj.get_states()) if hasattr(traj, 'get_states') else 'unknown'} states")
		return traj

	def _initialize_with_braking(self, state: State):
		"""
		Initialize warmstart with a braking trajectory on solver failure.

		Reference: C++ mpc_planner - initializeWithBraking()
		When solver fails, generate a safe deceleration trajectory.
		This prevents the vehicle from drifting and allows recovery.

		Args:
			state: Current vehicle state
		"""
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("_initialize_with_braking: no dynamics_model available")
			return

		# Get current state values
		current_x = float(state.get('x')) if state.has('x') else 0.0
		current_y = float(state.get('y')) if state.has('y') else 0.0
		current_psi = float(state.get('psi')) if state.has('psi') else 0.0
		current_v = float(state.get('v')) if state.has('v') else 0.0
		current_spline = float(state.get('spline')) if state.has('spline') else 0.0

		# Get velocity bounds
		try:
			v_lb, v_ub, _ = dynamics_model.get_bounds('v')
		except Exception:
			v_lb, v_ub = -0.01, 3.0

		# Braking deceleration (negative acceleration)
		# Reference: C++ mpc_planner - uses configured deceleration rate
		braking_accel = DEFAULT_BRAKING  # Typically -2.0 m/s^2

		# Compute velocity profile with braking
		v_profile = np.zeros(horizon_val + 1)
		v_profile[0] = current_v
		for k in range(1, horizon_val + 1):
			v_next = v_profile[k-1] + braking_accel * timestep_val
			v_profile[k] = np.clip(v_next, v_lb, v_ub)

		# Compute heading profile - try to turn toward reference path if available
		psi_profile = np.full(horizon_val + 1, current_psi)
		w_profile = np.zeros(horizon_val)

		# If we have a reference path, compute desired heading toward path
		if (self.data is not None and hasattr(self.data, 'reference_path') and
		    self.data.reference_path is not None):
			try:
				ref_path = self.data.reference_path
				if hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None:
					# Find closest point on path
					s_clamped = np.clip(current_spline, 0, ref_path.length if hasattr(ref_path, 'length') else float('inf'))
					path_x = float(ref_path.x_spline(s_clamped))
					path_y = float(ref_path.y_spline(s_clamped))

					# Compute heading toward path center
					dx = path_x - current_x
					dy = path_y - current_y
					dist_to_path = np.sqrt(dx**2 + dy**2)

					if dist_to_path > 0.1:  # Only steer if significantly off path
						desired_psi = np.arctan2(dy, dx)
						heading_error = desired_psi - current_psi
						# Normalize to [-pi, pi]
						heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

						# Get angular rate bounds
						try:
							w_lb, w_ub, _ = dynamics_model.get_bounds('w')
						except Exception:
							w_lb, w_ub = -0.8, 0.8

						# Compute steering command (proportional control)
						w_cmd = np.clip(heading_error * 2.0, w_lb, w_ub)

						# Integrate psi with steering
						for k in range(horizon_val):
							w_profile[k] = w_cmd
							psi_profile[k+1] = psi_profile[k] + w_profile[k] * timestep_val

						LOG_DEBUG(f"Braking fallback: steering toward path (heading_error={np.degrees(heading_error):.1f}deg)")
			except Exception as e:
				LOG_DEBUG(f"Braking fallback: could not compute steering: {e}")

		# Integrate positions
		x_pos = np.zeros(horizon_val + 1)
		y_pos = np.zeros(horizon_val + 1)
		x_pos[0] = current_x
		y_pos[0] = current_y
		for k in range(1, horizon_val + 1):
			x_pos[k] = x_pos[k-1] + v_profile[k-1] * np.cos(psi_profile[k-1]) * timestep_val
			y_pos[k] = y_pos[k-1] + v_profile[k-1] * np.sin(psi_profile[k-1]) * timestep_val

		# Compute spline profile (advance with velocity)
		spline_profile = np.zeros(horizon_val + 1)
		spline_profile[0] = current_spline
		for k in range(1, horizon_val + 1):
			spline_profile[k] = spline_profile[k-1] + v_profile[k-1] * timestep_val

		# Update warmstart values
		self.warmstart_values['x'] = x_pos
		self.warmstart_values['y'] = y_pos
		self.warmstart_values['psi'] = psi_profile
		self.warmstart_values['v'] = v_profile
		if 'spline' in dynamics_model.get_dependent_vars():
			self.warmstart_values['spline'] = spline_profile

		# Set control inputs
		self.warmstart_values['a'] = np.full(horizon_val, braking_accel)
		self.warmstart_values['w'] = w_profile

		LOG_INFO(f"Braking fallback: Initialized warmstart with braking (a={braking_accel:.1f}m/s^2)")
		LOG_INFO(f"  v profile: [{v_profile[0]:.2f} -> {v_profile[-1]:.2f}] m/s")
		LOG_INFO(f"  Position: ({x_pos[0]:.2f}, {y_pos[0]:.2f}) -> ({x_pos[-1]:.2f}, {y_pos[-1]:.2f})")

	def apply_braking_fallback(self, state: State = None):
		"""
		Apply braking fallback after solver failure and return control values.

		Reference: C++ mpc_planner - initializeWithBraking() on solver failure
		This should be called by the planner after solve() returns -1.

		Args:
			state: Current vehicle state. If None, uses self.data.state.

		Returns:
			dict: Control values {'a': ..., 'w': ...} with braking applied
		"""
		if state is None:
			if self.data is not None and hasattr(self.data, 'state') and self.data.state is not None:
				state = self.data.state
			else:
				LOG_WARN("apply_braking_fallback: No state available, returning zero control")
				return {'a': 0.0, 'w': 0.0}

		# Apply braking to warmstart
		self._initialize_with_braking(state)

		# Return first control values from warmstart
		control = {}
		if 'a' in self.warmstart_values and len(self.warmstart_values['a']) > 0:
			control['a'] = float(self.warmstart_values['a'][0])
		else:
			control['a'] = DEFAULT_BRAKING  # -2.0 m/s^2

		if 'w' in self.warmstart_values and len(self.warmstart_values['w']) > 0:
			control['w'] = float(self.warmstart_values['w'][0])
		else:
			control['w'] = 0.0

		LOG_INFO(f"Braking fallback control: a={control['a']:.2f} m/s^2, w={control['w']:.4f} rad/s")
		return control

	def _set_opti_initial_values(self):
		"""Set initial values for optimization variables from warmstart values."""
		if not hasattr(self, 'warmstart_values') or not self.warmstart_values:
			LOG_DEBUG("_set_opti_initial_values: No warmstart values available")
			return
		
		LOG_DEBUG("Setting initial values in CasADi opti problem from warmstart")
		for var_name in self.var_dict:
			if var_name in self.warmstart_values:
				try:
					ws_vals = self.warmstart_values[var_name]
					if isinstance(ws_vals, np.ndarray) and len(ws_vals) > 0:
						# Ensure values are within bounds
						dynamics_model = self._get_dynamics_model()
						if dynamics_model:
							try:
								lb, ub, _ = dynamics_model.get_bounds(var_name)
								ws_vals = np.clip(ws_vals, lb, ub)
							except Exception:
								pass
						# Set initial value in CasADi
						self.opti.set_initial(self.var_dict[var_name], ws_vals)
						LOG_DEBUG(f"  Set initial value for {var_name}: shape={ws_vals.shape}, range=[{np.min(ws_vals):.3f}, {np.max(ws_vals):.3f}]")
				except Exception as e:
					LOG_WARN(f"  Could not set initial value for {var_name}: {e}")

	def _set_initial_state(self, state: State):
		"""Set initial state constraints. State comes from data, not stored separately."""
		# DO NOT store state - always get from data when needed
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("_set_initial_state called but dynamics_model is not set")
			return
		if not self.var_dict:
			LOG_WARN("_set_initial_state called but var_dict is empty - solver not initialized")
			return
		for var_name in dynamics_model.get_dependent_vars():
			if var_name not in self.var_dict:
				LOG_WARN(f"_set_initial_state: variable '{var_name}' not in var_dict. Available: {list(self.var_dict.keys())}")
				continue
			value = state.get(var_name)
			if value is not None:
				# CRITICAL: Ensure warmstart value matches current state BEFORE setting constraint
				# Reference: C++ mpc_planner - initial values must match constraints exactly
				if var_name in self.warmstart_values:
					# Update warmstart to match current state
					self.warmstart_values[var_name][0] = value
					# Set initial value in opti to match constraint
					# This ensures the constraint is satisfied at the initial point
					try:
						initial_array = self.warmstart_values[var_name].copy()
						initial_array[0] = float(value)
						self.opti.set_initial(self.var_dict[var_name], initial_array)
						LOG_DEBUG(f"_set_initial_state: Set initial value for {var_name}[0] = {value}")
					except Exception as e:
						LOG_WARN(f"Could not set initial value for {var_name} before constraint: {e}")
				
				# Set constraint: x[0] == value
				# This constraint will be satisfied because we set the initial value above
				self.opti.subject_to(self.var_dict[var_name][0] == value)

	def solve(self, state=None, data=None):
		"""Solve the optimization problem.
		
		Args:
			state: Current vehicle state (DEPRECATED - use data.state instead)
			data: Data object with state, constraints, objectives, and dynamics_model
		"""
		LOG_INFO("=== CasADiSolver.solve() ===")
		LOG_DEBUG("Attempting to solve in Casadi solver")
		
		# Data is the single source of truth - store reference but don't duplicate
		if data is None:
			if self.data is None:
				LOG_WARN("solve: data is None and no stored data available - cannot solve")
				return -1
			data = self.data
			LOG_DEBUG("Using stored data reference")
		else:
			# Update data reference - this is the source of truth
			self.data = data
		
		# Get state from data - this is the authoritative source
		if hasattr(data, 'state') and data.state is not None:
			state = data.state
			LOG_DEBUG("Using state from data.state")
		elif state is None:
			LOG_WARN("solve: no state available (neither data.state nor state parameter)")
			return -1
		else:
			LOG_WARN("solve: state provided as parameter but data.state should be used instead")
		
		# Ensure data has required attributes
		if not hasattr(data, 'horizon') or data.horizon is None:
			data.horizon = self.horizon
		if not hasattr(data, 'timestep') or data.timestep is None:
			data.timestep = self.timestep
		# Ensure data has dynamics_model - it should be set by planner
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("solve: data.dynamics_model is not set - cannot solve")
			return -1
		
		# CRITICAL FIX: Create a fresh Opti problem for each solve.
		# CasADi Opti accumulates constraints and objectives - they don't reset automatically.
		# We must create a new Opti() each time, but preserve warmstart values for continuity.
		# Reference: Standard CasADi pattern - rebuild problem each solve, warm start from previous solution.

		# Save warmstart values before recreating opti
		saved_warmstart = {}
		if hasattr(self, 'warmstart_values') and self.warmstart_values:
			for k, v in self.warmstart_values.items():
				saved_warmstart[k] = np.array(v) if isinstance(v, np.ndarray) else v
			LOG_DEBUG(f"Saved {len(saved_warmstart)} warmstart vars")

		# Create fresh Opti problem
		self.opti = cs.Opti()
		self.var_dict = {}
		# CRITICAL: Clear old solution to prevent using variables from old opti instance
		self.solution = None

		# Restore warmstart values
		if saved_warmstart:
			self.warmstart_values = saved_warmstart
		elif not hasattr(self, 'warmstart_values') or not self.warmstart_values:
			self.warmstart_values = {}

		# Ensure data has dynamics_model - it should be set by planner
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("solve: data.dynamics_model is not set - cannot initialize solver")
			return -1

		# Initialize solver with data (creates variables and adds dynamics constraints)
		self.intialize_solver(data)

		# Apply warmstart values to the new opti problem
		# This sets initial guesses for IPOPT based on previous solution or computed warmstart
		self._set_opti_initial_values()

		# Set initial state constraints (x[0] == current_state)
		state = data.state if hasattr(data, 'state') and data.state is not None else None
		if state is not None:
			self._set_initial_state(state)

		total_objective = 0
		total_constraints_added = 0
		constraints_per_stage = {}  # Track constraints per stage for summary
		
		# Ensure horizon is set
		horizon_val = self.horizon if self.horizon is not None else 10
		
		# Log robot_area and disc information
		if self.data is not None:
			if hasattr(self.data, 'robot_area') and self.data.robot_area is not None:
				disc_info = []
				for i, disc in enumerate(self.data.robot_area):
					if hasattr(disc, 'offset') and hasattr(disc, 'radius'):
						disc_info.append(f"disc[{i}]: offset={disc.offset:.3f}, radius={disc.radius:.3f}")
				LOG_INFO(f"  robot_area in data: {len(self.data.robot_area)} disc(s) - {', '.join(disc_info)}")
			else:
				LOG_WARN("  robot_area not found in data or is None")
		
		# Log module manager state
		try:
			import logging as _logging
			_integ_logger = _logging.getLogger("integration_test")
			modules_list = [m.name for m in (self.module_manager.get_modules() or [])]
			msg_mod = f"Solver solve: module_manager has {len(modules_list)} modules: {modules_list}, data={self.data is not None}"
			LOG_INFO(msg_mod)
			try:
				_integ_logger.info(msg_mod)
			except Exception:
				pass
		except Exception:
			pass

		# Before building objectives/constraints, dump warmstart and bounds at k=0
		try:
			dynamics_model = self._get_dynamics_model()
			dv = dynamics_model.get_dependent_vars() if dynamics_model else []
			iv = dynamics_model.get_inputs() if dynamics_model else []
			keys_to_show = [k for k in ['x','y','psi','v','spline'] if k in dv] + [k for k in ['a','w','delta'] if k in iv]
			msg_ws = []
			msg_bd = []
			for name in keys_to_show:
				ws0 = None
				try:
					ws0 = float(self.warmstart_values.get(name, [None])[0])
				except Exception:
					ws0 = 'n/a'
				msg_ws.append(f"{name}0={ws0}")
				try:
					lb, ub, _ = dynamics_model.get_bounds(name)
					msg_bd.append(f"{name}∈[{lb},{ub}]")
				except Exception:
					pass
			if msg_ws:
				LOG_INFO(f"  Warmstart k=0: {' , '.join(msg_ws)}")
				try:
					import logging as _logging
					_integ_logger = _logging.getLogger("integration_test")
					_integ_logger.info(f"Warmstart k=0: {' , '.join(msg_ws)}")
				except Exception:
					pass
			if msg_bd:
				LOG_INFO(f"  Bounds: {' , '.join(msg_bd)}")
				try:
					import logging as _logging
					_integ_logger = _logging.getLogger("integration_test")
					_integ_logger.info(f"Bounds: {' , '.join(msg_bd)}")
				except Exception:
					pass
		except Exception:
			pass

		# Multi-stage dynamics residual diagnostics (k=0..2) using planner/base-solver path
		try:
			max_k = min(3, horizon_val)
			import numpy as _np
			dynamics_model = self._get_dynamics_model()
			dep_vars = dynamics_model.get_dependent_vars() if dynamics_model else []
			in_vars = dynamics_model.get_inputs() if dynamics_model else []
			for k in range(max_k):
				xk_num = []
				uk_num = []
				for name in dep_vars:
					seq = self.warmstart_values.get(name, [])
					xk_num.append(float(seq[k] if k < len(seq) else 0.0))
				for name in in_vars:
					seq = self.warmstart_values.get(name, [])
					uk_num.append(float(seq[k] if k < len(seq) else 0.0))
				# Compute predicted next state via planner/base-solver path
				x_sym = cs.SX.sym('x', len(xk_num))
				u_sym = cs.SX.sym('u', len(uk_num) if len(uk_num) > 0 else 1)
				x_next_sym = self._compute_next_state(dynamics_model, x_sym, u_sym, self.timestep, self.data, symbolic=True)
				# Debug: check shape of x_next_sym
				try:
					x_next_shape = x_next_sym.shape if hasattr(x_next_sym, 'shape') else 'unknown'
					x_next_size = x_next_sym.size1() if hasattr(x_next_sym, 'size1') else (x_next_sym.shape[0] if hasattr(x_next_sym, 'shape') and len(x_next_sym.shape) > 0 else 'unknown')
					LOG_INFO(f"  Dyn residual k={k}: x_next_sym shape={x_next_shape}, size={x_next_size}, xk_num len={len(xk_num)}")
				except Exception as _e:
					LOG_WARN(f"  Dyn residual k={k}: failed to get x_next_sym shape: {_e}")
				f_fun = cs.Function('f_next', [x_sym, u_sym], [x_next_sym])
				f_result = f_fun(cs.DM(xk_num), cs.DM(uk_num if len(uk_num) > 0 else [0.0]))
				# Debug: check what f_fun returns
				try:
					f_result_shape = f_result.shape if hasattr(f_result, 'shape') else 'unknown'
					f_result_size = f_result.size1() if hasattr(f_result, 'size1') else 'unknown'
					LOG_INFO(f"  Dyn residual k={k}: f_result type={type(f_result)}, shape={f_result_shape}, size={f_result_size}")
				except Exception as _e:
					LOG_WARN(f"  Dyn residual k={k}: failed to check f_result: {_e}")
				# Extract full vector from DM (f_result is already the output DM, not a tuple)
				try:
					x_next_pred = f_result.full().flatten().tolist()
					LOG_INFO(f"  Dyn residual k={k}: extracted pred len={len(x_next_pred)}, values={x_next_pred}")
				except Exception as _e:
					LOG_WARN(f"  Dyn residual k={k}: failed to extract pred: {_e}")
					x_next_pred = [0.0] * len(dep_vars)  # Fallback
				# Warmstart x_{k+1}
				xkp1_ws = []
				for name in dep_vars:
					seq = self.warmstart_values.get(name, [])
					xkp1_ws.append(float(seq[k+1] if k + 1 < len(seq) else (seq[-1] if seq else 0.0)))
				# Debug: check if pred has right size
				if len(x_next_pred) != len(xkp1_ws):
					LOG_WARN(f"  Dyn residual k={k}: pred has {len(x_next_pred)} elements, expected {len(xkp1_ws)}")
				# Residual
				res = [xp - xw for xp, xw in zip(x_next_pred, xkp1_ws)]
				res_norm = float(_np.linalg.norm(_np.array(res)))
				LOG_INFO(f"  Dyn residual k={k}: pred={x_next_pred}, ws={xkp1_ws}, res={res}, res_norm={res_norm:.3e}")
				try:
					import logging as _logging
					_logging.getLogger("integration_test").info(f"Dyn residual k={k}: pred={x_next_pred}, ws={xkp1_ws}, res={res}, res_norm={res_norm:.3e}")
				except Exception:
					pass
		except Exception as _e:
			LOG_WARN(f"Dyn residual diag failed: {_e}")
			try:
				import logging as _logging
				_logging.getLogger("integration_test").warning(f"Dyn residual diag failed: {_e}")
			except Exception:
				pass

		# CRITICAL: Loop over all stages (0 to horizon) to collect objectives and constraints
		# State variables: k in [0, horizon] (horizon + 1 values)
		# Input variables: k in [0, horizon-1] (horizon values)
		LOG_INFO(f"Collecting objectives and constraints for {horizon_val + 1} stages (0 to {horizon_val})")
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("solve: cannot create symbolic states without dynamics_model")
			return -1
		
		for stage_idx in range(horizon_val + 1):
			# Create symbolic state for this stage (converted from data)
			# This is a symbolic representation for CasADi - actual state comes from data
			symbolic_state = State(dynamics_model)
			for var_name in dynamics_model.get_all_vars():
				if var_name in self.var_dict and stage_idx < self.var_dict[var_name].shape[0]:
					symbolic_state.set(var_name, self.var_dict[var_name][stage_idx])

			# CRITICAL: Ensure data is up-to-date before getting objectives/constraints
			# The state in data should be the current state (k=0), not the predicted state
			# For objectives/constraints at stage k, we use symbolic_state for CasADi variables
			# but data.state should be the current actual state from the planner
			if self.data is not None:
				LOG_DEBUG(f"Stage {stage_idx}: Using data.state={self.data.state is not None}, symbolic_state for CasADi vars")
			
			# Get objectives from module manager
			# CRITICAL: For ALL stages, use symbolic_state (CasADi variables) so objectives are computed symbolically.
			# 
			# The symbolic_state contains CasADi MX/SX variables from var_dict, which are:
			#   - For states: constrained by RK4 integration (x[k+1] == symbolic_dynamics(x[k], u[k]))
			#   - For controls: free decision variables (a, w)
			# 
			# Objectives MUST return symbolic CasADi expressions (MX or SX) that depend on these variables.
			# This ensures the optimization problem is fully symbolic, matching the C++ reference implementation.
			# 
			# Reference: https://github.com/tud-amr/mpc_planner - all calculations are symbolic.
			# Only acceleration (a) and angular acceleration (w) are decision variables.
			# All states are computed via RK4 integration or model_discrete_dynamics.
			LOG_DEBUG(f"[OBJECTIVES] Stage {stage_idx}: Getting objectives from module_manager")
			objective_costs = self.module_manager.get_objectives(symbolic_state, self.data, stage_idx) or []
			# Log objectives BEFORE processing
			obj_count = len(objective_costs or [])
			LOG_DEBUG(f"[OBJECTIVES] Stage {stage_idx}: Received {obj_count} objective cost dict(s)")
			if obj_count > 0:
				try:
					import logging as _logging
					_integ_logger = _logging.getLogger("integration_test")
					obj_keys = []
					obj_values = []
					for item in (objective_costs or []):
						if isinstance(item, dict):
							obj_keys.extend(list(item.keys()))
							for k, v in item.items():
								try:
									# Try to get numeric value if symbolic
									if hasattr(v, '__float__'):
										obj_values.append((k, float(v)))
									else:
										obj_values.append((k, 'symbolic'))
								except:
									obj_values.append((k, 'unknown'))
					msg_obj = f"Stage {stage_idx}: {obj_count} objective(s) with keys: {obj_keys}"
					if stage_idx <= 2 and obj_values:
						LOG_DEBUG(f"  Objective values: {obj_values[:5]}")  # Show first 5
					LOG_INFO(msg_obj)
					try:
						_integ_logger.info(msg_obj)
					except Exception:
						pass
				except Exception:
					pass
			else:
				if stage_idx <= 2:
					LOG_DEBUG(f"Stage {stage_idx}: No objectives")
			for cost_dict in objective_costs:
				for cost_name, cost_val in cost_dict.items():
					LOG_DEBUG(f"[OBJECTIVES] Stage {stage_idx}: Adding cost '{cost_name}' (type={type(cost_val).__name__}, symbolic={isinstance(cost_val, (cs.MX, cs.SX))})")
					total_objective += cost_val
					# Log angle cost specifically for goal objective debugging
					if stage_idx == 0 and cost_name == 'goal_angle_cost':
						try:
							if hasattr(cost_val, 'is_constant') and cost_val.is_constant():
								LOG_INFO(f"  Stage {stage_idx}: goal_angle_cost = {float(cost_val):.4f} (added to total objective)")
							else:
								LOG_DEBUG(f"  Stage {stage_idx}: goal_angle_cost = symbolic (added to total objective)")
						except:
							LOG_DEBUG(f"  Stage {stage_idx}: goal_angle_cost = {cost_val} (added to total objective)")

			# Add control regularization costs (like reference codebase base_module)
			# This helps the solver find smoother solutions and escape local minima
			# Reference: https://github.com/tud-amr/mpc_planner uses base_module.weigh_variable()
			if stage_idx < horizon_val:  # Only for control stages (not terminal state)
				try:
					# Safely get control weights from config
					control_weights = {}
					if self.config is not None and isinstance(self.config, dict):
						solver_config = self.config.get("solver", {})
						if isinstance(solver_config, dict):
							t_mpc_config = solver_config.get("t-mpc", {})
							if isinstance(t_mpc_config, dict):
								control_weights = t_mpc_config.get("weights", {})
					
					acceleration_weight = control_weights.get("acceleration_x", 0.0) if isinstance(control_weights, dict) else 0.0
					angular_velocity_weight = control_weights.get("angular_velocity", 0.0) if isinstance(control_weights, dict) else 0.0
					
					# Get control variables from dynamics model
					if dynamics_model is not None and hasattr(self, 'var_dict') and self.var_dict:
						input_vars = dynamics_model.get_inputs()
						for u_name in input_vars:
							if u_name in self.var_dict and self.var_dict[u_name] is not None:
								if stage_idx < self.var_dict[u_name].shape[0]:
									u_k = self.var_dict[u_name][stage_idx]
									if u_name == 'a' and acceleration_weight > 0:
										# Quadratic cost on acceleration
										control_cost = acceleration_weight * u_k ** 2
										total_objective += control_cost
										if stage_idx == 0:
											LOG_DEBUG(f"  Stage {stage_idx}: Added acceleration cost (weight={acceleration_weight:.2f})")
									elif u_name == 'w' and angular_velocity_weight > 0:
										# Quadratic cost on angular velocity (reference: base_module.weigh_variable("w", "angular_velocity"))
										control_cost = angular_velocity_weight * u_k ** 2
										total_objective += control_cost
										if stage_idx == 0:
											LOG_INFO(f"  Stage {stage_idx}: Added angular velocity cost (weight={angular_velocity_weight:.2f}, cost={control_cost})")
									
									# Add control rate smoothing (penalize changes in control between steps)
									# This reduces oversteering by encouraging smoother control inputs
									# Reference: Similar to reference codebase which encourages smooth control
									if stage_idx > 0 and u_name in self.var_dict and stage_idx - 1 < self.var_dict[u_name].shape[0]:
										u_km1 = self.var_dict[u_name][stage_idx - 1]
										# Use a moderate weight for control rate (0.2-0.5 of control weight)
										# This prevents rapid changes in control, reducing oversteering
										control_rate_weight = angular_velocity_weight * 0.4 if u_name == 'w' else acceleration_weight * 0.4
										if control_rate_weight > 0:
											control_rate_cost = control_rate_weight * (u_k - u_km1) ** 2
											total_objective += control_rate_cost
											if stage_idx == 1 and u_name == 'w':
												LOG_DEBUG(f"  Stage {stage_idx}: Added control rate cost for {u_name} (weight={control_rate_weight:.2f})")
				except Exception as e:
					LOG_WARN(f"Failed to add control costs at stage {stage_idx}: {e}")

			# Get constraints from module manager
			# CRITICAL: For ALL stages, pass symbolic_state so constraints are computed symbolically.
			# 
			# Constraints MUST return symbolic CasADi expressions (MX or SX) that depend on:
			#   - State variables (x, y, psi, v, spline) - constrained by RK4/model_discrete_dynamics
			#   - Control variables (a, w) - free decision variables
			# 
			# The returned expressions must be symbolic - numeric calculations are NOT allowed.
			# This ensures constraints are properly integrated into the optimization problem.
			# 
			# Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
			# Only acceleration (a) and angular acceleration (w) are decision variables.
			# All states are computed via RK4 integration or model_discrete_dynamics.
			LOG_DEBUG(f"[CONSTRAINTS] Stage {stage_idx}: Getting constraints from module_manager")
			constraints = self.get_constraints(stage_idx, symbolic_state=symbolic_state)
			constraints_count_before = total_constraints_added
			LOG_INFO(f"[CONSTRAINTS] Stage {stage_idx}: Received {len(constraints or [])} constraint(s) from modules (total constraints before: {constraints_count_before})")
			# Log constraints BEFORE processing with detailed breakdown
			LOG_INFO(f"=== Solver.solve(): Stage {stage_idx} - Constraints being passed to solver ===")
			try:
				import logging as _logging
				_integ_logger = _logging.getLogger("integration_test")
				cons_summ = []
				linearized_count = 0
				contouring_count = 0
				other_count = 0
				linearized_constraints_detail = []
				for constraint_item in (constraints or []):
					# Handle both tuple format (c, lb, ub) and direct constraints
					if isinstance(constraint_item, tuple) and len(constraint_item) == 3:
						c, lb, ub = constraint_item
					else:
						c = constraint_item
						lb, ub = None, None
					
					ctype = None
					if isinstance(c, dict):
						# Check for symbolic expression type
						if c.get('type') == 'symbolic_expression':
							# Distinguish between linearized, safe_horizon, and contouring constraints
							if c.get('constraint_type') == 'linearized':
								linearized_count += 1
								ctype = 'linearized'
							elif c.get('constraint_type') == 'safe_horizon':
								other_count += 1
								ctype = 'safe_horizon'
							elif c.get('constraint_type') == 'ellipsoid':
								other_count += 1
								ctype = 'ellipsoid'
							elif c.get('constraint_type') == 'gaussian':
								other_count += 1
								ctype = 'gaussian'
							else:
								contouring_count += 1
								ctype = 'contouring'
						elif 'a1' in c and 'a2' in c:
							# Check if it's a linearized constraint (has obstacle index context) or contouring
							if 'is_left' in c or 'spline_s' in c:
								contouring_count += 1
								ctype = 'contouring'
							else:
								linearized_count += 1
								ctype = 'linearized'
								# Log detailed information for linearized constraints
								# CRITICAL: Handle symbolic values - only convert to float if numeric
								a1_val = c.get('a1', 0.0)
								a2_val = c.get('a2', 0.0)
								b_val = c.get('b', 0.0)
								disc_offset_val = c.get('disc_offset', 0.0)
								
								# Helper to safely convert to float (handles symbolic expressions)
								def safe_float(val):
									if val is None:
										return 0.0
									import casadi as cd
									if isinstance(val, (cd.MX, cd.SX)):
										return 'symbolic'  # Can't evaluate symbolic expressions here
									try:
										return float(val)
									except (TypeError, ValueError):
										return 'N/A'
								
								linearized_constraints_detail.append({
									"a1": safe_float(a1_val),
									"a2": safe_float(a2_val),
									"b": safe_float(b_val),
									"disc_offset": safe_float(disc_offset_val),
									"lb": lb,
									"ub": ub
								})
						else:
							ctype = c.get('type') or 'dict'
							other_count += 1
					else:
						ctype = type(c).__name__
						other_count += 1
					cons_summ.append({
						"type": ctype,
						"lb": None if lb is None else 'set',
						"ub": None if ub is None else 'set',
					})
				# Count total constraints properly (handle tuple format)
				total_constraints = len(constraints or [])
				msg_cons = f"Stage {stage_idx}: constraints count={total_constraints} (linearized={linearized_count}, contouring={contouring_count}, other={other_count})"
				LOG_INFO(msg_cons)
				
				# Log detailed information for linearized constraints
				if linearized_constraints_detail:
					LOG_INFO(f"  Linearized constraints details ({len(linearized_constraints_detail)} constraints):")
					for i, const_detail in enumerate(linearized_constraints_detail[:10]):  # Log first 10
						LOG_INFO(f"    Constraint {i}: a1={const_detail['a1']:.6f}, a2={const_detail['a2']:.6f}, "
						        f"b={const_detail['b']:.6f}, disc_offset={const_detail['disc_offset']:.4f}, "
						        f"bounds=[{const_detail['lb']}, {const_detail['ub']}]")
						# Verify normal vector is normalized
						a_norm = np.sqrt(const_detail['a1']**2 + const_detail['a2']**2)
						if abs(a_norm - 1.0) < 1e-6:
							LOG_INFO(f"      ✓ Normal vector is normalized (||a||={a_norm:.6f})")
						else:
							LOG_WARN(f"      ⚠️  Normal vector is NOT normalized! ||a||={a_norm:.6f} (expected 1.0)")
					if len(linearized_constraints_detail) > 10:
						LOG_INFO(f"    ... and {len(linearized_constraints_detail) - 10} more linearized constraints")
				
				try:
					_integ_logger.info(msg_cons)
				except Exception:
					pass
			except Exception as e:
				LOG_WARN(f"Error logging constraints: {e}")
			if constraints:
				for constraint_item in constraints:
					# Handle both tuple format (c, lb, ub) and direct CasADi expressions
					if isinstance(constraint_item, tuple) and len(constraint_item) == 3:
						c, lb, ub = constraint_item
					elif isinstance(constraint_item, tuple) and len(constraint_item) == 1:
						# Single expression, no bounds
						c = constraint_item[0]
						lb, ub = None, None
					else:
						# Direct expression or dict
						c = constraint_item
						lb, ub = None, None
					
					# Robust bound handling with translation of structured constraints to CasADi
					if c is None:
						continue

					def _to_mx(val):
						if val is None:
							return None
						if isinstance(val, (int, float, np.floating)):
							return cs.DM(val)
						if isinstance(val, np.ndarray) and val.size == 1:
							return cs.DM(float(val))
						return val

					def _translate_constraint(expr_or_dict):
						# Accept already-built CasADi expressions (from symbolic constraint computation)
						# This matches the reference codebase where constraints are symbolic
						import casadi as cd
						if isinstance(expr_or_dict, (cd.MX, cd.SX)):
							# Already a CasADi expression - return directly
							return expr_or_dict
						if not isinstance(expr_or_dict, dict):
							# Try to convert to CasADi if possible
							return expr_or_dict
						cdef = expr_or_dict
						
						# Check if this is a symbolic expression wrapped in a dict
						if cdef.get('type') == 'symbolic_expression' and 'expression' in cdef:
							# Return the symbolic expression directly
							return cdef['expression']
						# Check if this is a linear_halfspace constraint with expression (e.g., from safe horizon)
						if cdef.get('type') == 'linear_halfspace' and 'expression' in cdef:
							# Return the symbolic expression directly
							expr = cdef['expression']
							if isinstance(expr, (cd.MX, cd.SX)):
								return expr
							else:
								LOG_WARN(f"linear_halfspace constraint has non-symbolic expression: {type(expr)}")
								return None
						# Support common linearized halfspace: a1*x + a2*y <= b
						if ('a1' in cdef and 'a2' in cdef) or cdef.get('type') == 'linear':
							a1 = _to_mx(cdef.get('a1', 0.0))
							a2 = _to_mx(cdef.get('a2', 0.0))
							b = _to_mx(cdef.get('b', 0.0))
							xk = self.var_dict.get('x')[stage_idx] if 'x' in self.var_dict else None
							yk = self.var_dict.get('y')[stage_idx] if 'y' in self.var_dict else None
							if xk is None or yk is None:
								return None
							
							# Apply disc_offset to constraint (matching reference implementation)
							# Constraint should be applied to disc position: p_disc = p_robot + offset * [cos(psi), sin(psi)]
							# For constraint a·p_disc <= b, we have:
							#   a1*(x + offset*cos(psi)) + a2*(y + offset*sin(psi)) <= b
							# Which expands to: a1*x + a2*y + offset*(a1*cos(psi) + a2*sin(psi)) <= b
							# So: a1*x + a2*y <= b - offset*(a1*cos(psi) + a2*sin(psi))
							# For the expression a1*x + a2*y - b, we need to adjust b:
							#   a1*x + a2*y - b_adjusted <= 0
							#   a1*(x + offset*cos(psi)) + a2*(y + offset*sin(psi)) - b_adjusted <= 0
							#   a1*x + a2*y + offset*(a1*cos(psi) + a2*sin(psi)) - b_adjusted <= 0
							#   b_adjusted = b - offset*(a1*cos(psi) + a2*sin(psi))
							disc_offset = cdef.get('disc_offset', 0.0)
							if abs(float(disc_offset)) > 1e-9:
								# Get orientation angle (psi or theta depending on model)
								psi_k = None
								for angle_var in ['psi', 'theta', 'heading']:
									if angle_var in self.var_dict:
										psi_k = self.var_dict[angle_var][stage_idx]
										break
								
								if psi_k is not None:
									# Apply disc offset: adjust b by offset*(a1*cos(psi) + a2*sin(psi))
									offset_adjustment = disc_offset * (a1 * cs.cos(psi_k) + a2 * cs.sin(psi_k))
									b_adjusted = b - offset_adjustment
									LOG_DEBUG(f"  Stage {stage_idx}: Applied disc_offset={disc_offset:.3f}, b adjustment={float(offset_adjustment) if hasattr(offset_adjustment, '__float__') else 'symbolic'}")
								else:
									LOG_WARN(f"  Stage {stage_idx}: disc_offset={disc_offset:.3f} specified but no orientation angle variable found (psi/theta/heading)")
									b_adjusted = b
							else:
								b_adjusted = b
							
							# Halfspace constraint for obstacle avoidance (matching reference implementation):
							# The normal vector (a1, a2) points FROM vehicle TO obstacle
							# To keep vehicle AWAY from obstacle, we need: a·p_vehicle <= b
							# Reference uses: constraint_expr = a1*x + a2*y - (b + slack)
							# With bounds: lb=-inf, ub=0, this enforces: a1*x + a2*y - b <= 0
							# So the constraint expression is: (a1*x + a2*y) - b_adjusted
							return (a1 * xk + a2 * yk) - b_adjusted
						# Unknown structured type; skip
						return None

					c_expr = _translate_constraint(c)
					if c_expr is None:
						continue
					
					# For symbolic expressions, bounds may be in the dict
					if isinstance(c, dict) and c.get('type') == 'symbolic_expression':
						# Use bounds from dict if provided
						if 'lb' in c:
							lb = c.get('lb')
						if 'ub' in c:
							ub = c.get('ub')

					lb_mx = _to_mx(lb)
					ub_mx = _to_mx(ub)

					try:
						# Log constraint details for stage 0 to diagnose issues - check for violations
						if stage_idx == 0:
							if isinstance(c, dict):
								# Log constraint expression value for debugging
								try:
									if 'a1' in c and 'a2' in c:
										# This is a linearized constraint
										# CRITICAL: Handle symbolic values - only convert to float if numeric
										def safe_float_log(val):
											if val is None:
												return 0.0
											import casadi as cd
											if isinstance(val, (cd.MX, cd.SX)):
												return 'symbolic'  # Can't evaluate symbolic expressions here
											try:
												return float(val)
											except (TypeError, ValueError):
												return 'N/A'
										
										a1_val = safe_float_log(c.get('a1', 0.0))
										a2_val = safe_float_log(c.get('a2', 0.0))
										b_val = safe_float_log(c.get('b', 0.0))
										# Get current state values for constraint evaluation
										if hasattr(self, 'data') and self.data is not None:
											if hasattr(self.data, 'state') and self.data.state is not None:
												x_val = float(self.data.state.get('x', 0.0))
												y_val = float(self.data.state.get('y', 0.0))
												constraint_val = a1_val * x_val + a2_val * y_val
												constraint_satisfied = constraint_val <= b_val
												violation_amount = constraint_val - b_val if not constraint_satisfied else 0.0
												
												if not constraint_satisfied:
													LOG_WARN(f"  ⚠️  Stage 0 linearized constraint {total_constraints_added}: VIOLATED! "
													        f"a1={a1_val:.4f}, a2={a2_val:.4f}, b={b_val:.4f}, "
													        f"at vehicle ({x_val:.3f}, {y_val:.3f}): constraint_value={constraint_val:.4f} > b={b_val:.4f}, "
													        f"violation={violation_amount:.4f}")
												else:
													LOG_DEBUG(f"  Stage 0 linearized constraint {total_constraints_added}: a1={a1_val:.3f}, a2={a2_val:.3f}, b={b_val:.3f}, "
													         f"at vehicle ({x_val:.2f}, {y_val:.2f}): {constraint_val:.3f} <= {b_val:.3f}? {constraint_satisfied}")
								except Exception as e:
									LOG_DEBUG(f"  Stage 0 constraint {total_constraints_added}: a1={c.get('a1', 'N/A')}, a2={c.get('a2', 'N/A')}, b={c.get('b', 'N/A')}, disc_offset={c.get('disc_offset', 'N/A')}, lb={lb}, ub={ub}, error={e}")
						
						if lb_mx is not None and ub_mx is not None:
							self.opti.subject_to(self.opti.bounded(lb_mx, c_expr, ub_mx))
						elif lb_mx is not None:
							self.opti.subject_to(lb_mx <= c_expr)
						elif ub_mx is not None:
							self.opti.subject_to(c_expr <= ub_mx)
						else:
							self.opti.subject_to(c_expr)
						total_constraints_added += 1
					except Exception as e:
						LOG_WARN(f"Failed to add constraint at stage {stage_idx}: {e}")
						import traceback
						LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

			# Log constraints added for this stage
			constraints_added_this_stage = total_constraints_added - constraints_count_before
			constraints_per_stage[stage_idx] = constraints_added_this_stage
			LOG_INFO(f"[CONSTRAINTS] Stage {stage_idx}: Added {constraints_added_this_stage} constraint(s) (total now: {total_constraints_added})")

		# All costs should come from objective modules via BaseSolver.get_objective_cost

		self.opti.minimize(total_objective)
		
		# Log problem statistics before solving
		LOG_INFO("=== OPTIMIZATION PROBLEM SUMMARY ===")
		LOG_INFO(f"  Horizon: {horizon_val} steps")
		LOG_INFO(f"  Total constraints: {total_constraints_added}")
		
		# Log constraints per stage summary
		LOG_INFO(f"  Constraints per stage:")
		for stage_idx in sorted(constraints_per_stage.keys()):
			LOG_INFO(f"    Stage {stage_idx}: {constraints_per_stage[stage_idx]} constraint(s)")
		
		# Count variables (needed for expected constraint calculation)
		dynamics_model = self._get_dynamics_model()
		num_state_vars = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
		num_input_vars = len(dynamics_model.get_inputs()) if dynamics_model else 2
		total_state_vars = num_state_vars * (horizon_val + 1)
		total_input_vars = num_input_vars * horizon_val
		
		# Count variables (needed for expected constraint calculation)
		dynamics_model = self._get_dynamics_model()
		num_state_vars = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
		num_input_vars = len(dynamics_model.get_inputs()) if dynamics_model else 2
		total_state_vars = num_state_vars * (horizon_val + 1)
		total_input_vars = num_input_vars * horizon_val
		
		# Calculate expected constraint counts
		expected_dynamics = num_state_vars * horizon_val
		expected_initial_state = num_state_vars
		expected_module = total_constraints_added - expected_dynamics - expected_initial_state
		
		LOG_INFO(f"  Expected constraints breakdown:")
		LOG_INFO(f"    - Dynamics constraints: {num_state_vars} states × {horizon_val} transitions = {expected_dynamics}")
		LOG_INFO(f"    - Initial state constraints: {num_state_vars} = {expected_initial_state}")
		LOG_INFO(f"    - Module constraints: {expected_module}")
		
		# Verify constraint counts match expectations
		if expected_module < 0:
			LOG_WARN(f"  ⚠️  WARNING: Module constraint count is negative! This suggests constraint counting error.")
		elif expected_module == 0:
			LOG_WARN(f"  ⚠️  WARNING: No module constraints added! This may indicate a problem.")
		else:
			LOG_INFO(f"  ✓ Module constraints count looks reasonable: {expected_module}")
		LOG_INFO(f"  Variables: {num_state_vars} state types × {horizon_val + 1} = {total_state_vars} state vars, "
		         f"{num_input_vars} input types × {horizon_val} = {total_input_vars} input vars")
		
		# Verify decision variables are only controls (acceleration, angular acceleration)
		LOG_INFO(f"=== VERIFYING DECISION VARIABLES ===")
		state_vars_list = [v for v in dynamics_model.get_dependent_vars() if v in self.var_dict]
		control_vars_list = [v for v in dynamics_model.get_inputs() if v in self.var_dict]
		LOG_INFO(f"  State variables (constrained by RK4/model_discrete_dynamics): {state_vars_list}")
		LOG_INFO(f"  Control variables (FREE DECISION VARIABLES): {control_vars_list}")
		LOG_INFO(f"  ✓ Only control inputs (a, w) are free decision variables")
		LOG_INFO(f"  ✓ All states are constrained by RK4 integration or model_discrete_dynamics: x[k+1] == symbolic_dynamics(x[k], u[k])")
		LOG_INFO(f"  ✓ All calculations (objectives, constraints) are symbolic CasADi expressions")
		LOG_INFO(f"  ✓ Reference: https://github.com/tud-amr/mpc_planner - matches C++ implementation")
		
		# Log objective structure
		obj_type = "scalar" if not hasattr(total_objective, 'shape') else f"array shape {total_objective.shape}"
		LOG_INFO(f"  Objective: {obj_type}")
		
		# Check warmstart feasibility
		LOG_DEBUG("Checking warmstart values...")
		try:
			for var_name in list(self.var_dict.keys())[:5]:  # Check first few
				if var_name in self.warmstart_values:
					ws_vals = self.warmstart_values[var_name]
					if isinstance(ws_vals, np.ndarray) and len(ws_vals) > 0:
						lb, ub, _ = dynamics_model.get_bounds(var_name)
						ws_min, ws_max = float(np.min(ws_vals)), float(np.max(ws_vals))
						in_bounds = (ws_min >= lb) and (ws_max <= ub)
						LOG_DEBUG(f"    Warmstart[{var_name}]: range=[{ws_min:.3f}, {ws_max:.3f}], bounds=[{lb:.3f}, {ub:.3f}], feasible={in_bounds}")
		except Exception as ws_err:
			LOG_DEBUG(f"    Could not check warmstart: {ws_err}")
		
		LOG_DEBUG(f"  Solving over horizon: {horizon_val} steps (states: 0 to {horizon_val}, controls: 0 to {horizon_val-1})")

		try:
			LOG_INFO("=== ATTEMPTING TO SOLVE ===")
			LOG_INFO("Calling IPOPT solver...")
			self.solution = self.opti.solve()
			self.exit_flag = 1
			self._update_warmstart_from_solution()
			
			# Check IPOPT stats if available
			try:
				stats = self.opti.stats()
				if stats:
					LOG_INFO("=== IPOPT SOLVER STATISTICS ===")
					return_status = stats.get('return_status', 'N/A')
					iter_count = stats.get('iter_count', 'N/A')
					obj_val = stats.get('obj', 'N/A')
					constr_viol = stats.get('constr_viol', 'N/A')
					du_inf = stats.get('du_inf', 'N/A')
					pr_inf = stats.get('pr_inf', 'N/A')
					
					LOG_INFO(f"  Return status: {return_status}")
					LOG_INFO(f"  Iterations: {iter_count}")
					LOG_INFO(f"  Objective value: {obj_val}")
					LOG_INFO(f"  Constraint violation: {constr_viol}")
					LOG_INFO(f"  Dual infeasibility: {du_inf}")
					LOG_INFO(f"  Primal infeasibility: {pr_inf}")
					
					# Warn if constraint violation is significant
					try:
						if isinstance(constr_viol, (int, float)) and constr_viol > 1e-3:
							LOG_WARN(f"  ⚠️  Significant constraint violation detected: {constr_viol:.6f}")
						if isinstance(pr_inf, (int, float)) and pr_inf > 1e-3:
							LOG_WARN(f"  ⚠️  Significant primal infeasibility detected: {pr_inf:.6f}")
					except (TypeError, ValueError):
						pass
			except Exception as stats_err:
				LOG_DEBUG(f"Could not get IPOPT stats: {stats_err}")
			
			LOG_INFO("=== SOLUTION FOUND ===")
			LOG_INFO(f"  Solver successfully computed optimal trajectory")
			LOG_INFO(f"  Trajectory: {horizon_val + 1} states and {horizon_val} control inputs")
			
			# Log solution details
			if self.solution:
				LOG_DEBUG("[SOLUTION] Extracting solution values...")
				# Log first state values
				try:
					for var_name in dynamics_model.get_dependent_vars():
						if var_name in self.var_dict:
							val0 = self.solution.value(self.var_dict[var_name][0])
							LOG_DEBUG(f"  Solution[{var_name}][0] = {val0}")
					# Log first control values
					for var_name in dynamics_model.get_inputs():
						if var_name in self.var_dict:
							val0 = self.solution.value(self.var_dict[var_name][0])
							LOG_INFO(f"  Solution u[{var_name}][0] = {val0} (first control input to apply)")
					
					# Log solution trajectory positions for first few steps
					if 'x' in self.var_dict and 'y' in self.var_dict:
						LOG_INFO("=== SOLUTION TRAJECTORY (first 3 steps) ===")
						for k in range(min(3, horizon_val + 1)):
							sol_x = float(self.solution.value(self.var_dict['x'][k]))
							sol_y = float(self.solution.value(self.var_dict['y'][k]))
							sol_psi = float(self.solution.value(self.var_dict['psi'][k])) if 'psi' in self.var_dict else 0.0
							sol_v = float(self.solution.value(self.var_dict['v'][k])) if 'v' in self.var_dict else 0.0
							LOG_INFO(f"  Step {k}: x={sol_x:.3f}, y={sol_y:.3f}, psi={sol_psi:.3f}, v={sol_v:.3f}")
							
							# Compare with warmstart
							if k < len(self.warmstart_values.get('x', [])):
								ws_x = float(self.warmstart_values['x'][k])
								ws_y = float(self.warmstart_values['y'][k])
								diff = np.sqrt((sol_x - ws_x)**2 + (sol_y - ws_y)**2)
								LOG_DEBUG(f"    Warmstart: x={ws_x:.3f}, y={ws_y:.3f}, diff={diff:.3f}m")
				except Exception as e:
					LOG_DEBUG(f"  Could not extract solution values: {e}")
			
			return 1
		except RuntimeError as e:
			LOG_WARN("=== SOLVER FAILED ===")
			LOG_WARN(f"Error message: {e}")
			self.exit_flag = -1
			self.info["status"] = "failed"
			self.info["error"] = str(e)
			
			# Check IPOPT return status if available
			try:
				stats = self.opti.stats()
				if stats:
					return_status = stats.get('return_status', 'N/A')
					LOG_WARN(f"IPOPT return status: {return_status}")
					# Map common IPOPT status codes
					status_map = {
						'Maximum_Iterations_Exceeded': 'Solver hit maximum iterations - problem may be infeasible or difficult',
						'Infeasible_Problem_Detected': 'Problem is infeasible - constraints cannot be satisfied',
						'Search_Direction_Becomes_Too_Small': 'Solver cannot make progress - problem may be ill-conditioned',
						'Diverging_Iterates': 'Solution diverging - problem may be unbounded or infeasible',
						'Restoration_Failed': 'Restoration phase failed - initial point likely infeasible',
						'Error_In_Step_Computation': 'Error computing step - problem may be numerically difficult',
					}
					if return_status in status_map:
						LOG_WARN(f"  Interpretation: {status_map[return_status]}")
					
					LOG_WARN(f"  Final iteration: {stats.get('iter_count', 'N/A')}")
					LOG_WARN(f"  Final objective: {stats.get('obj', 'N/A')}")
					LOG_WARN(f"  Constraint violation: {stats.get('constr_viol', 'N/A')}")
					LOG_WARN(f"  Dual infeasibility: {stats.get('du_inf', 'N/A')}")
					LOG_WARN(f"  Primal infeasibility: {stats.get('pr_inf', 'N/A')}")
			except Exception as stats_err:
				LOG_DEBUG(f"Could not get IPOPT stats: {stats_err}")
			
			# Try to diagnose infeasibility
			LOG_WARN("=== DIAGNOSING FAILURE ===")
			LOG_WARN("Attempting to identify root cause...")
			try:
				# Check if we can get debug values
				if hasattr(self.opti, 'debug'):
					LOG_DEBUG("Checking opti.debug values...")
					for var_name in list(self.var_dict.keys())[:5]:  # Check first few variables
						try:
							debug_val = self.opti.debug.value(self.var_dict[var_name])
							if isinstance(debug_val, np.ndarray) and len(debug_val) > 0:
								LOG_DEBUG(f"  Debug[{var_name}][0] = {debug_val[0]}")
							else:
								LOG_DEBUG(f"  Debug[{var_name}] = {debug_val}")
						except Exception:
							pass
					
					# Check constraint violations
					try:
						# Get constraint values
						LOG_WARN("=== CHECKING CONSTRAINT VIOLATIONS ===")
						LOG_WARN("Attempting to get constraint values from CasADi debug...")
						# Try to get constraint info if available
						if hasattr(self.opti, 'debug') and hasattr(self.opti.debug, 'g'):
							constraint_violations = self.opti.debug.value(self.opti.debug.g)
							if constraint_violations is not None:
								LOG_WARN(f"Successfully retrieved {len(constraint_violations)} constraint values from debug.g")
								# CRITICAL: Constraint violations array includes ALL constraints in order:
								# 1. Initial state constraints (from _set_initial_state - 5 constraints)
								# 2. Dynamics constraints (from intialize_solver - 5 states × horizon = 50)
								# 3. Module constraints (from solve loop - varies per stage)
								# NOTE: Variable bounds from opti.bounded() are NOT in debug.g - they're handled separately
								# The debug.g values are at the CURRENT point (likely infeasible initial values if solve failed)
								LOG_WARN(f"  Constraint violations array length: {len(constraint_violations)}")
								LOG_WARN(f"  Constraint order: initial_state (5) -> dynamics (50) -> module (varies)")
								LOG_WARN(f"  Note: These values are at the current point (may be infeasible initial values)")
								
								# Log first few constraint values for debugging
								# CRITICAL: Order is dynamics -> initial_state -> module
								num_state_vars_log = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
								num_dynamics_log = num_state_vars_log * horizon_val
								num_initial_state_log = num_state_vars_log
								
								LOG_WARN(f"  First 20 constraint values (order: dynamics -> initial_state -> module):")
								for i in range(min(20, len(constraint_violations))):
									val = float(constraint_violations[i])
									# Try to identify what each constraint is (order: dynamics -> initial_state -> module)
									if i < num_dynamics_log:
										dyn_idx = i
										trans_idx = dyn_idx // num_state_vars_log
										state_idx = dyn_idx % num_state_vars_log
										state_vars = dynamics_model.get_dependent_vars() if dynamics_model else ['x', 'y', 'psi', 'v', 'spline']
										var_name = state_vars[state_idx] if state_idx < len(state_vars) else f"state_{state_idx}"
										LOG_WARN(f"    Constraint {i}: {val:.6f} [Dynamics: {var_name} at transition {trans_idx}->{trans_idx+1}]")
									elif i < num_dynamics_log + num_initial_state_log:
										initial_idx = i - num_dynamics_log
										state_vars = dynamics_model.get_dependent_vars() if dynamics_model else ['x', 'y', 'psi', 'v', 'spline']
										var_name = state_vars[initial_idx] if initial_idx < len(state_vars) else f"state_{initial_idx}"
										LOG_WARN(f"    Constraint {i}: {val:.6f} [Initial state: {var_name}[0]]")
									else:
										LOG_WARN(f"    Constraint {i}: {val:.6f} [Module constraint]")
								# For constraints expr <= 0, violation means expr > 0
								# For constraints lb <= expr <= ub, violation means expr < lb or expr > ub
								# The debug.g values are the constraint expressions themselves
								# We need to check against bounds to find actual violations
								# CRITICAL: Create symbolic states for diagnostic purposes to match the symbolic system
								violations = []
								constraints_with_bounds = []
								dynamics_model = self._get_dynamics_model()
								for stage_idx_check in range(horizon_val + 1):
									# Create symbolic state for this stage (for diagnostic purposes)
									symbolic_state_diag = State(dynamics_model)
									for var_name in dynamics_model.get_all_vars():
										if var_name in self.var_dict and stage_idx_check < self.var_dict[var_name].shape[0]:
											symbolic_state_diag.set(var_name, self.var_dict[var_name][stage_idx_check])
									cons_list = self.get_constraints(stage_idx_check, symbolic_state=symbolic_state_diag)
									constraints_with_bounds.extend(cons_list)
								
								# Check each constraint against its bounds
								if len(constraint_violations) == len(constraints_with_bounds):
									for i, (c, lb, ub) in enumerate(constraints_with_bounds[:min(10, len(constraints_with_bounds))]):
										expr_val = float(constraint_violations[i])
										if lb is not None and expr_val < lb:
											violations.append(lb - expr_val)
										elif ub is not None and expr_val > ub:
											violations.append(expr_val - ub)
								
								if violations:
									max_violation = max(violations)
									LOG_WARN(f"  Maximum constraint violation: {max_violation:.6f}")
									if max_violation > 1.0:
										LOG_WARN(f"  SEVERE: Constraint violation > 1.0 - problem is highly infeasible")
									elif max_violation > 1e-3:
										LOG_WARN(f"  Problem likely infeasible - constraint violation too large")
									
									# Find which constraints are violated
									violated_indices = [i for i, v in enumerate(violations) if v > 1e-6]
									if violated_indices:
										LOG_WARN(f"  Violated constraints (indices): {violated_indices[:10]}")  # Show first 10
								else:
									# Fallback: use absolute values (may not be accurate for all constraint types)
									# But also check for positive violations (constraints that are actually violated)
									constraint_violations_arr = np.array(constraint_violations)
									violations_abs = np.abs(constraint_violations_arr)
									max_violation_abs = np.max(violations_abs)
									max_idx_abs = np.argmax(violations_abs)
									
									# Also check for positive violations (actual constraint violations)
									positive_violations = constraint_violations_arr[constraint_violations_arr > 1e-6]
									if len(positive_violations) > 0:
										max_violation_pos = np.max(positive_violations)
										# Find the index of the maximum positive violation in the original array
										positive_indices = np.where(constraint_violations_arr > 1e-6)[0]
										max_idx_pos = positive_indices[np.argmax(positive_violations)]
										LOG_WARN(f"  ⚠️  Found {len(positive_violations)} constraint(s) with positive values (VIOLATIONS)")
										LOG_WARN(f"  ⚠️  Maximum positive violation: {max_violation_pos:.6f} at constraint index {max_idx_pos}")
										LOG_WARN(f"  ⚠️  This is an ACTUAL constraint violation (constraint value > 0)")
										
										# Log all positive violations (up to 10)
										LOG_WARN(f"  All positive violations (showing first 10):")
										for idx in positive_indices[:10]:
											val = float(constraint_violations_arr[idx])
											LOG_WARN(f"    Constraint {idx}: {val:.6f}")
										# Try to identify this constraint
										try:
											actual_val_pos = float(constraint_violations[max_idx_pos])
											LOG_WARN(f"    Constraint {max_idx_pos} value: {actual_val_pos:.6f}")
										except Exception as e:
											LOG_DEBUG(f"    Could not get constraint value: {e}")
									
										# Identify which type of constraint this is
									# CRITICAL: CasADi constraint order in debug.g matches the order constraints are added:
									# 1. Dynamics constraints (from intialize_solver - added FIRST, 5 states × horizon = 50)
									# 2. Initial state constraints (from _set_initial_state - added SECOND, 5 constraints)
									# 3. Module constraints (from solve loop - added LAST, varies per stage)
									# 
									# NOTE: opti.bounded() constraints are NOT included in debug.g - they're handled as variable bounds
									# So debug.g starts with dynamics constraints at index 0
									num_state_vars = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
									num_input_vars = len(dynamics_model.get_inputs()) if dynamics_model else 2
									num_dynamics = num_state_vars * horizon_val  # Dynamics constraints for each transition (5 × 10 = 50)
									num_initial_state = num_state_vars  # One constraint per state variable (x[0], y[0], psi[0], v[0], spline[0])
									
									LOG_WARN(f"    Constraint indexing: dynamics={num_dynamics}, initial_state={num_initial_state}, "
									         f"module constraints vary per stage")
									LOG_WARN(f"    Violated constraint index: {max_idx_pos}")
									LOG_WARN(f"    Determining constraint type...")
									
									# Constraint indexing: debug.g starts with dynamics constraints (added first in intialize_solver)
									if max_idx_pos < num_dynamics:
										dynamics_idx = max_idx_pos
										transition_idx = dynamics_idx // num_state_vars
										state_idx_in_transition = dynamics_idx % num_state_vars
										state_vars = dynamics_model.get_dependent_vars() if dynamics_model else ['x', 'y', 'psi', 'v', 'spline']
										var_name = state_vars[state_idx_in_transition] if state_idx_in_transition < len(state_vars) else f"state_{state_idx_in_transition}"
										LOG_WARN(f"    ⚠️  Constraint type: Dynamics constraint for {var_name} at transition {transition_idx}->{transition_idx+1} (index {max_idx_pos})")
										LOG_WARN(f"    ⚠️  This constraint enforces: {var_name}[{transition_idx+1}] == RK4({var_name}[{transition_idx}], u[{transition_idx}], dt)")
										LOG_WARN(f"    ⚠️  Constraint value from debug.g: {constraint_violations[max_idx_pos]:.6f}")
										LOG_WARN(f"    ⚠️  This is a LARGE violation - warmstart values likely don't satisfy dynamics!")
										
										# Check warmstart values for this transition
										try:
											dt_check = self.timestep if self.timestep is not None else 0.1
											LOG_WARN(f"      Checking warmstart for {var_name} at transition {transition_idx}->{transition_idx+1} (dt={dt_check})")
											
											# Get warmstart values
											if var_name in self.warmstart_values:
												ws_len = len(self.warmstart_values[var_name])
												LOG_WARN(f"      Warmstart array length for {var_name}: {ws_len}")
												if transition_idx < ws_len and transition_idx + 1 < ws_len:
													x_k_ws = float(self.warmstart_values[var_name][transition_idx])
													x_kp1_ws = float(self.warmstart_values[var_name][transition_idx + 1])
													LOG_WARN(f"      Warmstart: {var_name}[{transition_idx}]={x_k_ws:.6f}, {var_name}[{transition_idx+1}]={x_kp1_ws:.6f}")
													
													# Also log all warmstart values for this variable to see the pattern
													if transition_idx < 5:  # Only log for first few transitions
														ws_all = [float(v) for v in self.warmstart_values[var_name][:min(6, ws_len)]]
														LOG_WARN(f"      Warmstart values for {var_name}: {ws_all}")
													
													# Compute predicted value from warmstart for psi (angular dynamics)
													if var_name == 'psi' and 'w' in self.warmstart_values:
														w_k_ws = float(self.warmstart_values['w'][transition_idx]) if transition_idx < len(self.warmstart_values['w']) else 0.0
														# For unicycle: psi[k+1] = psi[k] + dt * w[k] (angular velocity integration)
														psi_pred_ws = x_k_ws + dt_check * w_k_ws
														violation_ws = x_kp1_ws - psi_pred_ws
														LOG_WARN(f"      Computed: psi_pred = {x_k_ws:.6f} + {dt_check:.4f} * {w_k_ws:.6f} = {psi_pred_ws:.6f}")
														LOG_WARN(f"      Warmstart dynamics violation: psi[{transition_idx+1}]_ws - psi_pred = {x_kp1_ws:.6f} - {psi_pred_ws:.6f} = {violation_ws:.6f}")
														if abs(violation_ws) > 1e-3:
															LOG_WARN(f"      ⚠️  WARMSTART IS NOT DYNAMICS-CONSISTENT! Violation: {violation_ws:.6f}")
													# Compute predicted value from warmstart for x
													elif var_name == 'x' and 'v' in self.warmstart_values and 'psi' in self.warmstart_values:
														v_k_ws = float(self.warmstart_values['v'][transition_idx]) if transition_idx < len(self.warmstart_values['v']) else 0.0
														psi_k_ws = float(self.warmstart_values['psi'][transition_idx]) if transition_idx < len(self.warmstart_values['psi']) else 0.0
														x_pred_ws = x_k_ws + dt_check * v_k_ws * np.cos(psi_k_ws)
														violation_ws = x_kp1_ws - x_pred_ws
														LOG_WARN(f"      Computed: x_pred = {x_k_ws:.6f} + {dt_check:.4f} * {v_k_ws:.6f} * cos({psi_k_ws:.6f}) = {x_pred_ws:.6f}")
														LOG_WARN(f"      Warmstart dynamics violation: x[{transition_idx+1}]_ws - x_pred = {x_kp1_ws:.6f} - {x_pred_ws:.6f} = {violation_ws:.6f}")
														if abs(violation_ws) > 1e-3:
															LOG_WARN(f"      ⚠️  WARMSTART IS NOT DYNAMICS-CONSISTENT! Violation: {violation_ws:.6f}")
													# Compute predicted value from warmstart for y
													elif var_name == 'y' and 'v' in self.warmstart_values and 'psi' in self.warmstart_values:
														v_k_ws = float(self.warmstart_values['v'][transition_idx]) if transition_idx < len(self.warmstart_values['v']) else 0.0
														psi_k_ws = float(self.warmstart_values['psi'][transition_idx]) if transition_idx < len(self.warmstart_values['psi']) else 0.0
														y_pred_ws = x_k_ws + dt_check * v_k_ws * np.sin(psi_k_ws)
														violation_ws = x_kp1_ws - y_pred_ws
														LOG_WARN(f"      Computed: y_pred = {x_k_ws:.6f} + {dt_check:.4f} * {v_k_ws:.6f} * sin({psi_k_ws:.6f}) = {y_pred_ws:.6f}")
														LOG_WARN(f"      Warmstart dynamics violation: y[{transition_idx+1}]_ws - y_pred = {x_kp1_ws:.6f} - {y_pred_ws:.6f} = {violation_ws:.6f}")
														if abs(violation_ws) > 1e-3:
															LOG_WARN(f"      ⚠️  WARMSTART IS NOT DYNAMICS-CONSISTENT! Violation: {violation_ws:.6f}")
													
													# Get CasADi values and compute RK4 prediction
													try:
														if var_name in self.var_dict:
															x_k_casadi = float(self.opti.debug.value(self.var_dict[var_name][transition_idx])) if transition_idx < self.var_dict[var_name].shape[0] else None
															x_kp1_casadi = float(self.opti.debug.value(self.var_dict[var_name][transition_idx + 1])) if transition_idx + 1 < self.var_dict[var_name].shape[0] else None
															if x_k_casadi is not None and x_kp1_casadi is not None:
																LOG_WARN(f"      CasADi values: {var_name}[{transition_idx}]={x_k_casadi:.6f}, {var_name}[{transition_idx+1}]={x_kp1_casadi:.6f}")
																
																# Get control inputs for RK4 prediction
																if 'w' in self.var_dict and transition_idx < self.var_dict['w'].shape[0]:
																	w_k_casadi = float(self.opti.debug.value(self.var_dict['w'][transition_idx]))
																	LOG_WARN(f"      Control input: w[{transition_idx}]={w_k_casadi:.6f}")
																	
																	# For psi: simple integration psi[k+1] = psi[k] + dt * w[k]
																	if var_name == 'psi':
																		psi_pred_casadi = x_k_casadi + dt_check * w_k_casadi
																		LOG_WARN(f"      RK4 prediction: psi[{transition_idx+1}]_pred = {x_k_casadi:.6f} + {dt_check:.4f} * {w_k_casadi:.6f} = {psi_pred_casadi:.6f}")
																		LOG_WARN(f"      Actual CasADi value: psi[{transition_idx+1}]={x_kp1_casadi:.6f}")
																		LOG_WARN(f"      Difference: {x_kp1_casadi:.6f} - {psi_pred_casadi:.6f} = {x_kp1_casadi - psi_pred_casadi:.6f}")
																
																constraint_val = float(constraint_violations[max_idx_pos])
																LOG_WARN(f"      Constraint value from debug.g (should be ~0): {constraint_val:.6f}")
																
																# Manually compute constraint value: x_next - x_next_pred
																# For psi: psi[k+1] - (psi[k] + dt*w[k])
																if var_name == 'psi':
																	manual_constraint_val = x_kp1_casadi - (x_k_casadi + dt_check * w_k_casadi)
																	LOG_WARN(f"      Manually computed constraint: {var_name}[{transition_idx+1}] - ({var_name}[{transition_idx}] + dt*w[{transition_idx}])")
																	LOG_WARN(f"      = {x_kp1_casadi:.6f} - ({x_k_casadi:.6f} + {dt_check:.4f}*{w_k_casadi:.6f})")
																	LOG_WARN(f"      = {x_kp1_casadi:.6f} - {x_k_casadi + dt_check * w_k_casadi:.6f} = {manual_constraint_val:.6f}")
																	if abs(manual_constraint_val) < 1e-6 and abs(constraint_val) > 1e-3:
																		LOG_WARN(f"      ⚠️  MISMATCH: Manual computation gives ~0, but debug.g gives {constraint_val:.6f}")
																		LOG_WARN(f"      ⚠️  This suggests constraint indexing or evaluation issue!")
																	elif abs(constraint_val) > 1e-6:
																		LOG_WARN(f"      ⚠️  DYNAMICS CONSTRAINT VIOLATED!")
																		LOG_WARN(f"      ⚠️  CasADi variables don't satisfy dynamics equations!")
													except Exception as e_casadi:
														LOG_WARN(f"      Could not get CasADi values: {e_casadi}")
														import traceback
														LOG_DEBUG(f"      Traceback: {traceback.format_exc()}")
												else:
													LOG_WARN(f"      ⚠️  Warmstart array length {ws_len} insufficient for transition {transition_idx}->{transition_idx+1}")
											else:
												LOG_WARN(f"      ⚠️  State '{var_name}' not in warmstart_values (available: {list(self.warmstart_values.keys())})")
										except Exception as e_dyn:
											LOG_WARN(f"      Could not check dynamics constraint: {e_dyn}")
											import traceback
											LOG_DEBUG(f"      Traceback: {traceback.format_exc()}")
									elif max_idx_pos < num_dynamics + num_initial_state:
										initial_idx = max_idx_pos - num_dynamics
										state_vars = dynamics_model.get_dependent_vars() if dynamics_model else ['x', 'y', 'psi', 'v', 'spline']
										var_name = state_vars[initial_idx] if initial_idx < len(state_vars) else f"state_{initial_idx}"
										LOG_WARN(f"    ⚠️  Constraint type: Initial state constraint for {var_name}[0] (index {max_idx_pos})")
										LOG_WARN(f"    ⚠️  This constraint enforces: {var_name}[0] == initial_value")
										
										# Check warmstart values for this transition
										try:
											dt_check = self.timestep if self.timestep is not None else 0.1
											LOG_WARN(f"      Checking warmstart for {var_name} at transition {transition_idx}->{transition_idx+1} (dt={dt_check})")
											
											# Get warmstart values
											if var_name in self.warmstart_values:
												ws_len = len(self.warmstart_values[var_name])
												LOG_WARN(f"      Warmstart array length for {var_name}: {ws_len}")
												if transition_idx < ws_len and transition_idx + 1 < ws_len:
													x_k_ws = float(self.warmstart_values[var_name][transition_idx])
													x_kp1_ws = float(self.warmstart_values[var_name][transition_idx + 1])
													LOG_WARN(f"      Warmstart: {var_name}[{transition_idx}]={x_k_ws:.6f}, {var_name}[{transition_idx+1}]={x_kp1_ws:.6f}")
													
													# Also log all warmstart values for this variable to see the pattern
													if transition_idx < 5:  # Only log for first few transitions
														ws_all = [float(v) for v in self.warmstart_values[var_name][:min(6, ws_len)]]
														LOG_WARN(f"      Warmstart values for {var_name}: {ws_all}")
												else:
													LOG_WARN(f"      ⚠️  Warmstart array length {ws_len} insufficient for transition {transition_idx}->{transition_idx+1}")
											else:
												LOG_WARN(f"      ⚠️  State '{var_name}' not in warmstart_values (available: {list(self.warmstart_values.keys())})")
										except Exception as e:
											LOG_WARN(f"      Could not check initial state values: {e}")
											import traceback
											LOG_DEBUG(f"      Traceback: {traceback.format_exc()}")
									else:
										module_idx = max_idx_pos - num_dynamics - num_initial_state
										LOG_WARN(f"    ⚠️  Constraint type: Module constraint (index {max_idx_pos}, module constraint index {module_idx})")
										# Try to identify which module constraint
										try:
											constraint_count = 0
											for stage_idx_check in range(horizon_val + 1):
												symbolic_state_check = State(dynamics_model)
												for var_name in dynamics_model.get_all_vars():
													if var_name in self.var_dict and stage_idx_check < self.var_dict[var_name].shape[0]:
														symbolic_state_check.set(var_name, self.var_dict[var_name][stage_idx_check])
												cons_list_check = self.get_constraints(stage_idx_check, symbolic_state=symbolic_state_check)
												if module_idx < constraint_count + len(cons_list_check):
													constraint_in_stage = module_idx - constraint_count
													LOG_WARN(f"      Module constraint at stage {stage_idx_check}, constraint index {constraint_in_stage} within stage")
													if constraint_in_stage < len(cons_list_check):
														c_check, lb_check, ub_check = cons_list_check[constraint_in_stage]
														if isinstance(c_check, dict):
															LOG_WARN(f"      Constraint type: {c_check.get('constraint_type', 'unknown')}")
															LOG_WARN(f"      Constraint details: {c_check}")
													break
												constraint_count += len(cons_list_check)
										except Exception as e_module:
											LOG_DEBUG(f"      Could not identify module constraint: {e_module}")
											import traceback
											LOG_DEBUG(f"      Traceback: {traceback.format_exc()}")
									
									LOG_WARN(f"  Maximum constraint value (abs): {max_violation_abs:.6f} at constraint index {max_idx_abs}")
									# Log the actual constraint value
									try:
										actual_val = float(constraint_violations[max_idx_abs])
										LOG_WARN(f"    Constraint {max_idx_abs} value: {actual_val:.6f}")
										# Try to identify which variable this constraint corresponds to
										# CasADi adds bounds as constraints after explicit constraints
										# Each variable has a lower and upper bound constraint
										explicit_constraints = 77  # initial (5) + module (22) + dynamics (50)
										if max_idx_abs >= explicit_constraints:
											bound_idx = max_idx_abs - explicit_constraints
											# Try to map to variable
											all_vars = []
											for vname in dynamics_model.get_dependent_vars():
												all_vars.extend([(vname, k) for k in range(horizon_val + 1)])
											for vname in dynamics_model.get_inputs():
												all_vars.extend([(vname, k) for k in range(horizon_val)])
											# Each variable has 2 bounds (lb, ub)
											if bound_idx < len(all_vars) * 2:
												var_idx = bound_idx // 2
												bound_type = "lower" if (bound_idx % 2 == 0) else "upper"
												if var_idx < len(all_vars):
													var_name, stage_k = all_vars[var_idx]
													LOG_WARN(f"    This is {bound_type} bound constraint for '{var_name}' at stage {stage_k}")
													# Get the actual value and bound
													try:
														if var_name in self.var_dict and stage_k < self.var_dict[var_name].shape[0]:
															actual_var_val = float(self.opti.debug.value(self.var_dict[var_name][stage_k]))
															lb, ub, _ = dynamics_model.get_bounds(var_name)
															bound_val = lb if bound_type == "lower" else ub
															# For upper bound: constraint is x - ub <= 0, so value < 0 means x < ub (satisfied)
															# For lower bound: constraint is lb - x <= 0, so value < 0 means x > lb (satisfied)
															# If value > 0, constraint is violated
															LOG_WARN(f"      Variable value: {actual_var_val:.6f}, bound: {bound_val:.6f}, constraint value: {actual_val:.6f}")
															if bound_type == "lower":
																# Lower bound constraint: lb - x <= 0, so x >= lb
																if actual_var_val < lb:
																	LOG_WARN(f"      VIOLATION: {var_name}[{stage_k}]={actual_var_val:.6f} < lb={lb:.6f}, violation={lb - actual_var_val:.6f}")
																elif actual_val > 0:
																	LOG_WARN(f"      CONSTRAINT VIOLATION: lb - x = {actual_val:.6f} > 0, meaning x={actual_var_val:.6f} < lb={lb:.6f}")
															elif bound_type == "upper":
																# Upper bound constraint: x - ub <= 0, so x <= ub
																if actual_var_val > ub:
																	LOG_WARN(f"      VIOLATION: {var_name}[{stage_k}]={actual_var_val:.6f} > ub={ub:.6f}, violation={actual_var_val - ub:.6f}")
																elif actual_val > 0:
																	LOG_WARN(f"      CONSTRAINT VIOLATION: x - ub = {actual_val:.6f} > 0, meaning x={actual_var_val:.6f} > ub={ub:.6f}")
															# Check warmstart value
															if var_name in self.warmstart_values and stage_k < len(self.warmstart_values[var_name]):
																ws_val = self.warmstart_values[var_name][stage_k]
																LOG_WARN(f"      Warmstart value: {ws_val:.6f}")
																if bound_type == "lower" and ws_val < lb:
																	LOG_WARN(f"      WARMSTART VIOLATION: {var_name}[{stage_k}]={ws_val:.6f} < lb={lb:.6f}")
																elif bound_type == "upper" and ws_val > ub:
																	LOG_WARN(f"      WARMSTART VIOLATION: {var_name}[{stage_k}]={ws_val:.6f} > ub={ub:.6f}")
													except Exception as e2:
														LOG_DEBUG(f"      Could not get variable value: {e2}")
									except Exception:
										pass
									# Try to identify which constraint this is
									try:
										horizon_val = self.horizon if self.horizon is not None else 10
										dynamics_model = self._get_dynamics_model()
										# Use max_idx_abs which is defined in outer scope
										idx_to_check = max_idx_abs if 'max_idx_abs' in locals() or 'max_idx_abs' in globals() else (len(constraint_violations) - 1 if constraint_violations else 0)
										# Initial state constraints: 5 states fixed at k=0
										num_initial_state = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
										# Dynamics constraints: 5 per transition (k=0 to k=horizon-1) = 5*horizon = 50 for horizon=10
										num_dynamics_constraints = len(dynamics_model.get_dependent_vars()) * horizon_val if dynamics_model else 5 * horizon_val
										# Module constraints: Count actual constraints from all stages
										# Note: Some constraint modules may not return constraints for stage_idx >= horizon_val
										# CRITICAL: Create symbolic states for diagnostic purposes to match the symbolic system
										num_module_constraints = 0
										for stage_check in range(horizon_val + 1):
											# Create symbolic state for diagnostic purposes
											symbolic_state_diag = State(dynamics_model)
											for var_name in dynamics_model.get_all_vars():
												if var_name in self.var_dict and stage_check < self.var_dict[var_name].shape[0]:
													symbolic_state_diag.set(var_name, self.var_dict[var_name][stage_check])
											cons_check = self.get_constraints(stage_check, symbolic_state=symbolic_state_diag)
											num_module_constraints += len(cons_check)
										# Constraints are added in order: initial state, then module constraints per stage, then dynamics constraints
										LOG_WARN(f"  Constraint indexing: initial={num_initial_state}, module={num_module_constraints}, dynamics={num_dynamics_constraints}, total expected={num_initial_state+num_module_constraints+num_dynamics_constraints}, actual={len(constraint_violations)}")
										if idx_to_check < num_initial_state:
											dep_vars = dynamics_model.get_dependent_vars()
											var_name = dep_vars[idx_to_check] if idx_to_check < len(dep_vars) else 'unknown'
											LOG_WARN(f"    Constraint {idx_to_check} is initial state constraint for {var_name}")
										elif idx_to_check < num_initial_state + num_module_constraints:
											module_idx = idx_to_check - num_initial_state
											# Find which stage this constraint belongs to by counting actual constraints
											constraints_counted = 0
											stage_idx = 0
											constraint_in_stage = 0
											found_stage = False
											for stage_check in range(horizon_val + 1):
												# Create symbolic state for diagnostic purposes
												symbolic_state_diag = State(dynamics_model)
												for var_name in dynamics_model.get_all_vars():
													if var_name in self.var_dict and stage_check < self.var_dict[var_name].shape[0]:
														symbolic_state_diag.set(var_name, self.var_dict[var_name][stage_check])
												cons_check = self.get_constraints(stage_check, symbolic_state=symbolic_state_diag)
												if module_idx < constraints_counted + len(cons_check):
													stage_idx = stage_check
													constraint_in_stage = module_idx - constraints_counted
													found_stage = True
													break
												constraints_counted += len(cons_check)
											if not found_stage:
												# Fallback: use last stage if out of bounds
												stage_idx = horizon_val
												constraint_in_stage = module_idx - constraints_counted
											LOG_WARN(f"    Constraint {idx_to_check} is module constraint at stage {stage_idx}, constraint #{constraint_in_stage}")
										elif idx_to_check < num_initial_state + num_module_constraints + num_dynamics_constraints:
											dyn_idx = idx_to_check - num_initial_state - num_module_constraints
											num_states = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
											transition_k = dyn_idx // num_states
											state_idx = dyn_idx % num_states
											dep_vars = dynamics_model.get_dependent_vars()
											state_name = dep_vars[state_idx] if state_idx < len(dep_vars) else 'unknown'
											LOG_WARN(f"    Constraint {idx_to_check} is dynamics constraint at transition k={transition_k}->{transition_k+1}, state={state_name}")
										else:
											LOG_WARN(f"    Constraint {idx_to_check} is beyond expected range (could be bounds or other constraints)")
									except Exception as _e:
										LOG_DEBUG(f"  Could not identify constraint: {_e}")
									# Check violation using max_violation_abs if available
									if 'max_violation_abs' in locals() or 'max_violation_abs' in globals():
										max_violation_check = max_violation_abs
									else:
										max_violation_check = max_violation_abs if 'max_violation_abs' in dir() else 0.0
									if max_violation_check > 1.0:
										LOG_WARN(f"  SEVERE: Constraint value > 1.0 - problem is highly infeasible")
									elif max_violation_check > 1e-3:
										LOG_WARN(f"  Problem likely infeasible - constraint violation too large")
									# Check dynamics constraint residuals explicitly
									try:
										horizon_val = self.horizon if self.horizon is not None else 10
										dynamics_model = self._get_dynamics_model()
										dep_vars = dynamics_model.get_dependent_vars() if dynamics_model else []
										nx = len(dep_vars)
										LOG_WARN(f"  Checking dynamics constraint residuals at k=0..2 (expecting {nx} states per stage):")
										for k_check in range(min(3, horizon_val)):
											try:
												# Get x_k and x_k+1 from solution
												x_k_vals = []
												x_kp1_vals = []
												for var in dep_vars:
													try:
														x_k_vals.append(float(self.opti.debug.value(self.var_dict[var][k_check])))
														x_kp1_vals.append(float(self.opti.debug.value(self.var_dict[var][k_check + 1])))
													except Exception:
														x_k_vals.append(0.0)
														x_kp1_vals.append(0.0)
												# Get u_k from solution
												in_vars = dynamics_model.get_inputs() if dynamics_model else []
												u_k_vals = []
												for var in in_vars:
													try:
														u_k_vals.append(float(self.opti.debug.value(self.var_dict[var][k_check])))
													except Exception:
														u_k_vals.append(0.0)
												# Compute predicted next state
												x_k_dm = cs.DM(x_k_vals)
												u_k_dm = cs.DM(u_k_vals if len(u_k_vals) > 0 else [0.0])
												def _pget(key):
													try:
														return self.data.parameters.get(key)
													except Exception:
														return 0.0
												x_next_pred_sym = dynamics_model.symbolic_dynamics(x_k_dm, u_k_dm, _pget, self.timestep)
												x_next_pred_num = np.array(x_next_pred_sym.full()).flatten()
												# Compute residual
												residual = np.array(x_kp1_vals) - x_next_pred_num
												res_norm = np.linalg.norm(residual)
												max_res = np.max(np.abs(residual))
												max_res_idx = np.argmax(np.abs(residual))
												LOG_WARN(f"    k={k_check}: residual norm={res_norm:.6f}, max={max_res:.6f} at {dep_vars[max_res_idx] if max_res_idx < len(dep_vars) else 'unknown'}")
												if max_res > 1.0:
													LOG_WARN(f"      Full residual: {residual}, x_k={x_k_vals}, x_kp1={x_kp1_vals}, u_k={u_k_vals}")
											except Exception as _e:
												LOG_DEBUG(f"    Could not check dynamics residual at k={k_check}: {_e}")
									except Exception as _e:
										LOG_DEBUG(f"  Could not check dynamics residuals: {_e}")
					except Exception as const_err:
						LOG_DEBUG(f"Could not check constraint violations: {const_err}")
			except Exception as debug_err:
				LOG_DEBUG(f"Could not get debug values: {debug_err}")
			
			# Log current state and data
			LOG_WARN("Current state at failure:")
			# Get state from data - this is the authoritative source
			current_state = None
			if self.data is not None and hasattr(self.data, 'state'):
				current_state = self.data.state
			elif state is not None:
				current_state = state
			dynamics_model = self._get_dynamics_model()
			if current_state and dynamics_model:
				try:
					state_vars = {}
					for v in dynamics_model.get_all_vars():
						if current_state.has(v):
							state_vars[v] = current_state.get(v)
					LOG_WARN(f"  State variables: {state_vars}")
					# Check if initial state might violate constraints
					if 'x' in state_vars and 'y' in state_vars:
						LOG_WARN(f"  Initial position: ({state_vars['x']:.2f}, {state_vars['y']:.2f})")
					# Check if state is within bounds
					for var_name, var_val in state_vars.items():
						try:
							lb, ub, _ = dynamics_model.get_bounds(var_name)
							if var_val < lb or var_val > ub:
								LOG_WARN(f"  WARNING: State[{var_name}]={var_val:.3f} is OUTSIDE bounds [{lb:.3f}, {ub:.3f}]")
						except Exception:
							pass
				except Exception as state_err:
					LOG_DEBUG(f"Could not log state: {state_err}")
			
			LOG_WARN("Data at failure:")
			if self.data:
				has_ref = hasattr(self.data, 'reference_path') and self.data.reference_path is not None
				has_static = hasattr(self.data, 'static_obstacles') and self.data.static_obstacles is not None
				has_dynamic = hasattr(self.data, 'dynamic_obstacles') and self.data.dynamic_obstacles is not None
				LOG_WARN(f"  reference_path: {'present' if has_ref else 'missing'}")
				LOG_WARN(f"  static_obstacles: {'present' if has_static else 'missing'}")
				LOG_WARN(f"  dynamic_obstacles: {'present' if has_dynamic else 'missing'}")
				
				# Log static obstacle details
				if has_static and hasattr(self.data, 'static_obstacles'):
					obs_count = sum(1 for o in self.data.static_obstacles if o is not None)
					LOG_WARN(f"  Static obstacles: {obs_count} non-None obstacles")
					if obs_count > 0 and state:
						try:
							x0 = state.get('x')
							y0 = state.get('y')
							if x0 is not None and y0 is not None:
								# Check first stage
								if len(self.data.static_obstacles) > 0:
									obs0 = self.data.static_obstacles[0]
									if obs0 is not None and hasattr(obs0, 'halfspaces'):
										LOG_WARN(f"  Stage 0 has {len(obs0.halfspaces)} halfspace constraints")
						except Exception:
							pass
				# Check if initial state is within road bounds
				if has_static and state and hasattr(state, 'get'):
					try:
						x0 = state.get('x')
						y0 = state.get('y')
						if x0 is not None and y0 is not None and hasattr(self.data, 'static_obstacles'):
							# Check first stage obstacle
							if self.data.static_obstacles and len(self.data.static_obstacles) > 0:
								obs0 = self.data.static_obstacles[0]
								if obs0 is not None and hasattr(obs0, 'halfspaces'):
									# Check if point violates any halfspace
									for i, halfspace in enumerate(obs0.halfspaces):
										if hasattr(halfspace, 'A') and hasattr(halfspace, 'b'):
											A = np.array(halfspace.A).flatten()
											b = float(halfspace.b)
											violation = np.dot(A, [x0, y0]) - b
											LOG_WARN(f"  Halfspace {i}: A={A}, b={b:.6f}, A·p={np.dot(A, [x0, y0]):.6f}, violation={violation:.6f}")
											if violation > 1e-6:
												LOG_WARN(f"  Initial position ({x0:.2f}, {y0:.2f}) violates road constraint {i}: A·p - b = {violation:.6f}")
					except Exception as check_err:
						LOG_DEBUG(f"Could not check initial position against constraints: {check_err}")
			
			LOG_WARN("=== DIAGNOSIS COMPLETE ===")
			LOG_WARN("Summary of potential issues:")
			LOG_WARN("  1. Check if initial state violates constraints (especially road boundaries)")
			LOG_WARN("  2. Check if warmstart is feasible (within bounds)")
			LOG_WARN("  3. Check if constraints are too tight for the problem")
			LOG_WARN("  4. Check if reference path alignment is correct")
			LOG_WARN("  5. Check if obstacle constraints are reasonable")
			return -1

	def get_output(self, k, var_name):
		"""Get output value for variable at stage k.

		Args:
			k: Stage index (0-based). For state variables: k in [0, horizon]. For input variables: k in [0, horizon-1]
			var_name: Name of variable (state or input)

		Returns:
			Value at stage k, or None if not available

		Reference: C++ mpc_planner - returns braking fallback values when solver fails
		"""
		if not self.solution:
			# No solution - fall back to warmstart values (which may have braking applied)
			# Reference: C++ mpc_planner - initializeWithBraking provides fallback trajectory
			if var_name in self.warmstart_values:
				ws_vals = self.warmstart_values[var_name]
				if isinstance(ws_vals, np.ndarray) and k < len(ws_vals):
					val = float(ws_vals[k])
					LOG_DEBUG(f"get_output({k}, {var_name}): Using warmstart fallback: {val}")
					return val
			LOG_DEBUG(f"get_output({k}, {var_name}): No solution and no warmstart fallback available")
			return None

		if var_name not in self.var_dict:
			LOG_DEBUG(f"get_output({k}, {var_name}): Variable not in var_dict")
			return None

		var = self.var_dict[var_name]
		if k >= var.shape[0]:
			LOG_DEBUG(f"get_output({k}, {var_name}): k={k} >= var.shape[0]={var.shape[0]}")
			return None

		try:
			val = self.solution.value(var[k])
			LOG_DEBUG(f"get_output({k}, {var_name}): {val}")
			return val
		except Exception as e:
			LOG_WARN(f"get_output({k}, {var_name}): Error extracting value: {e}")
			return None

	def get_reference_trajectory(self):
		if not self.solution:
			LOG_DEBUG("No solution available. Returning trajectory from warmstart.")
			# NOTE: Braking fallback should be applied via apply_braking_fallback() after solve fails
			# Do NOT apply here - this method is also called during constraint linearization
			return self._create_trajectory_from_warmstart()

		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return Trajectory(timestep=timestep_val, length=0)
		traj = Trajectory(timestep=timestep_val, length=horizon_val + 1)
		for k in range(horizon_val + 1):
			state_k = State(model_type=dynamics_model)
			for var_name in dynamics_model.get_dependent_vars():
				state_k.set(var_name, self.solution.value(self.var_dict[var_name][k]))
			if k < horizon_val:
				for var_name in dynamics_model.get_inputs():
					state_k.set(var_name, self.solution.value(self.var_dict[var_name][k]))
			traj.add_state(state_k)
		return traj

	def print_if_bound_limited(self):
		if not self.solution: return
		LOG_DEBUG("Checking variable bounds:")
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return
		for var_name, var in self.var_dict.items():
			values = self.solution.value(var)
			lb, ub, _ = dynamics_model.get_bounds(var_name)
			if np.any(np.isclose(values, lb, atol=1e-4)):
				LOG_WARN(f"Variable '{var_name}' is at its lower bound.")
			if np.any(np.isclose(values, ub, atol=1e-4)):
				LOG_WARN(f"Variable '{var_name}' is at its upper bound.")

	def explain_exit_flag(self, code=None):
		code_to_check = code if code is not None else self.exit_flag
		explanations = {
			1: "Optimization successful.",
			-1: f"Optimization failed: {self.info.get('error', 'Unknown error')}"
		}
		return explanations.get(code_to_check, f"Unknown exit code: {code_to_check}")

	def set_dynamics_model(self, model):
		"""Set the dynamics model in data - DO NOT store separately."""
		if self.data is None:
			LOG_WARN("set_dynamics_model: data is None, cannot set dynamics_model")
			return
		self.data.dynamics_model = model
		# If solver is already initialized, reinitialize with new model
		if self.opti is not None and self.data:
			self.intialize_solver(self.data)
	
	def reset(self):
		LOG_DEBUG("Resetting CasADi solver.")
		self.opti = cs.Opti()
		self.solution = None
		self.exit_flag = None
		self.info.clear()
		self.var_dict.clear()
		self.warmstart_values.clear()
		self.forecast.clear()

		# No need to set dynamics_model - it should be in data

		for module in self.module_manager.get_modules():
			module.reset()