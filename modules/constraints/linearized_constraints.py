import numpy as np

from modules.constraints.base_constraint import BaseConstraint
from planning.types import Data, State, PredictionType
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_tools import rotation_matrix
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN



class LinearizedConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = "linearized_constraints"

		LOG_DEBUG("Initializing Linearized Constraints")

		# Solver will be set by framework later
		self.solver = None
		num_discs_val = self.get_config_value("num_discs")
		self.num_discs = int(num_discs_val) if num_discs_val is not None else 1
		self.num_other_halfspaces = self.get_config_value("linearized_constraints.add_halfspaces") or 0
		self.max_obstacles = self.get_config_value("max_obstacles") or 10
		self.filter_distant_obstacles = self.get_config_value("linearized_constraints.filter_distant_obstacles") or False
		self.max_num_constraints = self.max_obstacles + self.num_other_halfspaces

		self.use_guidance = False
		self.use_slack = self.get_config_value("linearized_constraints.use_slack")

		self.use_topology_constraints = self.get_config_value("linearized_constraints.use_topology_constraints")
		if self.use_topology_constraints:
			self.num_discs = 1  # Only one disc is used for the topology constraints
			self.use_guidance = True

		self.disc_radius = self.get_config_value("disc_radius", 1.0)
		
		# Get halfspace offset from obstacles (additional safety margin for constraints)
		# This offset is added to the safe_distance when computing constraint b values
		self.halfspace_offset = self.get_config_value("linearized_constraints.halfspace_offset", 0.0)
		if self.halfspace_offset is None:
			self.halfspace_offset = 0.0
		else:
			self.halfspace_offset = float(self.halfspace_offset)
		LOG_DEBUG(f"LinearizedConstraints: halfspace_offset={self.halfspace_offset:.3f}m")

		# Use horizon with fallback - solver will be set later by framework
		horizon_val = 10  # Default, will be updated when solver is set

		self._a1 = [None] * self.num_discs
		self._a2 = [None] * self.num_discs
		self._b = [None] * self.num_discs
		# Store vehicle and obstacle positions used during update() for verification in calculate_constraints()
		self._vehicle_positions = [None] * horizon_val  # Vehicle position at each step
		self._obstacle_positions = [None] * horizon_val  # List of obstacle positions at each step

		for disc_id in range(self.num_discs):
			self._a1[disc_id] = [None] * horizon_val
			self._a2[disc_id] = [None] * horizon_val
			self._b[disc_id] = [None] * horizon_val
			for step in range(horizon_val):
				self._a1[disc_id][step] = [None] * self.max_num_constraints
				self._a2[disc_id][step] = [None] * self.max_num_constraints
				self._b[disc_id][step] = [None] * self.max_num_constraints

		self.num_active_obstacles = 0

		# Store dummy values for invalid states
		self._dummy_a1 = 0.0
		self._dummy_a2 = 0.0
		self._dummy_b = 100.0

		LOG_DEBUG("Linearized Constraints successfully initialized")

	def update(self, state: State, data: Data):
		"""
		Per-iteration update: prepare module for constraint computation.
		This is called once per planning iteration (matching reference C++ pattern).
		Per-step constraint computation happens in update_step() or calculate_constraints().
		"""
		LOG_DEBUG("LinearizedConstraints.update: Per-iteration setup")

		self._dummy_b = state.get("x") + 100.0

		# Thread safe copy of obstacles
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			LOG_WARN("No dynamic obstacles available")
			self.num_active_obstacles = 0
			self._copied_dynamic_obstacles = []
			self._ref_states = []
			return

		self._copied_dynamic_obstacles = data.dynamic_obstacles
		# Note: filter_distant_obstacles import adjusted elsewhere if needed
		# if self.filter_distant_obstacles:
		# 	self._copied_dynamic_obstacles = filter_distant_obstacles(data.dynamic_obstacles, state, 5)
		self.num_active_obstacles = len(self._copied_dynamic_obstacles)
		
		# Get reference trajectory - this should be the warmstart trajectory
		# CRITICAL: If reference trajectory is empty, use warmstart values directly
		ref_trajectory = self.solver.get_reference_trajectory()
		ref_states = ref_trajectory.get_states() if ref_trajectory else []
		
		# If reference trajectory is empty, try to create states from warmstart values
		if len(ref_states) == 0:
			LOG_WARN("LinearizedConstraints.update: Reference trajectory is EMPTY! Attempting to use warmstart values directly.")
			has_warmstart = hasattr(self.solver, 'warmstart_values') and self.solver.warmstart_values
			LOG_INFO(f"LinearizedConstraints.update: warmstart_values available: {has_warmstart}")
			if has_warmstart:
				ws_keys = list(self.solver.warmstart_values.keys()) if self.solver.warmstart_values else []
				LOG_INFO(f"LinearizedConstraints.update: warmstart_values keys: {ws_keys}")
				if 'x' in self.solver.warmstart_values:
					x_len = len(self.solver.warmstart_values['x']) if hasattr(self.solver.warmstart_values['x'], '__len__') else 'N/A'
					LOG_INFO(f"LinearizedConstraints.update: warmstart x length: {x_len}")
				
				# Create reference states from warmstart values
				horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
				from planning.types import State
				from planning.dynamic_models import DynamicsModel
				# Get dynamics model from data
				dynamics_model = None
				if data is not None and hasattr(data, 'dynamics_model'):
					dynamics_model = data.dynamics_model
				elif hasattr(self.solver, '_get_dynamics_model'):
					dynamics_model = self.solver._get_dynamics_model()
				
				LOG_INFO(f"LinearizedConstraints.update: dynamics_model available: {dynamics_model is not None}")
				
				if dynamics_model and 'x' in self.solver.warmstart_values and 'y' in self.solver.warmstart_values:
					ref_states = []
					x_vals = self.solver.warmstart_values['x']
					y_vals = self.solver.warmstart_values['y']
					psi_vals = self.solver.warmstart_values.get('psi', [0.0] * (horizon_val + 1))
					
					# Ensure arrays are the right length
					if not hasattr(x_vals, '__len__'):
						LOG_WARN(f"LinearizedConstraints.update: warmstart x is not an array: {type(x_vals)}")
					else:
						for k in range(min(horizon_val + 1, len(x_vals))):
							ref_state = State(model_type=dynamics_model)
							ref_state.set("x", float(x_vals[k]))
							ref_state.set("y", float(y_vals[k]))
							if k < len(psi_vals):
								ref_state.set("psi", float(psi_vals[k]))
							if 'v' in self.solver.warmstart_values and k < len(self.solver.warmstart_values['v']):
								ref_state.set("v", float(self.solver.warmstart_values['v'][k]))
							ref_states.append(ref_state)
						LOG_INFO(f"LinearizedConstraints.update: Created {len(ref_states)} reference states from warmstart values")
				else:
					missing = []
					if not dynamics_model:
						missing.append("dynamics_model")
					if 'x' not in self.solver.warmstart_values:
						missing.append("warmstart x")
					if 'y' not in self.solver.warmstart_values:
						missing.append("warmstart y")
					LOG_WARN(f"LinearizedConstraints.update: Cannot create reference states from warmstart - missing: {missing}")
			else:
				LOG_WARN("LinearizedConstraints.update: No warmstart values available either!")
		
		# Store reference states (for potential use, though constraints are now computed symbolically)
		self._ref_states = ref_states
		self._current_state = state
		self._current_data = data
		
		LOG_INFO(f"LinearizedConstraints.update: Using {len(ref_states)} reference states for data preparation")
		if len(ref_states) > 0:
			# Log first few reference states to verify they're correct
			for i in range(min(3, len(ref_states))):
				ref_state = ref_states[i]
				ref_x = ref_state.get("x") if ref_state.has("x") else None
				ref_y = ref_state.get("y") if ref_state.has("y") else None
				LOG_INFO(f"  Reference state {i}: x={ref_x:.3f}, y={ref_y:.3f}" if ref_x is not None and ref_y is not None else f"  Reference state {i}: x={ref_x}, y={ref_y}")

		# NOTE: update_step() is no longer called for constraint computation because constraints are computed symbolically
		# in calculate_constraints() using symbolic states from the solver.
		# However, we still call update_step(0, ...) to populate _a1, _a2, _b arrays for stage 0
		# so that visualization code can extract constraint parameters.
		# This is for visualization only - actual constraint computation happens symbolically in calculate_constraints().
		if len(ref_states) > 0 or state is not None:
			try:
				self.update_step(0, state, data)
			except Exception as e:
				LOG_WARN(f"LinearizedConstraints.update: Failed to call update_step(0) for visualization: {e}")
		
		# CRITICAL: Project warmstart to satisfy constraints
		# Reference: C++ mpc_planner - warmstart is projected to ensure feasibility
		# This prevents solver failures and improves convergence
		self._project_warmstart_to_safety(data)
		
		LOG_DEBUG("LinearizedConstraints.update done")

	def update_step(self, step: int, state: State, data: Data):
		"""
		Per-rollout-step update: compute constraints for a single step.
		This can be called per step if needed, or all steps can be computed in update().
		Matching reference C++ pattern where constraints are computed per step.
		"""
		# Use stored obstacles and reference states from update() if available
		copied_dynamic_obstacles = getattr(self, '_copied_dynamic_obstacles', data.dynamic_obstacles if data.has("dynamic_obstacles") else [])
		ref_states = getattr(self, '_ref_states', [])
		
		# Initialize obstacle positions list for this step
		if step >= len(self._obstacle_positions):
			self._obstacle_positions.extend([None] * (step + 1 - len(self._obstacle_positions)))
		self._obstacle_positions[step] = []
		
		# For stage 0, use current state position; for stage >= 1, use reference trajectory
		if step == 0:
			# Use current vehicle position from state (matching C++: state.get("x"))
			ego_position = np.array([
				state.get("x"),
				state.get("y")
			])
			ego_psi = state.get("psi")
			LOG_INFO(f"LinearizedConstraints.update_step: step={step} (stage 0), using CURRENT state position: ({ego_position[0]:.3f}, {ego_position[1]:.3f}), psi={ego_psi:.3f}")
			
			# CRITICAL FIX: Also project step 0 position away from obstacles if too close
			# This ensures constraints are feasible even when vehicle is near obstacles
			robot_radius = self.get_config_value("robot_radius", 0.5) or self.disc_radius
			safety_margin = 0.3
			
			# Find closest obstacle
			closest_obs_id = None
			closest_dist = float('inf')
			closest_obs_pos = None
			closest_min_safe_distance = None
			
			for obs_id in range(len(copied_dynamic_obstacles)):
				obstacle = copied_dynamic_obstacles[obs_id]
				if hasattr(obstacle, 'velocity') and obstacle.velocity is not None:
					vel_norm = np.linalg.norm(obstacle.velocity)
					is_static = vel_norm < 1e-3
				else:
					is_static = True
				
				if is_static:
					obs_pos = np.array([obstacle.position[0], obstacle.position[1]])
				else:
					obs_pos = np.array([obstacle.position[0], obstacle.position[1]])  # Use current position for step 0
				
				diff = ego_position - obs_pos
				dist = np.linalg.norm(diff)
				
				obstacle_radius = obstacle.radius if hasattr(obstacle, 'radius') else 0.35
				min_safe_distance = robot_radius + obstacle_radius + safety_margin
				
				if dist < closest_dist:
					closest_dist = dist
					closest_obs_id = obs_id
					closest_obs_pos = obs_pos
					closest_min_safe_distance = min_safe_distance
			
			# Project away if too close
			if closest_obs_id is not None and closest_dist < closest_min_safe_distance:
				if closest_dist < 1e-6:
					# Use path tangent direction
					if hasattr(self.solver, 'data') and self.solver.data is not None:
						if hasattr(self.solver.data, 'reference_path') and self.solver.data.reference_path is not None:
							try:
								ref_states = self.solver.get_reference_trajectory().get_states()
								if len(ref_states) > 0:
									ref_psi = ref_states[0].get("psi") if ref_states[0].has("psi") else ego_psi
									direction = np.array([np.cos(ref_psi), np.sin(ref_psi)])
								else:
									direction = np.array([np.cos(ego_psi), np.sin(ego_psi)])
							except:
								direction = np.array([np.cos(ego_psi), np.sin(ego_psi)])
						else:
							direction = np.array([np.cos(ego_psi), np.sin(ego_psi)])
					else:
						direction = np.array([np.cos(ego_psi), np.sin(ego_psi)])
				else:
					direction = (ego_position - closest_obs_pos) / closest_dist
				
				original_ego_pos = ego_position.copy()
				ego_position = closest_obs_pos + direction * closest_min_safe_distance
				projection_distance = np.linalg.norm(ego_position - original_ego_pos)
				LOG_INFO(f"  ⚠️  PROJECTED step 0 position away from obstacle {closest_obs_id}: "
				         f"original=({original_ego_pos[0]:.3f}, {original_ego_pos[1]:.3f}), "
				         f"projected=({ego_position[0]:.3f}, {ego_position[1]:.3f}), "
				         f"distance {closest_dist:.3f}m -> {closest_min_safe_distance:.3f}m, "
				         f"projection_offset={projection_distance:.3f}m")
		else:
			# Use reference trajectory position (matching C++: _solver->getEgoPrediction(k, "x"))
			if step < len(ref_states):
				ego_position = np.array([
					ref_states[step].get("x"),
					ref_states[step].get("y")
				])
				ego_psi = ref_states[step].get("psi")
				LOG_INFO(f"LinearizedConstraints.update_step: step={step}, using REFERENCE TRAJECTORY position: ({ego_position[0]:.3f}, {ego_position[1]:.3f}), psi={ego_psi:.3f}")
			else:
				# Fallback to state if reference trajectory is not long enough
				ego_position = np.array([
					state.get("x"),
					state.get("y")
				])
				ego_psi = state.get("psi")
				LOG_WARN(f"LinearizedConstraints.update_step: step={step}, reference trajectory too short (len={len(ref_states)}), using CURRENT state position: ({ego_position[0]:.3f}, {ego_position[1]:.3f})")
				
				# CRITICAL FIX: Project reference trajectory position away from obstacles
				# Matching reference implementation: linearization must happen at feasible points (outside obstacles)
				# The reference trajectory may go through obstacles, making linearization ineffective.
				# Project the position away from nearby obstacles to ensure constraints are linearized
				# at feasible positions. This is critical for solver convergence.
				robot_radius = self.get_config_value("robot_radius", 0.5) or self.disc_radius
				safety_margin = 0.3  # Increased safety margin for better feasibility
				
				# Find closest obstacle and project away from it
				closest_obs_id = None
				closest_dist = float('inf')
				closest_obs_pos = None
				closest_min_safe_distance = None
				
				for obs_id in range(len(copied_dynamic_obstacles)):
					obstacle = copied_dynamic_obstacles[obs_id]
					# Get obstacle position (use current position for static obstacles)
					if hasattr(obstacle, 'velocity') and obstacle.velocity is not None:
						vel_norm = np.linalg.norm(obstacle.velocity)
						is_static = vel_norm < 1e-3
					else:
						is_static = True  # Assume static if no velocity
					
					if is_static:
						obs_pos = np.array([obstacle.position[0], obstacle.position[1]])
					else:
						# For dynamic obstacles, use predicted position at step-1
						if step - 1 < len(obstacle.prediction.steps):
							obs_pos = np.array([
								obstacle.prediction.steps[step - 1].position[0],
								obstacle.prediction.steps[step - 1].position[1]
							])
						else:
							obs_pos = np.array([obstacle.position[0], obstacle.position[1]])
					
					# Compute distance to obstacle
					diff = ego_position - obs_pos
					dist = np.linalg.norm(diff)
					
					# Get combined radius (robot + obstacle)
					obstacle_radius = obstacle.radius if hasattr(obstacle, 'radius') else 0.35
					min_safe_distance = robot_radius + obstacle_radius + safety_margin
					
					# Track closest obstacle
					if dist < closest_dist:
						closest_dist = dist
						closest_obs_id = obs_id
						closest_obs_pos = obs_pos
						closest_min_safe_distance = min_safe_distance
				
				# Project away from closest obstacle if too close (even if just slightly inside)
				# This ensures linearization point is always feasible
				if closest_obs_id is not None and closest_dist < closest_min_safe_distance:
					if closest_dist < 1e-6:
						# Ego position is exactly at obstacle center - project in arbitrary direction
						# Use path tangent direction if available, otherwise use [1, 0]
						if hasattr(self.solver, 'data') and self.solver.data is not None:
							if hasattr(self.solver.data, 'reference_path') and self.solver.data.reference_path is not None:
								try:
									# Try to get path tangent at this step
									ref_states = self.solver.get_reference_trajectory().get_states()
									if step < len(ref_states):
										ref_psi = ref_states[step].get("psi")
										direction = np.array([np.cos(ref_psi), np.sin(ref_psi)])
									else:
										direction = np.array([1.0, 0.0])
								except:
									direction = np.array([1.0, 0.0])
							else:
								direction = np.array([1.0, 0.0])
						else:
							direction = np.array([1.0, 0.0])
					else:
						# Normalize direction vector (away from obstacle)
						direction = (ego_position - closest_obs_pos) / closest_dist
					
					# Project to safe distance (ensure we're outside the obstacle)
					original_ego_pos = ego_position.copy()
					ego_position = closest_obs_pos + direction * closest_min_safe_distance
					projection_distance = np.linalg.norm(ego_position - original_ego_pos)
					LOG_INFO(f"  ⚠️  PROJECTED reference position at step {step} away from obstacle {closest_obs_id}: "
					         f"original=({original_ego_pos[0]:.3f}, {original_ego_pos[1]:.3f}), "
					         f"projected=({ego_position[0]:.3f}, {ego_position[1]:.3f}), "
					         f"distance {closest_dist:.3f}m -> {closest_min_safe_distance:.3f}m, "
					         f"projection_offset={projection_distance:.3f}m")
		
		# Store vehicle position for this step (for verification in calculate_constraints)
		if step >= len(self._vehicle_positions):
			self._vehicle_positions.extend([None] * (step + 1 - len(self._vehicle_positions)))
		self._vehicle_positions[step] = ego_position.copy()

		for disc_id in range(self.num_discs):
			if not self.use_guidance:  # Use discs and their positions
				if not data.has("robot_area") or data.robot_area is None or disc_id >= len(data.robot_area):
					LOG_WARN(f"Robot area not available for disc {disc_id}")
					continue

				active_disc = data.robot_area[disc_id]
				disc_position = active_disc.get_position(ego_position, ego_psi)
				self.project_to_safety(copied_dynamic_obstacles, step, disc_position)
				ego_position = disc_position

			else:  # Use the robot position
				self.project_to_safety(copied_dynamic_obstacles, step, ego_position)

			# For all obstacles (matching C++ implementation)
			for obs_id in range(len(copied_dynamic_obstacles)):
				target_obstacle = copied_dynamic_obstacles[obs_id]
				
				# Get obstacle position
				# For static obstacles (zero velocity), always use the initial position
				# For dynamic obstacles, use predicted position at step k-1 (C++: obstacle.prediction.modes[0][k-1].position)
				# Check if obstacle is static by checking if velocity is zero or if all prediction steps have the same position
				is_static = False
				if hasattr(target_obstacle, 'velocity') and target_obstacle.velocity is not None:
					vel_norm = np.linalg.norm(target_obstacle.velocity)
					is_static = vel_norm < 1e-3
				elif len(target_obstacle.prediction.steps) > 1:
					# Check if all prediction steps have the same position (static obstacle)
					first_pos = target_obstacle.prediction.steps[0].position
					all_same = all(np.linalg.norm(step.position - first_pos) < 1e-3 
					                for step in target_obstacle.prediction.steps[1:])
					is_static = all_same
				
				if is_static:
					# For static obstacles, always use the initial position (obstacle doesn't move)
					target_obstacle_pos = np.array([
						target_obstacle.position[0],
						target_obstacle.position[1]
					])
					LOG_DEBUG(f"  Static obstacle {obs_id} at step {step}: using fixed position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
				else:
					# For dynamic obstacles, prioritize actual current position for stage 0
					# For future stages, use predicted positions but ensure they're reasonable
					if step == 0:
						# For stage 0, always use actual current obstacle position
						# This ensures constraints remain accurate even when obstacles turn rapidly
						target_obstacle_pos = np.array([
							target_obstacle.position[0],
							target_obstacle.position[1]
						])
						LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: using actual current position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
					else:
						# For future stages, use predicted position at step k-1
						# But validate that prediction is reasonable (not too far from current position)
						current_obstacle_pos = np.array([
							target_obstacle.position[0],
							target_obstacle.position[1]
						])
						
						# Try to get predicted position
						predicted_pos = None
						if step - 1 < len(target_obstacle.prediction.steps):
							predicted_pos = np.array([
								target_obstacle.prediction.steps[step - 1].position[0],
								target_obstacle.prediction.steps[step - 1].position[1]
							])
						elif len(target_obstacle.prediction.steps) > 0:
							# Use last available prediction
							last_step = len(target_obstacle.prediction.steps) - 1
							predicted_pos = np.array([
								target_obstacle.prediction.steps[last_step].position[0],
								target_obstacle.prediction.steps[last_step].position[1]
							])
						
						# Validate prediction is reasonable (not moving faster than solver can handle)
						# Maximum reasonable distance: max_velocity * timestep * step
						# Use conservative estimate: 3 m/s * 0.1s * step = 0.3 * step meters
						if predicted_pos is not None:
							prediction_distance = np.linalg.norm(predicted_pos - current_obstacle_pos)
							max_reasonable_distance = 0.3 * step  # Conservative: 3 m/s * 0.1s * step
							
							if prediction_distance > max_reasonable_distance:
								# Prediction seems unreasonable (obstacle moving too fast)
								# Use extrapolated position based on current velocity instead
								if hasattr(target_obstacle, 'velocity') and target_obstacle.velocity is not None:
									# Limit velocity to reasonable maximum (3 m/s)
									vel = np.array(target_obstacle.velocity)
									vel_norm = np.linalg.norm(vel)
									if vel_norm > 3.0:
										vel = vel / vel_norm * 3.0
									
									# Extrapolate position: current_pos + velocity * timestep * step
									timestep = 0.1  # Default timestep (should match solver timestep)
									if hasattr(self.solver, 'timestep') and self.solver.timestep is not None:
										timestep = float(self.solver.timestep)
									target_obstacle_pos = current_obstacle_pos + vel * timestep * step
									LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: prediction distance {prediction_distance:.2f}m > max {max_reasonable_distance:.2f}m, using velocity-extrapolated position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
								else:
									# No velocity available, use current position
									target_obstacle_pos = current_obstacle_pos
									LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: prediction unreasonable, no velocity available, using current position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
							else:
								# Prediction is reasonable, use it
								target_obstacle_pos = predicted_pos
								LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: using validated prediction step {step-1} position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
						else:
							# No prediction available, extrapolate from current position and velocity
							if hasattr(target_obstacle, 'velocity') and target_obstacle.velocity is not None:
								vel = np.array(target_obstacle.velocity)
								vel_norm = np.linalg.norm(vel)
								if vel_norm > 3.0:
									vel = vel / vel_norm * 3.0
								
								timestep = 0.1
								if hasattr(self.solver, 'timestep') and self.solver.timestep is not None:
									timestep = float(self.solver.timestep)
								target_obstacle_pos = current_obstacle_pos + vel * timestep * step
								LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: no prediction available, using velocity-extrapolated position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
							else:
								# No velocity available, use current position
								target_obstacle_pos = current_obstacle_pos
								LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: no prediction or velocity, using current position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")

				# Compute difference vector FROM ego TO obstacle (C++: diff_x = obstacle_pos(0) - pos(0))
				diff_x = target_obstacle_pos[0] - ego_position[0]
				diff_y = target_obstacle_pos[1] - ego_position[1]
				
				# Compute distance (C++: dist = (obstacle_pos - pos).norm())
				dist = np.linalg.norm(target_obstacle_pos - ego_position)
				
				# Normalize direction vector (C++: _a1[d][k](obs_id) = diff_x / dist)
				# This points FROM vehicle TO obstacle
				if dist < 1e-6:
					# Avoid division by zero - use arbitrary direction
					diff_x, diff_y = 1.0, 0.0
					dist = 1.0
				
				self._a1[disc_id][step][obs_id] = float(diff_x / dist)
				self._a2[disc_id][step][obs_id] = float(diff_y / dist)

				# Get obstacle radius (C++: radius = _use_guidance ? 1e-3 : obstacle.radius)
				target_obstacle_radius = 1e-3 if self.use_guidance else target_obstacle.radius
				
				# Get robot radius from config (C++: CONFIG["robot_radius"])
				robot_radius = self.get_config_value("robot_radius", 0.5)
				if robot_radius is None:
					robot_radius = self.disc_radius  # Fallback to disc_radius
				else:
					robot_radius = float(robot_radius)

				# Compute b value (C++: _b[d][k](obs_id) = _a1[d][k](obs_id) * obstacle_pos(0) + 
				#                                    _a2[d][k](obs_id) * obstacle_pos(1) - 
				#                                    (radius + CONFIG["robot_radius"] + halfspace_offset))
				# The constraint a1*x + a2*y <= b means the vehicle must satisfy this inequality.
				# Since (a1, a2) points FROM vehicle TO obstacle, and b = a·obstacle - safe_distance,
				# the constraint keeps the vehicle at least safe_distance away from the obstacle.
				# The halfspace_offset adds an additional safety margin to the constraint.
				safe_distance = target_obstacle_radius + robot_radius + self.halfspace_offset
				if step == 0 and obs_id == 0 and self.halfspace_offset > 1e-6:
					LOG_INFO(f"  Using halfspace_offset={self.halfspace_offset:.3f}m (safe_distance={safe_distance:.3f}m = {target_obstacle_radius:.3f} + {robot_radius:.3f} + {self.halfspace_offset:.3f})")
				self._b[disc_id][step][obs_id] = (self._a1[disc_id][step][obs_id] * target_obstacle_pos[0] +
												  self._a2[disc_id][step][obs_id] * target_obstacle_pos[1] -
												  safe_distance)

				# Store obstacle position for this step (for verification in calculate_constraints)
				if len(self._obstacle_positions[step]) <= obs_id:
					# Extend list if needed
					while len(self._obstacle_positions[step]) <= obs_id:
						self._obstacle_positions[step].append(None)
				self._obstacle_positions[step][obs_id] = target_obstacle_pos.copy()

				# Log constraint computation details for verification
				# Also evaluate constraint at ego_position to check if it's satisfied
				constraint_value = (self._a1[disc_id][step][obs_id] * ego_position[0] + 
				                   self._a2[disc_id][step][obs_id] * ego_position[1])
				constraint_satisfied = constraint_value <= self._b[disc_id][step][obs_id]
				violation_amount = constraint_value - self._b[disc_id][step][obs_id] if not constraint_satisfied else 0.0
				
				# Verify that (a1, a2) is the normalized vehicle-to-obstacle direction
				vehicle_to_obs = target_obstacle_pos - ego_position
				vehicle_to_obs_dist = np.linalg.norm(vehicle_to_obs)
				if vehicle_to_obs_dist > 1e-6:
					vehicle_to_obs_normalized = vehicle_to_obs / vehicle_to_obs_dist
					a_vec = np.array([self._a1[disc_id][step][obs_id], self._a2[disc_id][step][obs_id]])
					# Check if a_vec matches vehicle_to_obs_normalized (should be identical)
					dot_product = np.dot(a_vec, vehicle_to_obs_normalized)
					angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180.0 / np.pi
					norm_error = np.linalg.norm(a_vec - vehicle_to_obs_normalized)
					
					LOG_INFO(f"  Step {step}, disc {disc_id}, obstacle {obs_id}: ego_pos=({ego_position[0]:.3f}, {ego_position[1]:.3f}), "
					        f"obs_pos=({target_obstacle_pos[0]:.3f}, {target_obstacle_pos[1]:.3f}), "
					        f"dist={dist:.3f}, a1={self._a1[disc_id][step][obs_id]:.4f}, a2={self._a2[disc_id][step][obs_id]:.4f}, "
					        f"b={self._b[disc_id][step][obs_id]:.4f}, constraint_val={constraint_value:.4f}, "
					        f"satisfied={constraint_satisfied}, violation={violation_amount:.4f}")
					
					# Verify constraint normal vector matches vehicle-to-obstacle direction
					if norm_error < 1e-6 and abs(angle_error) < 1e-3:
						LOG_INFO(f"    ✓ VERIFIED: Constraint normal (a1, a2) matches vehicle-to-obstacle direction (error={norm_error:.2e}, angle_error={angle_error:.4f}°)")
					else:
						LOG_WARN(f"    ⚠️  WARNING: Constraint normal (a1, a2) does NOT match vehicle-to-obstacle direction! "
						        f"norm_error={norm_error:.6f}, angle_error={angle_error:.4f}°")
						LOG_WARN(f"      Expected: ({vehicle_to_obs_normalized[0]:.6f}, {vehicle_to_obs_normalized[1]:.6f}), "
						        f"Got: ({a_vec[0]:.6f}, {a_vec[1]:.6f})")
				else:
					LOG_WARN(f"    ⚠️  WARNING: Vehicle and obstacle are too close (dist={vehicle_to_obs_dist:.6f}), cannot verify constraint normal")
				
				if not constraint_satisfied and step == 0:
					LOG_WARN(f"  ⚠️  Step {step} constraint VIOLATED at linearization point! This may cause solver issues.")

			# Handle static obstacles (matching C++: module_data.static_obstacles[k])
			# In Python, data.static_obstacles is a list of StaticObstacle objects, one per stage
			# Each StaticObstacle has a halfspaces attribute which is a list of Halfspace objects
			if data.has("static_obstacles") and data.static_obstacles is not None:
				# Get static obstacle for this stage (C++: module_data.static_obstacles[k])
				if step < len(data.static_obstacles) and data.static_obstacles[step] is not None:
					static_obstacle = data.static_obstacles[step]
					
					# Get halfspaces for this stage (C++: module_data.static_obstacles[k].size())
					if hasattr(static_obstacle, 'halfspaces') and static_obstacle.halfspaces is not None:
						num_halfspaces = min(len(static_obstacle.halfspaces), self.num_other_halfspaces)
						
						# Process each halfspace (C++: for (int h = 0; h < num_halfspaces; h++))
						for halfspace_id in range(num_halfspaces):
							target_obs_id = len(copied_dynamic_obstacles) + halfspace_id
							if target_obs_id < self.max_num_constraints:
								halfspace = static_obstacle.halfspaces[halfspace_id]
								# Extract A and b from halfspace (C++: module_data.static_obstacles[k][h].A(0), etc.)
								if hasattr(halfspace, 'A') and hasattr(halfspace, 'b'):
									A = halfspace.A
									if isinstance(A, np.ndarray) and A.size >= 2:
										self._a1[disc_id][step][target_obs_id] = float(A[0])
										self._a2[disc_id][step][target_obs_id] = float(A[1])
										self._b[disc_id][step][target_obs_id] = float(halfspace.b)
									elif hasattr(halfspace, 'a1') and hasattr(halfspace, 'a2'):
										# Alternative format: halfspace has a1, a2, b attributes
										self._a1[disc_id][step][target_obs_id] = float(halfspace.a1)
										self._a2[disc_id][step][target_obs_id] = float(halfspace.a2)
										self._b[disc_id][step][target_obs_id] = float(halfspace.b)

	def project_to_safety(self, copied_obstacles, step, pos):
		# Placeholder projection if needed; left as-is
			return
	
	def _project_warmstart_to_safety(self, data):
		"""
		Project warmstart trajectory to satisfy linearized constraints.
		
		Reference: C++ mpc_planner - warmstart is projected to ensure feasibility
		Similar to SafeHorizonConstraint._project_warmstart_to_safety and GaussianConstraints._project_warmstart_to_gaussian_safety.
		
		This method:
		1. Checks if warmstart positions violate linearized constraints
		2. Projects violating positions away from obstacles to satisfy constraints
		3. Ensures warmstart is feasible, preventing solver failures
		
		Args:
			data: Data object containing obstacles and other information
		"""
		if not hasattr(self, 'solver') or self.solver is None:
			return
		
		if not hasattr(self.solver, 'warmstart_values') or not self.solver.warmstart_values:
			return
		
		ws_vals = self.solver.warmstart_values
		if 'x' not in ws_vals or 'y' not in ws_vals:
			return
		
		# Get obstacles
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			return
		
		copied_dynamic_obstacles = data.dynamic_obstacles
		horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
		x_ws = ws_vals['x']
		y_ws = ws_vals['y']
		
		# Get robot radius and obstacle radius
		robot_radius = float(self.get_config_value("robot.radius", 0.5)) or self.disc_radius
		halfspace_offset = float(self.get_config_value("linearized_constraints.halfspace_offset", 0.5))
		
		# Check each stage that has constraints
		projections_made = 0
		max_stage_for_constraints = min(horizon_val + 1, len(x_ws))
		
		for stage_idx in range(max_stage_for_constraints):
			if stage_idx >= len(x_ws) or stage_idx >= len(y_ws):
				continue
			
			robot_pos = np.array([float(x_ws[stage_idx]), float(y_ws[stage_idx])])
			
			# Check each obstacle for constraint violations
			for obs_id, obstacle in enumerate(copied_dynamic_obstacles[:self.max_obstacles]):
				# Get obstacle position (use predicted position if available)
				obstacle_pos = None
				if hasattr(obstacle, 'prediction') and obstacle.prediction is not None:
					if hasattr(obstacle.prediction, 'steps') and obstacle.prediction.steps:
						if stage_idx < len(obstacle.prediction.steps):
							pred_step = obstacle.prediction.steps[stage_idx]
							if hasattr(pred_step, 'position') and pred_step.position is not None:
								obstacle_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
				
				if obstacle_pos is None:
					if hasattr(obstacle, 'position') and obstacle.position is not None:
						obstacle_pos = np.array([float(obstacle.position[0]), float(obstacle.position[1])])
					else:
						continue
				
				obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
				target_obstacle_radius = 1e-3 if self.use_guidance else obstacle_radius
				safe_distance = robot_radius + target_obstacle_radius + halfspace_offset
				
				# Compute constraint: a1*x + a2*y <= b
				# Normal vector points FROM vehicle TO obstacle
				diff = obstacle_pos - robot_pos
				dist = np.linalg.norm(diff)
				if dist < 1e-6:
					continue
				
				a1 = diff[0] / dist
				a2 = diff[1] / dist
				b = a1 * obstacle_pos[0] + a2 * obstacle_pos[1] - safe_distance
				
				# Check constraint violation: a1*x + a2*y - b <= 0
				constraint_value = a1 * robot_pos[0] + a2 * robot_pos[1] - b
				
				if constraint_value > 1e-6:  # Violation
					# Project robot position away from obstacle
					# Move robot away from obstacle by safe_distance + violation_amount
					required_dist = safe_distance + 0.1  # Add small margin
					
					if dist < required_dist:
						# Project robot position to satisfy constraint
						projection_factor = required_dist / dist
						new_pos = obstacle_pos + projection_factor * diff
						
						# CRITICAL: Check if projected position satisfies contouring constraints (if active)
						# Reference: C++ mpc_planner - warmstart must satisfy ALL active constraints
						is_valid_contouring = True
						contour_error = 0.0
						if hasattr(self.solver, 'module_manager') and self.solver.module_manager is not None:
							# Check if contouring constraints are active
							contouring_module = None
							for module in self.solver.module_manager.get_modules():
								if hasattr(module, 'name') and module.name == 'contouring_constraints':
									contouring_module = module
									break
							
							if contouring_module is not None:
								# Manual check using reference path
								if (hasattr(self.solver, 'data') and self.solver.data is not None and
									hasattr(self.solver.data, 'reference_path') and self.solver.data.reference_path is not None):
									try:
										ref_path = self.solver.data.reference_path
										if 'spline' in ws_vals and stage_idx < len(ws_vals['spline']):
											s_val = float(ws_vals['spline'][stage_idx])
											if hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None:
												path_x = float(ref_path.x_spline(s_val))
												path_y = float(ref_path.y_spline(s_val))
												path_dx = float(ref_path.x_spline.derivative()(s_val))
												path_dy = float(ref_path.y_spline.derivative()(s_val))
												norm = np.hypot(path_dx, path_dy)
												if norm > 1e-6:
													path_dx_norm = path_dx / norm
													path_dy_norm = path_dy / norm
													A = np.array([path_dy_norm, -path_dx_norm])
													diff_path = new_pos - np.array([path_x, path_y])
													contour_error = np.dot(A, diff_path)
													road_width_half = 3.5  # Default
													if hasattr(contouring_module, '_road_width_half') and contouring_module._road_width_half is not None:
														road_width_half = float(contouring_module._road_width_half)
													is_valid_contouring = (-road_width_half <= contour_error <= road_width_half)
									except Exception:
										pass
							
							if not is_valid_contouring:
								# Try to project to satisfy contouring constraints
								# For now, log warning - full implementation would project to nearest valid position
								LOG_WARN(f"LinearizedConstraints._project_warmstart_to_safety: Stage {stage_idx}, Obstacle {obs_id}: "
								         f"Projected position violates contouring constraints (contour_error={contour_error:.3f}). "
								         f"May need additional projection to satisfy road boundaries.")
						
						# Update warmstart values
						x_ws[stage_idx] = float(new_pos[0])
						y_ws[stage_idx] = float(new_pos[1])
						projections_made += 1
						
						if stage_idx < 3:  # Log first few projections
							LOG_INFO(f"LinearizedConstraints._project_warmstart_to_safety: Stage {stage_idx}, Obstacle {obs_id}: "
							         f"Projected warmstart from ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}) "
							         f"to ({new_pos[0]:.3f}, {new_pos[1]:.3f}) "
							         f"(violation={constraint_value:.3f}, required_dist={required_dist:.3f}, "
							         f"contouring_valid={is_valid_contouring}, contour_error={contour_error:.3f})")
		
		if projections_made > 0:
			LOG_INFO(f"LinearizedConstraints._project_warmstart_to_safety: Projected {projections_made} warmstart positions to satisfy constraints")
		else:
			LOG_DEBUG(f"LinearizedConstraints._project_warmstart_to_safety: No warmstart projections needed (all positions satisfy constraints)")

	def calculate_constraints(self, state: State, data: Data, stage_idx: int):
		"""Return structured linear constraints for the solver to convert.
		
		CRITICAL: For future stages (stage_idx > 0), constraints MUST be computed symbolically
		using the symbolic state (predicted position). This matches the reference codebase pattern.
		
		For stage 0, uses pre-computed numeric values from update_step().
		For future stages, computes constraints symbolically using predicted vehicle position.
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
		"""
		import casadi as cd
		
		constraints = []
		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		if stage_idx >= horizon_val:
			LOG_DEBUG(f"LinearizedConstraints.calculate_constraints: stage_idx={stage_idx} >= horizon={horizon_val}, returning empty constraints")
			return constraints

		LOG_INFO(f"=== LinearizedConstraints.calculate_constraints: stage_idx={stage_idx} ===")
		LOG_INFO(f"  num_discs={self.num_discs}, num_active_obstacles={self.num_active_obstacles}, num_other_halfspaces={self.num_other_halfspaces}")
		
		# CRITICAL: Always use symbolic computation - solver always provides symbolic states
		# Check if state is symbolic (should always be true in normal operation)
		pos_x = state.get("x") if state is not None and state.has("x") else None
		pos_y = state.get("y") if state is not None and state.has("y") else None
		is_symbolic = isinstance(pos_x, (cd.MX, cd.SX)) or isinstance(pos_y, (cd.MX, cd.SX))
		
		if is_symbolic:
			# SYMBOLIC computation (matching reference codebase)
			# All constraints are computed symbolically using CasADi variables
			# This ensures constraints are properly integrated into the optimization problem
			LOG_INFO(f"  Stage {stage_idx}: Computing constraints SYMBOLICALLY using symbolic vehicle position")
			return self._compute_symbolic_obstacle_constraints(state, data, stage_idx)
		else:
			# LEGACY: Numeric fallback should not happen in normal operation
			# If state is not symbolic, try to create symbolic state from solver's var_dict
			LOG_WARN(f"  Stage {stage_idx}: State is not symbolic - attempting to create symbolic state from solver")
			if hasattr(self, 'solver') and self.solver is not None:
				try:
					dynamics_model = self.solver._get_dynamics_model() if hasattr(self.solver, '_get_dynamics_model') else None
					if dynamics_model is not None and hasattr(self.solver, 'var_dict'):
						# Create symbolic state from solver's var_dict
						symbolic_state = State(dynamics_model)
						for var_name in dynamics_model.get_all_vars():
							if var_name in self.solver.var_dict and stage_idx < self.solver.var_dict[var_name].shape[0]:
								symbolic_state.set(var_name, self.solver.var_dict[var_name][stage_idx])
						LOG_INFO(f"  Stage {stage_idx}: Created symbolic state from solver var_dict, computing symbolically")
						return self._compute_symbolic_obstacle_constraints(symbolic_state, data, stage_idx)
				except Exception as e:
					LOG_WARN(f"  Stage {stage_idx}: Failed to create symbolic state: {e}")
			
			# Last resort: numeric fallback (should not happen in normal operation)
			LOG_WARN(f"  Stage {stage_idx}: Falling back to NUMERIC computation (legacy mode - should not happen)")
			return self._compute_numeric_obstacle_constraints(state, data, stage_idx)
	
	def _compute_symbolic_obstacle_constraints(self, state: State, data: Data, stage_idx: int):
		"""Compute obstacle constraints symbolically using predicted vehicle position.
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints computed symbolically.
		"""
		import casadi as cd
		import numpy as np
		
		constraints = []
		
		# Get symbolic vehicle position
		pos_x_sym = state.get("x")
		pos_y_sym = state.get("y")
		psi_sym = state.get("psi")
		
		if pos_x_sym is None or pos_y_sym is None:
			# Try to get from solver's var_dict if available
			LOG_WARN(f"  Stage {stage_idx}: Cannot get symbolic position from state, attempting to get from solver")
			if hasattr(self, 'solver') and self.solver is not None:
				try:
					dynamics_model = self.solver._get_dynamics_model() if hasattr(self.solver, '_get_dynamics_model') else None
					if dynamics_model is not None and hasattr(self.solver, 'var_dict'):
						if 'x' in self.solver.var_dict and 'y' in self.solver.var_dict:
							if stage_idx < self.solver.var_dict['x'].shape[0] and stage_idx < self.solver.var_dict['y'].shape[0]:
								pos_x_sym = self.solver.var_dict['x'][stage_idx]
								pos_y_sym = self.solver.var_dict['y'][stage_idx]
								psi_sym = self.solver.var_dict.get('psi', [None] * (stage_idx + 1))[stage_idx] if 'psi' in self.solver.var_dict else state.get("psi")
								LOG_INFO(f"  Stage {stage_idx}: Retrieved symbolic position from solver var_dict")
							else:
								LOG_WARN(f"  Stage {stage_idx}: stage_idx {stage_idx} out of bounds for var_dict")
								return []
						else:
							LOG_WARN(f"  Stage {stage_idx}: var_dict missing x or y")
							return []
					else:
						LOG_WARN(f"  Stage {stage_idx}: Cannot get dynamics_model or var_dict")
						return []
				except Exception as e:
					LOG_WARN(f"  Stage {stage_idx}: Error retrieving symbolic position: {e}")
					return []
			else:
				LOG_WARN(f"  Stage {stage_idx}: No solver available, cannot compute constraints")
				return []
		
		# Get obstacles (these are numeric, known positions)
		copied_dynamic_obstacles = getattr(self, '_copied_dynamic_obstacles', [])
		if not copied_dynamic_obstacles:
			if data.has("dynamic_obstacles") and data.dynamic_obstacles:
				copied_dynamic_obstacles = data.dynamic_obstacles
		
		robot_radius = self.get_config_value("robot_radius", 0.5) or self.disc_radius
		safety_margin = 0.3
		
		# Compute constraints for each disc and obstacle
		for disc_id in range(self.num_discs):
			# Get disc offset
			disc_offset = 0.0
			if not self.use_guidance and data.has("robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
				disc_offset = float(data.robot_area[disc_id].offset)
			
			# Compute disc position symbolically (if offset is non-zero)
			if abs(disc_offset) > 1e-9 and psi_sym is not None:
				disc_x_sym = pos_x_sym + disc_offset * cd.cos(psi_sym)
				disc_y_sym = pos_y_sym + disc_offset * cd.sin(psi_sym)
			else:
				disc_x_sym = pos_x_sym
				disc_y_sym = pos_y_sym
			
			# Compute constraints for each obstacle
			for obs_id in range(min(len(copied_dynamic_obstacles), self.num_active_obstacles)):
				obstacle = copied_dynamic_obstacles[obs_id]
				
				# Get obstacle position (numeric, known)
				if hasattr(obstacle, 'position'):
					obs_pos = np.array([float(obstacle.position[0]), float(obstacle.position[1])])
				else:
					continue
				
				obstacle_radius = obstacle.radius if hasattr(obstacle, 'radius') else 0.35
				# Get obstacle radius (matching C++: radius = _use_guidance ? 1e-3 : obstacle.radius)
				target_obstacle_radius = 1e-3 if self.use_guidance else obstacle_radius
				# Compute safe_distance (matching C++: radius + CONFIG["robot_radius"] + halfspace_offset)
				# Note: safety_margin (0.3) is used in projection logic, not in constraint b value
				# The halfspace_offset adds an additional safety margin to the constraint
				safe_distance = robot_radius + target_obstacle_radius + self.halfspace_offset
				
				# Compute constraint symbolically: a·p_disc <= b
				# CRITICAL: Normal vector points FROM vehicle TO obstacle (matching reference C++ implementation)
				# Reference: https://github.com/tud-amr/mpc_planner - _a1[d][k](obs_id) = diff_x / dist
				# where diff = obstacle_pos - ego_position (points FROM vehicle TO obstacle)
				diff_x_sym = obs_pos[0] - disc_x_sym  # obstacle - vehicle (points FROM vehicle TO obstacle)
				diff_y_sym = obs_pos[1] - disc_y_sym
				dist_sym = cd.sqrt(diff_x_sym * diff_x_sym + diff_y_sym * diff_y_sym)
				dist_sym = cd.fmax(dist_sym, 1e-6)  # Avoid division by zero
				
				# Normalized direction vector (points FROM vehicle TO obstacle)
				# Matching C++: _a1[d][k](obs_id) = diff_x / dist, _a2[d][k](obs_id) = diff_y / dist
				a1_sym = diff_x_sym / dist_sym
				a2_sym = diff_y_sym / dist_sym
				
				# Compute b: a·obs_pos - safe_distance
				# Matching C++: _b[d][k](obs_id) = _a1[d][k](obs_id) * obstacle_pos(0) + 
				#                                    _a2[d][k](obs_id) * obstacle_pos(1) - 
				#                                    (radius + CONFIG["robot_radius"])
				# The constraint a1*x + a2*y <= b means: a·vehicle <= a·obstacle - safe_distance
				# Rearranging: a·(vehicle - obstacle) <= -safe_distance
				# Since a = (obstacle - vehicle) / ||obstacle - vehicle||, this enforces:
				# ||obstacle - vehicle|| >= safe_distance (correct!)
				b_sym = a1_sym * obs_pos[0] + a2_sym * obs_pos[1] - safe_distance
				
				# Constraint expression: a1*x + a2*y - b <= 0
				# This enforces: a·disc_pos <= b, i.e., a·disc_pos <= a·obs_pos - safe_distance
				constraint_expr = a1_sym * disc_x_sym + a2_sym * disc_y_sym - b_sym
				
				# Return as symbolic expression (solver will handle it)
				# Mark as linearized constraint so solver can count it correctly
				constraints.append({
					"type": "symbolic_expression",
					"expression": constraint_expr,
					"ub": 0.0,  # expr <= 0
					"constraint_type": "linearized",  # Mark as linearized for counting
				})
		
		LOG_INFO(f"  Symbolic constraints: {len(constraints)} constraint expressions computed symbolically")
		return constraints
	
	def _compute_numeric_obstacle_constraints(self, state: State, data: Data, stage_idx: int):
		"""Compute obstacle constraints numerically (for stage 0 or warmstart)."""
		constraints = []
		
		# Get vehicle position at this rollout step (from update())
		vehicle_pos_at_step = None
		if stage_idx < len(self._vehicle_positions) and self._vehicle_positions[stage_idx] is not None:
			vehicle_pos_at_step = self._vehicle_positions[stage_idx]
			LOG_INFO(f"  Vehicle position at rollout step {stage_idx}: ({vehicle_pos_at_step[0]:.3f}, {vehicle_pos_at_step[1]:.3f})")
		else:
			# Fallback to current state if not available
			if state is not None and state.has("x") and state.has("y"):
				vehicle_pos_at_step = np.array([float(state.get("x")), float(state.get("y"))])
				LOG_INFO(f"  Vehicle position at rollout step {stage_idx}: ({vehicle_pos_at_step[0]:.3f}, {vehicle_pos_at_step[1]:.3f}) [from state]")
			else:
				LOG_WARN(f"  ⚠️  Vehicle position not available for step {stage_idx}")

		for disc_id in range(self.num_discs):
			# Resolve disc offset from robot_area if available
			disc_offset = 0.0
			if not self.use_guidance and data.has("robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
				disc_offset = float(data.robot_area[disc_id].offset)

			disc_constraints = 0
			for index in range(self.num_active_obstacles + self.num_other_halfspaces):
				a1 = self._a1[disc_id][stage_idx][index] if self._a1[disc_id][stage_idx][index] is not None else self._dummy_a1
				a2 = self._a2[disc_id][stage_idx][index] if self._a2[disc_id][stage_idx][index] is not None else self._dummy_a2
				b = self._b[disc_id][stage_idx][index] if self._b[disc_id][stage_idx][index] is not None else self._dummy_b
				# Skip degenerate constraints (dummy constraints have a1=0, a2=0)
				if abs(a1) < 1e-9 and abs(a2) < 1e-9:
					continue
				# Add constraint - ensure it's for a valid obstacle index
				# For actual obstacles (index < num_active_obstacles), always add
				# For other halfspaces (index >= num_active_obstacles), also add
				constraint_type = "obstacle" if index < self.num_active_obstacles else "halfspace"
				constraints.append({"a1": float(a1), "a2": float(a2), "b": float(b), "disc_offset": disc_offset})
				disc_constraints += 1
				
				# Get obstacle position for this constraint (from update())
				obstacle_pos = None
				if (stage_idx < len(self._obstacle_positions) and 
				    self._obstacle_positions[stage_idx] is not None and 
				    index < len(self._obstacle_positions[stage_idx]) and
				    self._obstacle_positions[stage_idx][index] is not None):
					obstacle_pos = self._obstacle_positions[stage_idx][index]
				
				# Log detailed constraint information with verification
				LOG_INFO(f"  Constraint [{disc_id}][{index}] ({constraint_type}): a1={a1:.6f}, a2={a2:.6f}, b={b:.6f}, disc_offset={disc_offset:.4f}")
				
				# Verify constraint represents halfspace perpendicular to vehicle-to-obstacle direction
				if vehicle_pos_at_step is not None and obstacle_pos is not None and constraint_type == "obstacle":
					vehicle_to_obs = obstacle_pos - vehicle_pos_at_step
					vehicle_to_obs_dist = np.linalg.norm(vehicle_to_obs)
					if vehicle_to_obs_dist > 1e-6:
						vehicle_to_obs_normalized = vehicle_to_obs / vehicle_to_obs_dist
						a_vec = np.array([a1, a2])
						# Check if a_vec matches vehicle_to_obs_normalized (should be identical)
						dot_product = np.dot(a_vec, vehicle_to_obs_normalized)
						angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180.0 / np.pi
						norm_error = np.linalg.norm(a_vec - vehicle_to_obs_normalized)
						
						# Verify constraint normal vector is normalized
						a_norm = np.linalg.norm(a_vec)
						if abs(a_norm - 1.0) < 1e-6:
							LOG_INFO(f"    ✓ Normal vector is normalized (||a||={a_norm:.6f})")
						else:
							LOG_WARN(f"    ⚠️  Normal vector is NOT normalized! ||a||={a_norm:.6f} (expected 1.0)")
						
						# Verify constraint normal matches vehicle-to-obstacle direction
						if norm_error < 1e-6 and abs(angle_error) < 1e-3:
							LOG_INFO(f"    ✓ VERIFIED: Constraint normal (a1, a2) = ({a1:.6f}, {a2:.6f}) matches vehicle-to-obstacle direction")
							LOG_INFO(f"      Vehicle pos: ({vehicle_pos_at_step[0]:.3f}, {vehicle_pos_at_step[1]:.3f}), "
							        f"Obstacle pos: ({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}), "
							        f"Distance: {vehicle_to_obs_dist:.3f}m")
						else:
							LOG_WARN(f"    ⚠️  WARNING: Constraint normal does NOT match vehicle-to-obstacle direction!")
							LOG_WARN(f"      Expected: ({vehicle_to_obs_normalized[0]:.6f}, {vehicle_to_obs_normalized[1]:.6f}), "
							        f"Got: ({a_vec[0]:.6f}, {a_vec[1]:.6f})")
							LOG_WARN(f"      norm_error={norm_error:.6f}, angle_error={angle_error:.4f}°")
					else:
						LOG_WARN(f"    ⚠️  Vehicle and obstacle are too close (dist={vehicle_to_obs_dist:.6f}), cannot verify")
				
				# Evaluate constraint at vehicle position
				if vehicle_pos_at_step is not None:
					constraint_value = a1 * vehicle_pos_at_step[0] + a2 * vehicle_pos_at_step[1]
					constraint_satisfied = constraint_value <= b
					violation_amount = constraint_value - b if not constraint_satisfied else 0.0
					
					if constraint_satisfied:
						LOG_INFO(f"    ✓ Constraint satisfied at vehicle position: {constraint_value:.6f} <= {b:.6f}")
					else:
						LOG_WARN(f"    ⚠️  Constraint VIOLATED at vehicle position: {constraint_value:.6f} > {b:.6f} (violation={violation_amount:.6f})")
			
			if disc_constraints == 0:
				LOG_DEBUG(f"  Disc {disc_id} at stage {stage_idx}: no valid constraints (all degenerate)")

		LOG_INFO(f"=== Returning {len(constraints)} linear constraints for stage {stage_idx} ===")
		return constraints

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		# For linearized constraints: expr = a1*x + a2*y - b
		# Matching reference implementation: lower_bound = -inf, upper_bound = 0.0
		# This enforces: a1*x + a2*y - b <= 0, i.e., a1*x + a2*y <= b
		count = len(self.calculate_constraints(state, data, stage_idx)) if (data is not None and stage_idx is not None) else 0
		return [-np.inf] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		# For linearized constraints: expr = a1*x + a2*y - b
		# Matching reference implementation: lower_bound = -inf, upper_bound = 0.0
		# This enforces: a1*x + a2*y - b <= 0, i.e., a1*x + a2*y <= b
		count = len(self.calculate_constraints(state, data, stage_idx)) if (data is not None and stage_idx is not None) else 0
		return [0.0] * count

	def is_data_ready(self, data):
		missing_data = ""
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			missing_data += "Dynamic Obstacles "
			LOG_DEBUG("Missing dynamic_obstacles: {}".format(missing_data))
		return len(missing_data) < 1

	def reset(self):
		super().reset()
		self.num_obstacles = 0
		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		# Reset position storage arrays
		self._vehicle_positions = [None] * horizon_val
		self._obstacle_positions = [None] * horizon_val
		for d in range(self.num_discs):
			for k in range(horizon_val):
				for i in range(self.num_active_obstacles):
					self._a1[d][k][i] = self._dummy_a1
					self._a2[d][k][i] = self._dummy_a2
					self._b[d][k][i] = self._dummy_b
