import numpy as np
import casadi as cd
from utils.math_tools import TKSpline, Spline2D, Spline3D

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN
from utils.math_tools import safe_norm, haar_difference_without_abs


class ContouringConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = "contouring_constraints"
		# Solver will be set by framework later
		self.solver = None
		# use robot discs if provided in Data
		self.num_discs = int(self.get_config_value("num_discs", 1))
		# Get slack parameter (adaptive slack increases with horizon)
		self.slack = float(self.get_config_value("contouring.slack", 1.0))
		# Number of spline segments to use per boundary at each step (default 3)
		# NOTE: For symbolic spline evaluation, we compute constraints along the entire path
		# to ensure coverage regardless of predicted position
		# Reference: C++ mpc_planner - uses fewer segments to avoid over-constraining
		# Too many segments can cause infeasibility, especially with obstacle constraints
		self.num_segments_per_boundary = int(self.get_config_value("contouring.num_segments_per_boundary", 3))
		# Store reference path data for dynamic constraint computation
		self._reference_path = None
		self._road_width_half = None
		self._width_left_spline = None  # Spline for left width (distance from centerline to left boundary)
		self._width_right_spline = None  # Spline for right width (distance from centerline to right boundary)
		LOG_DEBUG(f"ContouringConstraints initialized with {self.num_segments_per_boundary} segments per boundary")

	def update(self, state, data):
		"""Prepare reference path data for dynamic constraint computation.
		
		Analogous to C++ onDataReceived: stores reference path and width information.
		Constraints are computed dynamically in calculate_constraints based on predicted spline values.
		"""
		# REQUIREMENT CHECK: Ensure reference path starts at current vehicle position
		if hasattr(data, 'reference_path') and data.reference_path is not None and state is not None:
			try:
				vehicle_pos = state.get_position()
				if vehicle_pos is not None and len(vehicle_pos) >= 2:
					ref_path_start = (float(data.reference_path.x[0]), float(data.reference_path.y[0]))
					vehicle_pos_tuple = (float(vehicle_pos[0]), float(vehicle_pos[1]))
					
					dist = np.sqrt((ref_path_start[0] - vehicle_pos_tuple[0])**2 + 
								 (ref_path_start[1] - vehicle_pos_tuple[1])**2)
					
					if dist > 0.5:
						LOG_WARN(f"ContouringConstraints: Reference path start ({ref_path_start[0]:.2f}, {ref_path_start[1]:.2f}) "
								f"is far from vehicle position ({vehicle_pos_tuple[0]:.2f}, {vehicle_pos_tuple[1]:.2f}), "
								f"distance: {dist:.3f}m. This may cause infeasibility.")
			except Exception as e:
				LOG_DEBUG(f"Could not verify reference path start position: {e}")
		
		# Store reference path data for dynamic constraint computation (analogous to C++ onDataReceived)
		# CRITICAL: Always use the path from data, don't cache it, so it stays in sync with updates
		if hasattr(data, 'reference_path') and data.reference_path is not None:
			self._reference_path = data.reference_path  # Store reference, but always check data.reference_path for latest
			
			# Compute width splines from actual road boundaries (analogous to C++ onDataReceived)
			# C++ computes: widths_left[i] = distance(center, left), widths_right[i] = distance(center, right)
			# Then creates splines: _width_left->set_points(s, widths_left), _width_right->set_points(s, widths_right)
			self._width_left_spline = None
			self._width_right_spline = None
			
			# Check if we have left and right boundaries
			has_boundaries = False
			left_bound = None
			right_bound = None
			
			# Try to get boundaries from data object
			if hasattr(data, 'left_bound') and data.left_bound is not None:
				left_bound = data.left_bound
				has_boundaries = True
			if hasattr(data, 'right_bound') and data.right_bound is not None:
				right_bound = data.right_bound
				has_boundaries = True
			
			if has_boundaries and left_bound is not None and right_bound is not None:
				# Compute widths as distances from centerline to boundaries
				ref_path = data.reference_path
				s_arr = np.asarray(ref_path.s, dtype=float)
				
				widths_left = []
				widths_right = []
				
				# Get boundary points (as arrays or from Bound objects)
				left_x = np.asarray(left_bound.x, dtype=float) if hasattr(left_bound, 'x') else None
				left_y = np.asarray(left_bound.y, dtype=float) if hasattr(left_bound, 'y') else None
				right_x = np.asarray(right_bound.x, dtype=float) if hasattr(right_bound, 'x') else None
				right_y = np.asarray(right_bound.y, dtype=float) if hasattr(right_bound, 'y') else None
				
				# Also check for array-based boundaries
				if left_x is None and hasattr(data, 'left_boundary_x') and hasattr(data, 'left_boundary_y'):
					left_x = np.asarray(data.left_boundary_x, dtype=float)
					left_y = np.asarray(data.left_boundary_y, dtype=float)
				if right_x is None and hasattr(data, 'right_boundary_x') and hasattr(data, 'right_boundary_y'):
					right_x = np.asarray(data.right_boundary_x, dtype=float)
					right_y = np.asarray(data.right_boundary_y, dtype=float)
				
				if (left_x is not None and left_y is not None and 
					right_x is not None and right_y is not None and
					len(left_x) == len(ref_path.x) and len(right_x) == len(ref_path.x)):
					# Compute signed distances from centerline to boundaries along the normal vector
					# This is the correct way to compute widths for contouring constraints
					for i in range(len(ref_path.x)):
						center = np.array([float(ref_path.x[i]), float(ref_path.y[i])])
						left_point = np.array([float(left_x[i]), float(left_y[i])])
						right_point = np.array([float(right_x[i]), float(right_y[i])])
						
						# Get path tangent at this point (for computing normal)
						# Use spline derivative if available, otherwise estimate from neighbors
						try:
							s_val = float(s_arr[i])
							path_dx = float(ref_path.x_spline.derivative()(s_val))
							path_dy = float(ref_path.y_spline.derivative()(s_val))
						except Exception:
							# Fallback: estimate tangent from neighbors
							if i > 0 and i < len(ref_path.x) - 1:
								path_dx = float(ref_path.x[i+1] - ref_path.x[i-1])
								path_dy = float(ref_path.y[i+1] - ref_path.y[i-1])
							elif i > 0:
								path_dx = float(ref_path.x[i] - ref_path.x[i-1])
								path_dy = float(ref_path.y[i] - ref_path.y[i-1])
							else:
								path_dx = float(ref_path.x[i+1] - ref_path.x[i])
								path_dy = float(ref_path.y[i+1] - ref_path.y[i])
						
						# Normalize tangent
						norm = safe_norm(path_dx, path_dy)
						if norm < 1e-6:
							# Fallback to Euclidean distance if tangent is invalid
							width_left = np.linalg.norm(center - left_point)
							width_right = np.linalg.norm(center - right_point)
						else:
							path_dx_norm = path_dx / norm
							path_dy_norm = path_dy / norm
							
							# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
							normal = np.array([path_dy_norm, -path_dx_norm])
							
							# Compute signed distances along normal
							# Positive = left of path, negative = right of path
							left_vec = left_point - center
							right_vec = right_point - center
							
							# Signed distance = dot product with normal
							width_left = np.dot(normal, left_vec)  # Should be positive (left is in +normal direction)
							width_right = -np.dot(normal, right_vec)  # Should be positive (right is in -normal direction)
							
							# Ensure widths are positive (take absolute value if needed)
							width_left = abs(width_left)
							width_right = abs(width_right)
						
						widths_left.append(width_left)
						widths_right.append(width_right)
					
					# Create numeric splines for widths using TKSpline (analogous to C++ _width_left->set_points(s, widths_left))
					if len(widths_left) > 0 and len(widths_right) > 0:
						self._width_left_spline = TKSpline(s_arr, np.array(widths_left))
						self._width_right_spline = TKSpline(s_arr, np.array(widths_right))
						LOG_DEBUG(f"ContouringConstraints: Created width splines from road boundaries ({len(widths_left)} points)")
			
			# Fallback to fixed width if boundaries not available
			if self._width_left_spline is None or self._width_right_spline is None:
				self._road_width_half = float(self.get_config_value("road.width", 7.0)) / 2.0
				LOG_DEBUG(f"ContouringConstraints: Using fixed road width half={self._road_width_half:.3f}m (boundaries not available)")
			else:
				LOG_DEBUG("ContouringConstraints: Stored reference path data with width splines for dynamic constraint computation")
		
		# CRITICAL: According to C++ reference implementations (mpc_planner, ttk592/spline, oscardegroot/ros_tools):
		# - The reference path should remain FIXED (stationary) throughout the problem
		# - The spline parameter tracks the vehicle's position along the fixed path
		# - The spline is updated by the dynamics model based on TANGENTIAL progress only
		# - Lateral deviation (perpendicular to path) does NOT contribute to spline progression
		# - The update() method should ONLY initialize spline if None, NOT reset it every time
		# 
		# Reference: https://github.com/tud-amr/mpc_planner - path remains fixed, spline tracks position
		# Reference: https://github.com/ttk592/spline - spline parameter represents arc length along fixed curve
		# Reference: https://github.com/oscardegroot/ros_tools - path handling maintains arc length progression
		# 
		# The spline initialization should happen only if spline is None (first time).
		# After that, the dynamics model handles updates using tangential component only.
		if state is not None and hasattr(data, 'reference_path') and data.reference_path is not None:
			try:
				current_spline = state.get('spline')
				if current_spline is None:
					# Only initialize if spline is None (first time)
					# Find closest point on path to initialize spline
					vehicle_pos = state.get_position()
					if vehicle_pos is not None and len(vehicle_pos) >= 2:
						ref_path = data.reference_path
						
						# Check if reference path has s array (arc length parameterization)
						if hasattr(ref_path, 's') and ref_path.s is not None:
							s_arr = np.asarray(ref_path.s, dtype=float)
							if s_arr.size > 0:
								s_min = float(s_arr[0])
								s_max = float(s_arr[-1])
								
								# Sample path to find closest point
								num_samples = min(200, max(50, len(s_arr)))
								s_sample = np.linspace(s_min, s_max, num_samples)
								
								# Evaluate path at sample points
								x_sample = []
								y_sample = []
								for s_val in s_sample:
									try:
										if hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None:
											x_val = float(ref_path.x_spline(s_val))
											y_val = float(ref_path.y_spline(s_val))
										else:
											# Fallback: interpolate from discrete points
											idx = int((s_val - s_min) / (s_max - s_min) * (len(ref_path.x) - 1))
											idx = max(0, min(idx, len(ref_path.x) - 1))
											x_val = float(ref_path.x[idx])
											y_val = float(ref_path.y[idx])
										x_sample.append(x_val)
										y_sample.append(y_val)
									except Exception:
										continue
								
								if len(x_sample) > 0 and len(y_sample) > 0:
									x_sample = np.array(x_sample)
									y_sample = np.array(y_sample)
									
									# Find closest point
									vehicle_x = float(vehicle_pos[0])
									vehicle_y = float(vehicle_pos[1])
									distances = np.sqrt((x_sample - vehicle_x)**2 + (y_sample - vehicle_y)**2)
									closest_idx = np.argmin(distances)
									closest_s = float(s_sample[closest_idx])
									
									# Initialize spline to closest point (only if None)
									state.set('spline', closest_s)
									LOG_INFO(f"ContouringConstraints.update: Initialized spline to {closest_s:.4f} "
									         f"(first time, vehicle at ({vehicle_x:.2f}, {vehicle_y:.2f}), "
									         f"closest path point at ({x_sample[closest_idx]:.2f}, {y_sample[closest_idx]:.2f}))")
						else:
							# Fallback: use discrete path points if s array not available
							x_arr = np.asarray(ref_path.x, dtype=float)
							y_arr = np.asarray(ref_path.y, dtype=float)
							if x_arr.size > 0 and y_arr.size > 0:
								vehicle_x = float(vehicle_pos[0])
								vehicle_y = float(vehicle_pos[1])
								distances = np.sqrt((x_arr - vehicle_x)**2 + (y_arr - vehicle_y)**2)
								closest_idx = np.argmin(distances)
								
								# Estimate s value from index (assuming uniform spacing)
								estimated_s = float(closest_idx) * (30.0 / len(x_arr)) if len(x_arr) > 0 else 0.0
								state.set('spline', estimated_s)
								LOG_INFO(f"ContouringConstraints.update: Initialized spline to estimated value {estimated_s:.4f} "
								         f"(using discrete path points, closest_idx={closest_idx})")
			except Exception as e:
				LOG_DEBUG(f"ContouringConstraints.update: Could not initialize spline: {e}")
				import traceback
				LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
		
		# CRITICAL: Project warmstart to ensure it respects contouring constraints
		# Reference: C++ mpc_planner - warmstart must satisfy all constraints
		# This ensures the vehicle doesn't violate road boundaries even when avoiding obstacles
		self._project_warmstart_to_contouring_safety(data)
		
		return
	
	def _project_warmstart_to_contouring_safety(self, data):
		"""
		Project warmstart trajectory to satisfy contouring constraints (road boundaries).
		
		Reference: C++ mpc_planner - warmstart is projected to ensure feasibility
		This ensures all warmstart positions are within road boundaries, using TKSpline
		to find closest points and project positions correctly.
		
		Args:
			data: Data object containing reference path and other information
		"""
		if not hasattr(self, 'solver') or self.solver is None:
			return
		
		if not hasattr(self.solver, 'warmstart_values') or not self.solver.warmstart_values:
			return
		
		ws_vals = self.solver.warmstart_values
		if 'x' not in ws_vals or 'y' not in ws_vals:
			return
		
		# Get reference path
		if not hasattr(data, 'reference_path') or data.reference_path is None:
			return
		
		ref_path = data.reference_path
		if not hasattr(ref_path, 'x_spline') or ref_path.x_spline is None:
			return
		
		# Get road width
		road_width_half = self._road_width_half if self._road_width_half is not None else 3.5
		
		horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
		x_ws = ws_vals['x']
		y_ws = ws_vals['y']
		
		# Get spline values (will update if needed)
		if 'spline' not in ws_vals:
			ws_vals['spline'] = [0.0] * (horizon_val + 1)
		spline_ws = ws_vals['spline']
		
		# Ensure spline array is correct length
		while len(spline_ws) < horizon_val + 1:
			spline_ws.append(0.0)
		
		projections_made = 0
		max_stage_for_constraints = min(horizon_val + 1, len(x_ws))
		
		for stage_idx in range(max_stage_for_constraints):
			if stage_idx >= len(x_ws) or stage_idx >= len(y_ws):
				continue
			
			robot_pos = np.array([float(x_ws[stage_idx]), float(y_ws[stage_idx])])
			
			# CRITICAL: Find closest point on path using TKSpline
			# Reference: C++ mpc_planner - spline parameter tracks closest point on path
			try:
				# Sample path to find closest point
				s_arr = np.asarray(ref_path.s, dtype=float) if hasattr(ref_path, 's') and ref_path.s is not None else None
				if s_arr is None or s_arr.size < 2:
					continue
				
				s_min = float(s_arr[0])
				s_max = float(s_arr[-1])
				
				# Use current spline value as starting point for search (if available)
				current_s = float(spline_ws[stage_idx]) if stage_idx < len(spline_ws) else s_min
				current_s = max(s_min, min(s_max, current_s))
				
				# Search around current_s first (most likely location)
				search_radius = 5.0  # Search within 5m of current s
				s_search_min = max(s_min, current_s - search_radius)
				s_search_max = min(s_max, current_s + search_radius)
				
				# Sample path points around current_s
				num_samples = 100
				s_samples = np.linspace(s_search_min, s_search_max, num_samples)
				
				# Evaluate path at sample points
				x_samples = np.array([float(ref_path.x_spline(s)) for s in s_samples])
				y_samples = np.array([float(ref_path.y_spline(s)) for s in s_samples])
				
				# Find closest point
				distances = np.sqrt((x_samples - robot_pos[0])**2 + (y_samples - robot_pos[1])**2)
				closest_idx = np.argmin(distances)
				closest_s = float(s_samples[closest_idx])
				closest_point = np.array([x_samples[closest_idx], y_samples[closest_idx]])
				min_distance = float(distances[closest_idx])
				
				# If closest point is at boundary, expand search
				if closest_idx == 0 or closest_idx == len(s_samples) - 1:
					# Expand search to full path
					s_samples_full = np.linspace(s_min, s_max, num_samples * 2)
					x_samples_full = np.array([float(ref_path.x_spline(s)) for s in s_samples_full])
					y_samples_full = np.array([float(ref_path.y_spline(s)) for s in s_samples_full])
					distances_full = np.sqrt((x_samples_full - robot_pos[0])**2 + (y_samples_full - robot_pos[1])**2)
					closest_idx_full = np.argmin(distances_full)
					if distances_full[closest_idx_full] < min_distance:
						closest_s = float(s_samples_full[closest_idx_full])
						closest_point = np.array([x_samples_full[closest_idx_full], y_samples_full[closest_idx_full]])
						min_distance = float(distances_full[closest_idx_full])
				
				# Update spline value to closest point
				spline_ws[stage_idx] = closest_s
				
				# Get path tangent at closest point
				path_dx = float(ref_path.x_spline.derivative()(closest_s))
				path_dy = float(ref_path.y_spline.derivative()(closest_s))
				norm = np.hypot(path_dx, path_dy)
				if norm < 1e-6:
					continue
				
				path_dx_norm = path_dx / norm
				path_dy_norm = path_dy / norm
				
				# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
				A = np.array([path_dy_norm, -path_dx_norm])
				
				# Compute contour error
				diff = robot_pos - closest_point
				contour_error = np.dot(A, diff)
				
				# Check if position violates contouring constraints
				# CRITICAL: Use a small margin to ensure positions are well within boundaries
				# Reference: C++ mpc_planner - warmstart should be safely within constraints
				margin = 0.1  # 10cm margin for safety
				if abs(contour_error) > (road_width_half - margin):
					# Project position to nearest valid point within road boundaries (with margin)
					contour_error_clamped = np.clip(contour_error, -(road_width_half - margin), road_width_half - margin)
					projected_pos = closest_point + contour_error_clamped * A
					
					# Update warmstart values
					x_ws[stage_idx] = float(projected_pos[0])
					y_ws[stage_idx] = float(projected_pos[1])
					# CRITICAL: Update spline value to match closest point
					spline_ws[stage_idx] = closest_s
					projections_made += 1
					
					if stage_idx < 3:  # Log first few projections
						LOG_INFO(f"ContouringConstraints._project_warmstart_to_contouring_safety: Stage {stage_idx}: "
						         f"Projected warmstart from ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}) "
						         f"to ({projected_pos[0]:.3f}, {projected_pos[1]:.3f}) "
						         f"(contour_error={contour_error:.3f} -> {contour_error_clamped:.3f}, closest_s={closest_s:.3f})")
			except Exception as e:
				LOG_DEBUG(f"ContouringConstraints._project_warmstart_to_contouring_safety: Error at stage {stage_idx}: {e}")
				continue
		
		if projections_made > 0:
			LOG_INFO(f"ContouringConstraints._project_warmstart_to_contouring_safety: Projected {projections_made} warmstart positions to satisfy contouring constraints")
		else:
			LOG_DEBUG(f"ContouringConstraints._project_warmstart_to_contouring_safety: No warmstart projections needed (all positions satisfy constraints)")
	
	def _compute_constraint_for_s(self, cur_s, stage_idx, state=None):
		"""Compute road boundary constraints for a given arc length s.
		
		Analogous to C++ setParameters: computes constraints based on the predicted spline value.
		This is called dynamically in calculate_constraints for each stage.
		"""
		if self._reference_path is None:
			return []
		
		ref_path = self._reference_path
		s_arr = np.asarray(ref_path.s, dtype=float)
		if s_arr.size < 2:
			return []
		
		s0 = float(s_arr[0])
		s_end = float(s_arr[-1])
		
		# Clamp s to valid range
		cur_s = max(s0, min(s_end, float(cur_s)))
		
		# Get path point and derivatives (analogous to C++ module_data.path->getPoint(cur_s) and getOrthogonal(cur_s))
		# Use TKSpline's callable interface (compatible with scipy CubicSpline)
		try:
			path_point_x = ref_path.x_spline(cur_s)  # TKSpline supports __call__
			path_point_y = ref_path.y_spline(cur_s)
			path_point = np.array([float(path_point_x), float(path_point_y)])
			
			# Get path tangent and normalize using TKSpline's derivative() method
			path_dx = float(ref_path.x_spline.derivative()(cur_s))
			path_dy = float(ref_path.y_spline.derivative()(cur_s))
			
			norm = safe_norm(path_dx, path_dy)
			if norm < 1e-6:
				return []
			path_dx_norm = path_dx / norm
			path_dy_norm = path_dy / norm
		except Exception as e:
			LOG_DEBUG(f"  Stage {stage_idx}: Failed to evaluate path at s={cur_s:.3f}: {e}")
			return []
		
		# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
		# Contour error = path_dy_norm * (x - path_x) - path_dx_norm * (y - path_y)
		# = [path_dy_norm, -path_dx_norm] · [x - path_x, y - path_y]
		# So A = [path_dy_norm, -path_dx_norm] points LEFT (positive contour_error = left of path)
		A = np.array([path_dy_norm, -path_dx_norm])  # Normal pointing left
		
		# Get robot parameters
		robot_radius = 0.5
		vehicle_width = 2.0
		lr = 1.0
		if hasattr(self, 'solver') and self.solver is not None:
			if hasattr(self.solver, 'data') and self.solver.data is not None:
				data = self.solver.data
				if hasattr(data, 'robot_area') and data.robot_area and len(data.robot_area) > 0:
					robot_radius = float(data.robot_area[0].radius)
				if hasattr(data, 'vehicle_width') and data.vehicle_width is not None:
					try:
						vehicle_width = float(data.vehicle_width)
					except (TypeError, ValueError):
						pass
				if hasattr(data, 'lr') and data.lr is not None:
					try:
						lr = float(data.lr)
					except (TypeError, ValueError):
						pass
		
		# Calculate effective vehicle width for constraint (w_cur in C++ code)
		w_cur_estimate = robot_radius
		
		# Compute width_right and width_left (analogous to C++ _width_right->operator()(cur_s) and _width_left->operator()(cur_s))
		if self._width_right_spline is not None and self._width_left_spline is not None:
			# Use width splines computed from actual road boundaries
			try:
				width_right = float(self._width_right_spline(cur_s))
				width_left = float(self._width_left_spline(cur_s))
			except Exception as e:
				LOG_DEBUG(f"  Stage {stage_idx}: Failed to evaluate width splines at s={cur_s:.3f}: {e}, using fallback")
				width_right = self._road_width_half if self._road_width_half is not None else 3.5
				width_left = self._road_width_half if self._road_width_half is not None else 3.5
		else:
			# Fallback to fixed width if splines not available
			width_right = self._road_width_half if self._road_width_half is not None else 3.5
			width_left = self._road_width_half if self._road_width_half is not None else 3.5
		
		# Strict boundary constraints: vehicle must stay within road bounds
		# CRITICAL FIX: Constraint signs corrected to match C++ reference
		# A = [path_dy_norm, -path_dx_norm] points LEFT (positive contour_error = left of path)
		# 
		# RIGHT boundary: Prevent vehicle from going too far RIGHT
		# We want: contour_error >= -width_right + w_cur
		# Which means: A·(p - path_point) >= -width_right + w_cur
		# Rearranging: A·p >= A·path_point - width_right + w_cur
		# Or: -A·p <= -A·path_point + width_right - w_cur
		# So RIGHT boundary uses -A (points RIGHT) to limit rightward movement
		b_right = np.dot(-A, path_point) + width_right - w_cur_estimate
		
		# LEFT boundary: Prevent vehicle from going too far LEFT
		# We want: contour_error <= width_left - w_cur
		# Which means: A·(p - path_point) <= width_left - w_cur
		# Rearranging: A·p <= A·path_point + width_left - w_cur
		# So LEFT boundary uses A (points LEFT) to limit leftward movement
		b_left = np.dot(A, path_point) + width_left - w_cur_estimate
		
		# Note: Slack is removed to strictly enforce road boundaries
		# If slack is needed for feasibility, it should be handled separately
		
		# Return constraints as (A, b, is_left) tuples
		constraints = [
			(-A, b_right, False),  # Right boundary (is_left=False) - uses -A to limit rightward movement
			(A, b_left, True),    # Left boundary (is_left=True) - uses A to limit leftward movement
		]
		
		return constraints

	def calculate_constraints(self, state, data, stage_idx):
		"""Return linear halfspace constraints computed dynamically based on predicted spline value.
		
		CRITICAL FIX: Constraints must be computed using the SYMBOLIC spline value, not a numeric estimate.
		In the reference codebase (mpc_planner), constraints are computed symbolically based on the
		predicted spline value at each stage. Using a numeric estimate causes constraints to be computed
		for the wrong path location, allowing the vehicle to exit road boundaries.
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
		"""
		LOG_INFO(f"ContouringConstraints.calculate_constraints: stage_idx={stage_idx}")
		
		if self._reference_path is None:
			LOG_INFO(f"  ContouringConstraints stage {stage_idx}: No reference path stored, skipping constraints")
			return []
		
		# Get predicted spline value for this stage (analogous to C++ _solver->getOutput(k, "spline"))
		spline_val = None
		if state is not None:
			if hasattr(state, 'has') and state.has('spline'):
				try:
					spline_val = state.get('spline')
					LOG_DEBUG(f"  Stage {stage_idx}: Got spline from state: type={type(spline_val).__name__}, value={spline_val if not hasattr(spline_val, '__class__') or 'MX' not in str(type(spline_val)) else 'symbolic'}")
				except Exception as e:
					LOG_DEBUG(f"  Could not get spline from state: {e}")
			else:
				LOG_DEBUG(f"  Stage {stage_idx}: State does not have 'spline' variable (has method: {hasattr(state, 'has')})")
		else:
			LOG_DEBUG(f"  Stage {stage_idx}: State is None")
		
		# CRITICAL: Check if spline is symbolic
		import casadi as cd
		is_symbolic = isinstance(spline_val, (cd.MX, cd.SX))
		if is_symbolic:
			LOG_INFO(f"  Stage {stage_idx}: Spline is SYMBOLIC (CasADi {type(spline_val).__name__})")
		elif spline_val is not None:
			LOG_INFO(f"  Stage {stage_idx}: Spline is NUMERIC: {float(spline_val):.3f}")
		else:
			LOG_INFO(f"  Stage {stage_idx}: Spline is None - will try to get from warmstart")
		
		if is_symbolic:
			# SYMBOLIC spline: Compute constraints symbolically using the symbolic spline value
			# This is the correct approach - constraints will be evaluated at the actual predicted spline value
			LOG_INFO(f"  Stage {stage_idx}: Computing constraints SYMBOLICALLY using symbolic spline value")
			return self._compute_symbolic_constraints(spline_val, state, data, stage_idx)
		else:
			# LEGACY: Numeric spline should not happen in normal operation
			# Try to get symbolic spline from solver's var_dict
			LOG_WARN(f"  Stage {stage_idx}: Spline is not symbolic - attempting to get from solver var_dict")
			if hasattr(self, 'solver') and self.solver is not None:
				try:
					dynamics_model = self.solver._get_dynamics_model() if hasattr(self.solver, '_get_dynamics_model') else None
					if dynamics_model is not None and hasattr(self.solver, 'var_dict'):
						if 'spline' in self.solver.var_dict:
							if stage_idx < self.solver.var_dict['spline'].shape[0]:
								spline_sym = self.solver.var_dict['spline'][stage_idx]
								# Also update state with symbolic spline for consistency
								if state is not None:
									state.set('spline', spline_sym)
								LOG_INFO(f"  Stage {stage_idx}: Retrieved symbolic spline from solver var_dict, computing symbolically")
								return self._compute_symbolic_constraints(spline_sym, state, data, stage_idx)
							else:
								LOG_WARN(f"  Stage {stage_idx}: stage_idx {stage_idx} out of bounds for spline var_dict")
								return []
						else:
							LOG_WARN(f"  Stage {stage_idx}: 'spline' not in var_dict")
							return []
					else:
						LOG_WARN(f"  Stage {stage_idx}: Cannot get dynamics_model or var_dict")
						return []
				except Exception as e:
					LOG_WARN(f"  Stage {stage_idx}: Error retrieving symbolic spline: {e}")
					return []
			else:
				# Last resort: numeric fallback (should not happen in normal operation)
				LOG_WARN(f"  Stage {stage_idx}: No solver available, falling back to NUMERIC computation (legacy mode)")
				# Get numeric value for fallback
				cur_s = None
				if spline_val is not None:
					try:
						cur_s = float(spline_val)
					except Exception as e:
						LOG_DEBUG(f"  Error converting spline to float: {e}")
				
				# Fallback: get from warmstart or current state
				if cur_s is None:
					if hasattr(self, 'solver') and self.solver is not None:
						if hasattr(self.solver, 'warmstart_values') and 'spline' in self.solver.warmstart_values:
							if stage_idx < len(self.solver.warmstart_values['spline']):
								cur_s = float(self.solver.warmstart_values['spline'][stage_idx])
					
					if cur_s is None:
						s_arr = np.asarray(self._reference_path.s, dtype=float)
						if s_arr.size > 0:
							cur_s = float(s_arr[0])
							LOG_DEBUG(f"  Using path start s0={cur_s:.3f} as fallback")
						else:
							LOG_WARN(f"  Stage {stage_idx}: Cannot determine spline value, skipping constraints")
							return []
				
				LOG_WARN(f"  Stage {stage_idx}: Computing constraints NUMERICALLY for cur_s={cur_s:.3f} (legacy fallback)")
				return self._compute_numeric_constraints(cur_s, state, data, stage_idx)
	
	def _compute_symbolic_constraints(self, spline_sym, state, data, stage_idx):
		"""Compute constraints symbolically using the symbolic spline value.
		
		CRITICAL: This uses Spline2D with path parameters when available (matching ContouringObjective),
		or falls back to CasADi interpolants. This ensures constraints are evaluated at the actual
		predicted spline value, matching the C++ reference implementation.
		
		CRITICAL FIX: When vehicle temporarily doesn't make forward progress, ensure constraints
		are computed at appropriate segments. The spline value should be at least as far as the
		vehicle's current position to maintain proper constraint coverage.
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
		"""
		import casadi as cd
		from utils.math_tools import Spline2D
		
		# Get path bounds
		s_arr = np.asarray(self._reference_path.s, dtype=float)
		s_min = float(s_arr[0])
		s_max = float(s_arr[-1])
		
		# CRITICAL FIX: Ensure spline value doesn't go backward too far
		# When vehicle temporarily doesn't make forward progress, we should still maintain
		# constraints at appropriate segments. However, we need to prevent the spline from
		# going backward beyond the vehicle's actual position.
		# 
		# For stage 0 (current state), use the current spline value directly
		# For future stages, use the predicted spline value but ensure it's reasonable
		# Reference: C++ mpc_planner - constraints maintain coverage even when progress is slow
		
		# Get current vehicle position to estimate minimum valid spline value
		current_s_min = s_min  # Default to path start
		if stage_idx == 0 and state is not None:
			# For stage 0, try to get current spline value from state or warmstart
			try:
				if hasattr(self, 'solver') and self.solver is not None:
					if hasattr(self.solver, 'warmstart_values') and 'spline' in self.solver.warmstart_values:
						if len(self.solver.warmstart_values['spline']) > 0:
							current_s_min = float(self.solver.warmstart_values['spline'][0])
							current_s_min = max(s_min, min(s_max, current_s_min))  # Clamp to valid range
			except:
				pass
		
		# Clamp symbolic spline to valid range (in arc length units)
		# CRITICAL: Ensure spline doesn't go backward beyond current position
		# Use fmax to ensure spline is at least at current position (prevents backward drift)
		spline_clamped = cd.fmax(current_s_min, cd.fmin(s_max, spline_sym))
		
		# Normalize spline to [0,1] for Spline2D evaluation
		s_normalized = (spline_clamped - s_min) / cd.fmax(s_max - s_min, 1e-6)
		s_normalized = cd.fmax(0.0, cd.fmin(1.0, s_normalized))
		
		# Try to use Spline2D with path parameters (preferred method, matches C++ reference)
		use_spline2d = False
		if hasattr(self, 'solver') and self.solver is not None:
			if hasattr(self.solver, 'parameter_manager') and self.solver.parameter_manager is not None:
				try:
					# Get parameters for current stage
					params_dict = self.solver.parameter_manager.get_all(stage_idx)
					
					# Check if path parameters are available
					has_path_params = params_dict.get('path_0_start') is not None
					if has_path_params:
						# Create parameter wrapper for Spline2D
						class ParamWrapper:
							def __init__(self, params_dict):
								self.params_dict = params_dict
							def get(self, key, default=None):
								return self.params_dict.get(key, default)
							def has_parameter(self, key):
								return key in self.params_dict
						
						param_wrapper = ParamWrapper(params_dict)
						
						# Get num_segments and check for 3D contouring from config or default
						num_segments = 10
						three_dimensional = False
						if hasattr(self, 'solver') and hasattr(self.solver, 'data'):
							if hasattr(self.solver.data, 'reference_path'):
								# Try to get from contouring objective config
								try:
									from modules.objectives.contouring_objective import ContouringObjective
									if hasattr(self.solver, 'module_manager'):
										for module in self.solver.module_manager.modules:
											if isinstance(module, ContouringObjective):
												num_segments = module.num_segments
												three_dimensional = module.three_dimensional_contouring
												break
								except:
									pass
						
						# Check if z parameters are available for 3D
						has_z_params = three_dimensional and params_dict.get('path_z_0_a') is not None
						
						# Create Spline2D or Spline3D instance
						if has_z_params:
							path = Spline3D(param_wrapper, num_segments, s_normalized)
							path_x_sym, path_y_sym, path_z_sym = path.at(s_normalized)
							path_dx_norm_sym, path_dy_norm_sym, path_dz_norm_sym = path.deriv_normalized(s_normalized)
							LOG_DEBUG(f"  Stage {stage_idx}: Using Spline3D with path parameters for symbolic constraints")
						else:
							path = Spline2D(param_wrapper, num_segments, s_normalized)
							path_x_sym, path_y_sym = path.at(s_normalized)
							path_dx_norm_sym, path_dy_norm_sym = path.deriv_normalized(s_normalized)
							path_z_sym = None
							path_dz_norm_sym = None
							LOG_DEBUG(f"  Stage {stage_idx}: Using Spline2D with path parameters for symbolic constraints")
						
						use_spline2d = True
				except Exception as e:
					LOG_DEBUG(f"  Stage {stage_idx}: Failed to use Spline2D, falling back to interpolants: {e}")
		
		if not use_spline2d:
			# Fallback: Create CasADi interpolants from TKSpline (numeric sampling)
			# Sample path at discrete points for interpolation
			s_vals = s_arr if s_arr.size > 0 else np.linspace(s_min, s_max, 100)
			# Use TKSpline for numeric sampling (reference_path.x_spline is now TKSpline)
			x_vals = np.array([float(self._reference_path.x_spline.at(si)) for si in s_vals])
			y_vals = np.array([float(self._reference_path.y_spline.at(si)) for si in s_vals])
			
			# Create CasADi interpolants for path position
			x_interp = cd.interpolant('x_interp', 'linear', [s_vals], x_vals)
			y_interp = cd.interpolant('y_interp', 'linear', [s_vals], y_vals)
			
			# Evaluate path point symbolically (using arc length, not normalized)
			path_x_sym = x_interp(spline_clamped)
			path_y_sym = y_interp(spline_clamped)
			
			# Compute path tangent using finite differences
			eps = 1e-3
			dx_sym = (x_interp(spline_clamped + eps) - x_interp(spline_clamped - eps)) / (2 * eps)
			dy_sym = (y_interp(spline_clamped + eps) - y_interp(spline_clamped - eps)) / (2 * eps)
			norm_sym = cd.sqrt(dx_sym*dx_sym + dy_sym*dy_sym)
			norm_sym = cd.fmax(norm_sym, 1e-6)
			path_dx_norm_sym = dx_sym / norm_sym
			path_dy_norm_sym = dy_sym / norm_sym
		
		# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
		A_sym = cd.vertcat(path_dy_norm_sym, -path_dx_norm_sym)
		
		# Get robot parameters
		robot_radius = 0.5
		if hasattr(self, 'solver') and self.solver is not None:
			if hasattr(self.solver, 'data') and self.solver.data is not None:
				data = self.solver.data
				if hasattr(data, 'robot_area') and data.robot_area and len(data.robot_area) > 0:
					robot_radius = float(data.robot_area[0].radius)
		
		# Get road width (for now, use fixed width - can be made symbolic if width splines exist)
		width_right = self._road_width_half if self._road_width_half is not None else 3.5
		width_left = self._road_width_half if self._road_width_half is not None else 3.5
		w_cur_estimate = robot_radius
		
		# Get vehicle position symbolically
		pos_x_sym = state.get('x')
		pos_y_sym = state.get('y')
		psi_sym = state.get('psi')
		
		if pos_x_sym is None or pos_y_sym is None:
			# Try to get from solver's var_dict if available
			LOG_WARN(f"  Stage {stage_idx}: Cannot get symbolic position, attempting to get from solver var_dict")
			if hasattr(self, 'solver') and self.solver is not None:
				try:
					dynamics_model = self.solver._get_dynamics_model() if hasattr(self.solver, '_get_dynamics_model') else None
					if dynamics_model is not None and hasattr(self.solver, 'var_dict'):
						if 'x' in self.solver.var_dict and 'y' in self.solver.var_dict:
							if stage_idx < self.solver.var_dict['x'].shape[0] and stage_idx < self.solver.var_dict['y'].shape[0]:
								pos_x_sym = self.solver.var_dict['x'][stage_idx]
								pos_y_sym = self.solver.var_dict['y'][stage_idx]
								# Update state with symbolic values
								if state is not None:
									state.set('x', pos_x_sym)
									state.set('y', pos_y_sym)
								LOG_INFO(f"  Stage {stage_idx}: Retrieved symbolic position from solver var_dict")
								# Retry with updated state
								return self._compute_symbolic_constraints(spline_sym, state, data, stage_idx)
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
		
		# CRITICAL FIX: Compute constraints at MULTIPLE segments along the path to ensure full coverage
		# Reference: C++ mpc_planner computes constraints at multiple segments to prevent vehicle from exiting boundaries
		# The numeric version uses num_segments_per_boundary segments, so the symbolic version should too
		# Generate segment s values centered around predicted spline value
		constraints = []
		psi_sym = state.get('psi')
		
		# Determine segment spacing (similar to numeric version)
		# Estimate segment spacing based on path geometry
		segment_spacing = 1.0  # meters
		try:
			# Estimate ds_per_meter from path tangent at predicted position
			if use_spline2d:
				# Use Spline2D derivative to estimate spacing
				path_dx_est = path_dx_norm_sym * cd.sqrt(path_dx_norm_sym*path_dx_norm_sym + path_dy_norm_sym*path_dy_norm_sym)
				path_dy_est = path_dy_norm_sym * cd.sqrt(path_dx_norm_sym*path_dx_norm_sym + path_dy_norm_sym*path_dy_norm_sym)
				ds_per_meter = 1.0 / cd.fmax(cd.sqrt(path_dx_est*path_dx_est + path_dy_est*path_dy_est), 1e-6)
			else:
				# Use interpolant to estimate spacing
				eps_est = 0.1
				dx_est = (x_interp(spline_clamped + eps_est) - x_interp(spline_clamped - eps_est)) / (2 * eps_est)
				dy_est = (y_interp(spline_clamped + eps_est) - y_interp(spline_clamped - eps_est)) / (2 * eps_est)
				ds_per_meter = 1.0 / cd.fmax(cd.sqrt(dx_est*dx_est + dy_est*dy_est), 1e-6)
			ds_segment = segment_spacing * ds_per_meter
		except:
			ds_segment = (s_max - s_min) * 0.1
		
		# Generate segment s values centered around predicted spline value
		# CRITICAL: Use symbolic operations to create segment s values
		# CRITICAL FIX: Ensure segments don't extend backward beyond current position
		# When vehicle doesn't make forward progress, constraints should still look forward
		if self.num_segments_per_boundary == 1:
			segment_s_sym_list = [spline_clamped]
		else:
			# Create segment s values symbolically around predicted spline
			# Total span = (num_segments - 1) * ds_segment
			total_span = (self.num_segments_per_boundary - 1) * ds_segment
			
			# CRITICAL: For forward-looking constraints, bias segments forward from current position
			# This ensures constraints cover the path ahead even when progress is slow
			# Reference: C++ mpc_planner - constraints are forward-looking
			s_center = spline_clamped
			
			# For stage 0, ensure we don't look too far backward
			if stage_idx == 0:
				# Get current spline value as minimum
				try:
					if hasattr(self, 'solver') and self.solver is not None:
						if hasattr(self.solver, 'warmstart_values') and 'spline' in self.solver.warmstart_values:
							if len(self.solver.warmstart_values['spline']) > 0:
								current_s = float(self.solver.warmstart_values['spline'][0])
								current_s = max(s_min, min(s_max, current_s))
								# Ensure center is at least at current position
								s_center = cd.fmax(current_s, spline_clamped)
				except:
					pass
			
			# Generate segments forward-biased: more segments ahead, fewer behind
			# This ensures constraints cover the path the vehicle will travel
			s_start_sym = cd.fmax(s_min, s_center - total_span * 0.3)  # 30% backward, 70% forward
			s_end_sym = cd.fmin(s_max, s_center + total_span * 0.7)
			
			# CRITICAL FIX: Use CasADi's if_else for boundary adjustments
			# Cannot use Python if/else with symbolic values - causes "Can only determine truth value" error
			# Check if at start boundary (s_start_sym would be clamped to s_min)
			at_start = s_center - total_span * 0.3 < s_min
			# Check if at end boundary (s_end_sym would be clamped to s_max)
			at_end = s_center + total_span * 0.7 > s_max
			
			# At start boundary: extend forward (s_end = s_start + total_span)
			s_end_extended = cd.fmin(s_max, s_min + total_span)
			# At end boundary: extend backward (s_start = s_end - total_span)
			s_start_extended = cd.fmax(s_min, s_max - total_span)
			
			# Apply boundary adjustments symbolically using if_else
			s_end_sym = cd.if_else(at_start, s_end_extended, s_end_sym)
			s_start_sym = cd.if_else(at_end, s_start_extended, s_start_sym)
			
			# Create evenly spaced segment s values symbolically
			segment_s_sym_list = []
			if self.num_segments_per_boundary > 1:
				for i in range(self.num_segments_per_boundary):
					alpha = float(i) / max(1, self.num_segments_per_boundary - 1) if self.num_segments_per_boundary > 1 else 0.0
					segment_s_sym = s_start_sym + alpha * (s_end_sym - s_start_sym)
					# CRITICAL: Ensure segments don't go backward beyond current position
					segment_s_sym = cd.fmax(s_min, cd.fmin(s_max, segment_s_sym))
					segment_s_sym_list.append(segment_s_sym)
			else:
				segment_s_sym_list = [spline_clamped]
		
		# Compute constraints for each segment
		for segment_idx, segment_s_sym in enumerate(segment_s_sym_list):
			# Evaluate path at this segment symbolically
			if use_spline2d:
				# Normalize segment s for Spline2D
				segment_s_normalized = (segment_s_sym - s_min) / cd.fmax(s_max - s_min, 1e-6)
				segment_s_normalized = cd.fmax(0.0, cd.fmin(1.0, segment_s_normalized))
				
				if has_z_params:
					path_seg = Spline3D(param_wrapper, num_segments, segment_s_normalized)
					path_x_seg_sym, path_y_seg_sym, _ = path_seg.at(segment_s_normalized)
					path_dx_norm_seg_sym, path_dy_norm_seg_sym, _ = path_seg.deriv_normalized(segment_s_normalized)
				else:
					path_seg = Spline2D(param_wrapper, num_segments, segment_s_normalized)
					path_x_seg_sym, path_y_seg_sym = path_seg.at(segment_s_normalized)
					path_dx_norm_seg_sym, path_dy_norm_seg_sym = path_seg.deriv_normalized(segment_s_normalized)
			else:
				# Use interpolants
				path_x_seg_sym = x_interp(segment_s_sym)
				path_y_seg_sym = y_interp(segment_s_sym)
				eps_seg = 1e-3
				dx_seg_sym = (x_interp(segment_s_sym + eps_seg) - x_interp(segment_s_sym - eps_seg)) / (2 * eps_seg)
				dy_seg_sym = (y_interp(segment_s_sym + eps_seg) - y_interp(segment_s_sym - eps_seg)) / (2 * eps_seg)
				norm_seg_sym = cd.sqrt(dx_seg_sym*dx_seg_sym + dy_seg_sym*dy_seg_sym)
				norm_seg_sym = cd.fmax(norm_seg_sym, 1e-6)
				path_dx_norm_seg_sym = dx_seg_sym / norm_seg_sym
				path_dy_norm_seg_sym = dy_seg_sym / norm_seg_sym
			
			# Normal vector pointing left for this segment: A_seg = [path_dy_norm_seg, -path_dx_norm_seg]
			# CRITICAL FIX: Constraint signs corrected to match numeric version and C++ reference
			# RIGHT boundary: Prevent vehicle from going too far RIGHT
			# We want: contour_error >= -width_right + w_cur
			# Which means: A_seg·(p - path_point) >= -width_right + w_cur
			# Rearranging: -A_seg·p <= -A_seg·path_point + width_right - w_cur
			path_point_dot_A_seg = path_dy_norm_seg_sym * path_x_seg_sym - path_dx_norm_seg_sym * path_y_seg_sym
			b_right_seg_sym = -path_point_dot_A_seg + width_right - w_cur_estimate
			
			# LEFT boundary: Prevent vehicle from going too far LEFT
			# We want: contour_error <= width_left - w_cur
			# Which means: A_seg·(p - path_point) <= width_left - w_cur
			# Rearranging: A_seg·p <= A_seg·path_point + width_left - w_cur
			b_left_seg_sym = path_point_dot_A_seg + width_left - w_cur_estimate
			
			# Right boundary constraint for this segment: -A_seg·[x, y] <= b_right_seg
			# CRITICAL FIX: Constraint expression must be: (-A_seg·[x, y]) - b_right_seg <= 0
			# This enforces: -A_seg·[x, y] <= b_right_seg
			constraint_right_seg_expr = -path_dy_norm_seg_sym * pos_x_sym + path_dx_norm_seg_sym * pos_y_sym - b_right_seg_sym
			
			# Left boundary constraint for this segment: A_seg·[x, y] <= b_left_seg
			# CRITICAL FIX: Constraint expression must be: (A_seg·[x, y]) - b_left_seg <= 0
			# This enforces: A_seg·[x, y] <= b_left_seg
			constraint_left_seg_expr = path_dy_norm_seg_sym * pos_x_sym - path_dx_norm_seg_sym * pos_y_sym - b_left_seg_sym
			
			# Apply disc offset if needed (for each disc)
			for disc_id in range(self.num_discs):
				disc_offset = 0.0
				if hasattr(data, "robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
					disc_offset = float(data.robot_area[disc_id].offset)
				
				if abs(disc_offset) > 1e-9 and psi_sym is not None:
					# Adjust constraints for disc offset
					# Right boundary uses -A_seg, so offset adjustment is: -A_seg·offset
					offset_adjustment_right_seg = disc_offset * (-path_dy_norm_seg_sym * cd.cos(psi_sym) + path_dx_norm_seg_sym * cd.sin(psi_sym))
					constraint_right_seg_expr_disc = constraint_right_seg_expr - offset_adjustment_right_seg
					
					# Left boundary uses A_seg, so offset adjustment is: A_seg·offset
					offset_adjustment_left_seg = disc_offset * (path_dy_norm_seg_sym * cd.cos(psi_sym) - path_dx_norm_seg_sym * cd.sin(psi_sym))
					constraint_left_seg_expr_disc = constraint_left_seg_expr - offset_adjustment_left_seg
				else:
					constraint_right_seg_expr_disc = constraint_right_seg_expr
					constraint_left_seg_expr_disc = constraint_left_seg_expr
				
				# Add constraints for this segment and disc
				constraints.append({
					"type": "symbolic_expression",
					"expression": constraint_right_seg_expr_disc,
					"ub": 0.0,  # expr <= 0
					"constraint_type": "contouring",
					"segment_idx": segment_idx,
					"disc_id": disc_id,
					"is_left": False,
				})
				constraints.append({
					"type": "symbolic_expression",
					"expression": constraint_left_seg_expr_disc,
					"ub": 0.0,  # expr <= 0
					"constraint_type": "contouring",
					"segment_idx": segment_idx,
					"disc_id": disc_id,
					"is_left": True,
				})
		
		LOG_INFO(f"  Symbolic constraints: {len(constraints)} constraint expressions computed symbolically "
		         f"({self.num_segments_per_boundary} segments × 2 boundaries × {self.num_discs} disc(s))")
		return constraints
	
	def _compute_numeric_constraints(self, cur_s, state, data, stage_idx):
		"""Compute constraints numerically for a given spline value (for warmstart/initialization)."""
		s_arr = np.asarray(self._reference_path.s, dtype=float)
		s_min = float(s_arr[0])
		s_max = float(s_arr[-1])
		
		# Clamp cur_s to valid range
		cur_s = max(s_min, min(s_max, float(cur_s)))
		
		# Determine segment spacing
		segment_spacing = 1.0  # meters
		try:
			dx = float(self._reference_path.x_spline.derivative()(cur_s))
			dy = float(self._reference_path.y_spline.derivative()(cur_s))
			ds_per_meter = 1.0 / (np.sqrt(dx**2 + dy**2) + 1e-6)
			ds_segment = segment_spacing * ds_per_meter
		except:
			ds_segment = (s_max - s_min) * 0.1
		
		# Generate segment s values centered around cur_s
		segment_s_values = []
		if self.num_segments_per_boundary == 1:
			segment_s_values = [cur_s]
		else:
			total_span = (self.num_segments_per_boundary - 1) * ds_segment
			s_start = max(s_min, cur_s - total_span / 2)
			s_end = min(s_max, cur_s + total_span / 2)
			
			if s_start == s_min and s_end < s_max:
				s_end = min(s_max, s_start + total_span)
			elif s_end == s_max and s_start > s_min:
				s_start = max(s_min, s_end - total_span)
			
			if self.num_segments_per_boundary > 1:
				segment_s_values = np.linspace(s_start, s_end, self.num_segments_per_boundary)
			else:
				segment_s_values = [cur_s]
		
		segment_s_values = [max(s_min, min(s_max, float(s))) for s in segment_s_values]
		
		constraints = []
		halfspace_count = 0
		
		for segment_s in segment_s_values:
			constraint_tuples = self._compute_constraint_for_s(segment_s, stage_idx, state)
			
			if not constraint_tuples:
				continue
			
			for A, b, is_left in constraint_tuples:
				halfspace_count += 1
				for disc_id in range(self.num_discs):
					disc_offset = 0.0
					disc_radius = 0.5
					if hasattr(data, "robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
						disc_offset = float(data.robot_area[disc_id].offset)
						disc_radius = float(data.robot_area[disc_id].radius)
					
					constraints.append({
						"type": "linear",
						"a1": float(A[0]),
						"a2": float(A[1]),
						"b": float(b),
						"disc_offset": disc_offset,
						"is_left": is_left,
						"spline_s": float(segment_s),
					})
		
		LOG_INFO(f"ContouringConstraints.calculate_constraints: stage_idx={stage_idx}, cur_s={cur_s:.3f}, "
		         f"segments={len(segment_s_values)}, returning {len(constraints)} constraint(s) "
		         f"({halfspace_count} halfspace(s) × {self.num_discs} disc(s))")
		
		if constraints:
			for i, const in enumerate(constraints[:3]):
				LOG_INFO(f"  Contouring constraint stage {stage_idx}, constraint {i}: a1={const.get('a1', 'N/A'):.6f}, "
				        f"a2={const.get('a2', 'N/A'):.6f}, b={const.get('b', 'N/A'):.6f}, "
				        f"disc_offset={const.get('disc_offset', 0.0):.4f}, is_left={const.get('is_left', 'N/A')}")
			if len(constraints) > 3:
				LOG_INFO(f"  ... and {len(constraints) - 3} more contouring constraints")
		else:
			LOG_INFO(f"  ContouringConstraints stage {stage_idx}: No constraints returned")
		
		return constraints

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		"""Return lower bounds for constraints (all -inf for halfspace constraints A·p <= b)."""
		# For A·p <= b, expr = A·p - b ≤ 0 → lb = -inf per constraint
		count = 0
		if data is not None and stage_idx is not None:
			# Estimate constraint count: (2 boundaries × num_segments_per_boundary) × num_discs
			count = 2 * self.num_segments_per_boundary * self.num_discs
		return [-np.inf] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		"""Return upper bounds for constraints (all 0 for halfspace constraints A·p <= b)."""
		# Upper bound 0 per constraint for A·p <= b
		count = 0
		if data is not None and stage_idx is not None:
			# Estimate constraint count: (2 boundaries × num_segments_per_boundary) × num_discs
			count = 2 * self.num_segments_per_boundary * self.num_discs
		return [0.0] * count

	def get_visualizer(self):
		class RoadBoundsVisualizer:
			def __init__(self, module):
				self.module = module
			def visualize(self, state, data, stage_idx=0):
				try:
					import matplotlib.pyplot as plt
				except Exception:
					return
				# Draw road bounds from reference_path and configured width
				if not hasattr(data, 'reference_path') or data.reference_path is None:
					return
				ref = data.reference_path
				if not hasattr(ref, 'x_spline') or not hasattr(ref, 'y_spline'):
					return
				
				# Sample along arc length
				s_vals = ref.s if hasattr(ref, 's') and ref.s is not None else np.linspace(0.0, 1.0, 100)
				if len(s_vals) == 0:
					return
				
				# Determine sampling range
				if len(s_vals) > 0 and (np.max(s_vals) <= 1.001):
					s_min = float(s_vals[0])
					s_max = float(s_vals[-1])
					s_sample = np.linspace(s_min, s_max, 200)  # More points for smoother visualization
				else:
					s_sample = np.linspace(float(s_vals[0]), float(s_vals[-1]), 200)
				
				xl = []
				yl = []
				xr = []
				yr = []
				
				# Check if width splines are available (more accurate)
				if (hasattr(self.module, '_width_left_spline') and self.module._width_left_spline is not None and
					hasattr(self.module, '_width_right_spline') and self.module._width_right_spline is not None):
					# Use width splines for accurate boundary visualization
					for s in s_sample:
						try:
							# Get centerline position using TKSpline
							x = ref.x_spline(s)  # TKSpline supports __call__
							y = ref.y_spline(s)
							
							# Get path tangent using TKSpline derivative
							dx = ref.x_spline.derivative()(s)
							dy = ref.y_spline.derivative()(s)
							norm = np.hypot(float(dx), float(dy))
							if norm < 1e-6:
								continue
							
							# Normalize tangent
							dx_norm = float(dx) / norm
							dy_norm = float(dy) / norm
							
							# Normal vector pointing left: [dy_norm, -dx_norm]
							nx = float(dy_norm)
							ny = float(-dx_norm)
							
							# Get widths from width splines (TKSpline supports __call__)
							width_left = float(self.module._width_left_spline(s))
							width_right = float(self.module._width_right_spline(s))
							
							# Compute boundary points
							xl.append(float(x) + width_left * nx)
							yl.append(float(y) + width_left * ny)
							xr.append(float(x) - width_right * nx)
							yr.append(float(y) - width_right * ny)
						except Exception as e:
							LOG_DEBUG(f"RoadBoundsVisualizer: Error evaluating at s={s:.3f}: {e}")
							continue
				else:
					# Fallback: use fixed width
					width = float(self.module.get_config_value("road.width", 7.0))
					half_w = 0.5 * width
					
					for s in s_sample:
						try:
							# Get centerline position using TKSpline
							x = ref.x_spline(s)  # TKSpline supports __call__
							y = ref.y_spline(s)
							
							# Get path tangent using TKSpline derivative
							dx = ref.x_spline.derivative()(s)
							dy = ref.y_spline.derivative()(s)
							norm = np.hypot(float(dx), float(dy))
							if norm < 1e-6:
								continue
							
							# Normalize tangent
							nx = -float(dy) / norm
							ny = float(dx) / norm
							
							# Compute boundary points with fixed width
							xl.append(float(x) + half_w * nx)
							yl.append(float(y) + half_w * ny)
							xr.append(float(x) - half_w * nx)
							yr.append(float(y) - half_w * ny)
						except Exception as e:
							LOG_DEBUG(f"RoadBoundsVisualizer: Error evaluating at s={s:.3f}: {e}")
							continue
				
				# Plot boundaries (using gray to distinguish from orange/cyan contouring constraints)
				if len(xl) > 0 and len(yl) > 0:
					ax = plt.gca()
					ax.plot(xl, yl, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Left Road Boundary')
					ax.plot(xr, yr, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Right Road Boundary')
		return RoadBoundsVisualizer(self)
