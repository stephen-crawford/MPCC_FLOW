"""
Solver diagnostics for MPC planning.

This module provides diagnostic utilities for identifying solver failures
and constraint violations in the MPC planning process.
"""

import traceback

import numpy as np
from typing import Optional, Dict, List, Any, TYPE_CHECKING

from utils.utils import LOG_WARN, LOG_DEBUG, LOG_INFO

if TYPE_CHECKING:
    from planning.planner import Planner


class SolverDiagnostics:
    """Diagnostic helper for solver failure analysis.

    This class provides methods for analyzing solver failures, including:
    - Checking constraint violations at current state
    - Checking warmstart feasibility
    - Identifying which constraints are violated (Gaussian vs contouring)
    - Analyzing current vehicle position vs obstacles
    - Analyzing current vehicle position vs path boundaries
    """

    def __init__(self, planner: 'Planner'):
        """Initialize with planner reference.

        Args:
            planner: The Planner instance to diagnose.
        """
        self.planner = planner

    @property
    def state(self):
        return self.planner.state

    @property
    def data(self):
        return self.planner.data

    @property
    def solver(self):
        return self.planner.solver

    def run_all_diagnostics(self):
        """Run all diagnostic checks and log results."""
        LOG_WARN("=== SOLVER FAILURE DIAGNOSTICS ===")

        vehicle_state = self._get_vehicle_state()
        if vehicle_state is None:
            LOG_WARN("=== END DIAGNOSTICS ===")
            return

        self._check_gaussian_constraints(vehicle_state)
        self._check_contouring_constraints(vehicle_state)
        self._check_warmstart_feasibility(vehicle_state)
        self._check_solver_constraint_violations()

        LOG_WARN("=== END DIAGNOSTICS ===")

    def _get_vehicle_state(self) -> Optional[Dict[str, float]]:
        """Get current vehicle state for diagnostics."""
        try:
            vehicle_x = float(self.state.get('x')) if self.state.has('x') else None
            vehicle_y = float(self.state.get('y')) if self.state.has('y') else None
            vehicle_psi = float(self.state.get('psi')) if self.state.has('psi') else None
            vehicle_v = float(self.state.get('v')) if self.state.has('v') else None

            if vehicle_x is not None and vehicle_y is not None:
                psi_str = f"{vehicle_psi:.3f}" if vehicle_psi is not None else "N/A"
                v_str = f"{vehicle_v:.3f}" if vehicle_v is not None else "N/A"
                LOG_WARN(f"Current vehicle state: x={vehicle_x:.3f}, y={vehicle_y:.3f}, psi={psi_str}, v={v_str}")
                return {'x': vehicle_x, 'y': vehicle_y, 'psi': vehicle_psi, 'v': vehicle_v}
            else:
                LOG_WARN("Cannot get vehicle position for diagnostics")
                return None
        except Exception as e:
            LOG_WARN(f"Error getting vehicle state: {e}")
            return None

    def _check_gaussian_constraints(self, vehicle_state: Dict[str, float]):
        """Check Gaussian constraint violations at current state."""
        try:
            LOG_WARN("--- Checking Gaussian Constraints ---")
            if not hasattr(self.data, 'dynamic_obstacles') or not self.data.dynamic_obstacles:
                LOG_WARN("  No dynamic obstacles to check")
                return

            from planning.types import PredictionType
            from scipy.stats import chi2

            gaussian_module = self._find_module('gaussian_constraints')
            if not gaussian_module:
                LOG_WARN("  No Gaussian constraints module found")
                return

            risk_level = float(gaussian_module.get_config_value("gaussian_constraints.risk_level", 0.05))
            chi_squared_threshold = chi2.ppf(1.0 - risk_level, df=2)
            robot_radius = float(gaussian_module.robot_radius) if gaussian_module.robot_radius else 0.5

            vehicle_x = vehicle_state['x']
            vehicle_y = vehicle_state['y']
            violations = []

            for obs_id, obstacle in enumerate(self.data.dynamic_obstacles):
                if (hasattr(obstacle, 'prediction') and obstacle.prediction is not None and
                        obstacle.prediction.type == PredictionType.GAUSSIAN and
                        hasattr(obstacle.prediction, 'steps') and len(obstacle.prediction.steps) > 0):

                    pred_step = obstacle.prediction.steps[0]
                    if hasattr(pred_step, 'position') and pred_step.position is not None:
                        mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
                        major_radius = float(getattr(pred_step, 'major_radius', 0.1))
                        minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
                        obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
                        safe_distance = robot_radius + obstacle_radius

                        sigma_x_eff = major_radius + safe_distance
                        sigma_y_eff = minor_radius + safe_distance

                        diff = np.array([vehicle_x, vehicle_y]) - mean_pos
                        mahalanobis_dist_sq = (diff[0]**2 / sigma_x_eff**2) + (diff[1]**2 / sigma_y_eff**2)

                        if mahalanobis_dist_sq < chi_squared_threshold:
                            violation = chi_squared_threshold - mahalanobis_dist_sq
                            violations.append((obs_id, violation, mean_pos, mahalanobis_dist_sq, chi_squared_threshold))

            if violations:
                LOG_WARN(f"  Found {len(violations)} Gaussian constraint violation(s):")
                for obs_id, violation, mean_pos, mah_dist_sq, threshold in violations:
                    euclidean_dist = np.linalg.norm(np.array([vehicle_x, vehicle_y]) - mean_pos)
                    LOG_WARN(f"    Obstacle {obs_id}: violation={violation:.6f}, "
                             f"mahalanobis_dist_sq={mah_dist_sq:.3f} < threshold={threshold:.3f}, "
                             f"euclidean_dist={euclidean_dist:.3f}m")
            else:
                LOG_WARN(f"  No Gaussian constraint violations at current position")

        except Exception as e:
            LOG_WARN(f"Error checking Gaussian constraints: {e}")
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

    def _check_contouring_constraints(self, vehicle_state: Dict[str, float]):
        """Check contouring constraint violations at current state."""
        try:
            LOG_WARN("--- Checking Contouring Constraints ---")
            if not hasattr(self.data, 'reference_path') or self.data.reference_path is None:
                LOG_WARN("  No reference path available")
                return

            contouring_module = self._find_module('contouring_constraints')
            if not contouring_module:
                LOG_WARN("  No contouring constraints module found")
                return

            if not hasattr(contouring_module, '_reference_path') or contouring_module._reference_path is None:
                LOG_WARN("  Contouring module has no reference path")
                return

            ref_path = contouring_module._reference_path
            vehicle_x = vehicle_state['x']
            vehicle_y = vehicle_state['y']

            # Get current spline value
            current_spline = self.state.get('spline') if self.state.has('spline') else None
            if current_spline is None:
                current_spline = self._estimate_spline_value(ref_path, vehicle_x, vehicle_y)

            if current_spline is None:
                LOG_WARN("  Could not determine spline value")
                return

            self._check_contouring_at_position(
                ref_path, contouring_module, vehicle_x, vehicle_y, current_spline
            )

        except Exception as e:
            LOG_WARN(f"Error checking contouring constraints: {e}")
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

    def _check_warmstart_feasibility(self, vehicle_state: Dict[str, float]):
        """Check warmstart feasibility at future stages."""
        try:
            LOG_WARN("--- Checking Warmstart Feasibility ---")
            if not hasattr(self.solver, 'warmstart_values') or not self.solver.warmstart_values:
                LOG_WARN("  No warmstart values available")
                return

            ws_vals = self.solver.warmstart_values
            if 'x' not in ws_vals or 'y' not in ws_vals or len(ws_vals['x']) == 0:
                LOG_WARN("  Warmstart missing x/y values")
                return

            vehicle_x = vehicle_state['x']
            vehicle_y = vehicle_state['y']
            ws_x = float(ws_vals['x'][0])
            ws_y = float(ws_vals['y'][0])
            ws_dist = np.hypot(ws_x - vehicle_x, ws_y - vehicle_y)

            LOG_WARN(f"  Warmstart stage 0: ({ws_x:.3f}, {ws_y:.3f})")
            LOG_WARN(f"  Current position: ({vehicle_x:.3f}, {vehicle_y:.3f})")
            LOG_WARN(f"  Distance: {ws_dist:.3f}m")

            if ws_dist > 1.0:
                LOG_WARN(f"  Warmstart is far from current position - may cause infeasibility")

        except Exception as e:
            LOG_WARN(f"Error checking warmstart feasibility: {e}")
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

    def _check_solver_constraint_violations(self):
        """Check solver-reported constraint violations."""
        try:
            LOG_WARN("--- Checking Solver Constraint Violations ---")
            if not hasattr(self.solver, 'opti') or self.solver.opti is None:
                LOG_WARN("  No solver opti available")
                return

            if not hasattr(self.solver.opti, 'debug') or not hasattr(self.solver.opti.debug, 'g'):
                LOG_WARN("  Solver debug info not available")
                return

            try:
                constraint_values = self.solver.opti.debug.value(self.solver.opti.debug.g)
                if constraint_values is None:
                    LOG_WARN("  No constraint values available")
                    return

                constraint_values_arr = np.array(constraint_values)
                positive_violations = constraint_values_arr[constraint_values_arr > 1e-6]

                if len(positive_violations) > 0:
                    max_violation = np.max(positive_violations)
                    max_idx = np.argmax(constraint_values_arr)
                    LOG_WARN(f"  Found {len(positive_violations)} constraint(s) with violations")
                    LOG_WARN(f"  Maximum violation: {max_violation:.6f} at index {max_idx}")
                else:
                    LOG_WARN(f"  No constraint violations found")

            except Exception as e:
                LOG_WARN(f"  Could not get constraint values: {e}")

        except Exception as e:
            LOG_WARN(f"Error checking solver constraints: {e}")
            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

    def _find_module(self, name: str) -> Optional[Any]:
        """Find a module by name in the solver's module manager."""
        if not hasattr(self.solver, 'module_manager'):
            return None
        for module in self.solver.module_manager.modules:
            if hasattr(module, 'name') and module.name == name:
                return module
        return None

    def _estimate_spline_value(self, ref_path, x: float, y: float) -> Optional[float]:
        """Estimate spline value from closest point on path."""
        if not hasattr(ref_path, 's') or ref_path.s is None:
            return None

        s_arr = np.asarray(ref_path.s, dtype=float)
        if s_arr.size == 0:
            return None

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

        if len(x_sample) == 0:
            return None

        distances = np.sqrt((np.array(x_sample) - x)**2 + (np.array(y_sample) - y)**2)
        closest_idx = np.argmin(distances)
        return float(s_sample[closest_idx])

    def _check_contouring_at_position(self, ref_path, contouring_module, x: float, y: float, spline_val: float):
        """Check contouring constraint at a specific position."""
        try:
            if not hasattr(ref_path, 'x_spline') or ref_path.x_spline is None:
                return

            path_x = float(ref_path.x_spline(spline_val))
            path_y = float(ref_path.y_spline(spline_val))
            path_dx = float(ref_path.x_spline.derivative()(spline_val))
            path_dy = float(ref_path.y_spline.derivative()(spline_val))

            norm = np.hypot(path_dx, path_dy)
            if norm <= 1e-9:
                return

            path_dx_norm = path_dx / norm
            path_dy_norm = path_dy / norm

            A = np.array([path_dy_norm, -path_dx_norm])
            path_point = np.array([path_x, path_y])
            vehicle_pos = np.array([x, y])

            contour_error = np.dot(A, vehicle_pos - path_point)

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
                LOG_WARN(f"  Contouring constraint violation:")
                LOG_WARN(f"    Contour error: {contour_error:.3f}m")
                LOG_WARN(f"    Allowed range: [{(-width_right + w_cur):.3f}, {(width_left - w_cur):.3f}]")
                if right_violation > 1e-6:
                    LOG_WARN(f"    RIGHT boundary violation: {right_violation:.3f}m")
                if left_violation > 1e-6:
                    LOG_WARN(f"    LEFT boundary violation: {left_violation:.3f}m")
            else:
                LOG_WARN(f"  No contouring constraint violations")

        except Exception as e:
            LOG_WARN(f"Error evaluating contouring constraints: {e}")


def diagnose_solver_failure(planner: 'Planner'):
    """Run diagnostics on a planner's solver failure.

    This is a convenience function that creates a SolverDiagnostics instance
    and runs all diagnostics.

    Args:
        planner: The Planner instance to diagnose.
    """
    diagnostics = SolverDiagnostics(planner)
    diagnostics.run_all_diagnostics()
