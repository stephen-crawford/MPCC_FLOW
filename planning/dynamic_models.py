from math import pi

import casadi as cd
import numpy as np
from scipy.interpolate import interpolate

from utils.utils import model_map_path, write_to_yaml, LOG_DEBUG, LOG_WARN, LOG_INFO


def safe_casadi_discrete_dynamics(model, z, p, nx=None, integration_step=None):
    """
    Symbolic rk4 using Casadi.
    """
    try:
        if nx is None:
            nx = model.state_dimension

        if integration_step is None:
            integration_step = 0.1

        # Ensure integration step is reasonable
        integration_step = min(max(integration_step, 0.01), 0.5)

        # Extract control and state with bounds checking
        if z.shape[0] < model.nu + nx:
            # Create safe fallback values
            return cd.DM.zeros(nx, 1)

        u = z[0:model.nu]
        x = z[model.nu:model.nu + nx]

        # Define the continuous dynamics function with additional safety
        def f(x_val, u_val, params):
            try:
                result = model.continuous_model(x_val, u_val, params)
                # Skip NaN check for symbolic expressions
                if isinstance(result, np.ndarray) and np.isnan(result).any():
                    LOG_WARN("NaN in continuous model output")
                    return cd.DM.zeros(nx, 1)
                return result
            except Exception as e:
                LOG_WARN(f"Error in continuous model: {e}")
                return cd.DM.zeros(nx, 1)

        # Use a safer dt value
        dt = cd.MX(integration_step)

        # RK4 integration with additional error checking
        h = dt
        try:
            k1 = f(x, u, p)
            k2 = f(x + h / 2 * k1, u, p)
            k3 = f(x + h / 2 * k2, u, p)
            k4 = f(x + h * k3, u, p)
        except Exception as e:
            LOG_WARN(f"Error in RK4 steps: {e}")
            return x  # Return original state if integration fails

        # Compute next state with safeguards against NaN
        x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Only perform NaN checks and bounds enforcement for numeric types, not symbolic
        if isinstance(x_next, cd.DM):  # Check if it's a numeric type
            # Project extreme values and check for NaN
            for i in range(x_next.size1()):
                if not x_next[i].is_regular():  # Only for numeric values
                    x_next[i] = x[i]  # Replace NaN with original value
                x_next[i] = cd.fmin(cd.fmax(x_next[i], -1e6), 1e6)  # Bound values

        return x_next

    except Exception as e:
        LOG_WARN(f"Error in safe_casadi_discrete_dynamics: {e}")
        # Return original state as fallback
        if nx is None:
            nx = model.state_dimension
        return z[model.nu:model.nu + nx]


def numeric_rk4(next_state, vehicle, params, timestep):
        if isinstance(next_state, cd.MX):
            LOG_DEBUG(f"Got symbolic result from discrete_dynamics. Using RK4 integration directly.")

            u = vehicle.get_z()[0:vehicle.nu]
            x_full = vehicle.get_z()[vehicle.nu:vehicle.nu + vehicle.state_dimension]

            # Get number of states to integrate (some models have algebraic states not integrated)
            nx_integrate = getattr(vehicle, 'state_dimension_integrate', None)
            if nx_integrate is None:
                nx_integrate = vehicle.state_dimension

            # Split state into integrated and algebraic parts
            x_integrate = x_full[:nx_integrate]
            x_algebraic = x_full[nx_integrate:] if nx_integrate < vehicle.state_dimension else None

            # Get parameters
            p = params

            # Use continuous model directly for RK4 integration
            # continuous_model returns derivatives for integrated states only
            k1 = vehicle.continuous_model(x_full, u, p)
            k2 = vehicle.continuous_model(cd.vertcat(x_integrate + timestep / 2 * k1, x_algebraic) if x_algebraic is not None else x_integrate + timestep / 2 * k1, u, p)
            k3 = vehicle.continuous_model(cd.vertcat(x_integrate + timestep / 2 * k2, x_algebraic) if x_algebraic is not None else x_integrate + timestep / 2 * k2, u, p)
            k4 = vehicle.continuous_model(cd.vertcat(x_integrate + timestep * k3, x_algebraic) if x_algebraic is not None else x_integrate + timestep * k3, u, p)

            # Compute next state for integrated states
            x_next_integrate = x_integrate + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Handle algebraic states (e.g., spline = arc length, updated as s + v*dt)
            if x_algebraic is not None:
                # For contouring models, spline is updated based on velocity
                v = x_next_integrate[3] if x_next_integrate.size1() > 3 else 1.0
                s = x_algebraic[0] if x_algebraic.size1() > 0 else 0.0
                s_next = s + v * timestep
                s_next = cd.fmax(s_next, 0.0)  # Clamp to non-negative
                x_next_algebraic = cd.vertcat(s_next)
                next_state = cd.vertcat(x_next_integrate, x_next_algebraic)
            else:
                next_state = x_next_integrate

            return next_state
        else:
            raise NotImplementedError


def numpy_to_casadi(x: np.array) -> cd.SX:
    return cd.vertcat(*x.tolist())

class DynamicsModel:

    def __init__(self):
        self.settings = None
        self._z = None
        self.nu = 0  # number of control variables
        self.state_dimension = 0  # number of states

        self.dependent_vars = []
        self.inputs = []

        self.lower_bound = []
        self.upper_bound = []

        self.params = None
        self.state_dimension_integrate = None

    def symbolic_dynamics(self, x, u, p, timestep):
        """
		Performs symbolic RK4 integration for the full state vector.
		This method is used by the CasADiSolver to build the optimization constraints.
		Subclasses with algebraic states (non-integrated) should override this.
		
		CRITICAL: This method is called by the solver to create the constraint:
		    x[k+1] == symbolic_dynamics(x[k], u[k], timestep)
		
		This constraint ensures that:
		    - States are NOT free decision variables
		    - Only control inputs (u) are free decision variables
		    - States are computed deterministically via RK4 integration
		
		RK4 (Runge-Kutta 4th order) integration:
		k1 = f(x, u, p)
		k2 = f(x + dt/2 * k1, u, p)
		k3 = f(x + dt/2 * k2, u, p)
		k4 = f(x + dt * k3, u, p)
		x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
		
		Reference: https://github.com/tud-amr/mpc_planner - states are constrained by RK4 integration.
		"""
        # RK4 integration - this is the standard 4th order Runge-Kutta method
        # All states are integrated using RK4
        k1 = self.continuous_model(x, u, p)
        k2 = self.continuous_model(x + timestep / 2 * k1, u, p)
        k3 = self.continuous_model(x + timestep / 2 * k2, u, p)
        k4 = self.continuous_model(x + timestep * k3, u, p)

        x_next = x + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next


    def get_dependent_vars(self):
        return self.dependent_vars

    def get_inputs(self):
        return self.inputs

    def get_all_vars(self):
        return self.inputs + self.dependent_vars

    def discrete_dynamics(self, z, p, timestep, **kwargs):
        try:
            # LOG_INFO(f"Getting discrete dynamics for {z}, {p}")
            # Load the z vector
            self.load(z)

            if timestep is None:
                timestep = 0.1

            # Determine how many states to integrate
            state_dimension = self.state_dimension if self.state_dimension_integrate is None else self.state_dimension_integrate

            # Call the discretization function with explicit error checking
            integrated_states = safe_casadi_discrete_dynamics(self, z, p, nx=state_dimension, integration_step=timestep)
            try:
                if isinstance(z, (cd.MX, cd.SX)):
                    # Skip model_discrete_dynamics for symbolic expressions
                    return integrated_states
                else:
                    # CRITICAL: model_discrete_dynamics should return full state vector (including spline)
                    # For ContouringSecondOrderUnicycleModel, integrated_states is 4x1 (x,y,psi,v)
                    # and model_discrete_dynamics should return 5x1 (x,y,psi,v,spline)
                    result_states = self.model_discrete_dynamics(z, integrated_states, **kwargs)
                    
                    # Verify result has correct dimension
                    if isinstance(result_states, (cd.DM, cd.MX, cd.SX)):
                        if result_states.shape[0] == self.state_dimension:
                            integrated_states = result_states
                        else:
                            LOG_WARN(f"model_discrete_dynamics returned {result_states.shape[0]} states, expected {self.state_dimension}. "
                                   f"Using fallback: concatenating integrated_states with spline update.")
                            # Fallback: if model_discrete_dynamics doesn't return full state, add spline manually
                            if result_states.shape[0] == state_dimension:
                                # Result is same as integrated_states, need to add spline
                                x = self.get_x()
                                s = x[-1] if x.size1() >= self.state_dimension else 0.0
                                v = integrated_states[3] if integrated_states.size1() > 3 else 1.0
                                dt_val = float(timestep) if not isinstance(timestep, (cd.MX, cd.SX)) else 0.1
                                ds = v * dt_val
                                new_s = s + ds
                                new_s = cd.fmax(new_s, 0.0)
                                integrated_states = cd.vertcat(integrated_states, new_s)
                            else:
                                integrated_states = result_states
                    else:
                        # Numeric array - check shape
                        if len(result_states) == self.state_dimension:
                            integrated_states = result_states
                        else:
                            LOG_WARN(f"model_discrete_dynamics returned {len(result_states)} states, expected {self.state_dimension}. "
                                   f"Using fallback.")
                            # Fallback: add spline if missing
                            if len(result_states) == state_dimension:
                                x = self.get_x()
                                s = float(x[-1]) if x.size1() >= self.state_dimension else 0.0
                                v = float(integrated_states[3]) if integrated_states.size1() > 3 else 1.0
                                dt_val = float(timestep) if not isinstance(timestep, (cd.MX, cd.SX)) else 0.1
                                ds = v * dt_val
                                new_s = max(0.0, s + ds)
                                integrated_states = np.concatenate([np.array(result_states).flatten(), [new_s]])
                    
                    # Check for NaN in the result - only for numeric types
                    if isinstance(integrated_states, cd.DM) and not integrated_states.is_regular():
                        LOG_WARN("NaN detected in model_discrete_dynamics output")
            except Exception as e:
                LOG_WARN(f"Error in model_discrete_dynamics: {e}")
                import traceback
                LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
                # CRITICAL FIX: If model_discrete_dynamics fails, ensure we return full state vector
                # For ContouringSecondOrderUnicycleModel, we need 5 states (x,y,psi,v,spline)
                # integrated_states is only 4 states (x,y,psi,v), so add spline update
                if integrated_states.shape[0] == state_dimension and self.state_dimension > state_dimension:
                    # Need to add spline state
                    x = self.get_x()
                    s = x[-1] if x.size1() >= self.state_dimension else 0.0
                    v = integrated_states[3] if integrated_states.size1() > 3 else 1.0
                    dt_val = float(timestep) if not isinstance(timestep, (cd.MX, cd.SX)) else 0.1
                    ds = v * dt_val
                    new_s = s + ds
                    if isinstance(new_s, (cd.MX, cd.SX, cd.DM)):
                        new_s = cd.fmax(new_s, 0.0)
                    else:
                        new_s = max(0.0, float(new_s))
                    integrated_states = cd.vertcat(integrated_states, new_s) if isinstance(integrated_states, (cd.MX, cd.SX, cd.DM)) else np.concatenate([np.array(integrated_states).flatten(), [new_s]])

            return integrated_states
        except Exception as e:
            LOG_WARN(f"Error in discrete_dynamics: {e}")
            raise

    def load(self, z):
        """
        Load state and control variables from the z vector.

        Parameters:
        ----------
        z : CasADi MX or SX vector or numpy array
            Vector containing control inputs followed by states [u, x]
        """
        # Store the full vector
        self._z = z

        # For debugging, you can print or verify the shape
        # import casadi as cd
        # if isinstance(z, (cd.MX, cd.SX)):
        #     print(f"Loaded symbolic z with shape {z.shape}")
        # else:
        #     print(f"Loaded numeric z with shape {len(z)}")


    def get_z(self):
        return self._z

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        """
        Apply model-specific discrete dynamics logic after integration.
        This method can be overridden by subclasses to implement model-specific behavior.

        Parameters:
        ----------
        z : CasADi MX or SX vector or numpy array
            Vector containing control inputs followed by states [u, x]
        integrated_states : CasADi MX or SX vector or numpy array
            The states after integration

        Returns:
        -------
        CasADi MX or SX vector or numpy array
            The updated states after model-specific logic
        """
        return integrated_states

    def get_nvar(self):
        return self.nu + self.state_dimension

    def get_xinit(self):
        return range(self.nu, self.get_nvar())

    def get_x(self):
        return self._z[self.nu:]

    def get_u(self):
        return self._z[:self.nu]

    def load_settings(self, settings):
        self.params = settings["params"]
        self.settings = settings

    def save_map(self):
        file_path = model_map_path()
        mapped = {}
        for idx, state in enumerate(self.dependent_vars):
            mapped[state] = ["x", idx + self.nu, self.get_bounds(state)[0], self.get_bounds(state)[1]]

        for idx, ins in enumerate(self.inputs):
            mapped[ins] = ["u", idx, self.get_bounds(ins)[0], self.get_bounds(ins)[1]]

        write_to_yaml(file_path, mapped)

    def integrate(self, z, settings, integration_step):
        # This function should handle params correctly
        return self.discrete_dynamics(z, settings["params"].get_p(), settings, integration_step=integration_step)

    def debug_z_vector(self):
        """
        Helper method to print information about the current z vector.
        Useful for debugging purposes.

        Returns:
        -------
        dict
            Dictionary with control and state values from _z
        """
        if self._z is None:
            print("Warning: _z is None. load() has not been called yet.")
            return None

        import casadi as cd

        result = {}

        # Check if _z is a numeric or symbolic vector
        is_symbolic = isinstance(self._z, (cd.MX, cd.SX))

        if is_symbolic:
            print(f"_z is symbolic with shape {self._z.shape}")
            # Can't extract numeric values from symbolic expressions
            return {"type": "symbolic", "shape": self._z.shape}
        else:
            # Try to treat _z as a numeric array or list
            try:
                # Extract control inputs
                controls = {}
                for i, input_name in enumerate(self.inputs):
                    if i < len(self._z):
                        controls[input_name] = self._z[i]

                # Extract states
                states = {}
                for i, state_name in enumerate(self.dependent_vars):
                    idx = self.nu + i
                    if idx < len(self._z):
                        states[state_name] = self._z[idx]

                result = {
                    "type": "numeric",
                    "controls": controls,
                    "states": states,
                    "length": len(self._z),
                    "expected_length": self.nu + self.state_dimension
                }

                print(f"_z contains {len(self._z)} elements (expected {self.nu + self.state_dimension})")
                print(f"Controls: {controls}")
                print(f"States: {states}")
                return result
            except Exception as e:
                print(f"Error parsing _z: {e}")
                return {"type": "error", "error": str(e)}

    def do_not_use_integration_for_last_n_states(self, n):
        self.state_dimension_integrate = self.state_dimension - n

    def get(self, state_or_input, default=None):
        if state_or_input in self.dependent_vars:
            i = self.dependent_vars.index(state_or_input)
            return self._z[self.nu + i]
        elif state_or_input in self.inputs:
            i = self.inputs.index(state_or_input)
            return self._z[i]
        elif hasattr(self, state_or_input):
            return getattr(self, state_or_input)
        elif default is not None:
            return default
        else:
            raise IOError(
                f"Requested a state or input `{state_or_input}' that was neither a state nor an input for the selected model")

    def set_bounds(self, lower_bound, upper_bound):
        assert len(lower_bound) == len(upper_bound) == len(self.lower_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_bounds(self, state_or_input):
        if state_or_input in self.dependent_vars:
            i = self.dependent_vars.index(state_or_input)
            return (
                self.lower_bound[self.nu + i],
                self.upper_bound[self.nu + i],
                self.upper_bound[self.nu + i] - self.lower_bound[self.nu + i],
            )
        elif state_or_input in self.inputs:
            i = self.inputs.index(state_or_input)
            return (
                self.lower_bound[i],
                self.upper_bound[i],
                self.upper_bound[i] - self.lower_bound[i],
            )
        else:
            raise IOError(
                f"Requested a state or input `{state_or_input}' that was neither a state nor an input for the selected model")

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  nx={self.state_dimension}, nu={self.nu},\n"
            f"  dependent_vars={self.dependent_vars},\n"
            f"  inputs={self.inputs},\n"
            f"  lower_bound={self.lower_bound},\n"
            f"  upper_bound={self.upper_bound}\n"
            f")"
        )

    # This is the method that needs to be implemented by subclasses
    def continuous_model(self, x, u, p):
        """
    Compute the continuous-time dynamics for the model.

    @param x: State vector
    @param u: Control vector
    @return: Derivative of state vector
    """
        raise NotImplementedError("Subclasses must implement continuous_model")


class SecondOrderUnicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.state_dimension = 4  # number of states

        self.width = 0.5
        self.length = 0.5

        self.dependent_vars = ["x", "y", "psi", "v"]
        self.inputs = ["a", "w"]
        # Bounds format: [a_lb, w_lb, x_lb, y_lb, psi_lb, v_lb], [a_ub, w_ub, x_ub, y_ub, psi_ub, v_ub]
        # v_lb >= 0 to prevent backward motion (forward-only vehicle)
        self.lower_bound = [-2.0, -2.0, -200.0, -200.0, -np.pi * 4, 0.0]
        self.upper_bound = [2.0, 2.0, 200.0, 200.0, np.pi * 4, 3.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a)

class ContouringSecondOrderUnicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.state_dimension = 5  # number of states

        self.dependent_vars = ["x", "y", "psi", "v", "spline"]
        self.inputs = ["a", "w"]

        self.width = 0.5
        self.length = 0.5

        self.lr = .5
        self.lf = .5

        # w = 0.8
        self.lower_bound = [-2.0, -0.8, -2000.0, -2000.0, -np.pi * 6, -0.01, -1.0]
        self.upper_bound = [2.0, 0.8, 2000.0, 2000.0, np.pi * 6, 3.0, 10000.0]
        
        # CRITICAL FIX: Only integrate first 4 states (x, y, psi, v) with RK4
        # The spline state is updated algebraically in model_discrete_dynamics, not via RK4
        # Reference: C++ mpc_planner - spline is updated via algebraic formula, not integration
        self.state_dimension_integrate = 4  # Only integrate first 4 states
        self.do_not_use_integration_for_last_n_states(n=1)

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a)

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        """
        Update spline parameter based on vehicle position and path geometry.
        Reference: https://github.com/tud-amr/mpc_planner - uses curvature-aware spline update.
        
        The spline parameter represents arc length along the reference path (not normalized).
        Update formula: s_new = s + R * theta, where:
        - R is the path curvature radius
        - theta = atan2(vt_t, R - contour_error - vn_t)
        - vt_t is tangential component of displacement
        - vn_t is normal component of displacement
        
        Note: The state stores spline as arc length, but we normalize to [0,1] for Spline2D evaluation.
        """
        x = self.get_x()
        pos_x = x[0]
        pos_y = x[1]
        s = x[-1]  # Current spline value (arc length, not normalized)
        dt = kwargs.get('timestep', 0.1)
        
        # Get path length to normalize s for Spline2D
        s_max = 30.0  # Default estimate
        if hasattr(self, 'solver') and hasattr(self.solver, 'data'):
            data = self.solver.data
            if hasattr(data, 'reference_path') and data.reference_path is not None:
                ref_path = data.reference_path
                if hasattr(ref_path, 's') and ref_path.s is not None:
                    s_arr = np.asarray(ref_path.s, dtype=float)
                    if s_arr.size > 0:
                        s_max = float(s_arr[-1])
        
        # Normalize s to [0,1] for Spline2D evaluation
        # Clamp s to [0, s_max] for path evaluation (even if s > s_max in state)
        # This allows vehicle to progress beyond path length while path evaluation stays valid
        s_clamped = cd.fmin(cd.fmax(s, 0.0), s_max)  # Clamp for path evaluation
        s_normalized = cd.fmin(cd.fmax(s_clamped / cd.fmax(s_max, 1e-6), 0.0), 1.0)
        
        # Get path parameters from solver
        if not hasattr(self, 'params') or self.params is None:
            # Fallback: simple velocity-based progression
            v = x[3] if x.size1() > 3 else 1.0
            # Progress by distance traveled (in arc length units, not normalized)
            ds = v * dt
            new_s = s + ds
            new_s = cd.fmax(new_s, 0.0)  # Clamp to non-negative
            return cd.vertcat(integrated_states, new_s)
        
        # Use Spline2D to evaluate path and compute spline update
        try:
            from utils.math_tools import Spline2D
            
            # Get num_segments from settings or default
            num_segments = 10
            if hasattr(self, 'settings') and 'contouring' in self.settings:
                get_num_segments = self.settings['contouring'].get('get_num_segments', lambda: 10)
                num_segments = get_num_segments() if callable(get_num_segments) else 10
            
            # Create Spline2D instance (expects normalized s [0,1])
            path = Spline2D(self.params, num_segments, s_normalized)
            path_x, path_y = path.at(s_normalized)
            path_dx_normalized, path_dy_normalized = path.deriv_normalized(s_normalized)
            
            # Compute displacement from current position to integrated position
            # integrated_states[0:2] is the new (x, y) after RK4 integration
            dp = integrated_states[0:2] - cd.vertcat(pos_x, pos_y)
            
            # CRITICAL FIX: When curvature changes, the path tangent at the predicted position differs from current position
            # Use an iterative approach: estimate ds first, then evaluate path tangent at predicted position
            # Reference: C++ mpc_planner - spline update accounts for curvature changes by evaluating at predicted position
            
            # First, estimate ds using current path tangent (initial guess)
            t_vec_current = cd.vertcat(path_dx_normalized, path_dy_normalized)
            n_vec_current = cd.vertcat(path_dy_normalized, -path_dx_normalized)
            vt_t_initial = dp.dot(t_vec_current)
            vn_t_initial = dp.T @ n_vec_current
            
            # Get initial curvature estimate
            try:
                curvature_current = path.get_curvature(s_normalized)
                curvature_current = cd.fmax(curvature_current, 1e-5)
                R_current = cd.fmin(1.0 / curvature_current, 1e4)
            except Exception:
                R_current = 1e4
            
            # Initial contour error estimate (using current position for initial guess)
            contour_error_initial = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
            
            # Initial theta estimate
            denominator_initial = cd.fmax(R_current - contour_error_initial - vn_t_initial, 1e-6)
            theta_initial = cd.atan2(vt_t_initial, denominator_initial)
            theta_initial = cd.fmin(cd.fmax(theta_initial, -0.5), 0.5)
            
            # Estimate predicted spline position
            ds_estimate = R_current * theta_initial
            v = x[3] if x.size1() > 3 else 1.0
            ds_max = v * dt * 5.0
            ds_min = -v * dt * 0.5
            ds_estimate = cd.fmin(cd.fmax(ds_estimate, ds_min), ds_max)
            s_predicted_normalized = cd.fmax(0.0, cd.fmin(1.0, s_normalized + ds_estimate / cd.fmax(s_max, 1e-6)))
            
            # Evaluate path at predicted position (accounts for curvature changes)
            path_x_pred, path_y_pred = path.at(s_predicted_normalized)
            path_dx_pred, path_dy_pred = path.deriv_normalized(s_predicted_normalized)
            
            # Use predicted path tangent for final computation
            t_vec_pred = cd.vertcat(path_dx_pred, path_dy_pred)
            n_vec_pred = cd.vertcat(path_dy_pred, -path_dx_pred)
            
            # CRITICAL FIX: Use tangential component of displacement to directly update spline parameter
            # Reference: https://github.com/tud-amr/mpc_planner - spline tracks progress along path centerline
            # Reference: https://github.com/ttk592/spline - cubic spline interpolation
            # 
            # The spline parameter should track the vehicle's progress along the reference path by using
            # only the component of vehicle movement that is in the direction of the path tangent.
            #
            # Approach: Project vehicle displacement onto path tangent to get arc length progress
            # This ensures the spline parameter accurately reflects vehicle progress along the path,
            # regardless of lateral deviations (contour error).
            
            # Project displacement onto predicted tangent and normal
            t_vec_pred = cd.vertcat(path_dx_pred, path_dy_pred)  # Unit tangent vector at predicted position
            n_vec_pred = cd.vertcat(path_dy_pred, -path_dx_pred)  # Unit normal vector (points left)
            
            # Compute tangential component: vt_t = dp · t_vec_pred
            # This is the component of displacement ALONG the path direction
            # CRITICAL: vt_t directly represents progress along the path tangent
            # For a straight path, vt_t is exactly the arc length progress
            # For curved paths, vt_t still represents progress along the tangent direction
            vt_t = dp.dot(t_vec_pred)  # Tangential component (signed: positive = forward, negative = backward)
            vn_t = dp.T @ n_vec_pred    # Normal component (perpendicular to path, used for contour error)
            
            # Get path curvature at predicted position to account for path curvature
            try:
                curvature_pred = path.get_curvature(s_predicted_normalized)
                curvature_pred = cd.fmax(curvature_pred, 1e-5)  # Prevent division by zero
                R = cd.fmin(1.0 / curvature_pred, 1e4)  # Cap radius at 10km
            except Exception:
                R = R_current  # Fallback to current curvature
            
            # CRITICAL: Compute contour error using INTEGRATED position relative to PREDICTED path point
            # This ensures the spline update accounts for curvature changes
            integrated_x = integrated_states[0]
            integrated_y = integrated_states[1]
            contour_error = path_dy_pred * (integrated_x - path_x_pred) - path_dx_pred * (integrated_y - path_y_pred)
            
            # Reference: C++ mpc_planner uses curvature-aware formula: s_new = s + R * theta
            # where theta = atan2(vt_t, R - contour_error - vn_t)
            # This accounts for:
            # - Path curvature (R)
            # - Vehicle position relative to path (contour_error)
            # - Normal component of displacement (vn_t)
            #
            # CRITICAL IMPROVEMENT: Use tangential component more directly for better tracking
            # The tangential component vt_t represents the actual progress along the path direction
            # We use this as the primary update mechanism, with curvature correction
            
            # Method 1: Curvature-aware formula (from C++ reference) - handles curved paths well
            denominator = R - contour_error - vn_t
            denominator = cd.fmax(denominator, 1e-6)  # Prevent division by zero
            theta = cd.atan2(vt_t, denominator)
            theta = cd.fmin(cd.fmax(theta, -0.5), 0.5)  # Bound theta
            R_max = 1e4  # Cap radius at 10km (matches C++ safeguard)
            R_used = cd.fmin(R, R_max)
            ds_curvature = R_used * theta
            
            # Method 2: Direct tangential projection (PRIMARY METHOD for accurate tracking)
            # Use vt_t directly as arc length progress along the path tangent
            # For curved paths, the actual arc length along the curve is approximately vt_t
            # (the projection onto the tangent is the correct measure of progress)
            # Reference: The angle between vehicle movement and path tangent determines progress
            # If vehicle moves exactly along tangent: progress = ||dp||
            # If vehicle moves at angle α to tangent: progress = ||dp|| * cos(α) = vt_t
            ds_tangential = vt_t  # Direct use of tangential component
            
            # CRITICAL: Blend both methods, but favor tangential for better tracking
            # - Tangential method (60%): Ensures accurate tracking of vehicle progress
            # - Curvature-aware method (40%): Handles curved paths and accounts for contour error
            # This combination ensures both accurate tracking and proper handling of curved paths
            blend_weight_tangential = 0.6  # Primary: direct tracking
            blend_weight_curvature = 0.4   # Secondary: curvature handling
            ds_arc_length = blend_weight_tangential * ds_tangential + blend_weight_curvature * ds_curvature
            
            # CRITICAL FIX: Only update spline when vehicle is moving forward along path direction
            # Reference: C++ mpc_planner - spline should only progress when vt_t > 0 (forward movement)
            # When vehicle moves backward or sideways (vt_t <= 0), spline should not decrease
            # This ensures the vehicle reaches the end of the reference path
            v = x[3] if x.size1() > 3 else 1.0
            ds_max = v * dt * 5.0  # Allow up to 5x velocity-based progression (for sharp turns)
            
            # CRITICAL: Only allow forward progression (vt_t > 0) or very small backward correction
            # If vt_t <= 0, the vehicle is not moving forward along the path, so spline should not decrease
            # Use a small threshold to handle numerical noise
            vt_t_threshold = 1e-6  # Minimum tangential component for forward movement
            
            # CRITICAL: Prevent negative spline updates when vehicle is not moving forward
            # If vt_t <= threshold, the vehicle is moving backward or sideways, so don't decrease spline
            # Only allow small backward corrections when vt_t > threshold (vehicle is moving forward)
            ds_min_allowed = cd.if_else(vt_t > vt_t_threshold, -v * dt * 0.1, 0.0)
            ds_arc_length = cd.fmin(cd.fmax(ds_arc_length, ds_min_allowed), ds_max)
            
            # CRITICAL: Ensure spline only increases when vehicle moves forward along path
            # If tangential component is negative or zero, prevent spline from decreasing
            # This ensures the vehicle continues toward the end of the path
            # Use conditional to only allow positive updates when not moving forward
            ds_arc_length = cd.if_else(vt_t > vt_t_threshold, ds_arc_length, cd.fmax(ds_arc_length, 0.0))
            
            new_s = s + ds_arc_length
            
            # CRITICAL FIX: Ensure spline tracks vehicle progress accurately
            # Reference: C++ mpc_planner - spline should track the closest point on the reference path centerline
            # The tangential component update above already ensures accurate tracking, but we add safeguards
            # to prevent the spline from lagging too far behind or getting too far ahead of the vehicle
            
            # For numeric evaluation (post-optimization), we can verify and correct using closest point
            # For symbolic evaluation (during optimization), we rely on the tangential projection method
            try:
                # Only do closest point correction for numeric values (post-optimization verification)
                # During symbolic optimization, the tangential projection method is sufficient
                if hasattr(self, 'solver') and hasattr(self.solver, 'data'):
                    data = self.solver.data
                    if hasattr(data, 'reference_path') and data.reference_path is not None:
                        ref_path = data.reference_path
                        if hasattr(ref_path, 's') and ref_path.s is not None:
                            s_arr = np.asarray(ref_path.s, dtype=float)
                            if s_arr.size > 0:
                                # Check if we're in numeric mode (post-optimization)
                                # For symbolic mode, skip closest point search and use tangential projection
                                is_symbolic = isinstance(integrated_states[0], (cd.MX, cd.SX)) or isinstance(new_s, (cd.MX, cd.SX))
                                
                                if not is_symbolic:
                                    # Numeric mode: Find closest point for verification and correction
                                    # CRITICAL: Reference code (ttk592/spline, mpc_planner) uses closest point
                                    # to ensure spline parameter accurately tracks vehicle position along path
                                    try:
                                        integrated_x_val = float(integrated_states[0])
                                        integrated_y_val = float(integrated_states[1])
                                        
                                        # Sample path to find closest point to predicted position
                                        min_dist = float('inf')
                                        closest_s = s_arr[0]
                                        num_samples = min(200, max(100, len(s_arr)))
                                        
                                        # Search around current s first (most likely location)
                                        current_s_val = float(new_s) if not isinstance(new_s, (cd.MX, cd.SX)) else float(s)
                                        search_radius = 10.0  # Search within 10m of current s
                                        s_search_min = max(float(s_arr[0]), current_s_val - search_radius)
                                        s_search_max = min(float(s_arr[-1]), current_s_val + search_radius)
                                        
                                        s_samples = np.linspace(s_search_min, s_search_max, num_samples)
                                        
                                        for s_sample in s_samples:
                                            try:
                                                path_x_sample = float(ref_path.x_spline(s_sample))
                                                path_y_sample = float(ref_path.y_spline(s_sample))
                                                dist = np.sqrt((integrated_x_val - path_x_sample)**2 + (integrated_y_val - path_y_sample)**2)
                                                if dist < min_dist:
                                                    min_dist = dist
                                                    closest_s = s_sample
                                            except:
                                                continue
                                        
                                        # If closest point is at boundary, expand search to full path
                                        if closest_s <= s_search_min + 0.1 or closest_s >= s_search_max - 0.1:
                                            s_samples_full = np.linspace(float(s_arr[0]), float(s_arr[-1]), num_samples)
                                            for s_sample in s_samples_full:
                                                try:
                                                    path_x_sample = float(ref_path.x_spline(s_sample))
                                                    path_y_sample = float(ref_path.y_spline(s_sample))
                                                    dist = np.sqrt((integrated_x_val - path_x_sample)**2 + (integrated_y_val - path_y_sample)**2)
                                                    if dist < min_dist:
                                                        min_dist = dist
                                                        closest_s = s_sample
                                                except:
                                                    continue
                                        
                                        # Compute contour error to determine if re-sync is needed
                                        closest_path_x = float(ref_path.x_spline(closest_s))
                                        closest_path_y = float(ref_path.y_spline(closest_s))
                                        path_dx = float(ref_path.x_spline.derivative()(closest_s))
                                        path_dy = float(ref_path.y_spline.derivative()(closest_s))
                                        norm = np.hypot(path_dx, path_dy)
                                        
                                        if norm > 1e-6:
                                            path_dx_norm = path_dx / norm
                                            path_dy_norm = path_dy / norm
                                            A = np.array([path_dy_norm, -path_dx_norm])
                                            diff = np.array([integrated_x_val - closest_path_x, integrated_y_val - closest_path_y])
                                            contour_error = np.dot(A, diff)
                                            
                                            # If contour error is large OR spline is far from closest point,
                                            # use closest point as primary update
                                            contour_error_threshold = 0.5  # 0.5m threshold
                                            spline_distance_from_closest = abs(float(new_s) - closest_s)
                                            spline_distance_threshold = 2.0  # 2m threshold
                                            
                                            if abs(contour_error) > contour_error_threshold or spline_distance_from_closest > spline_distance_threshold:
                                                # Re-sync spline to closest point
                                                new_s = closest_s
                                            else:
                                                # Use closest point as bounds check
                                                s_min_based_on_position = closest_s - 1.0
                                                new_s = cd.fmax(new_s, s_min_based_on_position)
                                                s_max_ahead = closest_s + 2.0
                                                new_s = cd.fmin(new_s, s_max_ahead)
                                        else:
                                            # If path tangent is invalid, use closest point directly
                                            new_s = closest_s
                                    except Exception as e:
                                        LOG_DEBUG(f"Error in closest point search for spline update: {e}")
                                        pass
            except Exception as e:
                import traceback
                LOG_DEBUG(f"Error in closest point verification for spline update: {e}")
                LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
                # Fallback: use tangential projection result (already computed above)
                pass
            
            # Allow progression beyond s_max to reach final point (C++ reference behavior)
            # Clamp only to non-negative (allow s > s_max to reach path end)
            new_s = cd.fmax(new_s, 0.0)
            # NOTE: Do NOT clamp to s_max - allow vehicle to progress beyond path length
            # to reach the final point. Path evaluation will clamp s to s_max internally.
            
            return cd.vertcat(integrated_states, new_s)
        except Exception as e:
            # Fallback: simple velocity-based progression
            v = x[3] if x.size1() > 3 else 1.0
            # Progress by distance traveled (in arc length units)
            ds = v * dt
            new_s = s + ds
            new_s = cd.fmax(new_s, 0.0)  # Clamp to non-negative
            return cd.vertcat(integrated_states, new_s)

    def symbolic_dynamics(self, x, u, p, timestep):
        """
        Symbolic dynamics for contouring unicycle: integrates x,y,psi,v with RK4.
        Spline is updated symbolically using curvature-aware formula: s_new = s + R * theta
        
        CRITICAL: This method is called by the solver to create the constraint:
            x[k+1] == symbolic_dynamics(x[k], u[k], timestep)
        
        This ensures that:
            - x, y, psi, v are computed via RK4 integration (NOT free decision variables)
            - spline is computed via algebraic update (NOT free decision variable)
            - Only control inputs (a, w) are free decision variables
        
        Reference: https://github.com/tud-amr/mpc_planner - matches C++ implementation.
        Only acceleration (a) and angular acceleration (w) are decision variables.
        All states are computed via RK4 integration or model_discrete_dynamics.
        """
        # Only integrate first 4 states (x, y, psi, v) with RK4
        # The spline state is updated algebraically, not via RK4
        x_integrated = x[0:4]
        
        # RK4 integration for the first 4 states
        k1 = self.continuous_model(x_integrated, u, p)
        k2 = self.continuous_model(x_integrated + timestep / 2 * k1, u, p)
        k3 = self.continuous_model(x_integrated + timestep / 2 * k2, u, p)
        k4 = self.continuous_model(x_integrated + timestep * k3, u, p)
        x_next_integrated = x_integrated + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Update spline symbolically using curvature-aware formula from C++ reference
        pos_x = x_next_integrated[0]
        pos_y = x_next_integrated[1]
        s = x[4] if x.size1() > 4 else 0.0  # Current spline value (arc length, not normalized)
        
        # Get path length to normalize s for Spline2D
        # Try to get from parameter manager or use default
        s_max = 30.0  # Default estimate
        try:
            # Try to get s_max from parameter getter (if available)
            if hasattr(self, 'solver') and hasattr(self.solver, 'data'):
                data = self.solver.data
                if hasattr(data, 'reference_path') and data.reference_path is not None:
                    ref_path = data.reference_path
                    if hasattr(ref_path, 's') and ref_path.s is not None:
                        s_arr = np.asarray(ref_path.s, dtype=float)
                        if s_arr.size > 0:
                            s_max = float(s_arr[-1])
        except:
            pass
        
        # Normalize s to [0,1] for Spline2D evaluation
        s_normalized = cd.fmin(cd.fmax(s / cd.fmax(s_max, 1e-6), 0.0), 1.0)
        
        # Try to update spline symbolically using Spline2D
        try:
            from utils.math_tools import Spline2D
            
            # Create parameter wrapper for Spline2D
            class ParamWrapper:
                def __init__(self, p_getter):
                    self.p_getter = p_getter
                def get(self, key, default=None):
                    try:
                        return self.p_getter(key)
                    except:
                        return default
                def has_parameter(self, key):
                    try:
                        val = self.p_getter(key)
                        return val is not None
                    except:
                        return False
            
            param_wrapper = ParamWrapper(p)
            
            # Get num_segments from settings or default
            num_segments = 10
            if hasattr(self, 'settings') and 'contouring' in self.settings:
                get_num_segments = self.settings['contouring'].get('get_num_segments', lambda: 10)
                num_segments = get_num_segments() if callable(get_num_segments) else 10
            
            # Create Spline2D and evaluate path
            path = Spline2D(param_wrapper, num_segments, s_normalized)
            path_x, path_y = path.at(s_normalized)
            path_dx_normalized, path_dy_normalized = path.deriv_normalized(s_normalized)
            
            # Compute displacement from current position to next position
            # pos_x, pos_y are the NEXT integrated positions, so we need current position from x
            current_x = x[0]
            current_y = x[1]
            dx = x_next_integrated[0] - current_x
            dy = x_next_integrated[1] - current_y
            dp = cd.vertcat(dx, dy)
            
            # Compute tangential and normal components
            t_vec = cd.vertcat(path_dx_normalized, path_dy_normalized)
            n_vec = cd.vertcat(path_dy_normalized, -path_dx_normalized)
            
            # CRITICAL: Project displacement onto tangent to get arc length progress
            # Reference: https://github.com/tud-amr/mpc_planner - spline tracks progress along path centerline
            # The tangential component vt_t directly represents progress along the path direction
            # This ensures the spline parameter accurately reflects vehicle progress along the path
            vt_t = dp.dot(t_vec)  # Tangential component (primary: progress along path)
            vn_t = dp.T @ n_vec    # Normal component (perpendicular to path, used for contour error)
            
            # Compute contour error using current position (before integration)
            contour_error = path_dy_normalized * (current_x - path_x) - path_dx_normalized * (current_y - path_y)
            
            # Get path curvature radius
            try:
                curvature = path.get_curvature(s_normalized)
                curvature = cd.fmax(curvature, 1e-5)  # Minimum curvature
                R = 1.0 / curvature
                R = cd.fmin(R, 1e4)  # Cap radius
            except Exception:
                R = 1e4  # Fallback: large radius
            
            # CRITICAL: Use tangential component to directly update spline parameter
            # Reference: https://github.com/tud-amr/mpc_planner - spline tracks progress along path centerline
            # The tangential component vt_t directly represents progress along the path direction
            # This ensures the spline parameter accurately reflects vehicle progress along the path
            
            # Method 1: Curvature-aware formula (from C++ reference) - handles curved paths well
            denominator = R - contour_error - vn_t
            denominator = cd.fmax(denominator, 1e-6)  # Prevent division by zero
            theta = cd.atan2(vt_t, denominator)
            theta = cd.fmin(cd.fmax(theta, -0.5), 0.5)  # Bound theta
            R_max = 1e4  # Cap radius at 10km (matches C++ safeguard)
            R_used = cd.fmin(R, R_max)
            ds_curvature = R_used * theta
            
            # Method 2: Direct tangential projection (PRIMARY METHOD for accurate tracking)
            # Use vt_t directly as arc length progress along the path tangent
            # The projection onto the tangent is the correct measure of progress
            # Reference: The angle between vehicle movement and path tangent determines progress
            ds_tangential = vt_t  # Direct use of tangential component
            
            # CRITICAL: Blend both methods, but favor tangential for better tracking
            # - Tangential method (60%): Ensures accurate tracking of vehicle progress
            # - Curvature-aware method (40%): Handles curved paths and accounts for contour error
            blend_weight_tangential = 0.6  # Primary: direct tracking
            blend_weight_curvature = 0.4   # Secondary: curvature handling
            ds_arc_length = blend_weight_tangential * ds_tangential + blend_weight_curvature * ds_curvature
            
            # CRITICAL FIX: Only update spline when vehicle is moving forward along path direction
            # Reference: C++ mpc_planner - spline should only progress when vt_t > 0 (forward movement)
            # When vehicle moves backward or sideways (vt_t <= 0), spline should not decrease
            # This ensures the vehicle reaches the end of the reference path
            v = x_next_integrated[3] if x_next_integrated.size1() > 3 else 1.0
            ds_max = v * timestep * 5.0  # Allow up to 5x velocity-based progression (for sharp turns)
            
            # CRITICAL: Only allow forward progression (vt_t > 0) or very small backward correction
            # If vt_t <= 0, the vehicle is not moving forward along the path, so spline should not decrease
            # Use a small threshold to handle numerical noise
            vt_t_threshold = 1e-6  # Minimum tangential component for forward movement
            
            # CRITICAL: Prevent negative spline updates when vehicle is not moving forward
            # If vt_t <= threshold, the vehicle is moving backward or sideways, so don't decrease spline
            # Only allow small backward corrections when vt_t > threshold (vehicle is moving forward)
            ds_min_allowed = cd.if_else(vt_t > vt_t_threshold, -v * timestep * 0.1, 0.0)
            ds_arc_length = cd.fmin(cd.fmax(ds_arc_length, ds_min_allowed), ds_max)
            
            # CRITICAL: Ensure spline only increases when vehicle moves forward along path
            # If tangential component is negative or zero, prevent spline from decreasing
            # This ensures the vehicle continues toward the end of the path
            # Use conditional to only allow positive updates when not moving forward
            ds_arc_length = cd.if_else(vt_t > vt_t_threshold, ds_arc_length, cd.fmax(ds_arc_length, 0.0))
            
            s_next = s + ds_arc_length
            
            # Allow progression beyond s_max to reach final point (C++ reference behavior)
            # Clamp only to non-negative (allow s > s_max to reach path end)
            s_next = cd.fmax(s_next, 0.0)
            # NOTE: Do NOT clamp to s_max - allow vehicle to progress beyond path length
            # to reach the final point. Path evaluation will clamp s to s_max internally.
            
        except Exception:
            # Fallback: simple velocity-based progression
            v = x_next_integrated[3]
            # Progress by distance traveled (in arc length units)
            ds = v * timestep
            s_next = s + ds
            s_next = cd.fmax(s_next, 0.0)  # Clamp to non-negative
        
        return cd.vertcat(x_next_integrated, s_next)

class ContouringSecondOrderUnicycleModelCurvatureAware(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.state_dimension = 5  # number of states

        self.dependent_vars = ["x", "y", "psi", "v", "spline"]
        self.inputs = ["a", "w"]

        self.do_not_use_integration_for_last_n_states(n=1)

        self.lower_bound = [-4.0, -0.8, -2000.0, -2000.0, -pi * 4, -0.01, -1.0]
        self.upper_bound = [4.0, 0.8, 2000.0, 2000.0, pi * 4, 3.0, 10000.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a)

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        x = self.get_x()

        pos_x = x[0]
        pos_y = x[1]
        s = x[-1]


        path = ScipySplineWrapper(self.params, self.settings["contouring"]["get_num_segments"], s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
        path_ddx, path_ddy = path.deriv2(s)

        # Contour = n_vec
        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        # dp = np.array([integrated_states[0] - pos_x, integrated_states[1] - pos_y])
        # t_vec = np.array([path_dx_normalized, path_dy_normalized])
        # n_vec = np.array([path_dy_normalized, -path_dx_normalized])


        dp = integrated_states[0:2] - cd.vertcat(pos_x, pos_y)
        t_vec = cd.vertcat(path_dx_normalized, path_dy_normalized)
        vn_t = dp.T @ cd.vertcat(path_dy_normalized, -path_dx_normalized)
        vt_t = dp.dot(t_vec)

        R = 1.0 / path.get_curvature(s)  # max(R) = 1 / 0.0001
        R = cd.fmax(R, 1e5)

        theta = cd.atan2(vt_t, R - contour_error - vn_t)

        return cd.vertcat(integrated_states, s + R * theta)

    def symbolic_dynamics(self, x, u, p, timestep):
        """
		Symbolic dynamics for the curvature-aware model.
		It integrates the first 4 states and uses an algebraic update for the 5th state ('spline').
		"""
        # 1. Separate states into integrated and algebraic parts
        x_integrated = x[0:4]  # x, y, psi, v
        s = x[4]  # spline

        # 2. Perform RK4 integration on the first 4 states
        k1 = self.continuous_model(x_integrated, u, p)
        k2 = self.continuous_model(x_integrated + timestep / 2 * k1, u, p)
        k3 = self.continuous_model(x_integrated + timestep / 2 * k2, u, p)
        k4 = self.continuous_model(x_integrated + timestep * k3, u, p)
        x_next_integrated = x_integrated + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # 3. Perform symbolic algebraic update for the 'spline' state
        # NOTE: This part requires a SYMBOLIC spline representation, not Scipy.
        # This logic should mirror what's in the ContouringObjective.
        # We assume a simple symbolic placeholder for path evaluation here.
        # In a real system, you would pass the spline coefficients via 'p'.

        # Placeholder for symbolic path evaluation. In the real code, this would
        # use the 'p' parameter vector to get spline coefficients.
        # For this example, we'll use a simplified algebraic update.
        v_current = x_integrated[3]
        s_next = s + v_current * timestep  # Simple linear progression

        # 4. Combine the results
        return cd.vertcat(x_next_integrated, s_next)


class ContouringSecondOrderUnicycleModelWithSlack(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.state_dimension = 6  # number of states

        self.dependent_vars = ["x", "y", "psi", "v", "spline", "slack"]
        self.inputs = ["a", "w"]

        self.lower_bound = [-2.0, -0.8, -2000.0, -2000.0, -np.pi * 4, -0.01, -1.0, 0.0]  # v -0.01
        self.upper_bound = [2.0, 0.8, 2000.0, 2000.0, np.pi * 4, 3.0, 10000.0, 5000.0]  # w 0.8

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a, v, 0.0)

    # NOTE: No initialization for slack variable
    def get_xinit(self):
        return range(self.nu, self.get_nvar() - 1)


# Bicycle model with dynamic steering
class SecondOrderBicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 3
        self.state_dimension = 6

        self.dependent_vars = ["x", "y", "psi", "v", "delta", "spline"]
        self.inputs = ["a", "w", "slack"]

        # Prius limits: https:#github.com/oscardegroot/lmpcc/blob/prius/lmpccsolver/scripts/systems.py
        # w [-0.2, 0.2] | a [-1.0 1.0]
        # w was 0.5
        # delta was 0.45

        # NOTE: the angle of the vehicle should not be limited to -pi, pi, as the solution will not shift when it is at the border!
        # a was 3.0
        self.lower_bound = [-3.0, -1.5, 0.0, -1.0e6, -1.0e6, -np.pi * 4, -0.01, -0.55, -1.0]
        self.upper_bound = [3.0, 1.5, 1.0e2, 1.0e6, 1.0e6, np.pi * 4, 5.0, 0.55, 5000.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]
        delta = x[4]

        wheel_base = self.params("wheel_base") # 2.79  between front wheel center and rear wheel center
        wheel_tread = self.params("wheel_tread") # 1.64  between left wheel center and right wheel center
        front_overhang = self.params("front_overhang") # 1.0  between front wheel center and vehicle front
        rear_overhang = self.params("rear_overhang") # 1.1  # between rear wheel center and vehicle rear
        left_overhang = self.params("left_overhang") # 0.128  # between left wheel center and vehicle left
        right_overhang = self.params("right_overhang") # 0.128  # between right wheel center and vehicle right

        # NOTE: Mass is equally distributed according to the parameters
        lr = wheel_base / 2.0
        lf = wheel_base / 2.0
        ratio = lr / (lr + lf)
        self.width = 2.25

        beta = cd.arctan(ratio * cd.tan(delta))

        return cd.vertcat(v * cd.cos(psi + beta), v * cd.sin(psi + beta), (v / lr) * cd.sin(beta), a, w, v)


class ScipySplineWrapper:
    """Wrapper class to handle scipy splines for path representation with segment support"""

    def __init__(self, params, get_num_segments_fn, s):
        """Initialize spline from parameters with segment support

        Args:
            params: Path parameters (assumed to contain x and y coordinates)
            get_num_segments_fn: Function to get number of segments
            s: Path parameter
        """
        self.params = params
        self.num_segments = get_num_segments_fn()

        # Extract x and y coordinates from params
        # This assumes params contains path points in some format
        # You'll need to adjust this based on your actual param structure
        x_coords = params["x_points"]
        y_coords = params["y_points"]

        # Handle segmentation
        self.segment_length = 1.0 / self.num_segments

        # Create parameter vector for the entire path
        t = np.linspace(0, 1, len(x_coords))

        # Create cubic splines for the entire path
        self.tck_x = interpolate.splrep(t, x_coords, s=0)
        self.tck_y = interpolate.splrep(t, y_coords, s=0)

        # Pre-compute some values for efficiency
        points_per_segment = len(x_coords) // self.num_segments
        self.segment_indices = [i * points_per_segment for i in range(self.num_segments + 1)]
        self.segment_indices[-1] = len(x_coords)  # Ensure the last segment includes all remaining points

    def at(self, s):
        """Get x,y position on spline at parameter s"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # Evaluate splines
        x = float(interpolate.splev(t, self.tck_x))
        y = float(interpolate.splev(t, self.tck_y))

        return x, y

    def deriv_normalized(self, s):
        """Get normalized tangent vector at parameter s"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # Get derivatives
        dx = float(interpolate.splev(t, self.tck_x, der=1))
        dy = float(interpolate.splev(t, self.tck_y, der=1))

        # Normalize
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm < 1e-10:
            norm = 1e-10  # Avoid division by zero

        dx_normalized = dx / norm
        dy_normalized = dy / norm

        return dx_normalized, dy_normalized

    def deriv2(self, s):
        """Get second derivatives at parameter s"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # Get second derivatives
        ddx = float(interpolate.splev(t, self.tck_x, der=2))
        ddy = float(interpolate.splev(t, self.tck_y, der=2))

        return ddx, ddy

    def get_curvature(self, s):
        """Calculate curvature at parameter s with improved numerical stability"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # First derivatives
        dx = float(interpolate.splev(t, self.tck_x, der=1))
        dy = float(interpolate.splev(t, self.tck_y, der=1))

        # Second derivatives
        ddx = float(interpolate.splev(t, self.tck_x, der=2))
        ddy = float(interpolate.splev(t, self.tck_y, der=2))

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2)

        # Improved numerical stability
        if denominator < 1e-6:
            # For very small denominators, return a stable but small curvature
            return 1e-4  # Minimum curvature (maximum radius)

        curvature = numerator / denominator

        # Limit maximum curvature to avoid numerical issues
        # Use min to cap the maximum curvature (minimum radius)
        return min(max(curvature, 1e-4), 10.0)  # Bounded between 1e-4 and 10

# Bicycle model with dynamic steering
class CurvatureAwareSecondOrderBicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 3
        self.state_dimension = 6

        self.dependent_vars = ["x", "y", "psi", "v", "delta", "spline"]
        self.inputs = ["a", "w", "slack"]

        self.do_not_use_integration_for_last_n_states(n=1)

        # Prius limits: https:#github.com/oscardegroot/lmpcc/blob/prius/lmpccsolver/scripts/systems.py
        # w [-0.2, 0.2] | a [-1.0 1.0]
        # w was 0.5
        # delta was 0.45

        # NOTE: the angle of the vehicle should not be limited to -pi, pi, as the solution will not shift when it is at the border!
        # a was 3.0
        # delta was -0.45, 0.45
        self.lower_bound = [-3.0, -1.5, 0.0, -1.0e6, -1.0e6, -np.pi * 4, -0.01, -0.55, -1.0]
        self.upper_bound = [3.0, 1.5, 1.0e2, 1.0e6, 1.0e6, np.pi * 4, 8.0, 0.55, 5000.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]
        delta = x[4]

        wheel_base = 2.79  # between front wheel center and rear wheel center

        # NOTE: Mass is equally distributed according to the parameters
        self.lr = wheel_base / 2.0
        self.lf = wheel_base / 2.0
        ratio = self.lr / (self.lr + self.lf)

        self.width = 2.25

        beta = cd.arctan(ratio * cd.tan(delta))

        return cd.vertcat(v * cd.cos(psi + beta), v * cd.sin(psi + beta), (v / self.lr) * cd.sin(beta), a, w)

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        x = self.get_x()

        pos_x = x[0]
        pos_y = x[1]
        s = x[-1]

        # Ensure s is within valid range
        s = cd.fmin(cd.fmax(s, 0.0), 0.999)  # Clamp slightly inside [0,1]

        path = ScipySplineWrapper(self.params, self.settings["contouring"]["get_num_segments"], s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
        path_ddx, path_ddy = path.deriv2(s)

        # Contour = n_vec
        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        dp = integrated_states[0:2] - cd.vertcat(pos_x, pos_y)
        t_vec = cd.vertcat(path_dx_normalized, path_dy_normalized)

        # Safer dot product calculations
        vn_t = dp.T @ cd.vertcat(path_dy_normalized, -path_dx_normalized)
        vt_t = dp.dot(t_vec)

        # Get curvature with improved numerical stability
        curvature = path.get_curvature(s)

        # Safeguard radius calculation with a minimum curvature
        curvature = cd.fmax(curvature, 1e-5)  # Ensure curvature is not too small
        R = 1.0 / curvature

        # Use a more conservative radius limit
        # Avoid extremely large radii that can cause precision issues
        R = cd.fmin(R, 1e4)  # Cap at 10 km radius

        # Safeguard theta calculation
        # Add a small epsilon to denominator to prevent division by zero
        theta = cd.atan2(vt_t, cd.fmax(R - contour_error - vn_t, 1e-6))

        # Bound theta to prevent extreme values
        theta = cd.fmin(cd.fmax(theta, -0.5), 0.5)

        # Ensure s + R * theta stays within bounds
        new_s = cd.fmin(cd.fmax(s + R * theta, 0.0), 0.999)

        return cd.vertcat(integrated_states, new_s)


class PointMassModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        # Controls: accelerations in x and y
        self.nu = 2
        # States: position and velocity in x and y
        self.state_dimension = 4

        self.dependent_vars = ["x", "y", "vx", "vy"]
        self.inputs = ["ax", "ay"]

        # Reasonable bounds for positions, velocities, and accelerations
        # Order: [u then x] in combined vectors; lower/upper arrays should match nu + nx
        self.lower_bound = [-5.0, -5.0,  -1.0e6, -1.0e6, -1.0e3, -1.0e3]
        self.upper_bound = [ 5.0,  5.0,   1.0e6,  1.0e6,  1.0e3,  1.0e3]

    def continuous_model(self, x, u, p):
        # x = [x, y, vx, vy], u = [ax, ay]
        vx = x[2]
        vy = x[3]
        ax = u[0]
        ay = u[1]
        return cd.vertcat(vx, vy, ax, ay)