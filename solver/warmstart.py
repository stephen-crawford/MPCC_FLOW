"""
Warmstart manager for CasADi solver.

This module provides utilities for managing warmstart values in the MPC solver.
Warmstarting is critical for solver performance as it provides a good initial
guess for the optimization problem based on previous solutions.
"""

import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO

if TYPE_CHECKING:
    from solver.casadi_solver import CasADiSolver
    from planning.types import State


class WarmstartManager:
    """Manager for warmstart values in the MPC solver.

    This class provides utilities for:
    - Initializing warmstart values from state
    - Shifting warmstart forward after each MPC step
    - Ensuring dynamics consistency in warmstart values
    - Updating warmstart from solutions
    """

    def __init__(self, solver: 'CasADiSolver' = None):
        """Initialize with optional solver reference.

        Args:
            solver: The CasADiSolver instance to manage warmstart for.
        """
        self._solver = solver
        self._warmstart_values: Dict[str, np.ndarray] = {}
        self._initialized = False
        self._path_length: Optional[float] = None  # Set by solver when reference path is available
    
    @property
    def path_length(self) -> Optional[float]:
        """Get the path length for spline clamping."""
        return self._path_length
    
    @path_length.setter
    def path_length(self, val: Optional[float]):
        """Set the path length for spline clamping."""
        self._path_length = val

    @property
    def values(self) -> Dict[str, np.ndarray]:
        """Get the warmstart values dictionary."""
        return self._warmstart_values

    @values.setter
    def values(self, val: Dict[str, np.ndarray]):
        """Set the warmstart values dictionary."""
        self._warmstart_values = val

    @property
    def is_initialized(self) -> bool:
        """Check if warmstart has been initialized."""
        return self._initialized and bool(self._warmstart_values)

    def initialize(self, state: 'State', horizon: int, state_vars: List[str],
                   input_vars: List[str], timestep: float = 0.1):
        """Initialize warmstart values from state.

        Args:
            state: Current state.
            horizon: Planning horizon.
            state_vars: List of state variable names.
            input_vars: List of input variable names.
            timestep: Timestep for dynamics.
        """
        LOG_DEBUG("Initializing warmstart values")

        # Initialize state variables (horizon + 1 values)
        for var_name in state_vars:
            self._warmstart_values[var_name] = np.zeros(horizon + 1)
            if state is not None and state.has(var_name):
                val = state.get(var_name)
                if val is not None:
                    self._warmstart_values[var_name][:] = float(val)

        # Initialize input variables (horizon values)
        for var_name in input_vars:
            self._warmstart_values[var_name] = np.zeros(horizon)

        self._initialized = True

    def shift_forward(self, timestep: float = 0.1):
        """Shift warmstart solution forward by one timestep.

        This is called after a successful MPC solve to prepare the warmstart
        for the next iteration.

        Args:
            timestep: Timestep for extrapolation.
        """
        if not self._warmstart_values:
            return

        LOG_DEBUG("Shifting warmstart forward by one step")

        for var_name, values in self._warmstart_values.items():
            if len(values) <= 1:
                continue

            if var_name == 'spline':
                # Special handling for spline - ensure monotonicity and clamp to path length
                old_spline = values.copy()
                values[:-1] = old_spline[1:]

                # Extrapolate last value based on velocity if available
                if 'v' in self._warmstart_values and len(self._warmstart_values['v']) > 0:
                    v_last = float(self._warmstart_values['v'][-1])
                    s_last = old_spline[-1] + v_last * timestep
                    values[-1] = s_last
                else:
                    values[-1] = old_spline[-1]

                # CRITICAL FIX: Clamp spline values to path length to prevent infeasible dynamics
                # When vehicle approaches path end, extrapolated spline can exceed path_length,
                # causing dynamics constraints to become infeasible
                if self._path_length is not None and self._path_length > 0:
                    # Clamp all spline values to path length
                    # When at path end, all future spline values should be at path_length
                    for k in range(len(values)):
                        if values[k] > self._path_length:
                            values[k] = self._path_length
                    LOG_DEBUG(f"Clamped spline warmstart to path_length={self._path_length:.2f}")

                # Ensure monotonicity (but allow equal values at path end)
                for k in range(1, len(values)):
                    if values[k] < values[k - 1]:
                        values[k] = values[k - 1]
            else:
                # Standard shift: copy forward and extrapolate last
                values[:-1] = values[1:]
                values[-1] = values[-2]

    def update_current_state(self, state: 'State', state_vars: List[str]):
        """Update stage 0 of warmstart to match current vehicle state.

        Args:
            state: Current state to update from.
            state_vars: List of state variable names to update.
        """
        if not self._warmstart_values or state is None:
            return

        LOG_DEBUG("Updating warmstart stage 0 to match current state")

        for var_name in state_vars:
            if var_name in self._warmstart_values and state.has(var_name):
                val = state.get(var_name)
                if val is not None:
                    self._warmstart_values[var_name][0] = float(val)

    def ensure_dynamics_consistency(self, timestep: float = 0.1):
        """Ensure warmstart values satisfy dynamics constraints.

        Specifically ensures psi[k+1] = psi[k] + w[k]*dt and updates
        position accordingly.

        Args:
            timestep: Timestep for dynamics.
        """
        if not self._warmstart_values:
            return

        if 'psi' not in self._warmstart_values or 'w' not in self._warmstart_values:
            return

        psi_profile = self._warmstart_values['psi']
        w_profile = self._warmstart_values['w']

        # Ensure consistent sizes
        n_psi = len(psi_profile)
        n_w = len(w_profile)

        if n_psi == 0 or n_w == 0:
            return

        changed = False
        for k in range(min(n_psi - 1, n_w)):
            expected_psi = psi_profile[k] + w_profile[k] * timestep
            actual_psi = psi_profile[k + 1]

            if abs(expected_psi - actual_psi) > 1e-3:
                psi_profile[k + 1] = expected_psi
                changed = True

        if changed:
            LOG_INFO("Fixed warmstart dynamics consistency")
            self._update_positions_from_psi(psi_profile, timestep)

    def _update_positions_from_psi(self, psi_profile: np.ndarray, timestep: float):
        """Update x, y positions based on psi and v profiles."""
        if 'x' not in self._warmstart_values or 'y' not in self._warmstart_values:
            return
        if 'v' not in self._warmstart_values:
            return

        x_pos = self._warmstart_values['x']
        y_pos = self._warmstart_values['y']
        v_profile = self._warmstart_values['v']

        n = min(len(x_pos) - 1, len(y_pos) - 1, len(v_profile), len(psi_profile) - 1)

        for k in range(n):
            v_k = float(v_profile[k])
            psi_k = float(psi_profile[k])
            x_pos[k + 1] = x_pos[k] + v_k * np.cos(psi_k) * timestep
            y_pos[k + 1] = y_pos[k] + v_k * np.sin(psi_k) * timestep

        self._warmstart_values['x'] = x_pos
        self._warmstart_values['y'] = y_pos
        self._warmstart_values['psi'] = psi_profile

    def update_from_solution(self, solution, var_dict: Dict[str, Any]):
        """Update warmstart values from a CasADi solution.

        Args:
            solution: CasADi solution object.
            var_dict: Dictionary mapping variable names to CasADi variables.
        """
        for var_name in var_dict:
            try:
                self._warmstart_values[var_name] = np.array(solution.value(var_dict[var_name]))
            except Exception as e:
                LOG_WARN(f"Could not update warmstart for {var_name}: {e}")

    def get_value(self, var_name: str, stage: int) -> Optional[float]:
        """Get warmstart value for a variable at a specific stage.

        Args:
            var_name: Variable name.
            stage: Stage index.

        Returns:
            The value or None if not available.
        """
        if var_name not in self._warmstart_values:
            return None
        vals = self._warmstart_values[var_name]
        if stage >= len(vals):
            return None
        return float(vals[stage])

    def set_value(self, var_name: str, stage: int, value: float):
        """Set warmstart value for a variable at a specific stage.

        Args:
            var_name: Variable name.
            stage: Stage index.
            value: Value to set.
        """
        if var_name not in self._warmstart_values:
            return
        vals = self._warmstart_values[var_name]
        if stage < len(vals):
            vals[stage] = value

    def get_position(self, stage: int) -> Optional[tuple]:
        """Get warmstart position (x, y) at a specific stage.

        Args:
            stage: Stage index.

        Returns:
            Tuple of (x, y) or None if not available.
        """
        x = self.get_value('x', stage)
        y = self.get_value('y', stage)
        if x is not None and y is not None:
            return (x, y)
        return None

    def clear(self):
        """Clear all warmstart values."""
        self._warmstart_values.clear()
        self._initialized = False

    def save_state(self) -> Dict[str, np.ndarray]:
        """Save current warmstart state for later restoration.

        Returns:
            Dictionary with saved warmstart values.
        """
        saved = {}
        for k, v in self._warmstart_values.items():
            saved[k] = np.array(v) if isinstance(v, np.ndarray) else v
        return saved

    def restore_state(self, saved: Dict[str, np.ndarray]):
        """Restore warmstart state from saved values.

        Args:
            saved: Previously saved warmstart values.
        """
        self._warmstart_values = saved
        self._initialized = bool(saved)

    def __contains__(self, var_name: str) -> bool:
        """Check if variable is in warmstart values."""
        return var_name in self._warmstart_values

    def __getitem__(self, var_name: str) -> np.ndarray:
        """Get warmstart values for a variable."""
        return self._warmstart_values[var_name]

    def __setitem__(self, var_name: str, values: np.ndarray):
        """Set warmstart values for a variable."""
        self._warmstart_values[var_name] = values
