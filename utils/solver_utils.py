"""
Solver utility classes for safely accessing solver attributes.

This module provides helper classes to reduce boilerplate code when
accessing solver attributes throughout the codebase.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from solver.casadi_solver import CasADiSolver


class SolverAccessor:
    """Helper for safely accessing solver attributes.

    This class wraps a solver instance and provides safe property access
    with sensible defaults, eliminating repetitive hasattr/getattr chains.

    Usage:
        accessor = SolverAccessor(self.solver)
        horizon = accessor.horizon  # Returns horizon or default (10)
        timestep = accessor.timestep  # Returns timestep or default (0.1)

        # Or use the convenience function:
        from utils.solver_utils import get_solver_accessor
        accessor = get_solver_accessor(module)
    """

    def __init__(self, solver: Optional['CasADiSolver'] = None):
        """Initialize with optional solver reference.

        Args:
            solver: The solver instance to wrap. Can be None.
        """
        self._solver = solver

    @property
    def solver(self) -> Optional['CasADiSolver']:
        """Get the underlying solver instance."""
        return self._solver

    @property
    def is_valid(self) -> bool:
        """Check if solver reference is valid."""
        return self._solver is not None

    @property
    def horizon(self) -> int:
        """Get solver horizon with default fallback.

        Returns:
            The solver's horizon value, or 10 if not available.
        """
        if self._solver is not None:
            val = getattr(self._solver, 'horizon', None)
            if val is not None:
                return int(val)
        return 10

    @property
    def timestep(self) -> float:
        """Get solver timestep with default fallback.

        Returns:
            The solver's timestep value, or 0.1 if not available.
        """
        if self._solver is not None:
            # Check both 'timestep' and 'dt' attributes
            val = getattr(self._solver, 'timestep', None)
            if val is not None:
                return float(val)
            val = getattr(self._solver, 'dt', None)
            if val is not None:
                return float(val)
        return 0.1

    @property
    def parameter_manager(self) -> Optional[Any]:
        """Get solver parameter manager.

        Returns:
            The solver's parameter_manager, or None if not available.
        """
        if self._solver is not None:
            return getattr(self._solver, 'parameter_manager', None)
        return None

    @property
    def has_parameter_manager(self) -> bool:
        """Check if solver has a valid parameter manager."""
        return self.parameter_manager is not None

    @property
    def module_manager(self) -> Optional[Any]:
        """Get solver module manager.

        Returns:
            The solver's module_manager, or None if not available.
        """
        if self._solver is not None:
            return getattr(self._solver, 'module_manager', None)
        return None

    @property
    def has_module_manager(self) -> bool:
        """Check if solver has a valid module manager."""
        return self.module_manager is not None

    @property
    def warmstart_values(self) -> Dict[str, List[float]]:
        """Get solver warmstart values.

        Returns:
            The solver's warmstart_values dict, or empty dict if not available.
        """
        if self._solver is not None:
            vals = getattr(self._solver, 'warmstart_values', None)
            if vals is not None:
                return vals
        return {}

    @property
    def has_warmstart(self) -> bool:
        """Check if solver has valid warmstart values."""
        return bool(self.warmstart_values)

    @property
    def data(self) -> Optional[Any]:
        """Get solver data object.

        Returns:
            The solver's data object, or None if not available.
        """
        if self._solver is not None:
            return getattr(self._solver, 'data', None)
        return None

    @property
    def has_data(self) -> bool:
        """Check if solver has valid data."""
        return self.data is not None

    @property
    def var_dict(self) -> Dict[str, Any]:
        """Get solver variable dictionary.

        Returns:
            The solver's var_dict, or empty dict if not available.
        """
        if self._solver is not None:
            vals = getattr(self._solver, 'var_dict', None)
            if vals is not None:
                return vals
        return {}

    @property
    def has_var_dict(self) -> bool:
        """Check if solver has valid variable dictionary."""
        return bool(self.var_dict)

    @property
    def opti(self) -> Optional[Any]:
        """Get solver opti instance.

        Returns:
            The solver's opti (CasADi Opti) instance, or None if not available.
        """
        if self._solver is not None:
            return getattr(self._solver, 'opti', None)
        return None

    @property
    def is_initialized(self) -> bool:
        """Check if solver is initialized (has valid opti)."""
        return self.opti is not None

    @property
    def dynamics_model(self) -> Optional[Any]:
        """Get solver dynamics model.

        Returns:
            The solver's dynamics model, or None if not available.
        """
        if self._solver is not None:
            # Try _get_dynamics_model method first
            if hasattr(self._solver, '_get_dynamics_model'):
                return self._solver._get_dynamics_model()
            # Fall back to direct attribute
            return getattr(self._solver, 'dynamics_model', None)
        return None

    @property
    def has_dynamics_model(self) -> bool:
        """Check if solver has a dynamics model."""
        return self.dynamics_model is not None

    def get_warmstart_position(self, stage_idx: int) -> Optional[tuple]:
        """Get warmstart position (x, y) at a specific stage.

        Args:
            stage_idx: The stage index.

        Returns:
            Tuple of (x, y) or None if not available.
        """
        ws = self.warmstart_values
        if 'x' in ws and 'y' in ws:
            x_vals = ws['x']
            y_vals = ws['y']
            if stage_idx < len(x_vals) and stage_idx < len(y_vals):
                return (float(x_vals[stage_idx]), float(y_vals[stage_idx]))
        return None

    def get_warmstart_value(self, var_name: str, stage_idx: int) -> Optional[float]:
        """Get warmstart value for a variable at a specific stage.

        Args:
            var_name: The variable name (e.g., 'x', 'y', 'psi', 'v').
            stage_idx: The stage index.

        Returns:
            The value or None if not available.
        """
        ws = self.warmstart_values
        if var_name in ws:
            vals = ws[var_name]
            if stage_idx < len(vals):
                return float(vals[stage_idx])
        return None


def get_solver_accessor(module_or_solver) -> SolverAccessor:
    """Create a SolverAccessor from a module or solver.

    Args:
        module_or_solver: Either a module with a .solver attribute,
                         or a solver instance directly.

    Returns:
        A SolverAccessor wrapping the solver.
    """
    if hasattr(module_or_solver, 'solver'):
        return SolverAccessor(module_or_solver.solver)
    return SolverAccessor(module_or_solver)


def get_horizon(solver, default: int = 10) -> int:
    """Get solver horizon with default fallback.

    Convenience function for quick access without creating an accessor.

    Args:
        solver: The solver instance.
        default: Default value if horizon not available.

    Returns:
        The horizon value.
    """
    if solver is not None:
        val = getattr(solver, 'horizon', None)
        if val is not None:
            return int(val)
    return default


def get_timestep(solver, default: float = 0.1) -> float:
    """Get solver timestep with default fallback.

    Convenience function for quick access without creating an accessor.

    Args:
        solver: The solver instance.
        default: Default value if timestep not available.

    Returns:
        The timestep value.
    """
    if solver is not None:
        val = getattr(solver, 'timestep', None)
        if val is not None:
            return float(val)
        val = getattr(solver, 'dt', None)
        if val is not None:
            return float(val)
    return default


def has_warmstart(solver) -> bool:
    """Check if solver has valid warmstart values.

    Args:
        solver: The solver instance.

    Returns:
        True if solver has warmstart values, False otherwise.
    """
    if solver is None:
        return False
    ws = getattr(solver, 'warmstart_values', None)
    return ws is not None and bool(ws)
