import copy
from abc import ABC, abstractmethod

from planning.types import Data
from planning.modules_manager import ModuleManager
from planning.parameter_manager import ParameterManager
from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import LOG_DEBUG


class BaseSolver(ABC):
    def __init__(self, config):
        self.config = config
        self.module_manager = ModuleManager()
        self.parameter_manager = ParameterManager()
        self.data = None
        self.planner = None

    def initialize_solver(self, data):
        pass

    def initialize_rollout(self, state, data, shift_forward=True):
        pass

    def _initialize_base_rollout(self, state, data):
        pass

    def set_planner(self, planner):
        """Attach a planner so the solver can delegate symbolic/numeric dynamics construction."""
        self.planner = planner

    def _compute_next_state(self, dynamics_model, x_k, u_k, timestep, data=None, symbolic=True):
        """
        Compute x_{k+1} from (x_k, u_k) using either symbolic or numeric integration.
        - If symbolic: prefer planner.get_symbolic_dynamics; fallback to model's symbolic_dynamics; else Euler on continuous_model.
        - If numeric: prefer planner.get_numeric_dynamics; fallback to Euler on continuous_model (assuming numeric arrays).
        """
        # Parameter getter callable for models expecting callable p
        def _param_getter(key):
            try:
                if data is not None and hasattr(data, 'parameters'):
                    return data.parameters.get(key)
            except Exception:
                pass
            return 0.0

        if symbolic:
            # Planner hook
            if self.planner is not None and hasattr(self.planner, 'get_symbolic_dynamics'):
                try:
                    return self.planner.get_symbolic_dynamics(dynamics_model, x_k, u_k, timestep, data)
                except Exception:
                    pass
            # Model-provided symbolic
            if hasattr(dynamics_model, 'symbolic_dynamics') and callable(getattr(dynamics_model, 'symbolic_dynamics')):
                try:
                    return dynamics_model.symbolic_dynamics(x_k, u_k, _param_getter, timestep)
                except Exception:
                    pass
            # Fallback Euler symbolic
            f_k = dynamics_model.continuous_model(x_k, u_k, _param_getter)
            try:
                return x_k + timestep * f_k
            except Exception:
                return x_k
        else:
            # Numeric path
            if self.planner is not None and hasattr(self.planner, 'get_numeric_dynamics'):
                try:
                    return self.planner.get_numeric_dynamics(dynamics_model, x_k, u_k, timestep, data)
                except Exception:
                    pass
            # Fallback simple Euler numeric; assumes x_k,u_k are numpy-like
            try:
                f_k = dynamics_model.continuous_model(x_k, u_k, _param_getter)
                return x_k + timestep * f_k
            except Exception:
                return x_k

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_output(self, k, var_name):
        pass

    @abstractmethod
    def explain_exit_flag(self, code):
        pass
    
    def define_parameters(self):
        """Define parameters for all modules."""
        for module in self.module_manager.get_modules():
            if hasattr(module, "define_parameters"):
                module.define_parameters(self.parameter_manager)

    # Standardized planner ↔ solver interface: fetch objectives/constraints via ModuleManager
    def get_objective_cost(self, state, stage_idx):
        """Return a list of objective term dicts for the given stage.
        
        CRITICAL: This method receives symbolic_state for future stages (stage_idx > 0),
        ensuring objectives are computed symbolically using predicted states.
        This matches the reference codebase pattern.
        
        Reference: https://github.com/tud-amr/mpc_planner - objectives are evaluated symbolically.
        
        Args:
            state: Symbolic state for this stage (predicted state for stage_idx > 0)
            stage_idx: Stage index (0 = current, >0 = future)
        """
        # Use the provided state directly (it's already symbolic for future stages)
        # This matches the reference codebase where objectives are evaluated symbolically
        return self.module_manager.get_objectives(state, self.data, stage_idx) or []

    def get_constraints(self, stage_idx, symbolic_state=None):
        """Return a list of (constraint, lb, ub) tuples for the given stage.
        
        CRITICAL: For ALL stages (including stage 0), constraints MUST be computed using
        symbolic_state (CasADi variables). This matches the reference codebase pattern
        where constraints are evaluated symbolically.
        
        Args:
            stage_idx: Stage index (0 = current, >0 = future)
            symbolic_state: Symbolic state for this stage (required for all stages)
        
        Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
        """
        # For ALL stages, use symbolic_state (CasADi variables)
        # This ensures constraints are properly integrated into the optimization problem
        if symbolic_state is None:
            from utils.utils import LOG_WARN
            LOG_WARN(f"get_constraints: stage_idx={stage_idx} but symbolic_state is None - constraints may be incorrect")
            # Fallback to current state (not ideal, but better than crashing)
            if self.data is not None and hasattr(self.data, 'state') and self.data.state is not None:
                state = self.data.state
            else:
                state = None
        else:
            state = symbolic_state
        
        # Ensure modules have solver reference before getting constraints
        for module in self.module_manager.get_modules():
            if not hasattr(module, 'solver') or module.solver is None:
                module.solver = self
        
        if hasattr(self.module_manager, 'get_constraints_with_bounds'):
            return self.module_manager.get_constraints_with_bounds(state, self.data, stage_idx)
        cons = self.module_manager.get_constraints(state, self.data, stage_idx) or []
        lbs = self.module_manager.get_lower_bounds(state, self.data, stage_idx) or []
        ubs = self.module_manager.get_upper_bounds(state, self.data, stage_idx) or []
        n = max(len(cons), len(lbs), len(ubs))
        result = []
        for i in range(n):
            c = cons[i] if i < len(cons) else None
            lb = lbs[i] if i < len(lbs) else None
            ub = ubs[i] if i < len(ubs) else None
            if c is not None:
                result.append((c, lb, ub))
        return result