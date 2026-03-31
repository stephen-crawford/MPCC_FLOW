from abc import abstractmethod

from planning.modules_manager import Module
from utils.const import OBJECTIVE
from utils.utils import read_config_file, LOG_DEBUG, LOG_INFO

class BaseObjective(Module):
    def __init__(self):
        super().__init__()
        self.name = "_objective".join(self.__class__.__name__.lower().split('objective'))
        self.module_type = OBJECTIVE
        self.controller = (self.module_type, None, self.name)
        self.solver = None
        # Config is already loaded in Module.__init__ as self.settings and self.config
        LOG_INFO(f"Initializing {self.name.title()} Objective")
        LOG_DEBUG(f"BaseObjective.__init__: name={self.name}, module_type={self.module_type}")

    def get_objective(self, state, data, stage_idx):
        """
        Get objective cost terms for this stage.
        
        CRITICAL: This method MUST return symbolic CasADi expressions (MX or SX) for MPC rollouts.
        The state parameter contains CasADi variables from the solver's var_dict:
            - State variables (x, y, psi, v, spline) - constrained by RK4/model_discrete_dynamics
            - Control variables (a, w) - free decision variables
        
        The returned expressions must be symbolic and depend on these variables.
        Numeric calculations are NOT allowed - all computations must be symbolic.
        
        Priority:
        1. get_stage_cost_symbolic() - preferred method returning symbolic expressions
        2. get_value() - legacy method (must also return symbolic expressions)
        
        Reference: https://github.com/tud-amr/mpc_planner - objectives are evaluated symbolically.
        Only acceleration (a) and angular acceleration (w) are decision variables.
        """
        LOG_DEBUG(f"BaseObjective.get_objective: {self.name}, stage_idx={stage_idx}")
        
        # Prefer get_stage_cost_symbolic if implemented (matches C++ pattern)
        if hasattr(self, 'get_stage_cost_symbolic'):
            try:
                result = self.get_stage_cost_symbolic(state, stage_idx)
                if result is not None:
                    LOG_DEBUG(f"  Using get_stage_cost_symbolic method, returned {len(result) if isinstance(result, (list, tuple)) else 1 if result else 0} term(s)")
                    return result
            except Exception as e:
                LOG_DEBUG(f"  get_stage_cost_symbolic failed: {e}, falling back to get_value")
        
        # Fallback to legacy get_value if implemented
        if hasattr(self, 'get_value'):
            LOG_DEBUG(f"  Using legacy get_value method")
            result = self.get_value(state, data, stage_idx)
            LOG_DEBUG(f"  get_value returned {len(result) if isinstance(result, (list, tuple)) else 1 if result else 0} term(s)")
            return result
        
        LOG_DEBUG(f"  No objective method found, returning empty list")
        return []

    @abstractmethod
    def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
        """
        Return symbolic objective cost expressions for this stage.
        
        CRITICAL: This method MUST return symbolic CasADi expressions (MX or SX) for MPC rollouts.
        The symbolic_state contains CasADi variables for the predicted state at this stage.
        
        Returns:
            dict: Dictionary mapping cost term names to symbolic CasADi expressions.
                 Example: {"contouring_lag_cost": cd.MX(...), "contouring_contour_cost": cd.MX(...)}
        
        Reference: https://github.com/tud-amr/mpc_planner - objectives are evaluated symbolically.
        """
        pass
