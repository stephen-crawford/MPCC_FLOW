from planning.modules_manager import Module
from utils.const import CONSTRAINT
from utils.utils import read_config_file, LOG_DEBUG, LOG_INFO
import casadi as cd
from abc import abstractmethod


class BaseConstraint(Module):
    def __init__(self, settings=None):
        super().__init__()  # This loads config as self.settings and self.config
        self.name = self.__class__.__name__.lower()
        self.module_type = CONSTRAINT
        # Override config if settings provided, otherwise use from Module base class
        if settings is not None:
            self.config = settings
            self.settings = settings
        LOG_INFO(f"Initializing {self.name.title()} Constraints")
        LOG_DEBUG(f"BaseConstraint.__init__: name={self.name}, module_type={self.module_type}")

    def calculate_constraints(self, state, data, stage_idx):
        """
        Calculate constraint expressions for this stage.
        
        CRITICAL: This method MUST return symbolic CasADi expressions (MX or SX) for MPC rollouts.
        The state parameter contains CasADi variables from the solver's var_dict:
            - State variables (x, y, psi, v, spline) - constrained by RK4/model_discrete_dynamics
            - Control variables (a, w) - free decision variables
        
        The returned expressions must be symbolic and depend on these variables.
        Numeric calculations are NOT allowed - all computations must be symbolic.
        
        Returns:
            list: List of constraint expressions or dicts containing constraint information.
                 Each constraint should be either:
                 - A CasADi expression (MX or SX) directly
                 - A dict with 'type': 'symbolic_expression' and 'expression': CasADi expression
                 - A dict with constraint parameters (a1, a2, b, etc.) for linear constraints
        
        Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
        Only acceleration (a) and angular acceleration (w) are decision variables.
        All states are computed via RK4 integration or model_discrete_dynamics.
        """
        # Default implementation: return empty list
        # Subclasses should override this method
        LOG_DEBUG(f"BaseConstraint.calculate_constraints: {self.name}, stage_idx={stage_idx} (no implementation)")
        return []

    def get_visualizer(self):
        """Return a visualizer for this constraint module. Override in subclasses."""
        return None

