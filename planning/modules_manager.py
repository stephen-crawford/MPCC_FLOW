import copy

from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import read_config_file
from utils.utils import print_value, print_header, LOG_DEBUG, LOG_INFO, LOG_WARN


class ModuleManager:
    """
  Module Manager handles collections of constraint and objective modules
  for MPC. It coordinates updates, parameter setting, and visualization.
  """

    def __init__(self):
        self.modules = []

    def add_module(self, module):
        """Add a module instance and check dependencies"""
        module_name = getattr(module, 'name', 'Unknown')
        module_type = getattr(module, 'module_type', 'Unknown')
        LOG_DEBUG(f"ModuleManager.add_module: Adding '{module_name}' (type: {module_type})")
        
        # Check if module has dependencies
        if hasattr(module, 'dependencies') and module.dependencies:
            LOG_DEBUG(f"Module '{module_name}' has dependencies: {module.dependencies}")
            for dep_name in module.dependencies:
                # Check if dependency exists in modules
                dep_found = any(m.name == dep_name for m in self.modules)
                if not dep_found:
                    error_msg = (
                        f"Module '{module.name}' requires dependency '{dep_name}' "
                        f"which is not present in the module manager. "
                        f"Please add '{dep_name}' before adding '{module.name}'."
                    )
                    LOG_WARN(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    LOG_DEBUG(f"  Dependency '{dep_name}' satisfied")
        else:
            LOG_DEBUG(f"Module '{module_name}' has no dependencies")
        
        self.modules.append(module)
        LOG_INFO(f"Module '{module_name}' added successfully. Total modules: {len(self.modules)}")

    def add_modules(self, modules):
        """Add a list of modules, preserving dependency checks."""
        for module in modules:
            self.add_module(module)

    def check_dependencies(self):
        """Verify that all declared dependencies are present among modules.

        Returns:
            bool: True if all dependencies satisfied, False otherwise.
        """
        available_names = {getattr(m, "name", None) for m in self.modules}
        for module in self.modules:
            deps = getattr(module, "dependencies", []) or []
            for dep_name in deps:
                if dep_name not in available_names:
                    return False
        return True

    def get_objective_modules(self):
        """Return the list of objective-type modules (or None if empty)."""
        objs = [m for m in self.modules if getattr(m, "module_type", None) == OBJECTIVE]
        return objs if objs else None

    def get_modules(self):
        return self.modules

    def is_data_ready(self, state, data):
        LOG_DEBUG("ModuleManager.is_data_ready: Checking all modules...")
        for module in self.modules:
            module_name = getattr(module, 'name', 'Unknown')
            is_ready = module.is_data_ready(state, data)
            LOG_DEBUG(f"  Module '{module_name}': data_ready={is_ready}")
            if not is_ready:
                LOG_WARN(f"Module '{module_name}' reports data is not ready")
                return False
        LOG_DEBUG("ModuleManager.is_data_ready: All modules report data is ready")
        return True

    def define_parameters(self, parameter_manager):
        """Define all module parameters for the solver"""
        LOG_DEBUG("ModuleManager.define_parameters: Defining parameters for all modules...")
        for module in self.modules:
            module_name = getattr(module, 'name', 'Unknown')
            if hasattr(module, "define_parameters"):
                LOG_DEBUG(f"  Defining parameters for '{module_name}'...")
                module.define_parameters(self, parameter_manager)
                LOG_DEBUG(f"  Parameters defined for '{module_name}'")
            else:
                LOG_DEBUG(f"  Module '{module_name}' has no define_parameters method")

    def get_objectives(self, state, data, stage_idx):
        """Collect objective contributions from all objective modules.

        Supports both `get_objective` and legacy `get_value` method names.
        Returns a flat list for the stage.
        """
        LOG_DEBUG(f"ModuleManager.get_objectives: stage_idx={stage_idx}")
        objectives = []
        objective_modules = [m for m in self.modules if getattr(m, "module_type", None) == OBJECTIVE]
        LOG_DEBUG(f"Found {len(objective_modules)} objective module(s) for stage {stage_idx}")
        
        for module in objective_modules:
            module_name = getattr(module, 'name', 'Unknown')
            LOG_DEBUG(f"  Getting objective from '{module_name}'...")
            
            if hasattr(module, "get_objective"):
                val = module.get_objective(state, data, stage_idx)
                if isinstance(val, (list, tuple)):
                    objectives.extend(val)
                    LOG_DEBUG(f"    '{module_name}' returned {len(val)} objective term(s)")
                elif val is not None:
                    objectives.append(val)
                    LOG_DEBUG(f"    '{module_name}' returned 1 objective term")
            elif hasattr(module, "get_value"):
                val = module.get_value(state, data, stage_idx)
                if isinstance(val, (list, tuple)):
                    objectives.extend(val)
                    LOG_DEBUG(f"    '{module_name}' returned {len(val)} objective term(s) (legacy get_value)")
                elif val is not None:
                    objectives.append(val)
                    LOG_DEBUG(f"    '{module_name}' returned 1 objective term (legacy get_value)")
                else:
                    LOG_DEBUG(f"    '{module_name}' returned None (legacy get_value)")
            else:
                LOG_WARN(f"    '{module_name}' has neither get_objective nor get_value method")
        
        LOG_DEBUG(f"ModuleManager.get_objectives: Returning {len(objectives)} total objective term(s) for stage {stage_idx}")
        return objectives

    def get_constraints(self, state, data, stage_idx):
        """Calculate constraint values from all constraint modules"""
        LOG_DEBUG(f"ModuleManager.get_constraints: stage_idx={stage_idx}")
        constraint_values = []
        constraint_modules = [m for m in self.modules if getattr(m, "module_type", None) == CONSTRAINT]
        LOG_DEBUG(f"Found {len(constraint_modules)} constraint module(s) for stage {stage_idx}")
        
        # Call update() on constraint modules before calculating constraints (for stage 0 only, to avoid redundant updates)
        if stage_idx == 0:
            for module in constraint_modules:
                if hasattr(module, "update"):
                    try:
                        # Ensure module has solver reference if it needs it
                        # Try multiple ways to get solver reference
                        if not hasattr(module, 'solver') or module.solver is None:
                            # Try to get solver from module manager's parent (solver)
                            # The module manager is typically owned by a solver
                            if hasattr(self, '_solver') and self._solver is not None:
                                module.solver = self._solver
                            elif hasattr(data, 'solver') and data.solver is not None:
                                module.solver = data.solver
                            # Also check if we can get it from the parent context
                            # (This is a fallback - ideally solver should be set when modules are added)
                        
                        LOG_DEBUG(f"ModuleManager.get_constraints: Calling update() on '{getattr(module, 'name', 'Unknown')}' (solver available: {hasattr(module, 'solver') and module.solver is not None})")
                        module.update(state, data)
                    except Exception as e:
                        LOG_WARN(f"Error calling update() on module '{getattr(module, 'name', 'Unknown')}': {e}")
                        import traceback
                        LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
        
        for module in constraint_modules:
            module_name = getattr(module, 'name', 'Unknown')
            LOG_DEBUG(f"  Getting constraints from '{module_name}'...")
            
            if hasattr(module, "calculate_constraints"):
                try:
                    cons = module.calculate_constraints(state, data, stage_idx)
                    if isinstance(cons, (list, tuple)):
                        constraint_values.extend(cons)
                        LOG_DEBUG(f"    '{module_name}' returned {len(cons)} constraint(s)")
                    elif cons is not None:
                        constraint_values.append(cons)
                        LOG_DEBUG(f"    '{module_name}' returned 1 constraint")
                    else:
                        LOG_DEBUG(f"    '{module_name}' returned None")
                except Exception as e:
                    LOG_WARN(f"    '{module_name}.calculate_constraints' raised exception: {e}")
            else:
                LOG_DEBUG(f"    '{module_name}' has no calculate_constraints method")
        
        LOG_DEBUG(f"ModuleManager.get_constraints: Returning {len(constraint_values)} total constraint(s) for stage {stage_idx}")
        return constraint_values

    def get_constraints_with_bounds(self, state, data, stage_idx):
        """Return constraints paired with their lower/upper bounds for this stage.
        Ensures alignment by zipping to the shortest list length.
        """
        cons = self.get_constraints(state, data, stage_idx) or []
        lbs = self.get_constraint_lower_bounds(state, data, stage_idx) or []
        ubs = self.get_constraint_upper_bounds(state, data, stage_idx) or []
        n = min(len(cons), len(lbs), len(ubs))
        paired = []
        for i in range(n):
            paired.append((cons[i], lbs[i], ubs[i]))
        # If lengths differ, append remaining with None bounds to avoid index errors
        if len(cons) > n:
            for i in range(n, len(cons)):
                lb = lbs[i] if i < len(lbs) else None
                ub = ubs[i] if i < len(ubs) else None
                paired.append((cons[i], lb, ub))
        return paired

    def get_constraint_lower_bounds(self, state, data, stage_idx):
        """Get lower bounds for all constraints"""
        bounds = []
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "lower_bounds"):
                    # Support signatures with or without args
                    try:
                        val = module.lower_bounds(state, data, stage_idx)
                    except TypeError:
                        val = module.lower_bounds()
                    if isinstance(val, (list, tuple)):
                        bounds.extend(val)
                    elif val is not None:
                        bounds.append(val)
        return bounds

    def get_constraint_upper_bounds(self, state, data, stage_idx):
        """Get upper bounds for all constraints"""
        bounds = []
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "upper_bounds"):
                    # Support signatures with or without args
                    try:
                        val = module.upper_bounds(state, data, stage_idx)
                    except TypeError:
                        val = module.upper_bounds()
                    if isinstance(val, (list, tuple)):
                        bounds.extend(val)
                    elif val is not None:
                        bounds.append(val)
        return bounds

    # Planner compatibility helpers: names without the "constraint_" prefix
    def get_lower_bounds(self, state, data, stage_idx):
        return self.get_constraint_lower_bounds(state, data, stage_idx)

    def get_upper_bounds(self, state, data, stage_idx):
        return self.get_constraint_upper_bounds(state, data, stage_idx)

    def get_constraint_number(self):
        """Get total number of constraints"""
        count = 0
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "n_constraints"):
                    count += module.n_constraints
        return count

    def update_parameters_all(self, parameter_manager):
        """Set parameters for all modules across all stages"""
        for module in self.modules:
            module.update_parameters(parameter_manager)

    def get_all_visualizers(self, state, data, step):
        """Trigger visualization for all modules"""
        visualizers = []
        for module in self.modules:
            module_visualizer = module.get_visualizer()
            if module_visualizer is not None:
                visualizers.append(module_visualizer)
        return visualizers

    def are_objectives_reached(self, state, data):
        """Check if all objectives have been reached"""
        objective_modules = []
        for module in self.modules:
            if module.module_type == OBJECTIVE:
                objective_modules.append(module)
        if len(objective_modules) == 0:
            return False

        for module in objective_modules:
            if hasattr(module, "is_objective_reached") and not module.is_objective_reached(state, data):
                return False
        return True

    def reset_all(self):
        """Reset all modules"""
        for module in self.modules:
            if hasattr(module, "reset"):
                module.reset()

    def __str__(self):
        result = "--- MPC Modules ---\n"
        for module in self.modules:
            module_name = getattr(module, "name", "Unnamed Module")
            result += f"{module_name}: {str(module)}\n"
        return result

    def print(self):
        print_header("MPC Modules")
        for module in self.modules:
            module_name = getattr(module, "name", "Unnamed Module")
            print_value(module_name, str(module), tab=True)

    def copy(self, new_solver=None):
        """
        Create a copy of the module manager.
        
        Args:
            new_solver: Optional new solver instance to attach modules to.
                       If None, creates a deep copy with original solver references.
        """
        if new_solver is None:
            return copy.deepcopy(self)
        
        # Create new manager and copy modules, updating solver references
        new_manager = ModuleManager()
        for module in self.modules:
            # Create a shallow copy of the module and update its solver reference
            module_copy = copy.copy(module)
            module_copy.solver = new_solver
            new_manager.modules.append(module_copy)
        return new_manager

class Module:

    def __init__(self):
        self.name = None
        self.settings = read_config_file()
        self.config = self.settings  # Alias for compatibility
        self.module_type = None
        self.description = ""

        self.dependencies = []
        self.visualizer = None
    
    def get_config_value(self, dotted_key: str, default=None):
        """Get a config value using dotted key notation (e.g., 'modules.constraints.contouring_constraints.num_discs').
        
        Args:
            dotted_key: Dot-separated key path into config dict
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        if self.settings is None:
            return default
        
        # Use get_config_dotted utility for consistent parsing
        from utils.utils import get_config_dotted
        return get_config_dotted(self.settings, dotted_key, default)

    def get_horizon(self, data=None, default: int = 10) -> int:
        """Resolve planning horizon using data first, then config, then default."""
        try:
            if data is not None and hasattr(data, 'horizon') and data.horizon is not None:
                return int(data.horizon)
        except Exception:
            pass
        cfg_h = self.get_config_value("planner.horizon", default)
        try:
            return int(cfg_h)
        except Exception:
            return default

    def get_timestep(self, data=None, default: float = 0.1) -> float:
        """Resolve timestep using data first, then config, then default."""
        try:
            if data is not None and hasattr(data, 'timestep') and data.timestep is not None:
                return float(data.timestep)
        except Exception:
            pass
        cfg_dt = self.get_config_value("planner.timestep", default)
        try:
            return float(cfg_dt)
        except Exception:
            return default

    def __str__(self):
        result = self.description
        return result

    def update_parameters(self, parameter_manager):
        pass

    def define_parameters(self, parameter_manager):
        pass

    def build_visualizer(self):
        pass

    def is_data_ready(self, data, state=None):
        """Check if required data is available. `state` is optional for planner compat."""
        return True

    def update(self, state, data):
        """Optional per-iteration module update."""
        pass

    def on_data_received(self, data):
        """Process incoming data by type"""
        pass

    def reset(self):
        """Reset constraint state"""
        pass

    def get_name(self):
        return self.name
    
    def get_visualizer(self):
        return self.visualizer

    def get_warmstart_position(self, stage_idx: int):
        """Get warmstart position (x, y) at a specific stage.

        Args:
            stage_idx: The stage index.

        Returns:
            Tuple of (x, y) or None if not available.
        """
        if not hasattr(self, 'solver') or self.solver is None:
            return None

        ws = getattr(self.solver, 'warmstart_values', None)
        if ws is None or not ws:
            return None

        if 'x' in ws and 'y' in ws:
            x_vals = ws['x']
            y_vals = ws['y']
            if stage_idx < len(x_vals) and stage_idx < len(y_vals):
                return (float(x_vals[stage_idx]), float(y_vals[stage_idx]))
        return None

    def get_effective_horizon(self, data=None) -> int:
        """Get the effective planning horizon.

        Checks solver, data, and config in order of priority.

        Args:
            data: Optional data object with horizon attribute.

        Returns:
            The effective horizon value.
        """
        # Try solver first
        if hasattr(self, 'solver') and self.solver is not None:
            h = getattr(self.solver, 'horizon', None)
            if h is not None:
                return int(h)

        # Try data
        if data is not None:
            h = getattr(data, 'horizon', None)
            if h is not None:
                return int(h)

        # Fall back to config
        return self.get_horizon(data)

    def filter_constraints_by_stage(self, constraints, stage_idx: int, horizon: int = None):
        """Filter constraints based on stage index.

        Some constraints only apply at specific stages (first, last, intermediate).
        This helper filters a list of constraints based on stage information.

        Args:
            constraints: List of constraint dicts or values.
            stage_idx: Current stage index.
            horizon: Total horizon length (for detecting final stage).

        Returns:
            Filtered list of constraints applicable to this stage.
        """
        if not constraints:
            return []

        if horizon is None:
            horizon = self.get_effective_horizon()

        is_first = (stage_idx == 0)
        is_last = (stage_idx >= horizon)

        filtered = []
        for constraint in constraints:
            # If constraint is a dict with stage specification, filter it
            if isinstance(constraint, dict):
                apply_at = constraint.get('apply_at', 'all')
                if apply_at == 'all':
                    filtered.append(constraint)
                elif apply_at == 'first' and is_first:
                    filtered.append(constraint)
                elif apply_at == 'last' and is_last:
                    filtered.append(constraint)
                elif apply_at == 'intermediate' and not is_first and not is_last:
                    filtered.append(constraint)
                elif apply_at == 'not_first' and not is_first:
                    filtered.append(constraint)
                elif apply_at == 'not_last' and not is_last:
                    filtered.append(constraint)
            else:
                # Non-dict constraints apply to all stages
                filtered.append(constraint)

        return filtered

    def copy(self):
        return copy.deepcopy(self)
