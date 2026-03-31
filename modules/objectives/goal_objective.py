import numpy as np
import casadi as cs
from modules.objectives.base_objective import BaseObjective
from planning.types import State
from utils.const import OBJECTIVE
from utils.math_tools import distance
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO

class GoalObjective(BaseObjective):

  def __init__(self):
    super().__init__()
    self.module_type = OBJECTIVE
    self.name = "goal_objective"
    LOG_INFO( "Initializing Goal Module")
    LOG_INFO( "Goal Module successfully initialized")
    self.goal_weight = self.get_config_value("weights.goal_weight")

  def update(self, state, data):
    return

  def define_parameters(self, params):

    print("Defining parameters for Goal Objective")

    params.add("goal_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["goal"]')
    params.add("goal_control_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["control"]')
    params.add("goal_terminal_angle_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["terminal_angle"]')
    params.add("goal_angle_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["angle"]')
    params.add("goal_terminal_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["terminal"]')
    params.add("goal_x")
    params.add("goal_y")

  def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
    """
    Return symbolic objective cost expressions for goal reaching.
    
    CRITICAL: This method returns symbolic CasADi expressions for MPC rollouts.
    The symbolic_state contains CasADi variables for the predicted state at this stage.
    
    Reference: https://github.com/tud-amr/mpc_planner - objectives are evaluated symbolically.
    """
    # Delegate to get_value for now (it already returns symbolic expressions)
    # This maintains backward compatibility while ensuring symbolic return
    return self.get_value(symbolic_state, self.solver.data if hasattr(self, 'solver') and self.solver else None, stage_idx)

  def get_value(self, state, data, stage_idx):
    """Compute goal objective cost symbolically.
    
    CRITICAL: This method receives symbolic_state for future stages (stage_idx > 0),
    ensuring objectives are computed symbolically using predicted states.
    This matches the reference codebase pattern.
    
    Reference: https://github.com/tud-amr/mpc_planner - objectives are evaluated symbolically.
    """
    # Get symbolic state variables (will be symbolic for future stages)
    pos_x = state.get("x")
    pos_y = state.get("y")
    psi = state.get("psi")

    # Get parameters from parameter_manager (like contouring objective does)
    goal_x = None
    goal_y = None
    
    # Log solver availability for debugging
    if stage_idx == 0:
        has_solver = hasattr(self, 'solver') and self.solver is not None
        has_param_mgr = has_solver and hasattr(self.solver, 'parameter_manager')
        LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}): has_solver={has_solver}, has_param_mgr={has_param_mgr}")
    
    if self.solver and hasattr(self.solver, 'parameter_manager'):
        params_dict = self.solver.parameter_manager.get_all(stage_idx)
        goal_x = params_dict.get("goal_x")
        goal_y = params_dict.get("goal_y")
        if stage_idx == 0:
            LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}): goal from parameter_manager: ({goal_x}, {goal_y}), all_params={list(params_dict.keys())[:5]}")
    
    # Fallback: try data.parameters
    if goal_x is None or goal_y is None:
        if data and hasattr(data, 'parameters') and data.parameters is not None:
            goal_x = data.parameters.get("goal_x")
            goal_y = data.parameters.get("goal_y")
            if stage_idx == 0:
                LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}): goal from data.parameters: ({goal_x}, {goal_y})")

    # If goal parameters not set, try to get from data.goal directly
    if goal_x is None or goal_y is None:
        if data and hasattr(data, 'goal') and data.goal is not None:
            goal_x = float(data.goal[0]) if len(data.goal) > 0 else None
            goal_y = float(data.goal[1]) if len(data.goal) > 1 else None
            if stage_idx == 0:
                LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}): goal from data.goal: ({goal_x}, {goal_y})")

    # If still None, return zero cost (can't compute goal error without goal)
    if goal_x is None or goal_y is None:
        LOG_WARN(f"GoalObjective.get_value(stage={stage_idx}): goal_x={goal_x}, goal_y={goal_y} - returning zero cost")
        LOG_WARN(f"  has_solver={hasattr(self, 'solver') and self.solver is not None}, has_data={data is not None}, has_data_goal={data and hasattr(data, 'goal') and data.goal is not None}")
        if data and hasattr(data, 'goal'):
            LOG_WARN(f"  data.goal={data.goal}")
        return {"goal_pos_cost": 0.0, "goal_angle_cost": 0.0}

    # Get weights with safe defaults - try parameter_manager first, then data.parameters, then config
    # Defaults from config file: goal_weight=10.0, goal_angle_weight=1.0
    goal_weight = self.get_config_value("weights.goal_weight", 10.0)
    angle_weight = self.get_config_value("weights.goal_angle_weight", 1.0)
    terminal_goal_weight = self.get_config_value("weights.goal_terminal_weight", 10.0)
    terminal_angle_weight = self.get_config_value("weights.goal_terminal_angle", 1.0)
    
    if self.solver and hasattr(self.solver, 'parameter_manager'):
        params_dict = self.solver.parameter_manager.get_all(stage_idx)
        goal_weight = params_dict.get("goal_weight", goal_weight)
        angle_weight = params_dict.get("goal_angle_weight", angle_weight)
        terminal_goal_weight = params_dict.get("goal_terminal_weight", terminal_goal_weight)
        terminal_angle_weight = params_dict.get("goal_terminal_angle_weight", terminal_angle_weight)
    elif data and hasattr(data, 'parameters') and data.parameters is not None:
        goal_weight = data.parameters.get("goal_weight", goal_weight)
        angle_weight = data.parameters.get("goal_angle_weight", angle_weight)
        terminal_goal_weight = data.parameters.get("goal_terminal_weight", terminal_goal_weight)
        terminal_angle_weight = data.parameters.get("goal_terminal_angle_weight", terminal_angle_weight)
    
    # Log weights for debugging
    if stage_idx == 0:
        LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}): weights - goal={goal_weight:.2f}, angle={angle_weight:.2f}, terminal_goal={terminal_goal_weight:.2f}, terminal_angle={terminal_angle_weight:.2f}")

    LOG_DEBUG("pos x : " + str(pos_x))
    LOG_DEBUG("pos y : " + str(pos_y))
    LOG_DEBUG("psi : " + str(psi))
    LOG_DEBUG("goal_x : " + str(goal_x))
    LOG_DEBUG("goal_y : " + str(goal_y))

    # Compute positional and angle error
    dx = pos_x - goal_x
    dy = pos_y - goal_y

    pos_error_sq = dx ** 2 + dy ** 2
    pos_error = cs.sqrt(pos_error_sq)  # For logging
    theta_goal = cs.atan2(goal_y - pos_y, goal_x - pos_x)
    # Angle error: difference between current heading and desired heading toward goal
    # Positive error means vehicle should turn left (increase psi), negative means turn right (decrease psi)
    angle_error = cs.fmod(psi - theta_goal + cs.pi, 2 * cs.pi) - cs.pi
    
    # Log angle error for debugging (only for stage 0 to avoid spam)
    if stage_idx == 0:
        try:
            # Try to evaluate if symbolic
            import casadi as cs_eval
            if hasattr(psi, 'is_constant') and psi.is_constant():
                psi_val = float(psi)
                theta_goal_val = float(theta_goal) if hasattr(theta_goal, 'is_constant') and theta_goal.is_constant() else 'symbolic'
                angle_error_val = float(angle_error) if hasattr(angle_error, 'is_constant') and angle_error.is_constant() else 'symbolic'
                LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}): angle_error computation - psi={psi_val:.3f} rad ({np.degrees(psi_val):.1f}°), theta_goal={theta_goal_val}, angle_error={angle_error_val}")
        except:
            pass

    # If at terminal stage, apply stronger penalty
    # Get horizon from data or use default
    horizon_val = 10
    if hasattr(self, 'solver') and hasattr(self.solver, 'horizon') and self.solver.horizon is not None:
      horizon_val = self.solver.horizon
    elif data and hasattr(data, 'horizon') and data.horizon is not None:
      horizon_val = data.horizon
    # Helper to safely format values (handles symbolic expressions)
    def safe_format(val, default='symbolic'):
        if val is None:
            return 'None'
        import casadi as cd
        if isinstance(val, (cd.MX, cd.SX)):
            return default  # Can't format symbolic expressions
        try:
            return f"{float(val):.3f}"
        except (TypeError, ValueError):
            return str(val)
    
    if stage_idx == horizon_val - 1:
      pos_cost = terminal_goal_weight * pos_error_sq
      angle_cost = terminal_angle_weight * angle_error ** 2
      if stage_idx == 0:  # Only log for first stage to avoid spam
        LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}, TERMINAL): pos=({safe_format(pos_x)}, {safe_format(pos_y)}), goal=({safe_format(goal_x)}, {safe_format(goal_y)}), error={safe_format(pos_error)}m, cost={safe_format(pos_cost)}")
    else:
      pos_cost = goal_weight * pos_error_sq
      angle_cost = angle_weight * angle_error ** 2
      if stage_idx == 0:  # Only log for first stage to avoid spam
        LOG_INFO(f"GoalObjective.get_value(stage={stage_idx}): pos=({safe_format(pos_x)}, {safe_format(pos_y)}), goal=({safe_format(goal_x)}, {safe_format(goal_y)}), error={safe_format(pos_error)}m, cost={safe_format(pos_cost)}")

    return {"goal_pos_cost": pos_cost, "goal_angle_cost": angle_cost}

  def set_parameters(self, parameter_manager, data, k):
    LOG_DEBUG("Setting parameters for Goal Objective")
    if k == 0:
      LOG_DEBUG( "Goal Module.set_parameters()")
      LOG_INFO(f"Goal Objective: Setting goal to ({data.goal[0]:.3f}, {data.goal[1]:.3f})")

    # Set goal for this specific stage (goal is the same for all stages)
    parameter_manager.set_parameter("goal_x", data.goal[0], stage_index=k)
    parameter_manager.set_parameter("goal_y", data.goal[1], stage_index=k)
    parameter_manager.set_parameter("goal_weight", self.goal_weight)
    # Set all weight parameters from config for proper convergence
    parameter_manager.set_parameter("goal_angle_weight", 
                                     self.get_config_value("weights.goal_angle_weight", 10.0))
    parameter_manager.set_parameter("goal_terminal_weight", 
                                     self.get_config_value("weights.goal_terminal_weight", 10.0))
    parameter_manager.set_parameter("goal_terminal_angle_weight", 
                                     self.get_config_value("weights.goal_terminal_angle", 1.0))
    parameter_manager.set_parameter("goal_control_weight", 
                                     self.get_config_value("weights.goal_control_weight", 1.0))

  def is_data_ready(self, data):
    missing_data = ""
    if not data.has("goal_received") or data.goal_received is None or data.goal_received == False:
      missing_data += "Goal "

    return len(missing_data) < 1

  def is_objective_reached(self, state: State, data):
    if not data.goal_received:
      return False

    # Check if we reached the goal
    reached = distance(state.get_position(), data.goal) < 1.0
    LOG_DEBUG(f"Goal Objective.is_objective_reached(): {reached}")
    return reached