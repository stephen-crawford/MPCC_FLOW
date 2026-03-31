"""
State and Disc classes - re-exported from types_impl for backward compatibility.
"""

from planning.types_impl import State, Disc, define_robot_area

# Export the nested ScenarioDisc class at module level for backward compatibility
ScenarioDisc = Disc.ScenarioDisc

__all__ = ['State', 'Disc', 'ScenarioDisc', 'define_robot_area']
