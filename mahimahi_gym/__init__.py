"""Gymnasium registration for mahimahi congestion-control environments."""

from gymnasium.envs.registration import register

from mahimahi_gym.mm_link import MahimahiSubprocessEnv
from mahimahi_gym.scenarios import CellularBurstyEnv, LeoSatelliteEnv, MultiFlowFairnessEnv, WiredBottleneckEnv

register(
    id="MahiWiredBottleneck-v0",
    entry_point="mahimahi_gym.scenarios:WiredBottleneckEnv",
)

register(
    id="MahiCellularBursty-v0",
    entry_point="mahimahi_gym.scenarios:CellularBurstyEnv",
)

register(
    id="MahiLeoSatellite-v0",
    entry_point="mahimahi_gym.scenarios:LeoSatelliteEnv",
)

register(
    id="MahiMultiFlowFairness-v0",
    entry_point="mahimahi_gym.scenarios:MultiFlowFairnessEnv",
)

__all__ = [
    "MahimahiSubprocessEnv",
    "WiredBottleneckEnv",
    "CellularBurstyEnv",
    "LeoSatelliteEnv",
    "MultiFlowFairnessEnv",
]
