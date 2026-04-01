"""Trace-driven link emulation for congestion control testing."""

from congestion_control.emulation.link_emulator import LinkEmulator, LinkConfig
from congestion_control.emulation.flow import Flow, FlowConfig
from congestion_control.emulation.runner import run_comparison, ComparisonResult

__all__ = [
    "LinkEmulator",
    "LinkConfig",
    "Flow",
    "FlowConfig",
    "run_comparison",
    "ComparisonResult",
]
