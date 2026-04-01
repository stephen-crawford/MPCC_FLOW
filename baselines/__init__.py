"""
Baseline congestion control algorithm wrappers for comparison with MPCC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentResult:
    algorithm: str
    trace: str
    delay_ms: int
    duration_s: float
    avg_throughput_mbps: float = 0.0
    p95_rtt_ms: float = 0.0
    median_rtt_ms: float = 0.0
    loss_rate: float = 0.0
    power: float = 0.0
    throughput_ts: list[tuple[float, float]] = field(default_factory=list)
    rtt_ts: list[tuple[float, float]] = field(default_factory=list)
    log_dir: Path | None = None


class BaselineAlgorithm(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(
        self,
        trace_name: str,
        delay_ms: int,
        duration_s: float,
        output_dir: Path,
    ) -> ExperimentResult:
        ...
