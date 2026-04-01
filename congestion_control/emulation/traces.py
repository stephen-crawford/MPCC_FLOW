"""Trace file utilities: create synthetic traces and locate real Mahimahi traces."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional


def create_constant_trace(rate_mbps: float, duration_ms: int = 60_000,
                          mtu: int = 1500) -> Path:
    """Create a Mahimahi-format trace file with constant link rate.

    Each line is a timestamp (ms) at which one MTU can be delivered.
    """
    packets_per_ms = (rate_mbps * 1e6) / (8 * mtu * 1000)
    interval_ms = 1.0 / max(packets_per_ms, 0.001)

    tmpf = tempfile.NamedTemporaryFile(mode="w", suffix=".trace",
                                        prefix="const_", delete=False)
    t = 0.0
    while t < duration_ms:
        tmpf.write(f"{int(t)}\n")
        t += interval_ms
    tmpf.close()
    return Path(tmpf.name)


def create_variable_trace(rates_mbps: List[float], segment_ms: int = 5000,
                           mtu: int = 1500) -> Path:
    """Create a trace with varying link rates (step function).

    Args:
        rates_mbps: list of rates, one per segment
        segment_ms: duration of each segment in ms
    """
    tmpf = tempfile.NamedTemporaryFile(mode="w", suffix=".trace",
                                        prefix="var_", delete=False)
    t = 0.0
    for rate in rates_mbps:
        seg_end = t + segment_ms
        if rate > 0:
            pps = (rate * 1e6) / (8 * mtu * 1000)
            interval = 1.0 / max(pps, 0.001)
            while t < seg_end:
                tmpf.write(f"{int(t)}\n")
                t += interval
        else:
            t = seg_end
    tmpf.close()
    return Path(tmpf.name)


def create_cellular_like_trace(avg_mbps: float = 5.0, duration_ms: int = 60_000,
                                burst_factor: float = 3.0, mtu: int = 1500) -> Path:
    """Create a trace that mimics cellular burstiness.

    Alternates between burst periods (burst_factor * avg) and quiet periods.
    """
    import random
    rng = random.Random(42)
    tmpf = tempfile.NamedTemporaryFile(mode="w", suffix=".trace",
                                        prefix="cell_", delete=False)
    t = 0.0
    while t < duration_ms:
        # Randomly choose burst or quiet
        if rng.random() < 0.3:
            # Quiet period: low rate
            rate = avg_mbps * 0.3
            dur = rng.uniform(50, 500)
        else:
            # Burst period
            rate = avg_mbps * rng.uniform(0.5, burst_factor)
            dur = rng.uniform(10, 200)

        seg_end = t + dur
        if rate > 0:
            pps = (rate * 1e6) / (8 * mtu * 1000)
            interval = 1.0 / max(pps, 0.001)
            while t < seg_end and t < duration_ms:
                tmpf.write(f"{int(t)}\n")
                t += interval
        else:
            t = seg_end
    tmpf.close()
    return Path(tmpf.name)


def find_mahimahi_traces_dir() -> Optional[Path]:
    """Locate the Mahimahi traces directory."""
    candidates = [
        Path(os.environ.get("MAHIMAHI_ROOT", "")) / "traces",
        Path(__file__).resolve().parents[2] / "third_party" / "mahimahi" / "traces",
        Path.home() / "mahimahi" / "traces",
    ]
    for p in candidates:
        if p.is_dir() and any(p.glob("*.up")):
            return p
    return None


def find_trace(name: str) -> Optional[Path]:
    """Find a specific trace file by name (e.g., 'Verizon-LTE-driving.up')."""
    traces_dir = find_mahimahi_traces_dir()
    if traces_dir is None:
        return None
    candidate = traces_dir / name
    if candidate.exists():
        return candidate
    return None
