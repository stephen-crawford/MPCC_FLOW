"""
Discover and label cellular link traces shipped with Mahimahi (NSDI 2013 Saturator).

Trace format: one timestamp (ms) per line per 1500-byte delivery opportunity; see
``traces/README`` in the Mahimahi tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from network_emulation.paths import mahimahi_traces_dir


@dataclass(frozen=True)
class CellularTracePair:
    """Uplink / downlink trace files for one recorded cellular scenario."""

    name: str
    uplink: Path
    downlink: Path
    carrier: str
    kind: str

    def exists(self) -> bool:
        return self.uplink.is_file() and self.downlink.is_file()


def _infer_carrier_kind(stem: str) -> tuple[str, str]:
    """Best-effort parse from filenames like ``Verizon-LTE-driving``."""
    parts = stem.split("-")
    carrier = parts[0] if parts else "unknown"
    rest = "-".join(parts[1:]) if len(parts) > 1 else stem
    return carrier, rest


def iter_cellular_pairs(traces_dir: Path | None = None) -> Iterator[CellularTracePair]:
    """
    Yield uplink/downlink pairs by matching ``*.up`` with ``*.down`` of the same basename.

    Skips pairs where either file is missing.
    """
    root = traces_dir or mahimahi_traces_dir()
    if not root.is_dir():
        return

    stems: set[str] = set()
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix == ".up":
            stems.add(p.stem)
        elif p.suffix == ".down":
            stems.add(p.stem)

    for stem in sorted(stems):
        up = root / f"{stem}.up"
        down = root / f"{stem}.down"
        if not up.is_file() or not down.is_file():
            continue
        carrier, kind = _infer_carrier_kind(stem)
        yield CellularTracePair(
            name=stem,
            uplink=up,
            downlink=down,
            carrier=carrier,
            kind=kind,
        )


def list_cellular_pairs(traces_dir: Path | None = None) -> list[CellularTracePair]:
    return list(iter_cellular_pairs(traces_dir))


def get_pair(name: str, traces_dir: Path | None = None) -> CellularTracePair | None:
    """Look up a pair by its stem (e.g. ``Verizon-LTE-short``)."""
    for pair in iter_cellular_pairs(traces_dir):
        if pair.name == name:
            return pair
    return None
