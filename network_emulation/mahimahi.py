"""
Helpers for invoking Mahimahi ``mm-link`` and plotting scripts.

Typical congestion-control experiment: run a shell (or client) under ``mm-link`` with
cellular uplink/downlink traces, record logs, optionally nest ``mm-delay`` for RTT.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from network_emulation.cellular_traces import CellularTracePair
from network_emulation.paths import mahimahi_root, mahimahi_scripts_dir


def find_mm_link() -> str | None:
    """Return ``mm-link`` on PATH, ``$MAHIMAHI_MM_LINK``, or an in-tree build under ``MAHIMAHI_ROOT``."""
    override = os.environ.get("MAHIMAHI_MM_LINK")
    if override and Path(override).is_file():
        return override
    w = shutil.which("mm-link")
    if w:
        return w
    root = mahimahi_root()
    for rel in (
        "src/frontend/mm-link",
        "src/frontend/.libs/mm-link",
    ):
        p = root / rel
        if p.is_file():
            return str(p)
    return None


def find_mm_graph() -> str | None:
    w = shutil.which("mm-graph")
    if w:
        return w
    candidate = mahimahi_scripts_dir() / "mm-graph"
    if candidate.is_file():
        return str(candidate)
    return None


@dataclass
class MmLinkSpec:
    """Command-line for ``mm-link uplink downlink [opts] -- command ...``."""

    uplink_trace: Path
    downlink_trace: Path
    inner_command: Sequence[str] = field(default_factory=lambda: ["/bin/bash"])
    uplink_log: Path | None = None
    downlink_log: Path | None = None
    mm_link_binary: str | None = None

    def argv(self) -> list[str]:
        """Full argument list (no shell); suitable for ``subprocess``."""
        exe = self.mm_link_binary or find_mm_link()
        if not exe:
            raise FileNotFoundError(
                "mm-link not found. Install Mahimahi (sudo make install) or set PATH."
            )
        args: list[str] = [exe, str(self.uplink_trace), str(self.downlink_trace)]
        if self.uplink_log is not None:
            args.append(f"--uplink-log={self.uplink_log}")
        if self.downlink_log is not None:
            args.append(f"--downlink-log={self.downlink_log}")
        args.append("--")
        args.extend(self.inner_command)
        return args


def mm_link_from_pair(
    pair: CellularTracePair,
    inner_command: Sequence[str],
    *,
    uplink_log: Path | None = None,
    downlink_log: Path | None = None,
) -> MmLinkSpec:
    return MmLinkSpec(
        uplink_trace=pair.uplink,
        downlink_trace=pair.downlink,
        inner_command=list(inner_command),
        uplink_log=uplink_log,
        downlink_log=downlink_log,
    )


def mm_graph_argv(
    log_path: Path,
    ms_per_bin: int,
    *,
    title: str | None = None,
    no_display: bool = True,
) -> list[str]:
    """Build ``mm-graph`` arguments (NSDI-style throughput/delay plot)."""
    script = find_mm_graph()
    if not script:
        raise FileNotFoundError(
            "mm-graph not found. Install Mahimahi or use MAHIMAHI_ROOT pointing at source."
        )
    args = [script, str(log_path), str(ms_per_bin)]
    if title:
        args.extend(["--title", title])
    if no_display:
        args.append("--no-display")
    return args


def check_mahimahi_tools() -> dict[str, str | None]:
    """Return which tools resolve (for diagnostics)."""
    return {
        "mm-link": find_mm_link(),
        "mm-graph": find_mm_graph(),
        "MAHIMAHI_ROOT": str(mahimahi_root()),
        "PATH": os.environ.get("PATH", ""),
    }
