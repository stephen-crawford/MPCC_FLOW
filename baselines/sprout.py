"""
Sprout baseline wrapper (Winstein & Balakrishnan, NSDI 2013).
Expects sproutbt2 built from https://github.com/keithw/sprout.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from baselines import BaselineAlgorithm, ExperimentResult
from network_emulation.cellular_traces import get_pair
from network_emulation.mahimahi import find_mm_link

logger = logging.getLogger(__name__)

SPROUT_DEFAULT_DIR = Path("third_party/sprout")


def find_sprout_binary(search_dir: Path | None = None) -> Path | None:
    candidates = [
        search_dir / "src" / "sproutbt2" if search_dir else None,
        SPROUT_DEFAULT_DIR / "src" / "sproutbt2",
        Path.home() / "sprout" / "src" / "sproutbt2",
    ]
    for c in candidates:
        if c and c.is_file():
            return c
    return None


class SproutBaseline(BaselineAlgorithm):

    def __init__(self, sprout_dir: Path | None = None):
        self._sprout_bin = find_sprout_binary(sprout_dir)

    @property
    def name(self) -> str:
        return "sprout"

    def run(self, trace_name, delay_ms, duration_s, output_dir):
        if not self._sprout_bin:
            raise FileNotFoundError("sproutbt2 not found")

        pair = get_pair(trace_name)
        if pair is None or not pair.exists():
            raise FileNotFoundError(f"Trace not found: {trace_name}")

        mm_link = find_mm_link()
        if not mm_link:
            raise FileNotFoundError("mm-link not found")

        output_dir.mkdir(parents=True, exist_ok=True)
        uplink_log = output_dir / "uplink.log"
        downlink_log = output_dir / "downlink.log"

        receiver_cmd = f"{self._sprout_bin}"
        inner_cmd = f"sh -c '{receiver_cmd} & sleep {duration_s}; kill %1'"

        cmd = [
            "mm-delay", str(delay_ms),
            mm_link,
            str(pair.uplink), str(pair.downlink),
            f"--uplink-log={uplink_log}",
            f"--downlink-log={downlink_log}",
            "--", "bash", "-c", inner_cmd,
        ]

        logger.info("Running Sprout: %s", " ".join(cmd))
        subprocess.run(cmd, capture_output=True, timeout=duration_s + 30)

        return ExperimentResult(
            algorithm="sprout",
            trace=trace_name,
            delay_ms=delay_ms,
            duration_s=duration_s,
            log_dir=output_dir,
        )
