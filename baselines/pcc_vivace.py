"""
PCC Vivace baseline wrapper (Goyal et al., NSDI 2020).
Expects PCC-Uspace built from https://github.com/PCCproject/PCC-Uspace.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from baselines import BaselineAlgorithm, ExperimentResult
from network_emulation.cellular_traces import get_pair
from network_emulation.mahimahi import find_mm_link

logger = logging.getLogger(__name__)

PCC_DEFAULT_DIR = Path("third_party/pcc-uspace")


def find_pcc_binaries(search_dir: Path | None = None) -> tuple[Path | None, Path | None]:
    dirs = [search_dir, PCC_DEFAULT_DIR, Path.home() / "PCC-Uspace"]
    server = client = None
    for d in dirs:
        if d is None:
            continue
        s = d / "src" / "app" / "pccserver"
        c = d / "src" / "app" / "pccclient"
        if s.is_file():
            server = s
        if c.is_file():
            client = c
        if server and client:
            break
    return server, client


class VivaceBaseline(BaselineAlgorithm):

    def __init__(self, pcc_dir: Path | None = None):
        self._server, self._client = find_pcc_binaries(pcc_dir)

    @property
    def name(self) -> str:
        return "vivace"

    def run(self, trace_name, delay_ms, duration_s, output_dir):
        if not self._server or not self._client:
            raise FileNotFoundError("PCC-Uspace binaries not found")

        pair = get_pair(trace_name)
        if pair is None or not pair.exists():
            raise FileNotFoundError(f"Trace not found: {trace_name}")

        mm_link = find_mm_link()
        if not mm_link:
            raise FileNotFoundError("mm-link not found")

        output_dir.mkdir(parents=True, exist_ok=True)
        uplink_log = output_dir / "uplink.log"
        port = 9876

        server_cmd = f"{self._server} recv {port}"
        inner_cmd = f"sh -c '{server_cmd} & sleep {duration_s}; kill %1'"

        cmd = [
            "mm-delay", str(delay_ms),
            mm_link,
            str(pair.uplink), str(pair.downlink),
            f"--uplink-log={uplink_log}",
            "--", "bash", "-c", inner_cmd,
        ]

        logger.info("Running PCC Vivace: %s", " ".join(cmd))
        subprocess.run(cmd, capture_output=True, timeout=duration_s + 30)

        return ExperimentResult(
            algorithm="vivace",
            trace=trace_name,
            delay_ms=delay_ms,
            duration_s=duration_s,
            log_dir=output_dir,
        )
