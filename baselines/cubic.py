"""
TCP Cubic baseline wrapper via iperf3.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from baselines import BaselineAlgorithm, ExperimentResult
from network_emulation.cellular_traces import get_pair
from network_emulation.mahimahi import find_mm_link

logger = logging.getLogger(__name__)


class CubicBaseline(BaselineAlgorithm):

    @property
    def name(self) -> str:
        return "cubic"

    def run(self, trace_name, delay_ms, duration_s, output_dir):
        pair = get_pair(trace_name)
        if pair is None or not pair.exists():
            raise FileNotFoundError(f"Trace not found: {trace_name}")

        mm_link = find_mm_link()
        if not mm_link:
            raise FileNotFoundError("mm-link not found")

        output_dir.mkdir(parents=True, exist_ok=True)
        uplink_log = output_dir / "uplink.log"
        iperf_json = output_dir / "iperf3_result.json"

        inner_cmd = "iperf3 -s -1 -J"

        cmd = [
            "mm-delay", str(delay_ms),
            mm_link,
            str(pair.uplink), str(pair.downlink),
            f"--uplink-log={uplink_log}",
            "--", "bash", "-c", inner_cmd,
        ]

        logger.info("Running TCP Cubic (iperf3): %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=duration_s + 30
        )

        avg_tput = 0.0
        if result.stdout:
            try:
                iperf_data = json.loads(result.stdout)
                end = iperf_data.get("end", {})
                sum_sent = end.get("sum_sent", {})
                avg_tput = sum_sent.get("bits_per_second", 0) / 1e6

                with open(iperf_json, "w") as f:
                    json.dump(iperf_data, f, indent=2)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse iperf3 output")

        return ExperimentResult(
            algorithm="cubic",
            trace=trace_name,
            delay_ms=delay_ms,
            duration_s=duration_s,
            avg_throughput_mbps=avg_tput,
            log_dir=output_dir,
        )
