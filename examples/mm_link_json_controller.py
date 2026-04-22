"""Example JSON-lines controller for real-Mahimahi scenario mode.

This is a protocol demo. In real use, replace the simulated measurement block
with code that drives your sender/receiver over the Mahimahi-emulated path and
reports measured throughput, RTT, queueing delay, and loss.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    capacity_mbps = 10.0
    base_rtt_ms = 40.0

    for line in sys.stdin:
        request = json.loads(line)
        rates = request.get("send_rates_mbps", [0.0])
        total_rate = float(sum(rates))
        overload = max(0.0, total_rate - capacity_mbps)
        throughput = min(total_rate, capacity_mbps)
        queue_delay = min(250.0, overload * 20.0)
        loss_fraction = overload / total_rate if total_rate > 0.0 else 0.0
        measurement = {
            "throughput_mbps": throughput,
            "rtt_ms": base_rtt_ms + queue_delay,
            "queue_delay_ms": queue_delay,
            "loss_fraction": loss_fraction,
            "capacity_mbps": capacity_mbps,
        }
        print(json.dumps(measurement), flush=True)


if __name__ == "__main__":
    main()
