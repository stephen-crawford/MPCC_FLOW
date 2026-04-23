"""
MPCC congestion control as a Portus/CCP datapath program.

Implements portus.AlgBase. On each ACK report from the kernel datapath,
the MPCC controller solves the CasADi NLP and outputs a new Cwnd.

Usage:
    python -m network_emulation.mpcc_cca --ipc netlink
"""

from __future__ import annotations

import argparse
import logging
import time

try:
    import pyportus as portus
    HAS_PORTUS = True
except ImportError:
    HAS_PORTUS = False

from planning.mpcc_controller import MPCCController, Observation

logger = logging.getLogger(__name__)

DATAPATH_PROGRAM = """\
(def (Report
    (volatile acked 0)
    (volatile sacked 0)
    (volatile loss 0)
    (volatile timeout false)
    (volatile rtt 0)
    (volatile inflight 0)
))
(when true
    (:= Report.inflight Flow.packets_in_flight)
    (:= Report.rtt Flow.rtt_sample_us)
    (:= Report.acked (+ Report.acked Ack.bytes_acked))
    (:= Report.sacked (+ Report.sacked Ack.packets_misordered))
    (:= Report.loss Ack.lost_pkts_sample)
    (:= Report.timeout Flow.was_timeout)
    (fallthrough)
)
(when (|| Report.timeout (> Report.loss 0))
    (report)
    (:= Micros 0)
)
(when (> Micros Flow.rtt_sample_us)
    (report)
    (:= Micros 0)
)
"""


class MPCCFlow:

    def __init__(self, datapath, datapath_info, config=None):
        self.datapath = datapath
        self.datapath_info = datapath_info
        self.mss = datapath_info.mss
        self.init_cwnd = float(self.mss * 10)

        cfg = config or {}
        solver_mode = cfg.get("solver_mode", "qp")
        self.controller = MPCCController(cfg, solver_type=solver_mode)
        self.last_report_time = time.monotonic()

        self.datapath.set_program("default", [("Cwnd", int(self.init_cwnd))])

    def on_report(self, r):
        now = time.monotonic()
        dt = now - self.last_report_time
        self.last_report_time = now

        rtt_s = r.rtt / 1e6
        tput = r.acked / max(dt, 0.001)
        queue_est = max(0, (rtt_s - self.controller.rtt_prop) * tput)

        obs = Observation(
            tput=tput,
            rtt=rtt_s,
            queue=queue_est,
            bw_est=tput / max(rtt_s / max(self.controller.rtt_prop, 0.001), 0.5),
            loss=r.loss,
        )

        send_rate = self.controller.step(obs)
        cwnd = max(int(send_rate * rtt_s), int(self.init_cwnd))
        self.datapath.update_field("Cwnd", cwnd)


if HAS_PORTUS:

    class MPCCAlgorithm(portus.AlgBase):

        def __init__(self, config=None):
            self.config = config

        def datapath_programs(self):
            return {"default": DATAPATH_PROGRAM}

        def new_flow(self, datapath, datapath_info):
            return MPCCFlow(datapath, datapath_info, self.config)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mpcc-cca")
    parser.add_argument("--ipc", default="netlink", choices=["netlink", "unix"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override planner horizon N.")
    parser.add_argument("--solver", default="qp", choices=["qp", "nlp"],
                        help="QP linearization (fast) or full NLP (reference).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not HAS_PORTUS:
        logger.error("pyportus not installed")
        return 1

    config: dict = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    if args.horizon is not None:
        config.setdefault("planner", {})["horizon"] = args.horizon
    config["solver_mode"] = args.solver

    alg = MPCCAlgorithm(config)
    portus.start(args.ipc, alg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
