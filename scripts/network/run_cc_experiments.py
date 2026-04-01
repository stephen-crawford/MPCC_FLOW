"""
Experiment orchestrator for MPCC congestion control evaluation.

Runs algorithms under mahimahi (mm-delay + mm-link) with real cellular traces.
MPCC runs as a CCP algorithm via Portus; baselines use their native binaries.

Usage:
    mpcc-cc run --traces Verizon-LTE-short --delays 25 50 --duration 60
    mpcc-cc list-traces
    mpcc-cc list-algorithms
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from baselines import ExperimentResult
from network_emulation.cellular_traces import list_cellular_pairs, get_pair
from network_emulation.mahimahi import find_mm_link, mm_link_from_pair, mm_graph_argv
from network_emulation.paths import mahimahi_root

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/cc_experiments")
MPCC_CCA_MODULE = "network_emulation.mpcc_cca"


def _find_mm_delay() -> str | None:
    import shutil
    w = shutil.which("mm-delay")
    if w:
        return w
    root = mahimahi_root()
    for rel in ("src/frontend/mm-delay", "src/frontend/.libs/mm-delay"):
        p = root / rel
        if p.is_file():
            return str(p)
    return None


def _parse_mm_link_log(log_path: Path, ms_per_bin: int = 500) -> list[tuple[float, float]]:
    if not log_path.is_file():
        return []
    throughput_ts = []
    try:
        with open(log_path) as f:
            lines = f.readlines()
        if len(lines) < 2:
            return []
        base_ms = None
        bin_bytes = 0
        bin_start = 0
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ts_ms = int(parts[0])
            nbytes = int(parts[1]) if len(parts) > 1 else 1500
            if base_ms is None:
                base_ms = ts_ms
                bin_start = ts_ms
            if ts_ms - bin_start >= ms_per_bin:
                duration_s = (ts_ms - bin_start) / 1000.0
                mbps = (bin_bytes * 8) / (duration_s * 1e6) if duration_s > 0 else 0
                throughput_ts.append(((bin_start - base_ms) / 1000.0, mbps))
                bin_bytes = 0
                bin_start = ts_ms
            bin_bytes += nbytes
    except Exception as e:
        logger.warning("Failed to parse mm-link log %s: %s", log_path, e)
    return throughput_ts


def run_mpcc(trace_name: str, delay_ms: int, duration_s: float,
             output_dir: Path, config_path: str | None = None) -> ExperimentResult:
    pair = get_pair(trace_name)
    if pair is None or not pair.exists():
        raise FileNotFoundError(f"Trace not found: {trace_name}")

    mm_link = find_mm_link()
    mm_delay = _find_mm_delay()
    if not mm_link:
        raise FileNotFoundError("mm-link not found. Build mahimahi first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    uplink_log = output_dir / "uplink.log"
    downlink_log = output_dir / "downlink.log"

    cca_cmd = ["python3", "-m", MPCC_CCA_MODULE, "--ipc", "netlink"]
    if config_path:
        cca_cmd.extend(["--config", config_path])

    iperf_server_cmd = "iperf3 -s -1 -D"
    cca_start = " ".join(cca_cmd)
    inner = f"bash -c '{iperf_server_cmd} && {cca_start} &' && sleep {duration_s}"

    cmd = []
    if mm_delay:
        cmd.extend([mm_delay, str(delay_ms)])
    cmd.extend([
        mm_link,
        str(pair.uplink), str(pair.downlink),
        f"--uplink-log={uplink_log}",
        f"--downlink-log={downlink_log}",
        "--", "bash", "-c", inner,
    ])

    logger.info("MPCC experiment: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, capture_output=True, timeout=duration_s + 60)
    except subprocess.TimeoutExpired:
        logger.warning("Experiment timed out")
    except FileNotFoundError as e:
        logger.error("Binary not found: %s", e)

    throughput_ts = _parse_mm_link_log(uplink_log)
    avg_tput = np.mean([t for _, t in throughput_ts]) if throughput_ts else 0.0

    return ExperimentResult(
        algorithm="mpcc",
        trace=trace_name,
        delay_ms=delay_ms,
        duration_s=duration_s,
        avg_throughput_mbps=avg_tput,
        throughput_ts=throughput_ts,
        log_dir=output_dir,
    )


def run_iperf_cubic(trace_name: str, delay_ms: int, duration_s: float,
                    output_dir: Path) -> ExperimentResult:
    pair = get_pair(trace_name)
    if pair is None or not pair.exists():
        raise FileNotFoundError(f"Trace not found: {trace_name}")

    mm_link = find_mm_link()
    mm_delay = _find_mm_delay()
    if not mm_link:
        raise FileNotFoundError("mm-link not found")

    output_dir.mkdir(parents=True, exist_ok=True)
    uplink_log = output_dir / "uplink.log"

    inner = f"iperf3 -s -1 -J --logfile {output_dir / 'iperf3.json'}"

    cmd = []
    if mm_delay:
        cmd.extend([mm_delay, str(delay_ms)])
    cmd.extend([
        mm_link,
        str(pair.uplink), str(pair.downlink),
        f"--uplink-log={uplink_log}",
        "--", "bash", "-c", inner,
    ])

    logger.info("Cubic experiment: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, capture_output=True, timeout=duration_s + 60)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Experiment issue: %s", e)

    throughput_ts = _parse_mm_link_log(uplink_log)
    avg_tput = np.mean([t for _, t in throughput_ts]) if throughput_ts else 0.0

    return ExperimentResult(
        algorithm="cubic",
        trace=trace_name,
        delay_ms=delay_ms,
        duration_s=duration_s,
        avg_throughput_mbps=avg_tput,
        throughput_ts=throughput_ts,
        log_dir=output_dir,
    )


def run_with_mm_graph(output_dir: Path, ms_per_bin: int = 500):
    uplink_log = output_dir / "uplink.log"
    if not uplink_log.is_file():
        return
    try:
        argv = mm_graph_argv(uplink_log, ms_per_bin, no_display=True)
        subprocess.run(argv, capture_output=True, timeout=30)
    except Exception as e:
        logger.debug("mm-graph failed: %s", e)


ALGORITHMS = {
    "mpcc": run_mpcc,
    "cubic": run_iperf_cubic,
}

DEFAULT_DELAYS = [25, 50, 100]
DEFAULT_DURATION = 60


def run_experiment_matrix(
    algorithms: list[str],
    traces: list[str],
    delays: list[int],
    duration_s: float,
    output_base: Path,
    n_trials: int = 3,
) -> list[ExperimentResult]:
    results = []
    total = len(algorithms) * len(traces) * len(delays) * n_trials
    done = 0

    for alg_name in algorithms:
        if alg_name not in ALGORITHMS:
            logger.warning("Unknown algorithm: %s", alg_name)
            continue

        run_fn = ALGORITHMS[alg_name]

        for trace in traces:
            for delay in delays:
                for trial in range(n_trials):
                    done += 1
                    run_dir = output_base / alg_name / trace / f"delay{delay}" / f"trial{trial}"
                    logger.info("[%d/%d] %s / %s / %dms / trial %d",
                                done, total, alg_name, trace, delay, trial)

                    try:
                        result = run_fn(trace, delay, duration_s, run_dir)
                        results.append(result)
                        run_with_mm_graph(run_dir)

                        run_dir.mkdir(parents=True, exist_ok=True)
                        with open(run_dir / "result.json", "w") as f:
                            json.dump(asdict(result), f, indent=2, default=str)
                    except Exception as e:
                        logger.error("Experiment failed: %s", e)

    summary_path = output_base / "summary.json"
    output_base.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    logger.info("Results saved to %s (%d experiments)", summary_path, len(results))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mpcc-cc", description="MPCC congestion control experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-traces")
    sp.set_defaults(func=_cmd_list_traces)

    sp = sub.add_parser("list-algorithms")
    sp.set_defaults(func=_cmd_list_algorithms)

    sp = sub.add_parser("run")
    sp.add_argument("--algorithms", "-a", nargs="+", default=["mpcc"])
    sp.add_argument("--traces", "-t", nargs="+", required=True)
    sp.add_argument("--delays", "-d", nargs="+", type=int, default=DEFAULT_DELAYS)
    sp.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    sp.add_argument("--trials", "-n", type=int, default=3)
    sp.add_argument("--output", "-o", type=Path, default=RESULTS_DIR)
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("check")
    sp.set_defaults(func=_cmd_check)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


def _cmd_list_traces(args) -> int:
    pairs = list_cellular_pairs()
    if not pairs:
        print("No traces found. Set MAHIMAHI_ROOT or check third_party/mahimahi.")
        return 1
    for pair in pairs:
        print(f"  {pair.name}\t{pair.carrier}\t{pair.kind}")
    return 0


def _cmd_list_algorithms(args) -> int:
    for name in ALGORITHMS:
        print(f"  {name}")
    return 0


def _cmd_run(args) -> int:
    results = run_experiment_matrix(
        algorithms=args.algorithms,
        traces=args.traces,
        delays=args.delays,
        duration_s=args.duration,
        output_base=args.output,
        n_trials=args.trials,
    )
    print(f"\nCompleted {len(results)} experiments. Results in {args.output}/")
    return 0


def _cmd_check(args) -> int:
    mm_link = find_mm_link()
    mm_delay = _find_mm_delay()
    pairs = list_cellular_pairs()
    try:
        import pyportus
        portus_ok = True
    except ImportError:
        portus_ok = False

    print(f"mm-link:    {mm_link or 'NOT FOUND'}")
    print(f"mm-delay:   {mm_delay or 'NOT FOUND'}")
    print(f"traces:     {len(pairs)} pairs found")
    print(f"pyportus:   {'OK' if portus_ok else 'NOT INSTALLED'}")

    if not mm_link:
        print("\nBuild mahimahi: cd /path/to/mahimahi && ./autogen.sh && ./configure && make && sudo make install")
    if not portus_ok:
        print("Build pyportus: cd /path/to/portus/python && pip install maturin && maturin develop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
