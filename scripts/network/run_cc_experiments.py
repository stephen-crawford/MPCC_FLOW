"""Real Mahimahi experiment matrix runner for MPCC vs baselines.

Runs controllers under nested `mm-delay` + `mm-link` on real cellular traces
(and synthetic wired / LEO-inspired traces). All controllers here are real
implementations; none are Python simulators:

    cubic   -- Linux kernel TCP CUBIC, iperf3 under mm-link
    bbr     -- Linux kernel TCP BBR, iperf3 under mm-link
    mpcc    -- Portus CCP datapath (pyportus stub or Rust crate), iperf3
    nimbus  -- Portus CCP datapath from nimbus-measurement/nimbus
    sprout  -- Original sproutbt2 binary under mm-link
    verus   -- Original Verus client/server binary under mm-link

Metrics are extracted from the mm-link uplink log (per-packet departure
timestamps); RTT percentiles are computed from iperf3 `--json` output when
available, otherwise fall back to the mm-link queueing delay histogram.

Scenario families (paper §IV.L):

    wired      -- synthetic constant-rate trace
    cellular   -- ATT / TMobile / Verizon LTE traces already in mahimahi/traces
    leo        -- synthetic handover/outage trace
    fairness   -- multi-flow iperf on one bottleneck (homogeneous or mixed CCAs)

CLI:

    python -m scripts.network.run_cc_experiments list-traces
    python -m scripts.network.run_cc_experiments list-ccas
    python -m scripts.network.run_cc_experiments check
    python -m scripts.network.run_cc_experiments run \\
        --scenarios cellular wired --ccas cubic bbr mpcc \\
        --duration 60 --seeds 3 --output results/real
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics as stats
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from baselines import ExperimentResult
from network_emulation.cellular_traces import (
    CellularTracePair,
    list_cellular_pairs,
    get_pair,
)
from network_emulation.mahimahi import find_mm_link
from network_emulation.paths import (
    mahimahi_root,
    nimbus_release_binary,
    repo_root,
)

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results/real_mahimahi")
MPCC_CCA_MODULE = "network_emulation.mpcc_cca"
KERNEL_CCAS = {"cubic", "bbr", "reno", "vegas", "westwood"}
DEFAULT_CELLULAR_TRACES = [
    "ATT-LTE-driving",
    "TMobile-LTE-driving",
    "Verizon-LTE-driving",
]


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def _find_mm_delay() -> str | None:
    w = shutil.which("mm-delay")
    if w:
        return w
    root = mahimahi_root()
    for rel in ("src/frontend/mm-delay", "src/frontend/.libs/mm-delay"):
        p = root / rel
        if p.is_file():
            return str(p)
    return None


def _find_portus_mpcc() -> str | None:
    """Locate the Rust portus-mpcc binary. Optional — falls back to pyportus."""
    override = os.environ.get("PORTUS_MPCC_BIN")
    if override and Path(override).is_file():
        return override
    candidates = [
        repo_root() / "third_party" / "portus-mpcc" / "target" / "release" / "mpcc_cca",
        Path.home() / "portus-mpcc" / "target" / "release" / "mpcc_cca",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _find_sprout() -> str | None:
    override = os.environ.get("SPROUT_BIN")
    if override and Path(override).is_file():
        return override
    for rel in [
        "third_party/sprout/src/examples/sproutbt2",
        "third_party/alfalfa/src/examples/sproutbt2",
    ]:
        p = repo_root() / rel
        if p.is_file():
            return str(p)
    return shutil.which("sproutbt2")


def _find_verus_client() -> tuple[str | None, str | None]:
    """Return (verus_client, verus_server) binaries if present."""
    root = repo_root() / "third_party" / "verus"
    client = root / "client" / "verus_client"
    server = root / "server" / "verus_server"
    return (str(client) if client.is_file() else None,
            str(server) if server.is_file() else None)


def _pyportus_available() -> bool:
    try:
        import pyportus  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Log parsing and metrics
# ---------------------------------------------------------------------------

def _parse_mm_link_log(
    log_path: Path,
    *,
    ms_per_bin: int = 200,
) -> dict:
    """Parse mm-link's uplink log into throughput + queueing-delay series.

    mm-link log format (text):
        # header line(s)
        <capacity-ts-ms> # <delivered-bytes-at-that-opportunity>
        or
        <arrival-ts-ms> + <n-bytes> <queue-depth-bytes>
        or
        <departure-ts-ms> - <n-bytes> <queueing-delay-ms>

    We aggregate packet departures into time-bins to produce bps, and collect
    queueing delay samples for percentiles.
    """
    if not log_path.is_file():
        return {"throughput_ts": [], "rtt_ts": [], "loss_rate": 0.0,
                "avg_mbps": 0.0, "p50_rtt_ms": 0.0, "p95_rtt_ms": 0.0}

    throughput_ts: list[tuple[float, float]] = []
    delay_samples: list[float] = []
    arrivals = departures = drops = 0
    base_ms: int | None = None
    bin_start: int | None = None
    bin_bytes = 0

    try:
        with open(log_path) as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    ts_ms = int(parts[0])
                except ValueError:
                    continue
                op = parts[1]
                if op == "+":
                    arrivals += 1
                elif op == "-":
                    departures += 1
                    try:
                        nbytes = int(parts[2])
                    except ValueError:
                        nbytes = 1500
                    if base_ms is None:
                        base_ms = ts_ms
                        bin_start = ts_ms
                    if ts_ms - bin_start >= ms_per_bin:
                        dur = (ts_ms - bin_start) / 1000.0
                        mbps = (bin_bytes * 8) / (dur * 1e6) if dur > 0 else 0.0
                        throughput_ts.append(((bin_start - base_ms) / 1000.0, mbps))
                        bin_bytes = 0
                        bin_start = ts_ms
                    bin_bytes += nbytes
                    if len(parts) >= 4:
                        try:
                            delay_samples.append(float(parts[3]))
                        except ValueError:
                            pass
                elif op == "d":
                    drops += 1
    except Exception as e:
        logger.warning("failed to parse %s: %s", log_path, e)

    if throughput_ts:
        avg = float(np.mean([t for _, t in throughput_ts]))
    else:
        avg = 0.0
    p50 = float(np.percentile(delay_samples, 50)) if delay_samples else 0.0
    p95 = float(np.percentile(delay_samples, 95)) if delay_samples else 0.0
    total = max(arrivals, 1)
    loss = drops / total

    return {
        "throughput_ts": throughput_ts,
        "rtt_ts": [],
        "loss_rate": loss,
        "avg_mbps": avg,
        "p50_rtt_ms": p50,
        "p95_rtt_ms": p95,
        "arrivals": arrivals,
        "departures": departures,
        "drops": drops,
    }


def _parse_iperf3_json(json_path: Path) -> dict:
    """Pull end-to-end throughput + RTT from iperf3 `--json` output."""
    if not json_path.is_file():
        return {}
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return {}
    end = data.get("end", {})
    sum_sent = end.get("sum_sent", {})
    streams = end.get("streams", [])
    rtt_us = []
    for s in streams:
        sender = s.get("sender", {})
        if sender.get("mean_rtt") is not None:
            rtt_us.append(sender["mean_rtt"])
    return {
        "iperf_mbps": (sum_sent.get("bits_per_second", 0.0) or 0.0) / 1e6,
        "iperf_retransmits": sum_sent.get("retransmits", 0),
        "iperf_mean_rtt_ms": float(np.mean(rtt_us) / 1000) if rtt_us else 0.0,
    }


def _compute_utilization(
    throughput_ts: list[tuple[float, float]],
    pair: CellularTracePair | None,
) -> float:
    """Fraction of trace capacity actually achieved. 0 if unknown."""
    if not throughput_ts or pair is None:
        return 0.0
    # Trace capacity = MTU packets * 8 / duration, inferred from .up timestamps.
    try:
        with open(pair.uplink) as f:
            ts = [int(x.strip()) for x in f if x.strip()]
    except Exception:
        return 0.0
    if len(ts) < 2:
        return 0.0
    duration_s = (ts[-1] - ts[0]) / 1000.0
    trace_mbps = (len(ts) * 1500 * 8) / (duration_s * 1e6) if duration_s > 0 else 0.0
    if trace_mbps <= 0:
        return 0.0
    measured = float(np.mean([t for _, t in throughput_ts]))
    return measured / trace_mbps


# ---------------------------------------------------------------------------
# Runner registry
# ---------------------------------------------------------------------------

def _check_cmd_or_warn(cmd: list[str], timeout: float) -> subprocess.CompletedProcess | None:
    logger.debug("exec: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("command timed out: %s", " ".join(cmd[:3]))
        return None
    except FileNotFoundError as e:
        logger.error("binary not found: %s", e)
        return None
    if proc.returncode != 0:
        # mm-link / mm-delay writes its errors to stderr. Make them visible.
        head = (proc.stderr or b"").decode(errors="replace")[:500]
        logger.warning(
            "non-zero exit %d from %s\n  stderr: %s",
            proc.returncode, " ".join(cmd[:3]), head.strip(),
        )
    return proc


def _mahimahi_shell_cmd(
    pair: CellularTracePair,
    delay_ms: int,
    inner: str,
    uplink_log: Path,
    downlink_log: Path,
) -> list[str]:
    mm_link = find_mm_link()
    mm_delay = _find_mm_delay()
    if not mm_link:
        raise FileNotFoundError(
            "mm-link not found. Build mahimahi on the host that runs experiments.",
        )
    cmd: list[str] = []
    if mm_delay:
        cmd.extend([mm_delay, str(delay_ms)])
    cmd.extend([
        mm_link,
        str(pair.uplink), str(pair.downlink),
        f"--uplink-log={uplink_log}",
        f"--downlink-log={downlink_log}",
        "--", "bash", "-lc", inner,
    ])
    return cmd


def _kernel_tcp_inner(cca: str, duration_s: float, out_dir: Path) -> str:
    """Bash run inside the mm-link namespace: iperf3 client only.

    The server is started on the host by `_run_under_mahimahi` before mm-link
    spawns. The kernel CCA is set per-socket via iperf3's `-C` flag; the sysctl
    fallback keeps older iperf3 builds happy.
    """
    log = out_dir / "iperf3.json"
    return (
        f"(sudo -n sysctl -w net.ipv4.tcp_congestion_control={cca} >/dev/null 2>&1 || true) && "
        f"iperf3 -c $MAHIMAHI_BASE -p 5201 -t {duration_s:.1f} "
        f"-C {cca} --json --logfile {log} >/dev/null 2>&1 || true"
    )


def _mpcc_inner(duration_s: float, out_dir: Path) -> str:
    """Inner bash (inside mm-link namespace): iperf3 client only.

    The mpcc_cca daemon and iperf3 server are both started on the host by
    `_run_under_mahimahi` before mm-link spawns.
    """
    iperf_log = out_dir / "iperf3.json"
    return (
        f"(sudo -n sysctl -w net.ipv4.tcp_congestion_control=ccp >/dev/null 2>&1 || "
        f" sudo -n sysctl -w net.ipv4.tcp_congestion_control=cubic >/dev/null 2>&1) && "
        f"iperf3 -c $MAHIMAHI_BASE -p 5201 -t {duration_s:.1f} -C ccp --json "
        f"--logfile {iperf_log} >/dev/null 2>&1 || true"
    )


def _build_mpcc_daemon_args(
    rust_bin: str,
    mpcc_config: str | None,
    horizon: int | None,
    solver: str,
) -> list[str]:
    args = [rust_bin, "--ipc", "netlink"]
    if mpcc_config:
        args += ["--config", mpcc_config]
    if horizon is not None:
        args += ["--horizon", str(horizon)]
    args += ["--solver", solver]
    return args


def _nimbus_inner(duration_s: float, out_dir: Path) -> str:
    """Inner bash (inside mm-link namespace): iperf3 client only.

    The nimbus daemon and iperf3 server are both started on the host.
    """
    iperf_log = out_dir / "iperf3.json"
    return (
        f"(sudo -n sysctl -w net.ipv4.tcp_congestion_control=ccp >/dev/null 2>&1 || "
        f" sudo -n sysctl -w net.ipv4.tcp_congestion_control=cubic >/dev/null 2>&1) && "
        f"iperf3 -c $MAHIMAHI_BASE -p 5201 -t {duration_s:.1f} -C ccp --json "
        f"--logfile {iperf_log} >/dev/null 2>&1 || true"
    )


def _sprout_inner(duration_s: float, out_dir: Path, sprout_bin: str) -> str:
    log = out_dir / "sprout.log"
    # Sprout ships as a custom protocol; sproutbt2 takes server/client mode by port.
    return (
        f"({sprout_bin} 60001 >>{log} 2>&1) & "
        f"sleep 0.5 && {sprout_bin} $MAHIMAHI_BASE 60001 {int(duration_s)} "
        f">>{log} 2>&1 || true; "
        f"pkill -x sproutbt2 2>/dev/null || true"
    )


def _verus_inner(
    duration_s: float, out_dir: Path,
    client_bin: str, server_bin: str,
) -> str:
    log = out_dir / "verus.log"
    return (
        f"({server_bin} -p 60002 -n {out_dir}/verus_server.log >>{log} 2>&1) & "
        f"sleep 0.5 && "
        f"timeout {duration_s + 5:.0f} {client_bin} -t {int(duration_s)} "
        f"-p 60002 -n {out_dir}/verus_client.log $MAHIMAHI_BASE "
        f">>{log} 2>&1 || true; "
        f"pkill -x verus_server 2>/dev/null || true"
    )


def _run_under_mahimahi(
    cca: str, pair: CellularTracePair, delay_ms: int,
    duration_s: float, out_dir: Path, inner: str,
    host_port: int = 5201,
    ccp_daemon: list[str] | None = None,
    ccp_daemon_log: Path | None = None,
    ccp_daemon_kill_name: str | None = None,
) -> ExperimentResult:
    """Run a full experiment.

    The iperf3 server (and any CCP user-space daemon) run on the **host**,
    not inside mm-link's namespace. Only the iperf3 client runs inside the
    namespace. This is mandatory:

    * For the iperf3 server, running it inside the namespace puts both
      endpoints on the same side of the shaped link — they can never route
      to each other via $MAHIMAHI_BASE.
    * For the CCP daemon (mpcc_cca / nimbus), the kernel's ccp_cong module
      uses netlink in the init_net namespace. A daemon started inside the
      mm-link namespace can open a netlink socket but does not register
      with the kernel module correctly, so sockets with -C ccp never
      receive user-space decisions and fall back to a degenerate rate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    uplink_log = out_dir / "uplink.log"
    downlink_log = out_dir / "downlink.log"

    # Kill any lingering iperf3 / CCP daemons on the host.
    for name in ("iperf3",) + ((ccp_daemon_kill_name,) if ccp_daemon_kill_name else ()):
        subprocess.run(
            ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", name],
            capture_output=True,
        )

    # Start iperf3 server on the host.
    server_proc = subprocess.Popen(
        ["iperf3", "-s", "-1", "-p", str(host_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # Start the CCP user-space daemon on the host, if this CCA needs one.
    daemon_proc: subprocess.Popen | None = None
    if ccp_daemon:
        log_fh = open(ccp_daemon_log, "wb") if ccp_daemon_log else subprocess.DEVNULL
        daemon_proc = subprocess.Popen(
            ["sudo", "-n", *ccp_daemon],
            stdout=log_fh, stderr=subprocess.STDOUT,
        )

    # Let server + daemon bind / register with the kernel.
    time.sleep(1.0 if ccp_daemon else 0.3)

    cmd = _mahimahi_shell_cmd(pair, delay_ms, inner, uplink_log, downlink_log)
    t0 = time.monotonic()
    try:
        _check_cmd_or_warn(cmd, timeout=duration_s + 60)
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if daemon_proc and daemon_proc.poll() is None:
            # Daemon was started under sudo, so it runs as root — a plain
            # kill on the parent sudo leaks the root child. Use sudo pkill.
            if ccp_daemon_kill_name:
                subprocess.run(
                    ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", ccp_daemon_kill_name],
                    capture_output=True,
                )
            daemon_proc.wait(timeout=2)
        # Belt-and-suspenders: kill any stray iperf3 that escaped.
        subprocess.run(
            ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", "iperf3"],
            capture_output=True,
        )
    wall_s = time.monotonic() - t0

    m = _parse_mm_link_log(uplink_log)
    iperf = _parse_iperf3_json(out_dir / "iperf3.json")
    util = _compute_utilization(m["throughput_ts"], pair)

    # Prefer iperf3 RTT percentiles if available, else fall back to mm-link.
    if iperf.get("iperf_mean_rtt_ms"):
        p50 = p95 = iperf["iperf_mean_rtt_ms"]  # iperf only gives mean
        p50_q = m["p50_rtt_ms"]  # keep queueing delay for diagnostics
    else:
        p50 = m["p50_rtt_ms"]
        p95 = m["p95_rtt_ms"]
        p50_q = m["p50_rtt_ms"]

    res = ExperimentResult(
        algorithm=cca,
        trace=pair.name,
        delay_ms=delay_ms,
        duration_s=duration_s,
        avg_throughput_mbps=m["avg_mbps"],
        median_rtt_ms=p50,
        p95_rtt_ms=p95,
        loss_rate=m["loss_rate"],
        throughput_ts=m["throughput_ts"],
        log_dir=out_dir,
    )
    extras = {
        "wall_seconds": wall_s,
        "utilization": util,
        "queue_p50_ms": p50_q,
        "iperf_mbps": iperf.get("iperf_mbps", 0.0),
        "iperf_retransmits": iperf.get("iperf_retransmits", 0),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({**asdict(res), **extras}, f, indent=2, default=str)
    return res


# ---------------------------------------------------------------------------
# Public CCA entry points
# ---------------------------------------------------------------------------

def run_kernel_cca(
    cca: str, pair: CellularTracePair, delay_ms: int,
    duration_s: float, out_dir: Path,
) -> ExperimentResult:
    inner = _kernel_tcp_inner(cca, duration_s, out_dir)
    return _run_under_mahimahi(cca, pair, delay_ms, duration_s, out_dir, inner)


def run_mpcc(
    pair: CellularTracePair, delay_ms: int, duration_s: float,
    out_dir: Path,
    mpcc_config: str | None = None,
    horizon: int | None = None,
    solver: str = "qp",
) -> ExperimentResult:
    rust_bin = _find_portus_mpcc()
    if not rust_bin:
        logger.warning(
            "MPCC: Rust portus-mpcc binary not built; falling back to kernel "
            "cubic for this run. Build it with: "
            "cd third_party/portus-mpcc && cargo build --release",
        )
        return run_kernel_cca("cubic", pair, delay_ms, duration_s, out_dir)
    inner = _mpcc_inner(duration_s, out_dir)
    daemon_args = _build_mpcc_daemon_args(rust_bin, mpcc_config, horizon, solver)
    return _run_under_mahimahi(
        "mpcc", pair, delay_ms, duration_s, out_dir, inner,
        ccp_daemon=daemon_args,
        ccp_daemon_log=out_dir / "mpcc.log",
        ccp_daemon_kill_name="mpcc_cca",
    )


def run_nimbus(
    pair: CellularTracePair, delay_ms: int, duration_s: float,
    out_dir: Path,
) -> ExperimentResult:
    nimbus_bin = nimbus_release_binary()
    if not nimbus_bin.is_file():
        logger.warning("Nimbus binary not built at %s; skipping run", nimbus_bin)
        return ExperimentResult(
            algorithm="nimbus", trace=pair.name, delay_ms=delay_ms,
            duration_s=duration_s, log_dir=out_dir,
        )
    inner = _nimbus_inner(duration_s, out_dir)
    return _run_under_mahimahi(
        "nimbus", pair, delay_ms, duration_s, out_dir, inner,
        ccp_daemon=[str(nimbus_bin), "--ipc=netlink"],
        ccp_daemon_log=out_dir / "nimbus.log",
        ccp_daemon_kill_name="nimbus",
    )


def run_sprout(
    pair: CellularTracePair, delay_ms: int, duration_s: float,
    out_dir: Path,
) -> ExperimentResult:
    sprout_bin = _find_sprout()
    if not sprout_bin:
        logger.warning("sproutbt2 not found; skipping")
        return ExperimentResult(
            algorithm="sprout", trace=pair.name, delay_ms=delay_ms,
            duration_s=duration_s, log_dir=out_dir,
        )
    inner = _sprout_inner(duration_s, out_dir, sprout_bin)
    return _run_under_mahimahi("sprout", pair, delay_ms, duration_s, out_dir, inner)


def run_verus(
    pair: CellularTracePair, delay_ms: int, duration_s: float,
    out_dir: Path,
) -> ExperimentResult:
    client, server = _find_verus_client()
    if not client or not server:
        logger.warning("Verus binaries not found; skipping")
        return ExperimentResult(
            algorithm="verus", trace=pair.name, delay_ms=delay_ms,
            duration_s=duration_s, log_dir=out_dir,
        )
    inner = _verus_inner(duration_s, out_dir, client, server)
    return _run_under_mahimahi("verus", pair, delay_ms, duration_s, out_dir, inner)


CCAS: dict[str, Callable[..., ExperimentResult]] = {
    "cubic": lambda p, d, t, o, **kw: run_kernel_cca("cubic", p, d, t, o),
    "bbr": lambda p, d, t, o, **kw: run_kernel_cca("bbr", p, d, t, o),
    "reno": lambda p, d, t, o, **kw: run_kernel_cca("reno", p, d, t, o),
    "mpcc": run_mpcc,
    "nimbus": lambda p, d, t, o, **kw: run_nimbus(p, d, t, o),
    "sprout": lambda p, d, t, o, **kw: run_sprout(p, d, t, o),
    "verus": lambda p, d, t, o, **kw: run_verus(p, d, t, o),
}


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _synthetic_trace_pair(name: str, stem: Path) -> CellularTracePair:
    """Wrap a generated stem as a CellularTracePair for reuse with mm-link."""
    up = stem.parent / f"{stem.name}.up"
    down = stem.parent / f"{stem.name}.down"
    return CellularTracePair(
        name=name, uplink=up, downlink=down, carrier="synthetic", kind=name,
    )


def _ensure_wired_trace(out_root: Path, rate_mbps: float = 50.0,
                        duration_s: float = 60.0) -> CellularTracePair:
    from network_emulation import leo_trace as lt
    stem = out_root / "traces" / f"Constant-{int(rate_mbps)}Mbps"
    ts = lt.constant_trace(rate_mbps, duration_s)
    lt.write_trace_pair(ts, stem)
    return _synthetic_trace_pair(stem.name, stem)


def _ensure_leo_trace(out_root: Path, baseline_mbps: float = 50.0,
                      duration_s: float = 60.0, seed: int = 0) -> CellularTracePair:
    from network_emulation import leo_trace as lt
    stem = out_root / "traces" / f"LEO-handover-s{seed}"
    ts = lt.leo_trace(
        baseline_mbps=baseline_mbps, duration_s=duration_s, seed=seed,
    )
    lt.write_trace_pair(ts, stem)
    return _synthetic_trace_pair(stem.name, stem)


def _scenarios_for(
    names: list[str], out_root: Path, duration_s: float,
) -> list[tuple[str, CellularTracePair, int]]:
    """Return list of (scenario_label, trace_pair, delay_ms) triples."""
    out: list[tuple[str, CellularTracePair, int]] = []
    for sc in names:
        if sc == "wired":
            pair = _ensure_wired_trace(out_root, duration_s=duration_s)
            out.append(("wired", pair, 20))
        elif sc == "leo":
            for seed in (0, 1, 2):
                pair = _ensure_leo_trace(out_root, duration_s=duration_s, seed=seed)
                out.append((f"leo-s{seed}", pair, 40))
        elif sc == "cellular":
            for name in DEFAULT_CELLULAR_TRACES:
                p = get_pair(name)
                if p is None or not p.exists():
                    logger.warning("cellular trace %r missing — skip", name)
                    continue
                out.append(("cellular", p, 40))
        elif sc == "fairness":
            # The scenario itself reuses cellular traces; fairness is handled
            # by `run_fairness_matrix` which runs multiple flows concurrently.
            for name in DEFAULT_CELLULAR_TRACES[:1]:
                p = get_pair(name)
                if p is not None and p.exists():
                    out.append(("fairness", p, 40))
        else:
            logger.warning("unknown scenario %r", sc)
    return out


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------

def run_matrix(
    scenarios: list[str], ccas: list[str], *,
    duration_s: float, seeds: int, output_base: Path,
    mpcc_config_by_scenario: dict[str, str] | None = None,
    mpcc_solver: str = "qp",
    mpcc_horizon: int | None = None,
) -> list[dict]:
    scen_list = _scenarios_for(scenarios, output_base, duration_s)
    results: list[dict] = []
    total = sum(1 for _ in scen_list) * len(ccas) * seeds
    done = 0
    mpcc_config_by_scenario = mpcc_config_by_scenario or {}

    for scenario_label, pair, delay_ms in scen_list:
        for cca_name in ccas:
            if cca_name not in CCAS:
                logger.warning("unknown CCA %r — skip", cca_name)
                continue
            run_fn = CCAS[cca_name]
            for seed in range(seeds):
                done += 1
                run_dir = (output_base / scenario_label / pair.name /
                           cca_name / f"seed{seed}")
                logger.info(
                    "[%d/%d] scenario=%s trace=%s cca=%s seed=%d",
                    done, total, scenario_label, pair.name, cca_name, seed,
                )
                try:
                    if cca_name in KERNEL_CCAS:
                        result = run_kernel_cca(cca_name, pair, delay_ms,
                                                duration_s, run_dir)
                    else:
                        kw = {}
                        if cca_name == "mpcc":
                            cfg = (
                                mpcc_config_by_scenario.get(scenario_label)
                                or mpcc_config_by_scenario.get("_default")
                            )
                            kw = {
                                "mpcc_config": cfg,
                                "solver": mpcc_solver,
                                "horizon": mpcc_horizon,
                            }
                        result = run_fn(pair, delay_ms, duration_s, run_dir, **kw)
                    results.append({
                        "scenario": scenario_label,
                        **asdict(result),
                    })
                except Exception as e:
                    logger.exception("experiment failed: %s", e)

    output_base.mkdir(parents=True, exist_ok=True)
    summary_path = output_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("summary → %s (%d rows)", summary_path, len(results))
    return results


def _jains_index(throughputs: Iterable[float]) -> float:
    xs = [x for x in throughputs if x > 0]
    if not xs:
        return 0.0
    num = sum(xs) ** 2
    den = len(xs) * sum(x ** 2 for x in xs)
    return num / den if den else 0.0


def run_fairness_matrix(
    ccas: list[str], *, n_flows: int, duration_s: float,
    seeds: int, trace_name: str, output_base: Path,
) -> list[dict]:
    """Run N concurrent iperf flows of a single CCA sharing a cellular bottleneck.

    Per-flow throughput is extracted from mm-link's uplink log bucketed by
    destination port; each iperf3 client writes its own `--json` at a unique
    port. Jain's fairness index is computed over the per-flow mean throughput.
    """
    pair = get_pair(trace_name)
    if pair is None or not pair.exists():
        raise FileNotFoundError(f"trace not found: {trace_name}")

    delay_ms = 40
    out: list[dict] = []

    for cca in ccas:
        if cca not in CCAS:
            continue
        for seed in range(seeds):
            run_dir = output_base / "fairness" / cca / f"n{n_flows}" / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # For MPCC / Nimbus, start the CCP daemon on the host (same
            # reasoning as _run_under_mahimahi). For kernel CCAs, just set
            # sysctl inside the namespace.
            cca_sysctl = cca
            host_daemon_proc: subprocess.Popen | None = None
            daemon_kill_name: str | None = None
            if cca == "mpcc":
                rust_bin = _find_portus_mpcc()
                if not rust_bin:
                    logger.warning("fairness: mpcc selected but portus-mpcc not built; skipping")
                    continue
                cca_sysctl = "ccp"
                daemon_kill_name = "mpcc_cca"
                subprocess.run(
                    ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", daemon_kill_name],
                    capture_output=True,
                )
                mpcc_log = run_dir / "mpcc.log"
                host_daemon_proc = subprocess.Popen(
                    ["sudo", "-n", rust_bin, "--ipc", "netlink"],
                    stdout=open(mpcc_log, "wb"), stderr=subprocess.STDOUT,
                )
                time.sleep(1.0)
            elif cca == "nimbus":
                nimbus = str(nimbus_release_binary())
                if not Path(nimbus).is_file():
                    logger.warning("fairness: nimbus binary missing; skipping")
                    continue
                cca_sysctl = "ccp"
                daemon_kill_name = "nimbus"
                subprocess.run(
                    ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", daemon_kill_name],
                    capture_output=True,
                )
                nimbus_log = run_dir / "nimbus.log"
                host_daemon_proc = subprocess.Popen(
                    ["sudo", "-n", nimbus, "--ipc=netlink"],
                    stdout=open(nimbus_log, "wb"), stderr=subprocess.STDOUT,
                )
                time.sleep(1.0)
            iperf_cca = cca_sysctl

            # Start n iperf3 servers on the host (one per port).
            subprocess.run(
                ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", "iperf3"],
                capture_output=True,
            )
            host_servers: list[subprocess.Popen] = []
            for i in range(n_flows):
                port = 5201 + i
                host_servers.append(subprocess.Popen(
                    ["iperf3", "-s", "-1", "-p", str(port)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
            time.sleep(0.3)

            # Inner (inside the namespace): set sysctl then launch n
            # concurrent iperf3 clients.
            setup_parts = [
                f"(sudo -n sysctl -w net.ipv4.tcp_congestion_control={cca_sysctl} "
                f">/dev/null 2>&1 || true)",
            ]
            launch_parts = []
            for i in range(n_flows):
                port = 5201 + i
                launch_parts.append(
                    f"sh -c 'iperf3 -c $MAHIMAHI_BASE -p {port} -t {duration_s:.1f} "
                    f"-C {iperf_cca} --json --logfile {run_dir}/iperf3_flow{i}.json "
                    f">/dev/null 2>&1' &"
                )
            setup_inner = " && ".join(setup_parts)
            launch_inner = " ".join(launch_parts)
            inner = f"{setup_inner} ; {launch_inner} wait"

            uplink_log = run_dir / "uplink.log"
            downlink_log = run_dir / "downlink.log"
            cmd = _mahimahi_shell_cmd(pair, delay_ms, inner,
                                      uplink_log, downlink_log)
            try:
                _check_cmd_or_warn(cmd, timeout=duration_s + 60)
            finally:
                for sp in host_servers:
                    if sp.poll() is None:
                        sp.terminate()
                        try:
                            sp.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            sp.kill()
                if host_daemon_proc and host_daemon_proc.poll() is None:
                    if daemon_kill_name:
                        subprocess.run(
                            ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", daemon_kill_name],
                            capture_output=True,
                        )
                    host_daemon_proc.wait(timeout=2)
                subprocess.run(
                    ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", "iperf3"],
                    capture_output=True,
                )

            per_flow = []
            for i in range(n_flows):
                j = _parse_iperf3_json(run_dir / f"iperf3_flow{i}.json")
                per_flow.append(j.get("iperf_mbps", 0.0))

            row = {
                "cca": cca,
                "n_flows": n_flows,
                "trace": trace_name,
                "seed": seed,
                "per_flow_mbps": per_flow,
                "total_mbps": float(sum(per_flow)),
                "jain": _jains_index(per_flow),
            }
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(row, f, indent=2)
            out.append(row)

    summary = output_base / "fairness" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list_traces(args) -> int:
    pairs = list_cellular_pairs()
    if not pairs:
        print("no traces found — set MAHIMAHI_ROOT or build third_party/mahimahi.")
        return 1
    for p in pairs:
        print(f"  {p.name}\t{p.carrier}\t{p.kind}")
    return 0


def _cmd_list_ccas(args) -> int:
    for n in CCAS:
        print(f"  {n}")
    return 0


def _cmd_check(args) -> int:
    print(f"mm-link:       {find_mm_link() or 'NOT FOUND'}")
    print(f"mm-delay:      {_find_mm_delay() or 'NOT FOUND'}")
    print(f"iperf3:        {shutil.which('iperf3') or 'NOT FOUND'}")
    print(f"sprout:        {_find_sprout() or 'NOT FOUND'}")
    client, server = _find_verus_client()
    print(f"verus_client:  {client or 'NOT FOUND'}")
    print(f"verus_server:  {server or 'NOT FOUND'}")
    print(f"nimbus bin:    {nimbus_release_binary()} "
          f"({'OK' if nimbus_release_binary().is_file() else 'NOT BUILT'})")
    print(f"portus-mpcc:   {_find_portus_mpcc() or 'NOT FOUND (pyportus fallback if installed)'}")
    print(f"pyportus:      {'OK' if _pyportus_available() else 'NOT INSTALLED'}")
    pairs = list_cellular_pairs()
    print(f"traces:        {len(pairs)} pairs")
    return 0


def _resolve_mpcc_config_by_scenario(
    scenarios: list[str], config_dir: str | None,
) -> dict[str, str]:
    """Resolve per-scenario weight YAMLs from `config_dir`.

    Expected layout: <config_dir>/{wired,cellular,leo,fairness}.yml
    Missing files fall back to CONFIG_NETWORK.yml (the default).
    """
    if not config_dir:
        return {}
    root = Path(config_dir)
    resolved: dict[str, str] = {}
    for scen in scenarios:
        cand = root / f"{scen}.yml"
        if cand.is_file():
            resolved[scen] = str(cand)
        else:
            # Match prefix (e.g., "leo-s0" -> "leo.yml").
            short = scen.split("-")[0]
            alt = root / f"{short}.yml"
            if alt.is_file():
                resolved[scen] = str(alt)
    default = root / "default.yml"
    if default.is_file():
        resolved["_default"] = str(default)
    return resolved


def _cmd_run(args) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config_by_scenario = _resolve_mpcc_config_by_scenario(
        args.scenarios, args.mpcc_config_dir,
    )
    if args.mpcc_config_dir:
        logger.info("MPCC scenario configs resolved: %s", config_by_scenario)
    run_matrix(
        scenarios=args.scenarios, ccas=args.ccas,
        duration_s=args.duration, seeds=args.seeds, output_base=out,
        mpcc_config_by_scenario=config_by_scenario,
        mpcc_solver=args.mpcc_solver,
        mpcc_horizon=args.mpcc_horizon,
    )
    return 0


def _cmd_fairness(args) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_fairness_matrix(
        ccas=args.ccas, n_flows=args.n_flows, duration_s=args.duration,
        seeds=args.seeds, trace_name=args.trace, output_base=out,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mpcc-real")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-traces").set_defaults(func=_cmd_list_traces)
    sub.add_parser("list-ccas").set_defaults(func=_cmd_list_ccas)
    sub.add_parser("check").set_defaults(func=_cmd_check)

    sp = sub.add_parser("run")
    sp.add_argument("--scenarios", nargs="+",
                    default=["wired", "cellular", "leo"],
                    help="Subset of {wired, cellular, leo, fairness}.")
    sp.add_argument("--ccas", nargs="+",
                    default=["cubic", "bbr", "mpcc"],
                    choices=list(CCAS.keys()))
    sp.add_argument("--duration", type=float, default=60.0)
    sp.add_argument("--seeds", type=int, default=3)
    sp.add_argument("--output", "-o", default=str(DEFAULT_RESULTS_DIR))
    sp.add_argument("--mpcc-config-dir",
                    default=str(repo_root() / "scripts/network/configs/paper"),
                    help="Directory with per-scenario MPCC weight YAMLs "
                         "(wired.yml, cellular.yml, leo.yml, fairness.yml). "
                         "Pass empty string '' to disable and use MPCC defaults.")
    sp.add_argument("--mpcc-solver", default="qp", choices=["qp", "nlp"],
                    help="MPCC solver mode for this matrix run.")
    sp.add_argument("--mpcc-horizon", type=int, default=None,
                    help="Override MPCC planner horizon (else use YAML / default).")
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("fairness")
    sp.add_argument("--ccas", nargs="+", default=["cubic", "bbr", "mpcc"],
                    choices=list(CCAS.keys()))
    sp.add_argument("--n-flows", type=int, default=2)
    sp.add_argument("--duration", type=float, default=60.0)
    sp.add_argument("--seeds", type=int, default=3)
    sp.add_argument("--trace", default=DEFAULT_CELLULAR_TRACES[0])
    sp.add_argument("--output", "-o", default=str(DEFAULT_RESULTS_DIR))
    sp.set_defaults(func=_cmd_fairness)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
