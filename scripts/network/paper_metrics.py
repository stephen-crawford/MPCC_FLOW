"""Paper-ready metrics extraction + aggregation.

Runs on a results directory produced by `run_cc_experiments.py run` or
`run_paper_matrix.sh`, and emits:

  - <run_dir>/paper_metrics.json per experiment run (one per seed)
  - <results_root>/paper_tables/<scenario>.csv       aggregated by CCA
  - <results_root>/paper_tables/<scenario>.tex       LaTeX-ready table
  - <results_root>/paper_figures/<scenario>_ts.png   time-series per CCA
  - <results_root>/paper_figures/solve_time.png      MPCC solve-time histogram
  - <results_root>/paper_figures/fairness.png        Jain's index bar chart

Metrics reported (paper §IV.K + §V.D):

  throughput_mean_mbps   mean over the uplink time-series
  rtt_p50_ms             median queueing delay from mm-link (or iperf RTT)
  rtt_p95_ms             95th percentile
  loss_pct               drops / arrivals at the bottleneck
  queue_p50_ms           50th percentile queueing delay
  rate_cov               coefficient of variation of sending rate (smoothness)
  jains_fairness         only for fairness runs; 0 otherwise
  solve_ms_mean, p99     from the Rust portus-mpcc info logs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger("paper_metrics")

MPCC_LOG_LINE = re.compile(
    r"mpcc_step\s+solve_ms=(?P<solve>[-0-9.eE+]+)\s+rate_bps=(?P<rate>[-0-9.eE+]+)\s+"
    r"tput_bps=(?P<tput>[-0-9.eE+]+)\s+rtt_us=(?P<rtt>[-0-9]+)\s+"
    r"q_bytes=(?P<q>[-0-9.eE+]+)\s+success=(?P<ok>\d)"
)


def _parse_mpcc_log(path: Path) -> dict:
    """Extract per-solve samples from mpcc.log produced by portus-mpcc."""
    if not path.is_file():
        return {}
    solves: list[float] = []
    rates: list[float] = []
    tputs: list[float] = []
    rtts: list[float] = []
    successes = 0
    total = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = MPCC_LOG_LINE.search(line)
                if not m:
                    continue
                total += 1
                solves.append(float(m.group("solve")))
                rates.append(float(m.group("rate")))
                tputs.append(float(m.group("tput")))
                rtts.append(float(m.group("rtt")))
                successes += int(m.group("ok"))
    except Exception as e:
        logger.warning("failed to parse %s: %s", path, e)
        return {}
    if not solves:
        return {}
    return {
        "solve_ms_mean": float(np.mean(solves)),
        "solve_ms_p50": float(np.percentile(solves, 50)),
        "solve_ms_p99": float(np.percentile(solves, 99)),
        "solve_count": total,
        "solve_success_rate": successes / max(total, 1),
        "rate_mean_mbps": float(np.mean(rates)) / 1e6,
        "rate_std_mbps": float(np.std(rates)) / 1e6,
        "rate_cov": (
            float(np.std(rates)) / float(np.mean(rates))
            if np.mean(rates) > 0 else 0.0
        ),
        "mpcc_tput_mean_mbps": float(np.mean(tputs)) / 1e6,
        "mpcc_rtt_mean_ms": float(np.mean(rtts)) / 1000.0,
    }


def _load_run_metrics(run_dir: Path) -> dict:
    """Aggregate everything interesting in a single run directory."""
    row: dict = {"run_dir": str(run_dir)}

    base = run_dir / "metrics.json"
    if base.is_file():
        try:
            with open(base) as f:
                row.update(json.load(f))
        except Exception as e:
            logger.warning("bad metrics.json in %s: %s", run_dir, e)

    # pull per-solve samples from the MPCC CCP log (no-op for non-MPCC runs).
    row.update(_parse_mpcc_log(run_dir / "mpcc.log"))

    # Rename to the paper's column names.
    mapping = {
        "avg_throughput_mbps": "throughput_mean_mbps",
        "median_rtt_ms": "rtt_p50_ms",
        "p95_rtt_ms": "rtt_p95_ms",
        "loss_rate": "loss_frac",
    }
    for src, dst in mapping.items():
        if src in row and dst not in row:
            row[dst] = row[src]

    if "loss_frac" in row and "loss_pct" not in row:
        row["loss_pct"] = row["loss_frac"] * 100.0

    return row


def _find_run_dirs(root: Path) -> Iterable[Path]:
    for metrics in root.rglob("metrics.json"):
        yield metrics.parent


def _aggregate_by_cca(rows: list[dict]) -> list[dict]:
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r.get("scenario", "?"), r.get("algorithm", r.get("cca", "?")))
        by.setdefault(key, []).append(r)

    aggregated: list[dict] = []
    for (scen, cca), group in sorted(by.items()):
        def _stat(field: str):
            xs = [r[field] for r in group
                  if field in r and isinstance(r[field], (int, float))]
            if not xs:
                return (0.0, 0.0)
            return (float(np.mean(xs)),
                    float(np.std(xs)) if len(xs) > 1 else 0.0)

        row = {
            "scenario": scen,
            "cca": cca,
            "n_seeds": len(group),
        }
        for field in [
            "throughput_mean_mbps", "rtt_p50_ms", "rtt_p95_ms",
            "loss_pct", "queue_p50_ms", "utilization",
            "solve_ms_mean", "solve_ms_p99", "rate_cov", "jain",
        ]:
            mean, std = _stat(field)
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        aggregated.append(row)
    return aggregated


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_tex(path: Path, rows: list[dict], scenario: str) -> None:
    """One LaTeX table per scenario. Columns: CCA, Throughput, p50 RTT, p95 RTT, Loss %."""
    rows = [r for r in rows if r["scenario"] == scenario]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("CCA & Throughput (Mbps) & p50 RTT (ms) & p95 RTT (ms) & "
                "Loss (\\%) & Rate CoV \\\\ \n")
        f.write("\\midrule\n")
        for r in sorted(rows, key=lambda x: x["cca"]):
            f.write(
                f"{r['cca']} & "
                f"{r['throughput_mean_mbps_mean']:.2f} $\\pm$ {r['throughput_mean_mbps_std']:.2f} & "
                f"{r['rtt_p50_ms_mean']:.1f} & "
                f"{r['rtt_p95_ms_mean']:.1f} & "
                f"{r['loss_pct_mean']:.2f} & "
                f"{r['rate_cov_mean']:.3f} \\\\ \n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def _plot_time_series(out_root: Path, scenario_rows: list[dict], scenario: str) -> None:
    """Overlay per-CCA throughput time-series on one figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return

    scenario_rows = [r for r in scenario_rows if r.get("scenario") == scenario]
    if not scenario_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = 0
    for r in scenario_rows:
        ts = r.get("throughput_ts") or []
        if not ts:
            continue
        if isinstance(ts[0], (list, tuple)):
            xs = [float(x) for x, _ in ts]
            ys = [float(y) for _, y in ts]
        else:
            continue
        ax.plot(xs, ys, alpha=0.6, label=f"{r.get('algorithm','?')} seed{r.get('seed', '')}")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("time (s)")
    ax.set_ylabel("throughput (Mbps)")
    ax.set_title(f"Scenario: {scenario}")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_path = out_root / f"{scenario}_ts.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_solve_time(out_root: Path, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    samples: list[float] = []
    for r in rows:
        if r.get("algorithm") != "mpcc":
            continue
        m = r.get("solve_ms_mean")
        if m:
            samples.append(m)
    if not samples:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=30)
    ax.set_xlabel("mean solve time per run (ms)")
    ax.set_ylabel("# runs")
    ax.set_title("MPCC solve time across the matrix")
    fig.tight_layout()
    out = out_root / "solve_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_fairness(out_root: Path, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fair_rows = [r for r in rows if r.get("jain")]
    if not fair_rows:
        return
    by_cca: dict[str, list[float]] = {}
    for r in fair_rows:
        by_cca.setdefault(r.get("cca", "?"), []).append(r["jain"])
    fig, ax = plt.subplots(figsize=(6, 4))
    names = sorted(by_cca.keys())
    means = [float(np.mean(by_cca[n])) for n in names]
    stds = [float(np.std(by_cca[n])) if len(by_cca[n]) > 1 else 0.0 for n in names]
    ax.bar(names, means, yerr=stds, capsize=4)
    ax.set_ylabel("Jain's fairness index")
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-flow fairness (higher = fairer)")
    fig.tight_layout()
    out = out_root / "fairness.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build_report(results_root: Path) -> None:
    """Walk `results_root`, emit per-scenario tables + figures."""
    raw_rows: list[dict] = []
    for run_dir in _find_run_dirs(results_root):
        r = _load_run_metrics(run_dir)
        rel = run_dir.relative_to(results_root)
        parts = rel.parts
        # Convention from run_cc_experiments: <scenario>/<trace>/<cca>/seed<N>/
        if len(parts) >= 4:
            r.setdefault("scenario", parts[0])
            r.setdefault("trace", parts[1])
            r.setdefault("cca", parts[2])
            r["algorithm"] = r.get("algorithm") or r["cca"]
            try:
                r["seed"] = int(parts[3].replace("seed", ""))
            except ValueError:
                pass
        with open(run_dir / "paper_metrics.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        raw_rows.append(r)

    # fairness runs live under results/<root>/fairness/<cca>/n<k>/seed<N>/
    for run_dir in (results_root / "fairness").rglob("metrics.json"):
        try:
            with open(run_dir) as f:
                j = json.load(f)
            j.setdefault("scenario", "fairness")
            j.setdefault("algorithm", j.get("cca", "?"))
            raw_rows.append(j)
        except Exception:
            continue

    if not raw_rows:
        logger.warning("no runs found under %s", results_root)
        return

    aggregated = _aggregate_by_cca(raw_rows)

    tables_dir = results_root / "paper_tables"
    figures_dir = results_root / "paper_figures"

    scenarios = sorted({r["scenario"] for r in aggregated})
    for scen in scenarios:
        scen_rows = [r for r in aggregated if r["scenario"] == scen]
        _write_csv(tables_dir / f"{scen}.csv", scen_rows)
        _write_tex(tables_dir / f"{scen}.tex", aggregated, scen)
        _plot_time_series(figures_dir, raw_rows, scen)

    _plot_solve_time(figures_dir, raw_rows)
    _plot_fairness(figures_dir, raw_rows)

    summary_path = results_root / "paper_summary.json"
    with open(summary_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    logger.info("wrote %s (%d aggregated rows across %d scenarios)",
                summary_path, len(aggregated), len(scenarios))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="paper_metrics")
    p.add_argument("results_root", help="Path to results/<timestamp>/ directory")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    root = Path(args.results_root)
    if not root.is_dir():
        p.error(f"not a directory: {root}")
    build_report(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
