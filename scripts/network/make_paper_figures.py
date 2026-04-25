"""Generate the LaTeX-ready figures for paper_state.tex from the matrix run.

Outputs under paper_figures/:
  - pareto.png           : throughput-delay Pareto scatter (wired, LEO)
  - fairness.png         : per-flow throughput bars + Jain annotations
  - solve_cdf.png        : per-ACK solve-time CDF across 11k+ solves
  - leo_timeseries.png   : throughput trace for one seed per CCA on LEO-s0

Reads a matrix output dir (default: the most recent results/paper_theta_*).
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CCAs = ["cubic", "reno", "bbr", "nimbus", "mpcc"]
COLORS = {
    "cubic":  "#377eb8",
    "reno":   "#4daf4a",
    "bbr":    "#ff7f00",
    "nimbus": "#984ea3",
    "mpcc":   "#e41a1c",
    "sprout": "#a65628",
    "verus":  "#f781bf",
}
MARKERS = {
    "cubic":  "o",
    "reno":   "s",
    "bbr":    "^",
    "nimbus": "D",
    "mpcc":   "*",
}


def latest_matrix_root() -> Path:
    dirs = sorted(glob.glob(
        str(Path(__file__).resolve().parents[2] / "results" / "paper_theta_*")
    ))
    if not dirs:
        raise SystemExit("no paper_theta_* results directory found")
    return Path(dirs[-1]) / "matrix"


def agg_tput_rtt(matrix_root: Path, scenario_prefix: str, cca: str):
    """Return (tputs, rtts) arrays across all seeds/traces for a scenario."""
    ts, rs = [], []
    for f in glob.glob(str(matrix_root / f"{scenario_prefix}*" / "*" / cca / "seed*" / "metrics.json")):
        r = json.load(open(f))
        t = r.get("avg_throughput_mbps", 0)
        if t > 0.5:
            ts.append(t)
            rs.append(r["median_rtt_ms"])
    return np.array(ts), np.array(rs)


def agg_tput_rtt_cellular(matrix_root: Path, carrier_dir: str, cca: str):
    """Return (tputs, rtts) for one cellular carrier directory."""
    ts, rs = [], []
    for f in glob.glob(str(matrix_root / "cellular" / carrier_dir / cca / "seed*" / "metrics.json")):
        r = json.load(open(f))
        t = r.get("avg_throughput_mbps", 0)
        if t > 0.5:
            ts.append(t)
            rs.append(r["median_rtt_ms"])
    return np.array(ts), np.array(rs)


def fig_pareto(matrix_root: Path, out: Path):
    """Clean three-panel Pareto plot (wired | cellular | LEO).

    One marker per CCA at its (mean p50 RTT, mean throughput); error bars
    show $\pm 1$ standard deviation. Non-dominated CCAs are red and
    connected by a monotone spline --- the Pareto front. Dominated CCAs
    are blue. CCA labels are placed by ``adjustText`` so they do not
    overlap each other or the markers.
    """
    import seaborn as sns
    from scipy.interpolate import PchipInterpolator
    from adjustText import adjust_text

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    def cca_means(prefix: str):
        out_ = {}
        for cca in CCAs:
            ts, rs = agg_tput_rtt(matrix_root, prefix, cca)
            if ts.size == 0:
                continue
            out_[cca] = (float(rs.mean()), float(ts.mean()),
                          float(rs.std()) if rs.size > 1 else 0.0,
                          float(ts.std()) if ts.size > 1 else 0.0)
        return out_

    def pareto(ccas):
        keep = set()
        for a, (ra, ta, _, _) in ccas.items():
            dominated = False
            for b, (rb, tb, _, _) in ccas.items():
                if a == b:
                    continue
                if rb <= ra and tb >= ta and (rb < ra or tb > ta):
                    dominated = True
                    break
            if not dominated:
                keep.add(a)
        return keep

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    panels = [
        ("wired",    axes[0], "Wired 50 Mbps"),
        ("cellular", axes[1], "Cellular LTE (ATT/TM/VZ)"),
        ("leo-s",    axes[2], "LEO handover (s0--s2)"),
    ]
    PARETO_RED = "#d62728"
    DOMIN_BLUE = "#3b6aa0"

    # Hand-tuned per-CCA label offsets (in display points) for every
    # panel. adjust_text was unreliable — on cellular it flung MPCC
    # across the panel, and on wired/LEO it sometimes overlapped the
    # Pareto line. Explicit offsets keep labels hugging their dots while
    # dodging neighboring markers and the Pareto curve.
    LABEL_OFFSETS = {
        "wired": {
            "mpcc":   ( 14,   0, "left",   "center"),
            "bbr":    ( 14,   0, "left",   "center"),
            "cubic":  (  0, -14, "center", "top"),
            "reno":   ( 10,   8, "left",   "bottom"),
            "nimbus": (  0,  12, "center", "bottom"),
        },
        "cellular": {
            "mpcc":   ( 14,   6, "left",   "bottom"),
            "bbr":    ( -6,  12, "right",  "bottom"),
            "reno":   ( 10,   8, "left",   "bottom"),
            "nimbus": ( 12,  -1, "left",   "center"),
            "cubic":  (  0, -14, "center", "top"),
        },
        "leo-s": {
            "mpcc":   ( 14,   0, "left",   "center"),
            "bbr":    (  0,  12, "center", "bottom"),
            "cubic":  (  0, -14, "center", "top"),
            # RENO dot is at the top of the panel — push the label
            # right instead of up so it doesn't collide with the title.
            "reno":   ( 10,   0, "left",   "center"),
            "nimbus": ( 14,   0, "left",   "center"),
        },
    }

    for prefix, ax, title in panels:
        ccas = cca_means(prefix)
        if not ccas:
            continue
        pf = pareto(ccas)

        # Plot the Pareto curve FIRST (behind points).
        if len(pf) >= 2:
            pts = sorted(((ccas[c][0], ccas[c][1]) for c in pf),
                         key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            try:
                xs_d = np.linspace(min(xs), max(xs), 200)
                curve = PchipInterpolator(xs, ys)(xs_d)
            except Exception:
                xs_d, curve = xs, ys
            ax.plot(xs_d, curve, color=PARETO_RED,
                    linewidth=2.5, zorder=2, alpha=0.95)

        # Markers only — error bars removed for clarity.
        panel_offsets = LABEL_OFFSETS.get(prefix, {})
        for cca, (r, t, _rs_std, _ts_std) in ccas.items():
            color = PARETO_RED if cca in pf else DOMIN_BLUE
            ax.plot(r, t, marker="o", markersize=12,
                    color=color, markeredgecolor="black",
                    markeredgewidth=0.7,
                    linestyle="none", zorder=4)
            nicename = "MPCC" if cca == "mpcc" else cca.upper()
            dx, dy, ha, va = panel_offsets.get(
                cca, (8, 8, "left", "bottom")
            )
            ax.annotate(
                nicename, xy=(r, t),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=11,
                fontweight="bold" if cca == "mpcc" else "normal",
                color="black", zorder=5, ha=ha, va=va,
            )

        if prefix == "cellular":
            ax.set_xscale("log")
            # Widened to 4 Mbps floor so the probe-BW MPCC point (~5.5 Mbps
            # at ~120 ms) stays comfortably above the x-axis.
            ax.set_ylim(4, 16)
            ax.set_xlim(80, 7000)

        ax.set_xlabel("Median RTT (ms) — lower is better")
        ax.set_ylabel("Throughput (Mbps) — higher is better")
        ax.set_title(title, fontsize=11)

    # One small caption-style legend at the bottom.
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="white",
                   markerfacecolor=PARETO_RED, markeredgecolor="black",
                   markersize=9, label="Pareto-optimal CCA"),
            Line2D([0], [0], marker="o", color="white",
                   markerfacecolor=DOMIN_BLUE, markeredgecolor="black",
                   markersize=9, label="Dominated CCA"),
            Line2D([0], [0], color=PARETO_RED, linewidth=2.5,
                   label="Pareto front"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False, fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_pareto_cellular(matrix_root: Path, out: Path):
    """Per-carrier cellular Pareto plot: ATT | T-Mobile | Verizon.

    Same visual language as ``fig_pareto`` but one column per LTE
    carrier trace instead of a single combined ``cellular`` panel.
    """
    import seaborn as sns
    from scipy.interpolate import PchipInterpolator

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    def cca_means(carrier_dir: str):
        out_ = {}
        for cca in CCAs:
            ts, rs = agg_tput_rtt_cellular(matrix_root, carrier_dir, cca)
            if ts.size == 0:
                continue
            out_[cca] = (float(rs.mean()), float(ts.mean()),
                          float(rs.std()) if rs.size > 1 else 0.0,
                          float(ts.std()) if ts.size > 1 else 0.0)
        return out_

    def pareto(ccas):
        keep = set()
        for a, (ra, ta, _, _) in ccas.items():
            dominated = False
            for b, (rb, tb, _, _) in ccas.items():
                if a == b:
                    continue
                if rb <= ra and tb >= ta and (rb < ra or tb > ta):
                    dominated = True
                    break
            if not dominated:
                keep.add(a)
        return keep

    PARETO_RED = "#d62728"
    DOMIN_BLUE = "#3b6aa0"

    carriers = [
        ("ATT-LTE-driving",     "ATT LTE (driving)"),
        ("TMobile-LTE-driving", "T-Mobile LTE (driving)"),
        ("Verizon-LTE-driving", "Verizon LTE (driving)"),
    ]

    # Default offsets are the same hand-tuned ones used for the combined
    # cellular panel; individual carriers can override if a dot lands
    # somewhere the default collides with the Pareto curve.
    DEFAULT_OFFSETS = {
        "mpcc":   ( 14,   6, "left",   "bottom"),
        "bbr":    ( -6,  12, "right",  "bottom"),
        "reno":   ( 10,   8, "left",   "bottom"),
        "nimbus": ( 12,  -1, "left",   "center"),
        "cubic":  (  0, -14, "center", "top"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    for (carrier_dir, title), ax in zip(carriers, axes):
        ccas = cca_means(carrier_dir)
        if not ccas:
            ax.set_title(f"{title} (no data)", fontsize=11)
            continue
        pf = pareto(ccas)

        if len(pf) >= 2:
            pts = sorted(((ccas[c][0], ccas[c][1]) for c in pf),
                         key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            try:
                xs_d = np.linspace(min(xs), max(xs), 200)
                curve = PchipInterpolator(xs, ys)(xs_d)
            except Exception:
                xs_d, curve = xs, ys
            ax.plot(xs_d, curve, color=PARETO_RED,
                    linewidth=2.5, zorder=2, alpha=0.95)

        for cca, (r, t, _rs_std, _ts_std) in ccas.items():
            color = PARETO_RED if cca in pf else DOMIN_BLUE
            ax.plot(r, t, marker="o", markersize=12,
                    color=color, markeredgecolor="black",
                    markeredgewidth=0.7,
                    linestyle="none", zorder=4)
            nicename = "MPCC" if cca == "mpcc" else cca.upper()
            dx, dy, ha, va = DEFAULT_OFFSETS.get(
                cca, (8, 8, "left", "bottom")
            )
            ax.annotate(
                nicename, xy=(r, t),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=11,
                fontweight="bold" if cca == "mpcc" else "normal",
                color="black", zorder=5, ha=ha, va=va,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Median RTT (ms) — lower is better")
        ax.set_ylabel("Throughput (Mbps) — higher is better")
        ax.set_title(title, fontsize=11)

    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="white",
                   markerfacecolor=PARETO_RED, markeredgecolor="black",
                   markersize=9, label="Pareto-optimal CCA"),
            Line2D([0], [0], marker="o", color="white",
                   markerfacecolor=DOMIN_BLUE, markeredgecolor="black",
                   markersize=9, label="Dominated CCA"),
            Line2D([0], [0], color=PARETO_RED, linewidth=2.5,
                   label="Pareto front"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False, fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_fairness(matrix_root: Path, out: Path):
    """Grouped bar chart of per-flow throughput for the 4-flow fairness run.
    Each CCA contributes a group of 4 bars (one per flow), colored by the CCA.
    Jain's index annotated above each group."""
    ccas = ["cubic", "mpcc", "bbr"]
    per_flow_means = {}
    jains = {}
    for cca in ccas:
        flows, js = [], []
        for f in sorted(glob.glob(str(matrix_root / "fairness" / cca / "n4" / "seed*" / "metrics.json"))):
            r = json.load(open(f))
            flows.append(r["per_flow_mbps"])
            js.append(r["jain"])
        flows = np.array(flows)
        # average per-flow over seeds: shape (n_seeds, 4) → mean over seeds
        per_flow_means[cca] = flows.mean(axis=0) if flows.size else np.zeros(4)
        jains[cca] = st.mean(js) if js else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    bar_w = 0.18
    x_groups = np.arange(len(ccas))
    for flow_idx in range(4):
        offsets = x_groups + (flow_idx - 1.5) * bar_w
        heights = [per_flow_means[c][flow_idx] for c in ccas]
        colors = [COLORS[c] for c in ccas]
        ax.bar(offsets, heights, width=bar_w,
               color=colors, edgecolor="black", linewidth=0.4,
               label=f"Flow {flow_idx+1}" if flow_idx < 4 else None,
               alpha=0.55 + 0.15 * flow_idx)
    # Jain annotation
    for i, cca in enumerate(ccas):
        height = max(per_flow_means[cca]) if per_flow_means[cca].size else 0.0
        ax.text(x_groups[i], height + 0.04,
                f"Jain {jains[cca]:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_groups)
    ax.set_xticklabels([c.upper() if c != "mpcc" else "MPCC" for c in ccas])
    ax.set_ylabel("Per-flow mean throughput (Mbps)")
    ax.set_title("4-flow fairness on ATT-LTE (3 seeds)")
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim(0, max(0.75, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_solve_cdf(matrix_root: Path, out: Path):
    """Per-ACK solve-time CDF aggregated over every MPCC run in the matrix.
    Useful evidence that the controller is real-time-deployable at RTT
    granularity."""
    times_ms = []
    for f in glob.glob(str(matrix_root / "*" / "*" / "mpcc" / "seed*" / "mpcc.log")):
        try:
            for line in open(f):
                m = re.search(r"solve_ms=([0-9.]+)", line)
                if m:
                    times_ms.append(float(m.group(1)))
        except Exception:
            pass
    if not times_ms:
        raise SystemExit("no MPCC solve-time samples found")
    times_us = np.array(times_ms) * 1000.0
    sorted_us = np.sort(times_us)
    ccdf = np.arange(1, sorted_us.size + 1) / sorted_us.size

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.plot(sorted_us, ccdf, color=COLORS["mpcc"], linewidth=1.8)
    # Mark mean / p50 / p99
    mean_us = times_us.mean()
    p50 = np.percentile(times_us, 50)
    p99 = np.percentile(times_us, 99)
    for x, label, y_anchor in [
        (p50, f"p50 = {p50:.0f} μs", 0.52),
        (mean_us, f"mean = {mean_us:.0f} μs", 0.70),
        (p99, f"p99 = {p99:.0f} μs", 0.95),
    ]:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.annotate(label, xy=(x, y_anchor), xytext=(6, 0),
                    textcoords="offset points", fontsize=9,
                    va="center", color="gray")
    # Control-interval reference
    ax.axvline(20_000, color="black", linestyle=":", linewidth=0.8)
    ax.annotate(
        "20 ms control interval",
        xy=(20_000, 0.5), xytext=(-6, 0),
        textcoords="offset points", fontsize=9, ha="right", color="black",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Per-ACK solve time (μs, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(f"MPCC per-ACK solve-time CDF ({times_us.size:,} solves)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.set_ylim(0, 1.03)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_leo_timeseries(matrix_root: Path, out: Path):
    """Throughput time series for a single seed per CCA on LEO-s0. Shows how
    each CCA reacts to periodic handover capacity drops (50 → 5 Mbps)."""
    ccas = ["cubic", "bbr", "mpcc"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for cca in ccas:
        f = matrix_root / "leo-s0" / "LEO-handover-s0" / cca / "seed0" / "metrics.json"
        if not f.is_file():
            continue
        r = json.load(open(f))
        ts = r.get("throughput_ts") or []
        if not ts:
            continue
        t = [p[0] for p in ts]
        y = [p[1] for p in ts]
        ax.plot(t, y, label=cca.upper() if cca != "mpcc" else "MPCC",
                color=COLORS[cca], linewidth=1.3,
                alpha=0.9 if cca == "mpcc" else 0.65)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("LEO-s0 throughput trace (seed 0)")
    ax.legend(loc="lower right", ncol=3, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(0, 55)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, default=None,
                   help="results/paper_theta_*/matrix directory")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[2] / "paper_figures")
    args = p.parse_args()

    matrix_root = args.matrix or latest_matrix_root()
    args.out.mkdir(parents=True, exist_ok=True)

    fig_pareto(matrix_root, args.out / "pareto.png")
    fig_pareto_cellular(matrix_root, args.out / "pareto_cellular.png")
    fig_fairness(matrix_root, args.out / "fairness.png")
    fig_solve_cdf(matrix_root, args.out / "solve_cdf.png")
    fig_leo_timeseries(matrix_root, args.out / "leo_timeseries.png")

    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
