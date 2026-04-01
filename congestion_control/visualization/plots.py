"""Matplotlib-based visualization for congestion control experiments.

Produces publication-quality plots for:
- Per-CCA time series (throughput, RTT, cwnd)
- Throughput vs delay scatter (Pareto frontier)
- Comparison bar charts
- Per-trace dashboards
- Aggregate multi-trace reports
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from congestion_control.diagnostics import Diagnostics, FlowMetrics

logger = logging.getLogger("cc.viz")

# Consistent color scheme across all plots
CCA_COLORS = {
    "MPCC": "#1f77b4",       # blue
    "Sprout": "#ff7f0e",     # orange
    "SproutEWMA": "#d62728", # red
    "Verus": "#2ca02c",      # green
    "ABC": "#9467bd",        # purple
}
CCA_MARKERS = {
    "MPCC": "o",
    "Sprout": "s",
    "SproutEWMA": "^",
    "Verus": "D",
    "ABC": "v",
}

def _color(name: str) -> str:
    return CCA_COLORS.get(name, "#7f7f7f")

def _marker(name: str) -> str:
    return CCA_MARKERS.get(name, "x")


# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------

def plot_throughput_timeseries(
    diagnostics: Dict[str, Diagnostics],
    title: str = "Throughput Over Time",
    bin_s: float = 0.5,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot throughput time series for multiple CCAs."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, diag in diagnostics.items():
        ts, vals = diag.throughput_timeseries(bin_s=bin_s)
        if ts:
            t0 = ts[0]
            ax.plot([t - t0 for t in ts], vals, label=name,
                    color=_color(name), linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Throughput (Mbps)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_rtt_timeseries(
    diagnostics: Dict[str, Diagnostics],
    title: str = "RTT Over Time",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot RTT time series for multiple CCAs."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, diag in diagnostics.items():
        ts, vals = diag.rtt_timeseries()
        if ts:
            t0 = ts[0]
            # Subsample for readability
            step = max(1, len(ts) // 2000)
            ax.plot([ts[i] - t0 for i in range(0, len(ts), step)],
                    [vals[i] for i in range(0, len(vals), step)],
                    label=name, color=_color(name), linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("RTT (ms)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_cwnd_timeseries(
    diagnostics: Dict[str, Diagnostics],
    title: str = "Congestion Window Over Time",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot cwnd evolution for multiple CCAs."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, diag in diagnostics.items():
        ts, vals = diag.cwnd_timeseries()
        if ts:
            t0 = ts[0]
            # Convert to packets
            vals_pkts = [v / 1500 for v in vals]
            step = max(1, len(ts) // 2000)
            ax.plot([ts[i] - t0 for i in range(0, len(ts), step)],
                    [vals_pkts[i] for i in range(0, len(vals_pkts), step)],
                    label=name, color=_color(name), linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("CWND (packets)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Scatter / Pareto plots
# ---------------------------------------------------------------------------

def plot_throughput_delay_scatter(
    metrics: Dict[str, FlowMetrics],
    title: str = "Throughput vs Delay",
    save_path: Optional[Path] = None,
    use_p95: bool = True,
) -> plt.Figure:
    """Scatter plot of throughput vs delay with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))

    names, tputs, delays = [], [], []
    for name, m in metrics.items():
        delay = m.p95_rtt_ms if use_p95 else m.p50_rtt_ms
        ax.scatter(delay, m.avg_throughput_mbps, s=180, zorder=5,
                   color=_color(name), marker=_marker(name),
                   edgecolors="black", linewidth=0.8)
        ax.annotate(name, (delay, m.avg_throughput_mbps),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=10, fontweight="bold", color=_color(name))
        names.append(name)
        tputs.append(m.avg_throughput_mbps)
        delays.append(delay)

    # Draw "better" arrow
    ax.annotate("", xy=(min(delays) * 0.6, max(tputs) * 1.1),
                xytext=(max(delays) * 0.8, min(tputs) * 0.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.text(min(delays) * 0.5, max(tputs) * 1.05, "Better",
            fontsize=11, color="gray", fontstyle="italic")

    delay_label = "95th Percentile RTT (ms)" if use_p95 else "Median RTT (ms)"
    ax.set_xlabel(delay_label, fontsize=12)
    ax.set_ylabel("Average Throughput (Mbps)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_pareto_frontier(
    all_metrics: Dict[str, Dict[str, FlowMetrics]],
    title: str = "Throughput-Delay Pareto Frontier (All Traces)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot all CCA results across multiple traces with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect all points per CCA
    cca_points: Dict[str, List[Tuple[float, float]]] = {}
    for trace_name, metrics in all_metrics.items():
        for cca_name, m in metrics.items():
            cca_points.setdefault(cca_name, []).append(
                (m.p95_rtt_ms, m.avg_throughput_mbps))

    # Plot individual points (small) and mean (large)
    for cca_name, points in cca_points.items():
        delays = [p[0] for p in points]
        tputs = [p[1] for p in points]
        # Individual trace results
        ax.scatter(delays, tputs, s=40, alpha=0.4,
                   color=_color(cca_name), marker=_marker(cca_name))
        # Mean point (large)
        mean_d = np.mean(delays)
        mean_t = np.mean(tputs)
        ax.scatter(mean_d, mean_t, s=250, zorder=5,
                   color=_color(cca_name), marker=_marker(cca_name),
                   edgecolors="black", linewidth=1.2, label=cca_name)
        ax.annotate(cca_name, (mean_d, mean_t),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, fontweight="bold", color=_color(cca_name))

    # Draw Pareto frontier from mean points
    mean_pts = []
    for cca_name, points in cca_points.items():
        mean_pts.append((np.mean([p[0] for p in points]),
                         np.mean([p[1] for p in points]),
                         cca_name))
    # Sort by delay ascending
    mean_pts.sort(key=lambda x: x[0])
    # Compute Pareto frontier
    frontier = []
    best_tput = -1
    for d, t, n in mean_pts:
        if t > best_tput:
            frontier.append((d, t))
            best_tput = t
    if len(frontier) >= 2:
        fd = [p[0] for p in frontier]
        ft = [p[1] for p in frontier]
        ax.plot(fd, ft, "k--", linewidth=1.5, alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("95th Percentile RTT (ms)", fontsize=12)
    ax.set_ylabel("Average Throughput (Mbps)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

def plot_comparison_bars(
    metrics: Dict[str, FlowMetrics],
    title: str = "CCA Comparison",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing CCA metrics."""
    names = list(metrics.keys())
    # Sort: MPCC first
    names.sort(key=lambda n: (0 if n == "MPCC" else 1, n))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Throughput
    ax = axes[0]
    vals = [metrics[n].avg_throughput_mbps for n in names]
    colors = [_color(n) for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mbps")
    ax.set_title("Avg Throughput", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # p95 RTT
    ax = axes[1]
    vals = [metrics[n].p95_rtt_ms for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("ms")
    ax.set_title("p95 RTT", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # Loss rate
    ax = axes[2]
    vals = [metrics[n].loss_rate * 100 for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("%")
    ax.set_title("Loss Rate", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Link utilization
    ax = axes[3]
    vals = [metrics[n].link_utilization * 100 for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("%")
    ax.set_title("Link Utilization", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Dashboard: per-trace multi-panel view
# ---------------------------------------------------------------------------

def plot_trace_dashboard(
    diagnostics: Dict[str, Diagnostics],
    trace_name: str,
    save_path: Optional[Path] = None,
    bin_s: float = 0.5,
) -> plt.Figure:
    """Full dashboard for one trace: throughput, RTT, cwnd, and scatter."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # Throughput time series
    ax1 = fig.add_subplot(gs[0, :])
    for name, diag in diagnostics.items():
        ts, vals = diag.throughput_timeseries(bin_s=bin_s)
        if ts:
            t0 = ts[0]
            ax1.plot([t - t0 for t in ts], vals, label=name,
                     color=_color(name), linewidth=1.5, alpha=0.85)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Throughput (Mbps)")
    ax1.set_title(f"Throughput — {trace_name}", fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # RTT time series
    ax2 = fig.add_subplot(gs[1, 0])
    for name, diag in diagnostics.items():
        ts, vals = diag.rtt_timeseries()
        if ts:
            t0 = ts[0]
            step = max(1, len(ts) // 1500)
            ax2.plot([ts[i] - t0 for i in range(0, len(ts), step)],
                     [vals[i] for i in range(0, len(vals), step)],
                     label=name, color=_color(name), linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RTT (ms)")
    ax2.set_title("RTT", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # CWND time series
    ax3 = fig.add_subplot(gs[1, 1])
    for name, diag in diagnostics.items():
        ts, vals = diag.cwnd_timeseries()
        if ts:
            t0 = ts[0]
            vals_pkts = [v / 1500 for v in vals]
            step = max(1, len(ts) // 1500)
            ax3.plot([ts[i] - t0 for i in range(0, len(ts), step)],
                     [vals_pkts[i] for i in range(0, len(vals_pkts), step)],
                     label=name, color=_color(name), linewidth=0.8, alpha=0.7)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("CWND (packets)")
    ax3.set_title("Congestion Window", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Bar comparison
    ax4 = fig.add_subplot(gs[2, 0])
    names_sorted = sorted(diagnostics.keys(), key=lambda n: (0 if n == "MPCC" else 1, n))
    tputs = [diagnostics[n].compute_metrics().avg_throughput_mbps for n in names_sorted]
    colors = [_color(n) for n in names_sorted]
    bars = ax4.bar(names_sorted, tputs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, tputs):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                 f"{v:.1f}", ha="center", fontsize=8)
    ax4.set_ylabel("Mbps")
    ax4.set_title("Throughput Comparison", fontweight="bold")
    ax4.tick_params(axis="x", rotation=25)
    ax4.grid(axis="y", alpha=0.3)

    # Scatter: throughput vs p95 delay
    ax5 = fig.add_subplot(gs[2, 1])
    for name in names_sorted:
        m = diagnostics[name].compute_metrics()
        ax5.scatter(m.p95_rtt_ms, m.avg_throughput_mbps, s=150, zorder=5,
                    color=_color(name), marker=_marker(name),
                    edgecolors="black", linewidth=0.8)
        ax5.annotate(name, (m.p95_rtt_ms, m.avg_throughput_mbps),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=9, color=_color(name), fontweight="bold")
    ax5.set_xlabel("p95 RTT (ms)")
    ax5.set_ylabel("Throughput (Mbps)")
    ax5.set_title("Throughput vs Delay", fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(left=0)
    ax5.set_ylim(bottom=0)

    fig.suptitle(f"CCA Comparison Dashboard — {trace_name}",
                 fontsize=15, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Aggregate multi-trace report
# ---------------------------------------------------------------------------

def plot_aggregate_report(
    all_metrics: Dict[str, Dict[str, FlowMetrics]],
    title: str = "Aggregate CCA Performance",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Aggregate comparison across multiple traces: mean + error bars."""
    # Collect per-CCA values across traces
    cca_names_set = set()
    for m in all_metrics.values():
        cca_names_set.update(m.keys())
    cca_names = sorted(cca_names_set, key=lambda n: (0 if n == "MPCC" else 1, n))

    cca_tput = {n: [] for n in cca_names}
    cca_p95 = {n: [] for n in cca_names}
    cca_loss = {n: [] for n in cca_names}
    cca_util = {n: [] for n in cca_names}

    for trace_name, metrics in all_metrics.items():
        for cca_name in cca_names:
            if cca_name in metrics:
                m = metrics[cca_name]
                cca_tput[cca_name].append(m.avg_throughput_mbps)
                cca_p95[cca_name].append(m.p95_rtt_ms)
                cca_loss[cca_name].append(m.loss_rate * 100)
                cca_util[cca_name].append(m.link_utilization * 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def bar_with_error(ax, data_dict, ylabel, bar_title):
        means = [np.mean(data_dict[n]) for n in cca_names]
        stds = [np.std(data_dict[n]) for n in cca_names]
        colors = [_color(n) for n in cca_names]
        x = np.arange(len(cca_names))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5,
                      error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(cca_names, rotation=25)
        ax.set_ylabel(ylabel)
        ax.set_title(bar_title, fontweight="bold", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    bar_with_error(axes[0, 0], cca_tput, "Mbps", "Average Throughput (mean ± std)")
    bar_with_error(axes[0, 1], cca_p95, "ms", "95th Percentile RTT (mean ± std)")
    bar_with_error(axes[1, 0], cca_loss, "%", "Loss Rate (mean ± std)")
    bar_with_error(axes[1, 1], cca_util, "%", "Link Utilization (mean ± std)")

    n_traces = len(all_metrics)
    trace_list = ", ".join(all_metrics.keys())
    fig.suptitle(f"{title}\n({n_traces} traces: {trace_list})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig
