"""
Visualization utilities for MPCC congestion control experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from baselines import ExperimentResult


def plot_throughput_delay_timeseries(
    results: Sequence[ExperimentResult],
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    fig, (ax_tput, ax_rtt) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for r in results:
        if r.throughput_ts:
            t, tput = zip(*r.throughput_ts)
            ax_tput.plot(t, tput, label=r.algorithm, alpha=0.8)
        if r.rtt_ts:
            t, rtt = zip(*r.rtt_ts)
            ax_rtt.plot(t, rtt, label=r.algorithm, alpha=0.8)

    ax_tput.set_ylabel("Throughput (Mbps)")
    ax_tput.legend(loc="upper right")
    ax_tput.grid(True, alpha=0.3)
    ax_rtt.set_ylabel("RTT (ms)")
    ax_rtt.set_xlabel("Time (s)")
    ax_rtt.legend(loc="upper right")
    ax_rtt.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_throughput_delay_scatter(
    results: Sequence[ExperimentResult],
    bw_est: float | None = None,
    rtt_prop: float = 0.025,
    alpha: float = 0.150,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    if bw_est is not None:
        theta = np.linspace(0, 1, 100)
        tput_ref = bw_est * theta / 1e6
        rtt_ref = (rtt_prop + alpha * theta ** 2) * 1000
        ax.plot(rtt_ref, tput_ref, "k--", linewidth=2, label="Reference trajectory", zorder=10)
        corridor_width = 20
        ax.fill_betweenx(tput_ref, rtt_ref - corridor_width, rtt_ref + corridor_width,
                         alpha=0.1, color="gray", label="Corridor")

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(results), 1)))
    for r, color in zip(results, colors):
        if r.rtt_ts and r.throughput_ts:
            rtts = [rtt for _, rtt in r.rtt_ts]
            tputs = [tput for _, tput in r.throughput_ts]
            n = min(len(rtts), len(tputs))
            ax.scatter(rtts[:n], tputs[:n], s=5, alpha=0.3, color=color, label=r.algorithm)

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_comparison(
    results: Sequence[ExperimentResult],
    metric: str = "avg_throughput_mbps",
    ylabel: str = "Throughput (Mbps)",
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    alg_values: dict[str, list[float]] = {}
    for r in results:
        val = getattr(r, metric, 0.0)
        alg_values.setdefault(r.algorithm, []).append(val)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(alg_values.keys())
    means = [np.mean(vals) for vals in alg_values.values()]
    stds = [np.std(vals) for vals in alg_values.values()]

    ax.bar(names, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cdf(
    data_by_label: dict[str, Sequence[float]],
    xlabel: str = "Value",
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values in data_by_label.items():
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
