"""
Visualize MPCC network dynamics: forward simulation with reference trajectory.

State mapping: x[0]=throughput, x[1]=RTT, x[2]=psi, x[3]=queue, x[4]=spline

Usage (on compute node):
    python scripts/network/visualize_dynamics.py --save results/network_demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import casadi as cd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from planning.network_dynamics import NetworkDynamicsModel
from planning.network_reference import generate_network_reference


def simulate_scenario(
    model: NetworkDynamicsModel,
    send_rate_schedule: list[tuple[float, float]],
    bw_schedule: list[tuple[float, float]],
    dt: float = 0.05,
    total_time: float = 5.0,
) -> dict[str, np.ndarray]:
    n_steps = int(total_time / dt)

    bw_init = bw_schedule[0][1]
    x = cd.DM([0.0, model.rtt_prop, 0.0, 0.0, 0.0])
    p = cd.DM([])

    records = {k: [] for k in ["t", "q", "rtt", "tput", "bw_est", "spline", "send_rate"]}

    def get_value(schedule, t):
        val = schedule[0][1]
        for t_start, v in schedule:
            if t >= t_start:
                val = v
        return val

    for step in range(n_steps):
        t = step * dt
        sr = get_value(send_rate_schedule, t)
        bw = get_value(bw_schedule, t)

        records["t"].append(t)
        records["tput"].append(float(x[0]) / 1e6)
        records["rtt"].append(float(x[1]) * 1000)
        records["q"].append(float(x[3]))
        records["bw_est"].append(bw / 1e6)
        records["spline"].append(float(x[4]))
        records["send_rate"].append(sr / 1e6)

        u = cd.DM([sr])
        x = model.symbolic_dynamics(x, u, p, dt)

    return {k: np.array(v) for k, v in records.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize MPCC network dynamics")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    model = NetworkDynamicsModel()
    bw_nominal = 10e6

    ramp = simulate_scenario(
        model,
        send_rate_schedule=[(0.0, 2e6), (1.0, 5e6), (2.0, 8e6), (3.0, 9.5e6)],
        bw_schedule=[(0.0, bw_nominal)],
        total_time=5.0,
    )

    drop = simulate_scenario(
        model,
        send_rate_schedule=[(0.0, 8e6)],
        bw_schedule=[(0.0, 10e6), (2.0, 4e6), (3.5, 10e6)],
        total_time=5.0,
    )

    overload = simulate_scenario(
        model,
        send_rate_schedule=[(0.0, 15e6)],
        bw_schedule=[(0.0, 10e6)],
        total_time=5.0,
    )

    ref = generate_network_reference(bw_est=bw_nominal, rtt_prop=model.rtt_prop, alpha=0.150)
    ref_tput = np.array(ref.x) / 1e6
    ref_rtt = np.array(ref.y) * 1000

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ref_rtt, ref_tput, "k--", linewidth=2, label="Reference Γ(θ)", zorder=10)
    ax1.fill_betweenx(ref_tput, ref_rtt - 10, ref_rtt + 10, alpha=0.1, color="gray", label="Corridor")
    ax1.scatter(ramp["rtt"], ramp["tput"], s=8, alpha=0.6, c=ramp["t"], cmap="Blues", label="Ramp-up")
    ax1.scatter(drop["rtt"], drop["tput"], s=8, alpha=0.6, c=drop["t"], cmap="Reds", label="BW drop")
    ax1.scatter(overload["rtt"], overload["tput"], s=8, alpha=0.6, c=overload["t"], cmap="Greens", label="Overload")
    ax1.set_xlabel("RTT (ms)")
    ax1.set_ylabel("Throughput (Mbps)")
    ax1.set_title("Operating Points in Throughput-Delay Space")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(200, max(overload["rtt"]) * 1.1))

    ax2 = fig.add_subplot(gs[0, 1])
    theta = np.linspace(0, 1, 100)
    ax2.plot(theta, ref_tput[np.linspace(0, len(ref_tput) - 1, 100).astype(int)], "b-", linewidth=2, label="tput_ref(θ)")
    ax2_rtt = ax2.twinx()
    ax2_rtt.plot(theta, ref_rtt[np.linspace(0, len(ref_rtt) - 1, 100).astype(int)], "r-", linewidth=2, label="rtt_ref(θ)")
    ax2.set_xlabel("θ (utilization parameter)")
    ax2.set_ylabel("Throughput (Mbps)", color="b")
    ax2_rtt.set_ylabel("RTT (ms)", color="r")
    ax2.set_title("Reference Trajectory Γ(θ)")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ramp["t"], ramp["send_rate"], "g--", label="Send rate", alpha=0.7)
    ax3.plot(ramp["t"], ramp["tput"], "b-", label="Throughput", linewidth=2)
    ax3.plot(ramp["t"], ramp["bw_est"], "k:", label="Bandwidth", linewidth=1.5)
    ax3.set_ylabel("Rate (Mbps)")
    ax3.set_title("Scenario: Ramp-Up to Steady State")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3r = ax3.twinx()
    ax3r.plot(ramp["t"], ramp["rtt"], "r-", alpha=0.5, label="RTT")
    ax3r.set_ylabel("RTT (ms)", color="r")
    ax3.set_xlabel("Time (s)")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(drop["t"], drop["send_rate"], "g--", label="Send rate", alpha=0.7)
    ax4.plot(drop["t"], drop["tput"], "b-", label="Throughput", linewidth=2)
    ax4.plot(drop["t"], drop["bw_est"], "k:", label="Bandwidth", linewidth=1.5)
    ax4.set_ylabel("Rate (Mbps)")
    ax4.set_title("Scenario: Bandwidth Drop (Cellular Handoff)")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4r = ax4.twinx()
    ax4r.plot(drop["t"], drop["rtt"], "r-", alpha=0.5, label="RTT")
    ax4r.set_ylabel("RTT (ms)", color="r")
    ax4.set_xlabel("Time (s)")

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ramp["t"], np.array(ramp["q"]) / 1000, label="Ramp-up", linewidth=2)
    ax5.plot(drop["t"], np.array(drop["q"]) / 1000, label="BW drop", linewidth=2)
    ax5.plot(overload["t"], np.array(overload["q"]) / 1000, label="Overload", linewidth=2)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Queue (KB)")
    ax5.set_title("Queue Occupancy")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(ramp["t"], ramp["spline"], label="Ramp-up", linewidth=2)
    ax6.plot(drop["t"], drop["spline"], label="BW drop", linewidth=2)
    ax6.plot(overload["t"], overload["spline"], label="Overload", linewidth=2)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Spline Progress (utilization × time)")
    ax6.set_title("Reference Trajectory Progress")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle("MPCC Network Dynamics — Forward Simulation", fontsize=14, fontweight="bold")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if args.show:
        plt.show()
    else:
        print("Use --show to display interactively, or --save <path> to save.")


if __name__ == "__main__":
    main()
