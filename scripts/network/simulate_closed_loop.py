"""
Closed-loop MPCC simulation with synthetic bandwidth traces.

Runs the full CasADi solver in the loop: on each timestep, the MPCC
controller observes (throughput, RTT, queue) and outputs a send_rate.
The network dynamics model propagates the state forward.

Compares MPCC against a simple AIMD baseline on the same traces.

Usage (on compute node):
    python scripts/network/simulate_closed_loop.py --save results/closed_loop.png
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import casadi as cd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from planning.network_dynamics import NetworkDynamicsModel
from planning.network_reference import generate_network_reference
from planning.mpcc_controller import MPCCController, Observation


@dataclass
class SimResult:
    name: str
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    tput: np.ndarray = field(default_factory=lambda: np.array([]))
    rtt: np.ndarray = field(default_factory=lambda: np.array([]))
    queue: np.ndarray = field(default_factory=lambda: np.array([]))
    send_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    bw: np.ndarray = field(default_factory=lambda: np.array([]))
    solve_times: list[float] = field(default_factory=list)


def bandwidth_trace(t: float, scenario: str = "cellular") -> float:
    if scenario == "constant":
        return 10e6
    elif scenario == "step":
        return 10e6 if t < 3.0 else 5e6 if t < 6.0 else 10e6
    elif scenario == "cellular":
        base = 8e6
        slow_fade = 2e6 * np.sin(2 * np.pi * t / 4.0)
        fast_fade = 1e6 * np.sin(2 * np.pi * t / 0.3)
        return max(base + slow_fade + fast_fade, 1e6)
    elif scenario == "ramp":
        return min(2e6 + t * 2e6, 12e6)
    return 10e6


def simulate_mpcc(scenario: str, duration: float = 10.0, dt: float = 0.05,
                  solver_type: str = "nlp") -> SimResult:
    model = NetworkDynamicsModel()
    controller = MPCCController(solver_type=solver_type)

    x = cd.DM([0.0, model.rtt_prop, 0.0, 0.0, 0.0])
    p = cd.DM([])

    n_steps = int(duration / dt)
    result = SimResult(name="MPCC")
    records = {k: [] for k in ["t", "tput", "rtt", "queue", "send_rate", "bw"]}

    for step in range(n_steps):
        t = step * dt
        bw = bandwidth_trace(t, scenario)

        tput_val = float(x[0])
        rtt_val = float(x[1])
        queue_val = float(x[3])

        obs = Observation(tput=tput_val, rtt=rtt_val, queue=queue_val, bw_est=bw)

        t0 = time.monotonic()
        try:
            send_rate = controller.step(obs)
        except Exception:
            send_rate = controller._fallback_rate(obs)
        solve_time = time.monotonic() - t0
        result.solve_times.append(solve_time)

        records["t"].append(t)
        records["tput"].append(tput_val / 1e6)
        records["rtt"].append(rtt_val * 1000)
        records["queue"].append(queue_val / 1000)
        records["send_rate"].append(send_rate / 1e6)
        records["bw"].append(bw / 1e6)

        u = cd.DM([send_rate])
        x = model.symbolic_dynamics(x, u, p, dt)

    for k in records:
        setattr(result, k, np.array(records[k]))
    return result


def simulate_aimd(scenario: str, duration: float = 10.0, dt: float = 0.05) -> SimResult:
    model = NetworkDynamicsModel()

    x = cd.DM([0.0, model.rtt_prop, 0.0, 0.0, 0.0])
    p = cd.DM([])
    cwnd = 10 * 1460.0
    rtt_min = model.rtt_prop

    n_steps = int(duration / dt)
    result = SimResult(name="AIMD")
    records = {k: [] for k in ["t", "tput", "rtt", "queue", "send_rate", "bw"]}

    for step in range(n_steps):
        t = step * dt
        bw = bandwidth_trace(t, scenario)

        tput_val = float(x[0])
        rtt_val = max(float(x[1]), 0.001)
        queue_val = float(x[3])

        send_rate = cwnd / rtt_val

        if queue_val > 50000:
            cwnd *= 0.5
        else:
            cwnd += 1460.0 * (1460.0 / cwnd) * dt * 20

        cwnd = max(cwnd, 1460.0 * 2)
        cwnd = min(cwnd, model.rate_max * rtt_val)

        records["t"].append(t)
        records["tput"].append(tput_val / 1e6)
        records["rtt"].append(rtt_val * 1000)
        records["queue"].append(queue_val / 1000)
        records["send_rate"].append(send_rate / 1e6)
        records["bw"].append(bw / 1e6)

        u = cd.DM([send_rate])
        x = model.symbolic_dynamics(x, u, p, dt)

    for k in records:
        setattr(result, k, np.array(records[k]))
    return result


def plot_comparison(mpcc: SimResult, aimd: SimResult, scenario: str,
                    save_path: Path | None = None):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(mpcc.t, mpcc.bw, "k:", linewidth=1.5, label="Capacity")
    ax1.plot(mpcc.t, mpcc.send_rate, "b-", linewidth=1.5, label="MPCC send rate")
    ax1.plot(mpcc.t, mpcc.tput, "b--", alpha=0.7, label="MPCC throughput")
    ax1.set_ylabel("Rate (Mbps)")
    ax1.set_title(f"MPCC — {scenario}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(aimd.t, aimd.bw, "k:", linewidth=1.5, label="Capacity")
    ax2.plot(aimd.t, aimd.send_rate, "r-", linewidth=1.5, label="AIMD send rate")
    ax2.plot(aimd.t, aimd.tput, "r--", alpha=0.7, label="AIMD throughput")
    ax2.set_ylabel("Rate (Mbps)")
    ax2.set_title(f"AIMD — {scenario}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(mpcc.t, mpcc.rtt, "b-", label="MPCC")
    ax3.plot(aimd.t, aimd.rtt, "r-", label="AIMD")
    ax3.set_ylabel("RTT (ms)")
    ax3.set_title("RTT Comparison")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(mpcc.t, mpcc.queue, "b-", label="MPCC")
    ax4.plot(aimd.t, aimd.queue, "r-", label="AIMD")
    ax4.set_ylabel("Queue (KB)")
    ax4.set_title("Queue Occupancy")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ref = generate_network_reference(bw_est=8e6, rtt_prop=0.025, alpha=0.150)
    ref_tput = np.array(ref.x) / 1e6
    ref_rtt = np.array(ref.y) * 1000

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ref_rtt, ref_tput, "k--", linewidth=2, label="Reference", zorder=10)
    ax5.scatter(mpcc.rtt, mpcc.tput, s=5, alpha=0.4, c="blue", label="MPCC")
    ax5.scatter(aimd.rtt, aimd.tput, s=5, alpha=0.4, c="red", label="AIMD")
    ax5.set_xlabel("RTT (ms)")
    ax5.set_ylabel("Throughput (Mbps)")
    ax5.set_title("Throughput-Delay Space")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    metrics = {
        "Avg Tput\n(Mbps)": (np.mean(mpcc.tput), np.mean(aimd.tput)),
        "p95 RTT\n(ms)": (np.percentile(mpcc.rtt, 95), np.percentile(aimd.rtt, 95)),
        "Avg Queue\n(KB)": (np.mean(mpcc.queue), np.mean(aimd.queue)),
    }
    x_pos = np.arange(len(metrics))
    width = 0.35
    mpcc_vals = [v[0] for v in metrics.values()]
    aimd_vals = [v[1] for v in metrics.values()]
    ax6.bar(x_pos - width / 2, mpcc_vals, width, label="MPCC", color="blue", alpha=0.7)
    ax6.bar(x_pos + width / 2, aimd_vals, width, label="AIMD", color="red", alpha=0.7)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(list(metrics.keys()))
    ax6.set_title("Metrics Comparison")
    ax6.legend(fontsize=8)
    ax6.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"MPCC vs AIMD — Closed-Loop Simulation ({scenario})",
                 fontsize=14, fontweight="bold")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Closed-loop MPCC simulation")
    parser.add_argument("--scenario", default="cellular",
                        choices=["constant", "step", "cellular", "ramp"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    print(f"Running closed-loop simulation: scenario={args.scenario}, duration={args.duration}s")

    print("Simulating MPCC (NLP)...")
    mpcc_result = simulate_mpcc(args.scenario, args.duration)
    print(f"  Avg throughput: {np.mean(mpcc_result.tput):.2f} Mbps")
    print(f"  p95 RTT: {np.percentile(mpcc_result.rtt, 95):.1f} ms")
    print(f"  Avg solve time: {np.mean(mpcc_result.solve_times)*1000:.1f} ms")

    print("Simulating MPCC (QP)...")
    qp_result = simulate_mpcc(args.scenario, args.duration, solver_type="qp")
    qp_result.name = "MPCC-QP"
    print(f"  Avg throughput: {np.mean(qp_result.tput):.2f} Mbps")
    print(f"  p95 RTT: {np.percentile(qp_result.rtt, 95):.1f} ms")
    print(f"  Avg solve time: {np.mean(qp_result.solve_times)*1000:.1f} ms")

    print("Simulating AIMD...")
    aimd_result = simulate_aimd(args.scenario, args.duration)
    print(f"  Avg throughput: {np.mean(aimd_result.tput):.2f} Mbps")
    print(f"  p95 RTT: {np.percentile(aimd_result.rtt, 95):.1f} ms")

    plot_comparison(mpcc_result, aimd_result, args.scenario,
                    save_path=Path(args.save) if args.save else None)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
