"""Train a PPO policy on one of the four scenario environments."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import statistics
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import mahimahi_gym  # noqa: F401


SCENARIO_ENV_IDS = {
    "MahiWiredBottleneck-v0",
    "MahiCellularBursty-v0",
    "MahiLeoSatellite-v0",
    "MahiMultiFlowFairness-v0",
}


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


class ProgressCallback(BaseCallback):
    """Print compact training progress every few PPO updates."""

    def __init__(self, print_every_steps: int = 1024, label: str = "") -> None:
        super().__init__()
        self.print_every_steps = print_every_steps
        self.label = label
        self._next_print = print_every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_print:
            prefix = f"{self.label} " if self.label else ""
            ep_reward = self.logger.name_to_value.get("rollout/ep_rew_mean")
            if ep_reward is None:
                print(f"{prefix}timesteps={self.num_timesteps}")
            else:
                print(f"{prefix}timesteps={self.num_timesteps} ep_rew_mean={ep_reward:.3f}")
            self._next_print += self.print_every_steps
        return True


def make_env(args: argparse.Namespace, log_path: Path | None = None, seed: int | None = None):
    env_kwargs: dict[str, object] = {
        "episode_steps": args.episode_steps,
        "reward_mode": args.reward_mode,
        "mpcc_contour_weight": args.mpcc_contour_weight,
        "mpcc_lag_weight": args.mpcc_lag_weight,
        "mpcc_delay_weight": args.mpcc_delay_weight,
        "mpcc_power_weight": args.mpcc_power_weight,
        "mpcc_smoothing_weight": args.mpcc_smoothing_weight,
        "mpcc_fairness_weight": args.mpcc_fairness_weight,
        "mpcc_loss_weight": args.mpcc_loss_weight,
        "mpcc_reference_delay_alpha_ms": args.mpcc_reference_delay_alpha_ms,
        "mpcc_target_rtt_ms": args.mpcc_target_rtt_ms,
        "mpcc_throughput_tau_s": args.mpcc_throughput_tau_s,
        "mpcc_rtt_tau_s": args.mpcc_rtt_tau_s,
        "mpcc_base_rtt_ms": args.mpcc_base_rtt_ms,
    }
    if args.env_id == "MahiMultiFlowFairness-v0":
        env_kwargs["n_flows"] = args.n_flows
    if args.use_mahimahi:
        env_kwargs.update(
            {
                "controller_cmd": args.controller_cmd,
                "mahimahi_cmd": args.mahimahi_cmd,
                "use_mahimahi": True,
                "timeout_s": args.timeout_s,
                "restart_on_reset": args.restart_on_reset,
            }
        )
        if args.uplink_trace is not None:
            env_kwargs["uplink_trace"] = args.uplink_trace
        if args.downlink_trace is not None:
            env_kwargs["downlink_trace"] = args.downlink_trace

    env = gym.make(args.env_id, **env_kwargs)
    if seed is not None:
        env.action_space.seed(seed)
    return Monitor(
        env,
        filename=str(log_path) if log_path is not None else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-id",
        default="MahiWiredBottleneck-v0",
        choices=sorted(SCENARIO_ENV_IDS),
        help="Registered scenario environment ID.",
    )
    parser.add_argument("--episode-steps", type=int, default=20, help="Episode length.")
    parser.add_argument("--n-flows", type=int, default=3, help="Number of flows for MahiMultiFlowFairness-v0.")
    parser.add_argument(
        "--reward-mode",
        choices=["default", "mpcc"],
        default="default",
        help="Reward function to use in the scenario environments.",
    )
    parser.add_argument("--mpcc-contour-weight", type=float, default=1e-3, help="MPCC contouring-error weight.")
    parser.add_argument("--mpcc-lag-weight", type=float, default=1e-3, help="MPCC lag-error weight.")
    parser.add_argument("--mpcc-delay-weight", type=float, default=1e-4, help="MPCC target-RTT violation weight.")
    parser.add_argument("--mpcc-power-weight", type=float, default=1.0, help="MPCC Kleinrock power reward weight.")
    parser.add_argument("--mpcc-smoothing-weight", type=float, default=0.1, help="MPCC input smoothing weight.")
    parser.add_argument("--mpcc-fairness-weight", type=float, default=1e-2, help="MPCC multi-flow fair-share weight.")
    parser.add_argument("--mpcc-loss-weight", type=float, default=0.0, help="Optional MPCC loss-fraction penalty weight.")
    parser.add_argument(
        "--mpcc-reference-delay-alpha-ms",
        type=float,
        default=None,
        help="MPCC reference-curve delay alpha. Defaults to half the queue delay scale.",
    )
    parser.add_argument(
        "--mpcc-target-rtt-ms",
        type=float,
        default=None,
        help="MPCC target RTT threshold. Defaults to R0 plus one quarter of the queue delay scale.",
    )
    parser.add_argument("--mpcc-throughput-tau-s", type=float, default=0.3, help="MPCC throughput smoothing time constant.")
    parser.add_argument("--mpcc-rtt-tau-s", type=float, default=0.3, help="MPCC RTT smoothing time constant.")
    parser.add_argument("--mpcc-base-rtt-ms", type=float, default=20.0, help="MPCC base RTT for envs without propagation delay.")
    parser.add_argument(
        "--controller-cmd",
        default=f"{sys.executable} examples/mm_link_json_controller.py",
        help="Controller command used when --use-mahimahi is enabled.",
    )
    parser.add_argument("--uplink-trace", default=None, help="Uplink trace for mm-link.")
    parser.add_argument("--downlink-trace", default=None, help="Downlink trace for mm-link.")
    parser.add_argument("--mahimahi-cmd", default="mm-link", help="Path or command name for mm-link.")
    parser.add_argument("--use-mahimahi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-s", type=float, default=5.0, help="Controller measurement timeout.")
    parser.add_argument(
        "--restart-on-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restart the mm-link/controller subprocess on every episode reset.",
    )
    parser.add_argument("--timesteps", type=int, default=5_000, help="Total PPO training timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Deterministic evaluation episodes.")
    parser.add_argument("--seeds", default="7", help="Comma-separated random seeds.")
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_scenario"))
    parser.add_argument("--plot-path", type=Path, default=Path("plots/ppo_returns.png"))
    parser.add_argument("--monitor-path", type=Path, default=Path("runs/ppo_monitor"))
    parser.add_argument("--summary-path", type=Path, default=Path("runs/ppo_seed_summary.csv"))
    parser.add_argument("--aggregate-plot-path", type=Path, default=Path("plots/ppo_seed_returns.png"))
    return parser.parse_args()


def parse_seeds(seed_text: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def read_returns(monitor_path: Path) -> list[float]:
    returns: list[float] = []
    with monitor_path.open(newline="", encoding="utf-8") as monitor_file:
        rows = (row for row in monitor_file if not row.startswith("#"))
        for row in csv.DictReader(rows):
            returns.append(float(row["r"]))

    if not returns:
        raise RuntimeError(f"No episode returns found in {monitor_path}")
    return returns


def moving_average(values: list[float], window: int) -> list[float]:
    return [
        sum(values[max(0, idx - window + 1) : idx + 1]) / min(window, idx + 1)
        for idx in range(len(values))
    ]


def plot_returns(monitor_path: Path, plot_path: Path) -> None:
    returns = read_returns(monitor_path)
    window = min(25, len(returns))
    average = moving_average(returns, window)

    plot_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(returns, color="#6b7280", alpha=0.35, linewidth=1.0, label="episode return")
    plt.plot(average, color="#2563eb", linewidth=2.0, label=f"{window}-episode mean")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO Training Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def plot_seed_returns(monitor_paths: list[Path], plot_path: Path) -> None:
    seed_averages = []
    for monitor_path in monitor_paths:
        returns = read_returns(monitor_path)
        seed_averages.append(moving_average(returns, min(25, len(returns))))

    min_len = min(len(values) for values in seed_averages)
    aggregate = [
        statistics.fmean(values[idx] for values in seed_averages)
        for idx in range(min_len)
    ]

    plot_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 5))
    for values in seed_averages:
        plt.plot(values, color="#9ca3af", alpha=0.35, linewidth=1.0)
    plt.plot(aggregate, color="#dc2626", linewidth=2.2, label="mean across seeds")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO Return Across Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def seed_path(path: Path, seed: int, multi_seed: bool) -> Path:
    if not multi_seed:
        return path
    return path.with_name(f"{path.name}_seed_{seed}")


def confidence_interval_95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean

    sample_std = statistics.stdev(values)
    dof = len(values) - 1
    t_critical = T_CRITICAL_95.get(dof, 1.96)
    half_width = t_critical * sample_std / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def train_one_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> dict[str, float | int | str]:
    model_path = seed_path(args.model_path, seed, multi_seed)
    plot_path = seed_path(args.plot_path.with_suffix(""), seed, multi_seed).with_suffix(args.plot_path.suffix)
    monitor_path = seed_path(args.monitor_path, seed, multi_seed)
    monitor_csv_path = monitor_path.with_suffix(".monitor.csv")

    model_path.parent.mkdir(exist_ok=True)
    plot_path.parent.mkdir(exist_ok=True)
    monitor_path.parent.mkdir(exist_ok=True)

    env = make_env(args, monitor_path, seed=seed)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=8,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=seed,
        )

        model.learn(total_timesteps=args.timesteps, callback=ProgressCallback(label=f"seed={seed}"))

        eval_env = make_env(args, seed=seed + 10_000)
        try:
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
        finally:
            eval_env.close()

        model.save(model_path)
    finally:
        env.close()

    plot_returns(monitor_csv_path, plot_path)

    print(f"seed={seed} mean_reward={mean_reward:.3f} std_reward={std_reward:.3f}")
    print(f"seed={seed} saved_model={model_path}.zip")
    print(f"seed={seed} saved_plot={plot_path}")

    return {
        "seed": seed,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "model_path": f"{model_path}.zip",
        "plot_path": str(plot_path),
        "monitor_path": str(monitor_csv_path),
    }


def write_summary(rows: list[dict[str, float | int | str]], summary_path: Path) -> None:
    summary_path.parent.mkdir(exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=["seed", "mean_reward", "std_reward", "model_path", "plot_path", "monitor_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if isinstance(args.controller_cmd, str):
        args.controller_cmd = shlex.split(args.controller_cmd)
    seeds = parse_seeds(args.seeds)
    multi_seed = len(seeds) > 1
    rows = [train_one_seed(args, seed, multi_seed) for seed in seeds]

    write_summary(rows, args.summary_path)
    mean_rewards = [float(row["mean_reward"]) for row in rows]
    mean, lower, upper = confidence_interval_95(mean_rewards)

    if multi_seed:
        monitor_paths = [Path(str(row["monitor_path"])) for row in rows]
        plot_seed_returns(monitor_paths, args.aggregate_plot_path)
        print(f"aggregate_plot={args.aggregate_plot_path}")

    print(f"summary={args.summary_path}")
    print(f"mean_reward_across_seeds={mean:.3f}")
    print(f"confidence_interval_95=[{lower:.3f}, {upper:.3f}]")


if __name__ == "__main__":
    main()
