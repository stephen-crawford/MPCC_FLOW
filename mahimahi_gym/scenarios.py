"""Scenario environments inspired by mahimahi link shells."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mahimahi_gym.mm_link import MahimahiSubprocessEnv


DEFAULT_CONTROLLER_CMD = [sys.executable, "examples/mm_link_json_controller.py"]
WIRED_UPLINK_TRACE = "mahimahi/12mbps.trace"
WIRED_DOWNLINK_TRACE = "mahimahi/12mbps.trace"
CELLULAR_UPLINK_TRACE = "mahimahi/traces/TMobile-LTE-driving.up"
CELLULAR_DOWNLINK_TRACE = "mahimahi/traces/TMobile-LTE-driving.down"
LEO_UPLINK_TRACE = "mahimahi/traces/Verizon-LTE-short.up"
LEO_DOWNLINK_TRACE = "mahimahi/traces/Verizon-LTE-short.down"


@dataclass(frozen=True)
class LinkTrace:
    """Discrete bottleneck trace consumed by the in-process simulators.

    Each index represents one simulator step. During that step, the link uses
    the matching duration, service capacity, and propagation delay values.

    Args:
        durations_ms: Step duration for each trace entry, in milliseconds.
        capacities_mbps: Bottleneck service rate for each trace entry, in Mbps.
        delays_ms: Base propagation delay for each trace entry, in milliseconds.
    """

    durations_ms: np.ndarray
    capacities_mbps: np.ndarray
    delays_ms: np.ndarray

    @property
    def max_capacity_mbps(self) -> float:
        return float(self.capacities_mbps.max(initial=1.0))

    @property
    def max_delay_ms(self) -> float:
        return float(self.delays_ms.max(initial=1.0))


@dataclass(frozen=True)
class MpccRewardConfig:
    """Weights and time constants for the MPCC-style reward.

    The MPCC objective is a cost, while Gymnasium expects a reward, so the
    environments return the negative of this cost. The default weights are
    intentionally modest because contouring, lag, and delay errors are measured
    in physical units.

    Args:
        contour_weight: Weight ``w_c`` for squared contouring error.
        lag_weight: Weight ``w_l`` for squared lag error.
        delay_weight: Weight ``w_d`` for RTT above the target threshold.
        power_weight: Weight ``w_p`` for the normalized Kleinrock power term.
        smoothing_weight: Weight ``w_u`` for changes in sending rate.
        fairness_weight: Weight ``w_f`` for per-flow fair-share deviation.
        loss_weight: Optional extra weight for packet-loss fraction. This is
            not part of the MPCC expression in the paper text, so it defaults
            to zero.
        reference_delay_alpha_ms: Curve parameter ``alpha`` in
            ``Gamma(theta) = (C theta, R0 + alpha theta^2)``. If omitted, the
            environment uses half of its queue-size delay scale.
        target_rtt_ms: RTT threshold ``R*``. If omitted, the environment uses
            ``R0 + 0.25 * queue_size_ms``.
        throughput_tau_s: Smoothing time constant for ``hat T``.
        rtt_tau_s: Smoothing time constant for ``hat R``.
        base_rtt_ms: Baseline RTT used when a simulator does not have an
            explicit propagation-delay signal, such as the multi-flow scenario.
    """

    contour_weight: float = 1e-3
    lag_weight: float = 1e-3
    delay_weight: float = 1e-4
    power_weight: float = 1.0
    smoothing_weight: float = 0.1
    fairness_weight: float = 1e-2
    loss_weight: float = 0.0
    reference_delay_alpha_ms: float | None = None
    target_rtt_ms: float | None = None
    throughput_tau_s: float = 0.3
    rtt_tau_s: float = 0.3
    base_rtt_ms: float = 20.0


def _make_mpcc_config(
    *,
    mpcc_contour_weight: float = 1e-3,
    mpcc_lag_weight: float = 1e-3,
    mpcc_delay_weight: float = 1e-4,
    mpcc_power_weight: float = 1.0,
    mpcc_smoothing_weight: float = 0.1,
    mpcc_fairness_weight: float = 1e-2,
    mpcc_loss_weight: float = 0.0,
    mpcc_reference_delay_alpha_ms: float | None = None,
    mpcc_target_rtt_ms: float | None = None,
    mpcc_throughput_tau_s: float = 0.3,
    mpcc_rtt_tau_s: float = 0.3,
    mpcc_base_rtt_ms: float = 20.0,
) -> MpccRewardConfig:
    return MpccRewardConfig(
        contour_weight=float(mpcc_contour_weight),
        lag_weight=float(mpcc_lag_weight),
        delay_weight=float(mpcc_delay_weight),
        power_weight=float(mpcc_power_weight),
        smoothing_weight=float(mpcc_smoothing_weight),
        fairness_weight=float(mpcc_fairness_weight),
        loss_weight=float(mpcc_loss_weight),
        reference_delay_alpha_ms=None
        if mpcc_reference_delay_alpha_ms is None
        else float(mpcc_reference_delay_alpha_ms),
        target_rtt_ms=None if mpcc_target_rtt_ms is None else float(mpcc_target_rtt_ms),
        throughput_tau_s=float(mpcc_throughput_tau_s),
        rtt_tau_s=float(mpcc_rtt_tau_s),
        base_rtt_ms=float(mpcc_base_rtt_ms),
    )


def _smooth_measurement(previous: float, measurement: float, duration_s: float, tau_s: float) -> float:
    alpha = 1.0 if tau_s <= 0.0 else 1.0 - float(np.exp(-max(duration_s, 0.0) / tau_s))
    return previous + alpha * (measurement - previous)


def _mpcc_reward(
    *,
    smoothed_throughput_mbps: float,
    smoothed_rtt_ms: float,
    capacity_mbps: float,
    base_rtt_ms: float,
    queue_size_ms: float,
    current_rates_mbps: np.ndarray,
    previous_rates_mbps: np.ndarray,
    max_rate_mbps: float,
    loss_fraction: float,
    config: MpccRewardConfig,
) -> tuple[float, dict[str, float]]:
    capacity = max(float(capacity_mbps), 1e-6)
    base_rtt = max(float(base_rtt_ms), 1e-6)
    throughput_hat = max(0.0, float(smoothed_throughput_mbps))
    rtt_hat = max(1e-6, float(smoothed_rtt_ms))
    theta = float(np.clip(throughput_hat / capacity, 0.0, 1.0))

    reference_alpha = (
        float(config.reference_delay_alpha_ms)
        if config.reference_delay_alpha_ms is not None
        else 0.5 * float(queue_size_ms)
    )
    target_rtt = (
        float(config.target_rtt_ms)
        if config.target_rtt_ms is not None
        else base_rtt + 0.25 * float(queue_size_ms)
    )

    gamma_t = capacity * theta
    gamma_r = base_rtt + reference_alpha * theta**2
    tangent_norm = float(np.sqrt(capacity**2 + (2.0 * reference_alpha * theta) ** 2))
    tangent_t = capacity / max(tangent_norm, 1e-6)
    tangent_r = (2.0 * reference_alpha * theta) / max(tangent_norm, 1e-6)

    throughput_error = throughput_hat - gamma_t
    rtt_error = rtt_hat - gamma_r
    contour_error = tangent_r * throughput_error - tangent_t * rtt_error
    lag_error = tangent_t * throughput_error + tangent_r * rtt_error
    delay_excess = max(0.0, rtt_hat - target_rtt)
    power = (throughput_hat / capacity) / max(rtt_hat / base_rtt, 1e-6)

    rate_delta = current_rates_mbps - previous_rates_mbps
    smoothing_penalty = config.smoothing_weight * float(
        np.square(rate_delta).sum() / max(max_rate_mbps**2, 1e-6)
    )
    fair_share = capacity / max(len(current_rates_mbps), 1)
    fairness_penalty = 0.0
    if len(current_rates_mbps) > 1:
        fairness_penalty = config.fairness_weight * float(np.square(current_rates_mbps - fair_share).sum())

    contour_cost = config.contour_weight * contour_error**2
    lag_cost = config.lag_weight * lag_error**2
    delay_cost = config.delay_weight * delay_excess**2
    loss_cost = config.loss_weight * float(loss_fraction) ** 2
    stage_cost = (
        contour_cost
        + lag_cost
        + delay_cost
        + smoothing_penalty
        + fairness_penalty
        + loss_cost
        - config.power_weight * power
    )
    return -float(stage_cost), {
        "mpcc_stage_cost": float(stage_cost),
        "mpcc_contour_error": float(contour_error),
        "mpcc_lag_error": float(lag_error),
        "mpcc_delay_excess_ms": float(delay_excess),
        "mpcc_power": float(power),
        "mpcc_smoothing_penalty": float(smoothing_penalty),
        "mpcc_fairness_penalty": float(fairness_penalty),
        "mpcc_loss_cost": float(loss_cost),
        "mpcc_theta": float(theta),
        "mpcc_reference_throughput_mbps": float(gamma_t),
        "mpcc_reference_rtt_ms": float(gamma_r),
        "mpcc_smoothed_throughput_mbps": float(throughput_hat),
        "mpcc_smoothed_rtt_ms": float(rtt_hat),
    }


class TraceLinkEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-flow congestion-control task over a traced bottleneck link.

    The agent chooses one sending rate in Mbps. The simulator converts that
    rate into sent data for the current trace duration, serves queued data at
    the trace capacity, and drops any data that cannot fit in the finite queue.
    Observations are normalized to ``[0, 1]`` and include send rate,
    throughput, capacity, propagation delay, queueing delay, loss fraction, and
    queue occupancy. Reward is delivered throughput minus delay and loss costs.

    Args:
        trace: Link trace replayed by the simulator.
        max_rate_mbps: Maximum allowed action rate. If omitted, defaults to
            twice the maximum trace capacity, with a minimum of 1 Mbps.
        queue_size_ms: Bottleneck queue size expressed as the amount of data
            that would accumulate over this many milliseconds at current link
            capacity.
        delay_penalty: Reward penalty multiplier applied to RTT in
            milliseconds.
        loss_penalty: Reward penalty multiplier applied to dropped traffic
            fraction.
        episode_steps: Number of ``step`` calls before termination. If omitted,
            one episode spans the full trace length.
        render_mode: Gymnasium render mode. ``"human"`` prints a compact line
            of link statistics each step.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        trace: LinkTrace,
        max_rate_mbps: float | None = None,
        queue_size_ms: float = 300.0,
        delay_penalty: float = 0.015,
        loss_penalty: float = 10.0,
        reward_mode: str = "default",
        mpcc_contour_weight: float = 1e-3,
        mpcc_lag_weight: float = 1e-3,
        mpcc_delay_weight: float = 1e-4,
        mpcc_power_weight: float = 1.0,
        mpcc_smoothing_weight: float = 0.1,
        mpcc_fairness_weight: float = 1e-2,
        mpcc_loss_weight: float = 0.0,
        mpcc_reference_delay_alpha_ms: float | None = None,
        mpcc_target_rtt_ms: float | None = None,
        mpcc_throughput_tau_s: float = 0.3,
        mpcc_rtt_tau_s: float = 0.3,
        mpcc_base_rtt_ms: float = 20.0,
        episode_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.trace = trace
        self.max_rate_mbps = float(max_rate_mbps or max(1.0, trace.max_capacity_mbps * 2.0))
        self.queue_size_ms = float(queue_size_ms)
        self.delay_penalty = float(delay_penalty)
        self.loss_penalty = float(loss_penalty)
        self.reward_mode = reward_mode
        if self.reward_mode not in {"default", "mpcc"}:
            raise ValueError("reward_mode must be 'default' or 'mpcc'.")
        self.mpcc_config = _make_mpcc_config(
            mpcc_contour_weight=mpcc_contour_weight,
            mpcc_lag_weight=mpcc_lag_weight,
            mpcc_delay_weight=mpcc_delay_weight,
            mpcc_power_weight=mpcc_power_weight,
            mpcc_smoothing_weight=mpcc_smoothing_weight,
            mpcc_fairness_weight=mpcc_fairness_weight,
            mpcc_loss_weight=mpcc_loss_weight,
            mpcc_reference_delay_alpha_ms=mpcc_reference_delay_alpha_ms,
            mpcc_target_rtt_ms=mpcc_target_rtt_ms,
            mpcc_throughput_tau_s=mpcc_throughput_tau_s,
            mpcc_rtt_tau_s=mpcc_rtt_tau_s,
            mpcc_base_rtt_ms=mpcc_base_rtt_ms,
        )
        self.episode_steps = int(episode_steps or len(trace.capacities_mbps))
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.max_rate_mbps], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self._step_index = 0
        self._trace_index = 0
        self._queue_mbits = 0.0
        self._last_send_rate_mbps = 0.0
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = 0.0
        self._last_info: dict[str, float] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if hasattr(self, "_real_env"):
            return self._real_env.reset(seed=seed, options=options)
        super().reset(seed=seed)
        start_index = None if options is None else options.get("start_index")
        if start_index is None:
            max_start = max(1, len(self.trace.capacities_mbps) - self.episode_steps)
            self._trace_index = int(self.np_random.integers(0, max_start))
        else:
            self._trace_index = int(start_index) % len(self.trace.capacities_mbps)
        self._step_index = 0
        self._queue_mbits = 0.0
        self._last_send_rate_mbps = 0.0

        capacity = float(self.trace.capacities_mbps[self._trace_index])
        propagation_delay = float(self.trace.delays_ms[self._trace_index])
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = propagation_delay
        obs = self._make_obs(0.0, 0.0, capacity, propagation_delay, 0.0, 0.0)
        self._last_info = self._info(0.0, 0.0, capacity, propagation_delay, 0.0, 0.0, 0.0)
        return obs, dict(self._last_info)

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if hasattr(self, "_real_env"):
            return self._real_env.step(action)
        send_rate_mbps = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        send_rate_mbps = float(np.clip(send_rate_mbps, 0.0, self.max_rate_mbps))

        idx = self._trace_index % len(self.trace.capacities_mbps)
        duration_ms = float(self.trace.durations_ms[idx])
        capacity_mbps = float(self.trace.capacities_mbps[idx])
        propagation_delay_ms = float(self.trace.delays_ms[idx])
        duration_s = duration_ms / 1000.0

        sent_mbits = send_rate_mbps * duration_s
        service_mbits = capacity_mbps * duration_s
        available_buffer_mbits = self._queue_limit_mbits(capacity_mbps) - self._queue_mbits
        accepted_mbits = min(sent_mbits, max(0.0, service_mbits + available_buffer_mbits))
        dropped_mbits = max(0.0, sent_mbits - accepted_mbits)

        queued_before_service = self._queue_mbits + accepted_mbits
        delivered_mbits = min(queued_before_service, service_mbits)
        self._queue_mbits = max(0.0, queued_before_service - delivered_mbits)

        throughput_mbps = delivered_mbits / duration_s if duration_s > 0.0 else 0.0
        queue_delay_ms = self._queue_delay_ms(capacity_mbps)
        rtt_ms = propagation_delay_ms + queue_delay_ms
        loss_fraction = dropped_mbits / sent_mbits if sent_mbits > 0.0 else 0.0
        default_reward = throughput_mbps - self.delay_penalty * rtt_ms - self.loss_penalty * loss_fraction
        reward = default_reward
        reward_info: dict[str, float | str] = {"reward_mode": self.reward_mode, "default_reward": float(default_reward)}
        if self.reward_mode == "mpcc":
            self._smoothed_throughput_mbps = _smooth_measurement(
                self._smoothed_throughput_mbps,
                throughput_mbps,
                duration_s,
                self.mpcc_config.throughput_tau_s,
            )
            self._smoothed_rtt_ms = _smooth_measurement(
                self._smoothed_rtt_ms,
                rtt_ms,
                duration_s,
                self.mpcc_config.rtt_tau_s,
            )
            reward, mpcc_info = _mpcc_reward(
                smoothed_throughput_mbps=self._smoothed_throughput_mbps,
                smoothed_rtt_ms=self._smoothed_rtt_ms,
                capacity_mbps=capacity_mbps,
                base_rtt_ms=propagation_delay_ms,
                queue_size_ms=self.queue_size_ms,
                current_rates_mbps=np.array([send_rate_mbps], dtype=np.float32),
                previous_rates_mbps=np.array([self._last_send_rate_mbps], dtype=np.float32),
                max_rate_mbps=self.max_rate_mbps,
                loss_fraction=loss_fraction,
                config=self.mpcc_config,
            )
            reward_info.update(mpcc_info)

        obs = self._make_obs(
            send_rate_mbps,
            throughput_mbps,
            capacity_mbps,
            propagation_delay_ms,
            queue_delay_ms,
            loss_fraction,
        )
        self._last_info = self._info(
            send_rate_mbps,
            throughput_mbps,
            capacity_mbps,
            propagation_delay_ms,
            queue_delay_ms,
            loss_fraction,
            dropped_mbits,
        )
        self._last_info.update(reward_info)
        self._last_send_rate_mbps = send_rate_mbps

        self._trace_index += 1
        self._step_index += 1
        terminated = self._step_index >= self.episode_steps
        truncated = False
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, dict(self._last_info)

    def render(self) -> None:
        if hasattr(self, "_real_env"):
            self._real_env.render()
            return
        print(
            "rate={send_rate_mbps:.2f} throughput={throughput_mbps:.2f} "
            "capacity={capacity_mbps:.2f} rtt={rtt_ms:.1f} loss={loss_fraction:.3f}".format(**self._last_info)
        )

    def close(self) -> None:
        if hasattr(self, "_real_env"):
            self._real_env.close()

    def _queue_limit_mbits(self, capacity_mbps: float) -> float:
        return max(0.0, capacity_mbps * self.queue_size_ms / 1000.0)

    def _queue_delay_ms(self, capacity_mbps: float) -> float:
        return min(self.queue_size_ms, 1000.0 * self._queue_mbits / max(capacity_mbps, 1e-6))

    def _make_obs(
        self,
        send_rate_mbps: float,
        throughput_mbps: float,
        capacity_mbps: float,
        propagation_delay_ms: float,
        queue_delay_ms: float,
        loss_fraction: float,
    ) -> np.ndarray:
        queue_limit = self._queue_limit_mbits(max(capacity_mbps, 1e-6))
        queue_fraction = self._queue_mbits / queue_limit if queue_limit > 0.0 else 0.0
        return np.array(
            [
                send_rate_mbps / self.max_rate_mbps,
                throughput_mbps / self.max_rate_mbps,
                capacity_mbps / self.max_rate_mbps,
                propagation_delay_ms / max(self.trace.max_delay_ms, 1e-6),
                queue_delay_ms / self.queue_size_ms,
                loss_fraction,
                queue_fraction,
            ],
            dtype=np.float32,
        ).clip(0.0, 1.0)

    def _info(
        self,
        send_rate_mbps: float,
        throughput_mbps: float,
        capacity_mbps: float,
        propagation_delay_ms: float,
        queue_delay_ms: float,
        loss_fraction: float,
        dropped_mbits: float,
    ) -> dict[str, float]:
        return {
            "send_rate_mbps": send_rate_mbps,
            "throughput_mbps": throughput_mbps,
            "capacity_mbps": capacity_mbps,
            "propagation_delay_ms": propagation_delay_ms,
            "queue_delay_ms": queue_delay_ms,
            "rtt_ms": propagation_delay_ms + queue_delay_ms,
            "loss_fraction": loss_fraction,
            "queue_mbits": self._queue_mbits,
            "dropped_mbits": dropped_mbits,
        }


class WiredBottleneckEnv(TraceLinkEnv):
    """Stable wired bottleneck with fixed capacity and low propagation delay.

    This is the simplest scenario: every simulator step uses a 10 Mbps
    bottleneck, 20 ms propagation delay, and 100 ms duration. The useful policy
    is therefore easy to reason about: send near 10 Mbps, avoid building a
    persistent queue, and avoid loss.

    Args:
        episode_steps: Episode length in simulator steps. Also controls the
            length of the generated constant trace in simulator mode.
        **kwargs: Optional scenario settings. ``use_mahimahi`` switches from
            the in-process simulator to the real ``mm-link`` wrapper.
            ``max_rate_mbps`` sets the action upper bound and defaults to
            20 Mbps. ``queue_size_ms`` sets the finite queue size and defaults
            to 150 ms. In real Mahimahi mode, ``uplink_trace`` and
            ``downlink_trace`` override the default wired traces, and any other
            keyword accepted by ``SingleFlowMahimahiScenarioEnv`` is forwarded.
            In simulator mode, remaining keywords are forwarded to
            ``TraceLinkEnv``.
    """

    def __init__(self, episode_steps: int = 100, **kwargs: Any) -> None:
        use_mahimahi = bool(kwargs.pop("use_mahimahi", False))
        max_rate_mbps = kwargs.pop("max_rate_mbps", 20.0)
        queue_size_ms = kwargs.pop("queue_size_ms", 150.0)
        if use_mahimahi:
            self._real_env = SingleFlowMahimahiScenarioEnv(
                uplink_trace=kwargs.pop("uplink_trace", WIRED_UPLINK_TRACE),
                downlink_trace=kwargs.pop("downlink_trace", WIRED_DOWNLINK_TRACE),
                max_rate_mbps=max_rate_mbps,
                max_queue_delay_ms=queue_size_ms,
                episode_steps=episode_steps,
                **kwargs,
            )
            self.action_space = self._real_env.action_space
            self.observation_space = self._real_env.observation_space
            return
        trace = LinkTrace(
            durations_ms=np.full(episode_steps, 100.0, dtype=np.float32),
            capacities_mbps=np.full(episode_steps, 10.0, dtype=np.float32),
            delays_ms=np.full(episode_steps, 20.0, dtype=np.float32),
        )
        super().__init__(trace=trace, max_rate_mbps=max_rate_mbps, queue_size_ms=queue_size_ms, episode_steps=episode_steps, **kwargs)


class CellularBurstyEnv(TraceLinkEnv):
    """Cellular trace with bursty capacity variation and moderate base delay.

    The simulator replays a 20-step cellular-style trace where capacity can
    collapse to around 1 Mbps and later burst above 15 Mbps. Delay rises during
    weaker-capacity periods, so a good controller must react to changing link
    conditions without filling the queue.

    Args:
        episode_steps: Number of steps per episode. If omitted, simulator mode
            uses the full 20-entry trace and real Mahimahi mode uses 20 steps.
        **kwargs: Optional scenario settings. ``use_mahimahi`` switches from
            the in-process trace simulator to the real ``mm-link`` wrapper.
            ``queue_size_ms`` sets the finite queue size and defaults to
            300 ms. In real Mahimahi mode, ``max_rate_mbps`` defaults to
            30 Mbps, ``uplink_trace`` and ``downlink_trace`` default to the
            T-Mobile LTE driving traces, and remaining keywords are forwarded
            to ``SingleFlowMahimahiScenarioEnv``. In simulator mode, remaining
            keywords are forwarded to ``TraceLinkEnv``.
    """

    def __init__(self, episode_steps: int | None = None, **kwargs: Any) -> None:
        use_mahimahi = bool(kwargs.pop("use_mahimahi", False))
        queue_size_ms = kwargs.pop("queue_size_ms", 300.0)
        if use_mahimahi:
            self._real_env = SingleFlowMahimahiScenarioEnv(
                uplink_trace=kwargs.pop("uplink_trace", CELLULAR_UPLINK_TRACE),
                downlink_trace=kwargs.pop("downlink_trace", CELLULAR_DOWNLINK_TRACE),
                max_rate_mbps=kwargs.pop("max_rate_mbps", 30.0),
                max_queue_delay_ms=queue_size_ms,
                episode_steps=episode_steps or 20,
                **kwargs,
            )
            self.action_space = self._real_env.action_space
            self.observation_space = self._real_env.observation_space
            return
        capacities = np.array(
            [
                8.0,
                7.5,
                3.0,
                1.2,
                2.0,
                12.0,
                15.0,
                6.0,
                4.0,
                2.2,
                1.0,
                9.0,
                14.0,
                18.0,
                5.0,
                2.5,
                1.5,
                7.0,
                11.0,
                4.5,
            ],
            dtype=np.float32,
        )
        delays = np.array(
            [55, 58, 70, 95, 85, 60, 50, 65, 78, 90, 110, 62, 54, 48, 72, 88, 105, 68, 58, 76],
            dtype=np.float32,
        )
        trace = LinkTrace(
            durations_ms=np.full_like(capacities, 100.0),
            capacities_mbps=capacities,
            delays_ms=delays,
        )
        super().__init__(trace=trace, queue_size_ms=queue_size_ms, episode_steps=episode_steps or len(capacities), **kwargs)


class LeoSatelliteEnv(TraceLinkEnv):
    """LEO-inspired trace with abrupt bandwidth and delay changes.

    The simulator replays a 20-step trace with high-capacity windows, abrupt
    drops to single-digit Mbps, and propagation-delay spikes above 100 ms. The
    delay penalty is lower than the default single-flow penalty because delay
    variation is inherent to this scenario.

    Args:
        episode_steps: Number of steps per episode. If omitted, simulator mode
            uses the full 20-entry trace and real Mahimahi mode uses 20 steps.
        **kwargs: Optional scenario settings. ``use_mahimahi`` switches from
            the in-process trace simulator to the real ``mm-link`` wrapper.
            ``queue_size_ms`` sets the finite queue size and defaults to
            250 ms. ``delay_penalty`` defaults to 0.012. In real Mahimahi mode,
            ``max_rate_mbps`` defaults to 60 Mbps, ``uplink_trace`` and
            ``downlink_trace`` default to the Verizon LTE short traces, and
            remaining keywords are forwarded to ``SingleFlowMahimahiScenarioEnv``.
            In simulator mode, remaining keywords are forwarded to
            ``TraceLinkEnv``.
    """

    def __init__(self, episode_steps: int | None = None, **kwargs: Any) -> None:
        use_mahimahi = bool(kwargs.pop("use_mahimahi", False))
        queue_size_ms = kwargs.pop("queue_size_ms", 250.0)
        delay_penalty = kwargs.pop("delay_penalty", 0.012)
        if use_mahimahi:
            self._real_env = SingleFlowMahimahiScenarioEnv(
                uplink_trace=kwargs.pop("uplink_trace", LEO_UPLINK_TRACE),
                downlink_trace=kwargs.pop("downlink_trace", LEO_DOWNLINK_TRACE),
                max_rate_mbps=kwargs.pop("max_rate_mbps", 60.0),
                max_queue_delay_ms=queue_size_ms,
                delay_penalty=delay_penalty,
                episode_steps=episode_steps or 20,
                **kwargs,
            )
            self.action_space = self._real_env.action_space
            self.observation_space = self._real_env.observation_space
            return
        capacities = np.array(
            [35, 35, 32, 8, 6, 45, 48, 12, 10, 9, 50, 52, 18, 15, 4, 4, 30, 42, 46, 7],
            dtype=np.float32,
        )
        delays = np.array(
            [35, 36, 38, 80, 95, 42, 36, 75, 88, 92, 40, 34, 65, 70, 120, 125, 55, 44, 38, 100],
            dtype=np.float32,
        )
        trace = LinkTrace(
            durations_ms=np.full_like(capacities, 100.0),
            capacities_mbps=capacities,
            delays_ms=delays,
        )
        super().__init__(
            trace=trace,
            queue_size_ms=queue_size_ms,
            delay_penalty=delay_penalty,
            episode_steps=episode_steps or len(capacities),
            **kwargs,
        )


class MultiFlowFairnessEnv(gym.Env[np.ndarray, np.ndarray]):
    """Multi-flow bottleneck where reward balances throughput and fairness.

    The agent chooses one sending rate per flow. In simulator mode, all flows
    share one fixed-capacity bottleneck and one finite queue. Accepted traffic
    and delivered traffic are split proportionally across flows, then reward
    combines total throughput, Jain's fairness index, queueing delay, and loss.
    In real Mahimahi mode, the class delegates to
    ``MultiFlowMahimahiScenarioEnv`` while keeping the same action shape.

    Args:
        n_flows: Number of independently controlled flows and action
            dimensions.
        episode_steps: Number of ``step`` calls before termination.
        capacity_mbps: Shared bottleneck capacity in simulator mode.
        max_rate_mbps: Per-flow action upper bound in Mbps.
        queue_size_ms: Shared queue size expressed as milliseconds of data at
            ``capacity_mbps``.
        fairness_weight: Reward multiplier for Jain's fairness index.
        delay_penalty: Reward penalty multiplier applied to queueing delay in
            simulator mode, and forwarded to the real adapter in Mahimahi mode.
        loss_penalty: Reward penalty multiplier applied to dropped traffic
            fraction.
        use_mahimahi: If true, launch the real ``mm-link`` adapter instead of
            using the in-process simulator.
        controller_cmd: Command run inside ``mm-link`` in real Mahimahi mode.
            If omitted, the default JSON-lines example controller is used by
            the wrapper.
        uplink_trace: Uplink packet-delivery trace path for real Mahimahi mode.
        downlink_trace: Downlink packet-delivery trace path for real Mahimahi
            mode.
        mahimahi_cmd: Executable name or path for ``mm-link``.
        timeout_s: Seconds to wait for one controller measurement in real
            Mahimahi mode.
        restart_on_reset: If true in real Mahimahi mode, restart the controller
            subprocess on each reset.
        render_mode: Gymnasium render mode. ``"human"`` prints compact
            throughput, fairness, delay, and loss statistics.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        n_flows: int = 3,
        episode_steps: int = 100,
        capacity_mbps: float = 12.0,
        max_rate_mbps: float = 15.0,
        queue_size_ms: float = 200.0,
        fairness_weight: float = 8.0,
        delay_penalty: float = 0.015,
        loss_penalty: float = 4.0,
        reward_mode: str = "default",
        mpcc_contour_weight: float = 1e-3,
        mpcc_lag_weight: float = 1e-3,
        mpcc_delay_weight: float = 1e-4,
        mpcc_power_weight: float = 1.0,
        mpcc_smoothing_weight: float = 0.1,
        mpcc_fairness_weight: float = 1e-2,
        mpcc_loss_weight: float = 0.0,
        mpcc_reference_delay_alpha_ms: float | None = None,
        mpcc_target_rtt_ms: float | None = None,
        mpcc_throughput_tau_s: float = 0.3,
        mpcc_rtt_tau_s: float = 0.3,
        mpcc_base_rtt_ms: float = 20.0,
        use_mahimahi: bool = False,
        controller_cmd: list[str] | None = None,
        uplink_trace: str = WIRED_UPLINK_TRACE,
        downlink_trace: str = WIRED_DOWNLINK_TRACE,
        mahimahi_cmd: str = "mm-link",
        timeout_s: float = 5.0,
        restart_on_reset: bool = False,
        render_mode: str | None = None,
    ) -> None:
        if use_mahimahi:
            self._real_env = MultiFlowMahimahiScenarioEnv(
                n_flows=n_flows,
                controller_cmd=controller_cmd,
                uplink_trace=uplink_trace,
                downlink_trace=downlink_trace,
                mahimahi_cmd=mahimahi_cmd,
                max_rate_mbps=max_rate_mbps,
                max_queue_delay_ms=queue_size_ms,
                timeout_s=timeout_s,
                episode_steps=episode_steps,
                delay_penalty=delay_penalty,
                loss_penalty=loss_penalty,
                reward_mode=reward_mode,
                mpcc_contour_weight=mpcc_contour_weight,
                mpcc_lag_weight=mpcc_lag_weight,
                mpcc_delay_weight=mpcc_delay_weight,
                mpcc_power_weight=mpcc_power_weight,
                mpcc_smoothing_weight=mpcc_smoothing_weight,
                mpcc_fairness_weight=mpcc_fairness_weight,
                mpcc_loss_weight=mpcc_loss_weight,
                mpcc_reference_delay_alpha_ms=mpcc_reference_delay_alpha_ms,
                mpcc_target_rtt_ms=mpcc_target_rtt_ms,
                mpcc_throughput_tau_s=mpcc_throughput_tau_s,
                mpcc_rtt_tau_s=mpcc_rtt_tau_s,
                mpcc_base_rtt_ms=mpcc_base_rtt_ms,
                restart_on_reset=restart_on_reset,
                render_mode=render_mode,
            )
            self.action_space = self._real_env.action_space
            self.observation_space = self._real_env.observation_space
            return

        self.n_flows = int(n_flows)
        self.episode_steps = int(episode_steps)
        self.capacity_mbps = float(capacity_mbps)
        self.max_rate_mbps = float(max_rate_mbps)
        self.queue_size_ms = float(queue_size_ms)
        self.fairness_weight = float(fairness_weight)
        self.delay_penalty = float(delay_penalty)
        self.loss_penalty = float(loss_penalty)
        self.reward_mode = reward_mode
        if self.reward_mode not in {"default", "mpcc"}:
            raise ValueError("reward_mode must be 'default' or 'mpcc'.")
        self.mpcc_config = _make_mpcc_config(
            mpcc_contour_weight=mpcc_contour_weight,
            mpcc_lag_weight=mpcc_lag_weight,
            mpcc_delay_weight=mpcc_delay_weight,
            mpcc_power_weight=mpcc_power_weight,
            mpcc_smoothing_weight=mpcc_smoothing_weight,
            mpcc_fairness_weight=mpcc_fairness_weight,
            mpcc_loss_weight=mpcc_loss_weight,
            mpcc_reference_delay_alpha_ms=mpcc_reference_delay_alpha_ms,
            mpcc_target_rtt_ms=mpcc_target_rtt_ms,
            mpcc_throughput_tau_s=mpcc_throughput_tau_s,
            mpcc_rtt_tau_s=mpcc_rtt_tau_s,
            mpcc_base_rtt_ms=mpcc_base_rtt_ms,
        )
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.zeros(self.n_flows, dtype=np.float32),
            high=np.full(self.n_flows, self.max_rate_mbps, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2 * self.n_flows + 4,), dtype=np.float32)
        self._step_index = 0
        self._queue_mbits = 0.0
        self._last_send_rates = np.zeros(self.n_flows, dtype=np.float32)
        self._last_throughputs = np.zeros(self.n_flows, dtype=np.float32)
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = self.mpcc_config.base_rtt_ms
        self._last_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if hasattr(self, "_real_env"):
            return self._real_env.reset(seed=seed, options=options)
        super().reset(seed=seed)
        self._step_index = 0
        self._queue_mbits = 0.0
        self._last_send_rates = np.zeros(self.n_flows, dtype=np.float32)
        self._last_throughputs = np.zeros(self.n_flows, dtype=np.float32)
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = self.mpcc_config.base_rtt_ms
        obs = self._make_obs(0.0, 0.0, 1.0, 0.0)
        self._last_info = self._info(1.0, 0.0, 0.0, 0.0)
        return obs, dict(self._last_info)

    def step(self, action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if hasattr(self, "_real_env"):
            return self._real_env.step(action)
        duration_s = 0.1
        send_rates = np.clip(np.asarray(action, dtype=np.float32).reshape(self.n_flows), 0.0, self.max_rate_mbps)
        total_send_rate = float(send_rates.sum())
        sent_mbits = send_rates * duration_s
        total_sent_mbits = float(sent_mbits.sum())
        service_mbits = self.capacity_mbps * duration_s
        queue_limit_mbits = self.capacity_mbps * self.queue_size_ms / 1000.0
        available_buffer_mbits = max(0.0, queue_limit_mbits - self._queue_mbits)
        accepted_total = min(total_sent_mbits, service_mbits + available_buffer_mbits)
        accepted_fraction = accepted_total / total_sent_mbits if total_sent_mbits > 0.0 else 0.0
        accepted_mbits = sent_mbits * accepted_fraction
        dropped_mbits = sent_mbits - accepted_mbits

        queued_before_service = self._queue_mbits + float(accepted_mbits.sum())
        delivered_total = min(queued_before_service, service_mbits)
        delivery_fraction = delivered_total / max(float(accepted_mbits.sum()), 1e-6)
        throughputs = accepted_mbits * delivery_fraction / duration_s
        self._queue_mbits = max(0.0, queued_before_service - delivered_total)

        queue_delay_ms = min(self.queue_size_ms, 1000.0 * self._queue_mbits / max(self.capacity_mbps, 1e-6))
        loss_fraction = float(dropped_mbits.sum() / total_sent_mbits) if total_sent_mbits > 0.0 else 0.0
        total_throughput = float(throughputs.sum())
        fairness = self._jain_fairness(throughputs)
        default_reward = (
            total_throughput
            + self.fairness_weight * fairness
            - self.delay_penalty * queue_delay_ms
            - self.loss_penalty * loss_fraction
        )
        reward = default_reward
        reward_info: dict[str, Any] = {"reward_mode": self.reward_mode, "default_reward": float(default_reward)}
        if self.reward_mode == "mpcc":
            rtt_ms = self.mpcc_config.base_rtt_ms + queue_delay_ms
            self._smoothed_throughput_mbps = _smooth_measurement(
                self._smoothed_throughput_mbps,
                total_throughput,
                duration_s,
                self.mpcc_config.throughput_tau_s,
            )
            self._smoothed_rtt_ms = _smooth_measurement(
                self._smoothed_rtt_ms,
                rtt_ms,
                duration_s,
                self.mpcc_config.rtt_tau_s,
            )
            reward, mpcc_info = _mpcc_reward(
                smoothed_throughput_mbps=self._smoothed_throughput_mbps,
                smoothed_rtt_ms=self._smoothed_rtt_ms,
                capacity_mbps=self.capacity_mbps,
                base_rtt_ms=self.mpcc_config.base_rtt_ms,
                queue_size_ms=self.queue_size_ms,
                current_rates_mbps=send_rates,
                previous_rates_mbps=self._last_send_rates,
                max_rate_mbps=self.max_rate_mbps,
                loss_fraction=loss_fraction,
                config=self.mpcc_config,
            )
            reward_info.update(mpcc_info)

        self._last_send_rates = send_rates
        self._last_throughputs = throughputs.astype(np.float32)
        obs = self._make_obs(queue_delay_ms, loss_fraction, fairness, total_throughput)
        self._last_info = self._info(fairness, queue_delay_ms, loss_fraction, total_throughput)
        self._last_info.update(reward_info)

        self._step_index += 1
        terminated = self._step_index >= self.episode_steps
        truncated = False
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, dict(self._last_info)

    def render(self) -> None:
        if hasattr(self, "_real_env"):
            self._real_env.render()
            return
        print(
            "throughput={total_throughput_mbps:.2f} fairness={jain_fairness:.3f} "
            "delay={queue_delay_ms:.1f} loss={loss_fraction:.3f}".format(**self._last_info)
        )

    def close(self) -> None:
        if hasattr(self, "_real_env"):
            self._real_env.close()

    def _make_obs(self, queue_delay_ms: float, loss_fraction: float, fairness: float, total_throughput: float) -> np.ndarray:
        queue_fraction = self._queue_mbits / max(self.capacity_mbps * self.queue_size_ms / 1000.0, 1e-6)
        obs = np.concatenate(
            [
                self._last_send_rates / self.max_rate_mbps,
                self._last_throughputs / self.max_rate_mbps,
                np.array(
                    [
                        self.capacity_mbps / self.max_rate_mbps,
                        queue_delay_ms / self.queue_size_ms,
                        loss_fraction,
                        fairness if total_throughput > 0.0 else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        obs[-3] = min(1.0, queue_fraction)
        return obs.astype(np.float32).clip(0.0, 1.0)

    def _info(self, fairness: float, queue_delay_ms: float, loss_fraction: float, total_throughput: float) -> dict[str, Any]:
        return {
            "send_rates_mbps": self._last_send_rates.copy(),
            "throughputs_mbps": self._last_throughputs.copy(),
            "capacity_mbps": self.capacity_mbps,
            "queue_delay_ms": queue_delay_ms,
            "loss_fraction": loss_fraction,
            "jain_fairness": fairness,
            "total_throughput_mbps": total_throughput,
            "queue_mbits": self._queue_mbits,
        }

    def _jain_fairness(self, throughputs: np.ndarray) -> float:
        numerator = float(np.square(throughputs.sum()))
        denominator = float(self.n_flows * np.square(throughputs).sum())
        return numerator / denominator if denominator > 0.0 else 0.0


class SingleFlowMahimahiScenarioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Scenario-shaped wrapper around the real ``mm-link`` subprocess adapter.

    This class adapts the raw six-value ``MahimahiSubprocessEnv`` observation
    into the seven-value observation used by the single-flow scenario
    simulators. Each action is one send rate in Mbps. The wrapped subprocess
    controller receives the rate over JSON lines, runs or manages the real
    network experiment, and reports throughput, RTT, queueing delay, loss, and
    optional capacity measurements.

    Args:
        controller_cmd: Command launched inside ``mm-link``. If omitted, uses
            ``examples/mm_link_json_controller.py`` through the current Python
            executable.
        uplink_trace: Mahimahi uplink packet-delivery trace path.
        downlink_trace: Mahimahi downlink packet-delivery trace path.
        mahimahi_cmd: Executable name or path for ``mm-link``.
        max_rate_mbps: Action upper bound and normalization scale for send
            rate, throughput, and fallback capacity.
        max_queue_delay_ms: Normalization scale for reported queueing delay.
        timeout_s: Seconds to wait for one JSON measurement from the
            controller after sending an action.
        episode_steps: Number of ``step`` calls before termination if the
            controller does not report ``done`` first.
        delay_penalty: Reward penalty multiplier passed to
            ``MahimahiSubprocessEnv`` for default reward calculation.
        loss_penalty: Loss penalty multiplier passed to
            ``MahimahiSubprocessEnv`` for default reward calculation.
        restart_on_reset: If true, close and relaunch ``mm-link`` on every
            reset. If false, reuse the subprocess across episodes when it is
            still running.
        render_mode: Gymnasium render mode. ``"human"`` delegates rendering to
            the wrapped subprocess environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        controller_cmd: list[str] | None = None,
        uplink_trace: str = WIRED_UPLINK_TRACE,
        downlink_trace: str = WIRED_DOWNLINK_TRACE,
        mahimahi_cmd: str = "mm-link",
        max_rate_mbps: float = 50.0,
        max_queue_delay_ms: float = 300.0,
        timeout_s: float = 5.0,
        episode_steps: int = 100,
        delay_penalty: float = 0.015,
        loss_penalty: float = 10.0,
        reward_mode: str = "default",
        mpcc_contour_weight: float = 1e-3,
        mpcc_lag_weight: float = 1e-3,
        mpcc_delay_weight: float = 1e-4,
        mpcc_power_weight: float = 1.0,
        mpcc_smoothing_weight: float = 0.1,
        mpcc_fairness_weight: float = 1e-2,
        mpcc_loss_weight: float = 0.0,
        mpcc_reference_delay_alpha_ms: float | None = None,
        mpcc_target_rtt_ms: float | None = None,
        mpcc_throughput_tau_s: float = 0.3,
        mpcc_rtt_tau_s: float = 0.3,
        mpcc_base_rtt_ms: float = 20.0,
        restart_on_reset: bool = False,
        render_mode: str | None = None,
    ) -> None:
        self.max_rate_mbps = float(max_rate_mbps)
        self.max_queue_delay_ms = float(max_queue_delay_ms)
        self.reward_mode = reward_mode
        if self.reward_mode not in {"default", "mpcc"}:
            raise ValueError("reward_mode must be 'default' or 'mpcc'.")
        self.mpcc_config = _make_mpcc_config(
            mpcc_contour_weight=mpcc_contour_weight,
            mpcc_lag_weight=mpcc_lag_weight,
            mpcc_delay_weight=mpcc_delay_weight,
            mpcc_power_weight=mpcc_power_weight,
            mpcc_smoothing_weight=mpcc_smoothing_weight,
            mpcc_fairness_weight=mpcc_fairness_weight,
            mpcc_loss_weight=mpcc_loss_weight,
            mpcc_reference_delay_alpha_ms=mpcc_reference_delay_alpha_ms,
            mpcc_target_rtt_ms=mpcc_target_rtt_ms,
            mpcc_throughput_tau_s=mpcc_throughput_tau_s,
            mpcc_rtt_tau_s=mpcc_rtt_tau_s,
            mpcc_base_rtt_ms=mpcc_base_rtt_ms,
        )
        self._last_send_rate_mbps = 0.0
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = self.mpcc_config.base_rtt_ms
        self._env = MahimahiSubprocessEnv(
            controller_cmd=controller_cmd or DEFAULT_CONTROLLER_CMD,
            uplink_trace=uplink_trace,
            downlink_trace=downlink_trace,
            mahimahi_cmd=mahimahi_cmd,
            use_mahimahi=True,
            action_dim=1,
            max_rate_mbps=max_rate_mbps,
            max_queue_delay_ms=max_queue_delay_ms,
            timeout_s=timeout_s,
            episode_steps=episode_steps,
            delay_penalty=delay_penalty,
            loss_penalty=loss_penalty,
            restart_on_reset=restart_on_reset,
            render_mode=render_mode,
        )
        self.action_space = self._env.action_space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self._env.reset(seed=seed, options=options)
        self._last_send_rate_mbps = 0.0
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = float(info.get("rtt_ms", self.mpcc_config.base_rtt_ms))
        obs = self._make_obs(0.0, 0.0, info)
        info = self._single_flow_info(0.0, info)
        return obs, dict(info)

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self._env.step(action)
        default_reward = float(reward)
        send_rates = np.asarray(info.get("send_rates_mbps", action), dtype=np.float32).reshape(-1)
        send_rate = float(send_rates[0])
        obs = self._make_obs(send_rate, float(info.get("throughput_mbps", 0.0)), info)
        info = self._single_flow_info(send_rate, info)
        info["reward_mode"] = self.reward_mode
        info["default_reward"] = default_reward
        if self.reward_mode == "mpcc":
            throughput_mbps = float(info.get("throughput_mbps", 0.0))
            rtt_ms = float(info.get("rtt_ms", 0.0))
            queue_delay_ms = float(info.get("queue_delay_ms", 0.0))
            base_rtt_ms = max(0.0, rtt_ms - queue_delay_ms) or self.mpcc_config.base_rtt_ms
            self._smoothed_throughput_mbps = _smooth_measurement(
                self._smoothed_throughput_mbps,
                throughput_mbps,
                0.1,
                self.mpcc_config.throughput_tau_s,
            )
            self._smoothed_rtt_ms = _smooth_measurement(
                self._smoothed_rtt_ms,
                rtt_ms,
                0.1,
                self.mpcc_config.rtt_tau_s,
            )
            reward, mpcc_info = _mpcc_reward(
                smoothed_throughput_mbps=self._smoothed_throughput_mbps,
                smoothed_rtt_ms=self._smoothed_rtt_ms,
                capacity_mbps=float(info.get("capacity_mbps", self.max_rate_mbps)),
                base_rtt_ms=base_rtt_ms,
                queue_size_ms=self.max_queue_delay_ms,
                current_rates_mbps=np.array([send_rate], dtype=np.float32),
                previous_rates_mbps=np.array([self._last_send_rate_mbps], dtype=np.float32),
                max_rate_mbps=self.max_rate_mbps,
                loss_fraction=float(info.get("loss_fraction", 0.0)),
                config=self.mpcc_config,
            )
            info.update(mpcc_info)
        self._last_send_rate_mbps = send_rate
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()

    def _make_obs(self, send_rate_mbps: float, throughput_mbps: float, info: dict[str, Any]) -> np.ndarray:
        queue_delay_ms = float(info.get("queue_delay_ms", 0.0))
        rtt_ms = float(info.get("rtt_ms", queue_delay_ms))
        propagation_delay_ms = max(0.0, rtt_ms - queue_delay_ms)
        capacity_mbps = float(info.get("capacity_mbps", self.max_rate_mbps))
        return np.array(
            [
                send_rate_mbps / self.max_rate_mbps,
                throughput_mbps / self.max_rate_mbps,
                capacity_mbps / self.max_rate_mbps,
                propagation_delay_ms / max(rtt_ms, 1e-6),
                queue_delay_ms / self.max_queue_delay_ms,
                float(info.get("loss_fraction", 0.0)),
                queue_delay_ms / self.max_queue_delay_ms,
            ],
            dtype=np.float32,
        ).clip(0.0, 1.0)

    def _single_flow_info(self, send_rate_mbps: float, info: dict[str, Any]) -> dict[str, Any]:
        queue_delay_ms = float(info.get("queue_delay_ms", 0.0))
        rtt_ms = float(info.get("rtt_ms", queue_delay_ms))
        return {
            **info,
            "send_rate_mbps": send_rate_mbps,
            "throughput_mbps": float(info.get("throughput_mbps", 0.0)),
            "capacity_mbps": float(info.get("capacity_mbps", self.max_rate_mbps)),
            "queue_delay_ms": queue_delay_ms,
            "rtt_ms": rtt_ms,
            "loss_fraction": float(info.get("loss_fraction", 0.0)),
            "propagation_delay_ms": max(0.0, rtt_ms - queue_delay_ms),
            "queue_mbits": float(info.get("queue_mbits", 0.0)),
            "dropped_mbits": float(info.get("dropped_mbits", 0.0)),
        }


class MultiFlowMahimahiScenarioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Multi-flow scenario-shaped wrapper around the real ``mm-link`` adapter.

    This class adapts ``MahimahiSubprocessEnv`` to the observation layout used
    by ``MultiFlowFairnessEnv``. Each action contains one send rate per flow.
    The controller may report per-flow throughputs via ``throughputs_mbps``; if
    it only reports aggregate throughput, this wrapper distributes throughput
    across flows in proportion to the requested send rates so the observation
    and Jain fairness metric remain well-defined.

    Args:
        n_flows: Number of controlled flows and action dimensions.
        controller_cmd: Command launched inside ``mm-link``. If omitted, uses
            ``examples/mm_link_json_controller.py`` through the current Python
            executable.
        uplink_trace: Mahimahi uplink packet-delivery trace path.
        downlink_trace: Mahimahi downlink packet-delivery trace path.
        mahimahi_cmd: Executable name or path for ``mm-link``.
        max_rate_mbps: Per-flow action upper bound and normalization scale for
            per-flow send rates and throughputs.
        max_queue_delay_ms: Normalization scale for reported queueing delay.
        timeout_s: Seconds to wait for one JSON measurement from the
            controller after sending an action.
        episode_steps: Number of ``step`` calls before termination if the
            controller does not report ``done`` first.
        delay_penalty: Reward penalty multiplier passed to
            ``MahimahiSubprocessEnv`` for default reward calculation.
        loss_penalty: Loss penalty multiplier passed to
            ``MahimahiSubprocessEnv`` for default reward calculation.
        restart_on_reset: If true, close and relaunch ``mm-link`` on every
            reset. If false, reuse the subprocess across episodes when it is
            still running.
        render_mode: Gymnasium render mode. ``"human"`` delegates rendering to
            the wrapped subprocess environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        n_flows: int = 3,
        controller_cmd: list[str] | None = None,
        uplink_trace: str = WIRED_UPLINK_TRACE,
        downlink_trace: str = WIRED_DOWNLINK_TRACE,
        mahimahi_cmd: str = "mm-link",
        max_rate_mbps: float = 15.0,
        max_queue_delay_ms: float = 200.0,
        timeout_s: float = 5.0,
        episode_steps: int = 100,
        delay_penalty: float = 0.015,
        loss_penalty: float = 4.0,
        reward_mode: str = "default",
        mpcc_contour_weight: float = 1e-3,
        mpcc_lag_weight: float = 1e-3,
        mpcc_delay_weight: float = 1e-4,
        mpcc_power_weight: float = 1.0,
        mpcc_smoothing_weight: float = 0.1,
        mpcc_fairness_weight: float = 1e-2,
        mpcc_loss_weight: float = 0.0,
        mpcc_reference_delay_alpha_ms: float | None = None,
        mpcc_target_rtt_ms: float | None = None,
        mpcc_throughput_tau_s: float = 0.3,
        mpcc_rtt_tau_s: float = 0.3,
        mpcc_base_rtt_ms: float = 20.0,
        restart_on_reset: bool = False,
        render_mode: str | None = None,
    ) -> None:
        self.n_flows = int(n_flows)
        self.max_rate_mbps = float(max_rate_mbps)
        self.max_queue_delay_ms = float(max_queue_delay_ms)
        self.reward_mode = reward_mode
        if self.reward_mode not in {"default", "mpcc"}:
            raise ValueError("reward_mode must be 'default' or 'mpcc'.")
        self.mpcc_config = _make_mpcc_config(
            mpcc_contour_weight=mpcc_contour_weight,
            mpcc_lag_weight=mpcc_lag_weight,
            mpcc_delay_weight=mpcc_delay_weight,
            mpcc_power_weight=mpcc_power_weight,
            mpcc_smoothing_weight=mpcc_smoothing_weight,
            mpcc_fairness_weight=mpcc_fairness_weight,
            mpcc_loss_weight=mpcc_loss_weight,
            mpcc_reference_delay_alpha_ms=mpcc_reference_delay_alpha_ms,
            mpcc_target_rtt_ms=mpcc_target_rtt_ms,
            mpcc_throughput_tau_s=mpcc_throughput_tau_s,
            mpcc_rtt_tau_s=mpcc_rtt_tau_s,
            mpcc_base_rtt_ms=mpcc_base_rtt_ms,
        )
        self._last_send_rates = np.zeros(self.n_flows, dtype=np.float32)
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = self.mpcc_config.base_rtt_ms
        self._env = MahimahiSubprocessEnv(
            controller_cmd=controller_cmd or DEFAULT_CONTROLLER_CMD,
            uplink_trace=uplink_trace,
            downlink_trace=downlink_trace,
            mahimahi_cmd=mahimahi_cmd,
            use_mahimahi=True,
            action_dim=self.n_flows,
            max_rate_mbps=max_rate_mbps,
            max_queue_delay_ms=max_queue_delay_ms,
            timeout_s=timeout_s,
            episode_steps=episode_steps,
            delay_penalty=delay_penalty,
            loss_penalty=loss_penalty,
            restart_on_reset=restart_on_reset,
            render_mode=render_mode,
        )
        self.action_space = self._env.action_space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2 * self.n_flows + 4,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self._env.reset(seed=seed, options=options)
        send_rates = np.zeros(self.n_flows, dtype=np.float32)
        throughputs = np.zeros(self.n_flows, dtype=np.float32)
        self._last_send_rates = send_rates
        self._smoothed_throughput_mbps = 0.0
        self._smoothed_rtt_ms = float(info.get("rtt_ms", self.mpcc_config.base_rtt_ms))
        info = self._multi_flow_info(send_rates, throughputs, info)
        obs = self._make_obs(send_rates, throughputs, info)
        return obs, dict(info)

    def step(self, action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self._env.step(action)
        default_reward = float(reward)
        send_rates = np.clip(np.asarray(action, dtype=np.float32).reshape(self.n_flows), 0.0, self.max_rate_mbps)
        throughputs = self._throughputs_from_info(send_rates, info)
        info = self._multi_flow_info(send_rates, throughputs, info)
        obs = self._make_obs(send_rates, throughputs, info)
        info["reward_mode"] = self.reward_mode
        info["default_reward"] = default_reward
        if self.reward_mode == "mpcc":
            total_throughput = float(throughputs.sum())
            queue_delay_ms = float(info.get("queue_delay_ms", 0.0))
            rtt_ms = float(info.get("rtt_ms", self.mpcc_config.base_rtt_ms + queue_delay_ms))
            base_rtt_ms = max(0.0, rtt_ms - queue_delay_ms) or self.mpcc_config.base_rtt_ms
            self._smoothed_throughput_mbps = _smooth_measurement(
                self._smoothed_throughput_mbps,
                total_throughput,
                0.1,
                self.mpcc_config.throughput_tau_s,
            )
            self._smoothed_rtt_ms = _smooth_measurement(
                self._smoothed_rtt_ms,
                rtt_ms,
                0.1,
                self.mpcc_config.rtt_tau_s,
            )
            reward, mpcc_info = _mpcc_reward(
                smoothed_throughput_mbps=self._smoothed_throughput_mbps,
                smoothed_rtt_ms=self._smoothed_rtt_ms,
                capacity_mbps=float(info.get("capacity_mbps", self.max_rate_mbps * self.n_flows)),
                base_rtt_ms=base_rtt_ms,
                queue_size_ms=self.max_queue_delay_ms,
                current_rates_mbps=send_rates,
                previous_rates_mbps=self._last_send_rates,
                max_rate_mbps=self.max_rate_mbps,
                loss_fraction=float(info.get("loss_fraction", 0.0)),
                config=self.mpcc_config,
            )
            info.update(mpcc_info)
        self._last_send_rates = send_rates
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()

    def _make_obs(self, send_rates: np.ndarray, throughputs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        queue_delay_ms = float(info.get("queue_delay_ms", 0.0))
        loss_fraction = float(info.get("loss_fraction", 0.0))
        fairness = float(info.get("jain_fairness", self._jain_fairness(throughputs)))
        capacity_mbps = float(info.get("capacity_mbps", self.max_rate_mbps * self.n_flows))
        obs = np.concatenate(
            [
                send_rates / self.max_rate_mbps,
                throughputs / self.max_rate_mbps,
                np.array(
                    [
                        capacity_mbps / max(self.max_rate_mbps * self.n_flows, 1e-6),
                        queue_delay_ms / self.max_queue_delay_ms,
                        loss_fraction,
                        fairness if throughputs.sum() > 0.0 else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32).clip(0.0, 1.0)

    def _throughputs_from_info(self, send_rates: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        throughputs = info.get("throughputs_mbps")
        if throughputs is not None:
            return np.asarray(throughputs, dtype=np.float32).reshape(self.n_flows)

        total_throughput = float(info.get("total_throughput_mbps", info.get("throughput_mbps", 0.0)))
        total_send = float(send_rates.sum())
        if total_send <= 0.0:
            return np.zeros(self.n_flows, dtype=np.float32)
        return (send_rates / total_send * total_throughput).astype(np.float32)

    def _multi_flow_info(self, send_rates: np.ndarray, throughputs: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        fairness = float(info.get("jain_fairness", self._jain_fairness(throughputs)))
        return {
            **info,
            "send_rates_mbps": send_rates.copy(),
            "throughputs_mbps": throughputs.copy(),
            "capacity_mbps": float(info.get("capacity_mbps", self.max_rate_mbps * self.n_flows)),
            "queue_delay_ms": float(info.get("queue_delay_ms", 0.0)),
            "loss_fraction": float(info.get("loss_fraction", 0.0)),
            "jain_fairness": fairness,
            "total_throughput_mbps": float(throughputs.sum()),
            "queue_mbits": float(info.get("queue_mbits", 0.0)),
        }

    def _jain_fairness(self, throughputs: np.ndarray) -> float:
        numerator = float(np.square(throughputs.sum()))
        denominator = float(self.n_flows * np.square(throughputs).sum())
        return numerator / denominator if denominator > 0.0 else 0.0
