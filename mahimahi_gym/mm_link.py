"""Gymnasium adapter for real Mahimahi ``mm-link`` subprocess experiments."""

from __future__ import annotations

import json
import select
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MahimahiSubprocessEnv(gym.Env[np.ndarray, np.ndarray]):
    """Drive an external congestion-control process through ``mm-link``.

    The child process speaks a simple JSON-lines protocol:

    Agent action sent to child stdin:
        ``{"type": "action", "step": 0, "send_rates_mbps": [4.0]}``

    Measurement read from child stdout:
        ``{"throughput_mbps": 3.8, "rtt_ms": 45, "loss_fraction": 0.0}``

    When ``use_mahimahi=True`` the child command is launched as:
        ``mm-link uplink_trace downlink_trace -- <controller_cmd...>``

    This class intentionally does not prescribe the sender/receiver program.
    The controller command can be a Python shim, a compiled congestion-control
    binary, or a script that manages additional subprocesses and reports JSON
    measurements back to the Gym environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        controller_cmd: Sequence[str] | None = None,
        uplink_trace: str | Path | None = None,
        downlink_trace: str | Path | None = None,
        mahimahi_cmd: str = "mm-link",
        use_mahimahi: bool = True,
        action_dim: int = 1,
        max_rate_mbps: float = 50.0,
        max_throughput_mbps: float | None = None,
        max_rtt_ms: float = 500.0,
        max_queue_delay_ms: float = 300.0,
        timeout_s: float = 5.0,
        episode_steps: int = 100,
        delay_penalty: float = 0.015,
        loss_penalty: float = 10.0,
        restart_on_reset: bool = True,
        render_mode: str | None = None,
    ) -> None:
        if controller_cmd is None:
            raise ValueError("controller_cmd is required for MahimahiSubprocessEnv.")
        if use_mahimahi and (uplink_trace is None or downlink_trace is None):
            raise ValueError("uplink_trace and downlink_trace are required when use_mahimahi=True.")

        self.controller_cmd = [str(part) for part in controller_cmd]
        self.uplink_trace = None if uplink_trace is None else str(uplink_trace)
        self.downlink_trace = None if downlink_trace is None else str(downlink_trace)
        self.mahimahi_cmd = mahimahi_cmd
        self.use_mahimahi = bool(use_mahimahi)
        self.action_dim = int(action_dim)
        self.max_rate_mbps = float(max_rate_mbps)
        self.max_throughput_mbps = float(max_throughput_mbps or max_rate_mbps * max(1, action_dim))
        self.max_rtt_ms = float(max_rtt_ms)
        self.max_queue_delay_ms = float(max_queue_delay_ms)
        self.timeout_s = float(timeout_s)
        self.episode_steps = int(episode_steps)
        self.delay_penalty = float(delay_penalty)
        self.loss_penalty = float(loss_penalty)
        self.restart_on_reset = bool(restart_on_reset)
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.zeros(self.action_dim, dtype=np.float32),
            high=np.full(self.action_dim, self.max_rate_mbps, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        self._process: subprocess.Popen[str] | None = None
        self._step_index = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._last_obs = np.zeros(6, dtype=np.float32)
        self._last_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.restart_on_reset:
            self.close()
        if not self._process_is_running():
            self.close()
            self._start_process()
        self._step_index = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._last_obs = np.zeros(6, dtype=np.float32)
        self._last_info = {"controller_pid": self._process.pid if self._process is not None else None}
        return self._last_obs.copy(), dict(self._last_info)

    def step(self, action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Environment must be reset before step().")

        rates = np.clip(np.asarray(action, dtype=np.float32).reshape(self.action_dim), 0.0, self.max_rate_mbps)
        request = {
            "type": "action",
            "step": self._step_index,
            "send_rates_mbps": rates.astype(float).tolist(),
        }
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

        measurement = self._read_measurement()
        throughput_mbps = float(measurement.get("throughput_mbps", 0.0))
        rtt_ms = float(measurement.get("rtt_ms", measurement.get("delay_ms", 0.0)))
        queue_delay_ms = float(measurement.get("queue_delay_ms", max(0.0, rtt_ms - float(measurement.get("propagation_delay_ms", 0.0)))))
        loss_fraction = float(measurement.get("loss_fraction", 0.0))
        capacity_mbps = float(measurement.get("capacity_mbps", self.max_rate_mbps))
        fairness = float(measurement.get("jain_fairness", 1.0))

        reward = float(
            measurement.get(
                "reward",
                throughput_mbps * fairness - self.delay_penalty * rtt_ms - self.loss_penalty * loss_fraction,
            )
        )

        self._step_index += 1
        terminated = bool(measurement.get("done", False)) or self._step_index >= self.episode_steps
        truncated = False
        self._last_action = rates
        self._last_obs = self._make_obs(throughput_mbps, rtt_ms, queue_delay_ms, loss_fraction, capacity_mbps)
        self._last_info = {
            **measurement,
            "send_rates_mbps": rates.copy(),
            "throughput_mbps": throughput_mbps,
            "rtt_ms": rtt_ms,
            "queue_delay_ms": queue_delay_ms,
            "loss_fraction": loss_fraction,
            "capacity_mbps": capacity_mbps,
        }

        if self.render_mode == "human":
            self.render()
        return self._last_obs.copy(), reward, terminated, truncated, dict(self._last_info)

    def render(self) -> None:
        print(
            "rates={} throughput={:.2f} rtt={:.1f} loss={:.3f}".format(
                np.round(self._last_action, 3).tolist(),
                self._last_info.get("throughput_mbps", 0.0),
                self._last_info.get("rtt_ms", 0.0),
                self._last_info.get("loss_fraction", 0.0),
            )
        )

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def _start_process(self) -> None:
        cmd = self._build_command()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _process_is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _build_command(self) -> list[str]:
        if not self.use_mahimahi:
            return list(self.controller_cmd)

        if shutil.which(self.mahimahi_cmd) is None:
            raise FileNotFoundError(
                f"{self.mahimahi_cmd!r} was not found. Install Mahimahi or set use_mahimahi=False for protocol tests."
            )
        return [
            self.mahimahi_cmd,
            str(self.uplink_trace),
            str(self.downlink_trace),
            "--",
            *self.controller_cmd,
        ]

    def _read_measurement(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Process is not running.")

        ready, _, _ = select.select([self._process.stdout], [], [], self.timeout_s)
        if not ready:
            stderr = self._drain_stderr()
            raise TimeoutError(f"Timed out waiting for controller measurement. stderr={stderr!r}")

        line = self._process.stdout.readline()
        if line == "":
            stderr = self._drain_stderr()
            raise RuntimeError(f"Controller exited before producing a measurement. stderr={stderr!r}")

        try:
            measurement = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Controller emitted invalid JSON: {line!r}") from exc
        if not isinstance(measurement, dict):
            raise RuntimeError(f"Controller measurement must be a JSON object: {line!r}")
        return measurement

    def _drain_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        if self._process.poll() is None:
            return ""
        return self._process.stderr.read()

    def _make_obs(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        queue_delay_ms: float,
        loss_fraction: float,
        capacity_mbps: float,
    ) -> np.ndarray:
        total_send_rate = float(self._last_action.sum())
        return np.array(
            [
                total_send_rate / max(self.max_rate_mbps * self.action_dim, 1e-6),
                throughput_mbps / max(self.max_throughput_mbps, 1e-6),
                rtt_ms / max(self.max_rtt_ms, 1e-6),
                queue_delay_ms / max(self.max_queue_delay_ms, 1e-6),
                loss_fraction,
                capacity_mbps / max(self.max_throughput_mbps, 1e-6),
            ],
            dtype=np.float32,
        ).clip(0.0, 1.0)
