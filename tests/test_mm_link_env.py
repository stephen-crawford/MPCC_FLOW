import sys
from pathlib import Path

import numpy as np
import pytest

from mahimahi_gym.mm_link import MahimahiSubprocessEnv


def test_mm_link_subprocess_protocol_without_mahimahi() -> None:
    controller = Path("examples/mm_link_json_controller.py")
    env = MahimahiSubprocessEnv(
        controller_cmd=[sys.executable, str(controller)],
        use_mahimahi=False,
        episode_steps=2,
        max_rate_mbps=20.0,
    )

    obs, info = env.reset(seed=1)
    assert obs.shape == (6,)
    assert info["controller_pid"] is not None

    obs, reward, terminated, truncated, info = env.step(np.array([8.0], dtype=np.float32))
    assert obs.shape == (6,)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["throughput_mbps"] == pytest.approx(8.0)

    _, _, terminated, _, _ = env.step(np.array([8.0], dtype=np.float32))
    assert terminated
    env.close()


def test_mm_link_subprocess_can_reuse_process_across_resets() -> None:
    controller = Path("examples/mm_link_json_controller.py")
    env = MahimahiSubprocessEnv(
        controller_cmd=[sys.executable, str(controller)],
        use_mahimahi=False,
        episode_steps=1,
        restart_on_reset=False,
    )

    _, first_info = env.reset(seed=1)
    _, _, terminated, _, _ = env.step(np.array([5.0], dtype=np.float32))
    assert terminated

    _, second_info = env.reset(seed=2)
    assert second_info["controller_pid"] == first_info["controller_pid"]
    env.close()


def test_mm_link_subprocess_requires_mm_link_when_enabled() -> None:
    env = MahimahiSubprocessEnv(
        controller_cmd=[sys.executable, "examples/mm_link_json_controller.py"],
        uplink_trace="mahimahi/12mbps.trace",
        downlink_trace="mahimahi/12mbps.trace",
        mahimahi_cmd="definitely-not-mm-link",
        use_mahimahi=True,
    )

    with pytest.raises(FileNotFoundError):
        env.reset(seed=1)
    env.close()
