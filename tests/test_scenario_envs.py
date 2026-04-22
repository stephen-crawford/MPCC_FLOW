import gymnasium as gym
import numpy as np

import mahimahi_gym  # noqa: F401


def test_single_flow_scenario_envs_step() -> None:
    for env_id in ["MahiWiredBottleneck-v0", "MahiCellularBursty-v0", "MahiLeoSatellite-v0"]:
        env = gym.make(env_id, episode_steps=3)
        obs, info = env.reset(seed=1)
        assert obs.shape == (7,)
        assert "capacity_mbps" in info

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (7,)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["rtt_ms"] >= info["propagation_delay_ms"]


def test_multi_flow_fairness_env_step() -> None:
    env = gym.make("MahiMultiFlowFairness-v0", n_flows=3, episode_steps=2)
    obs, info = env.reset(seed=1)
    assert obs.shape == (10,)
    assert "jain_fairness" in info

    obs, reward, terminated, truncated, info = env.step(np.array([4.0, 4.0, 4.0], dtype=np.float32))
    assert obs.shape == (10,)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["jain_fairness"] > 0.99


def test_mpcc_reward_mode_exposes_tracking_terms() -> None:
    env = gym.make("MahiWiredBottleneck-v0", reward_mode="mpcc", episode_steps=2)
    _, _ = env.reset(seed=1)
    _, reward, _, _, info = env.step(np.array([8.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert info["reward_mode"] == "mpcc"
    assert "default_reward" in info
    assert "mpcc_stage_cost" in info
    assert "mpcc_contour_error" in info
    assert "mpcc_power" in info
    env.close()

    env = gym.make("MahiMultiFlowFairness-v0", reward_mode="mpcc", n_flows=3, episode_steps=2)
    _, _ = env.reset(seed=1)
    _, reward, _, _, info = env.step(np.array([4.0, 4.0, 4.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert info["reward_mode"] == "mpcc"
    assert "mpcc_fairness_penalty" in info
    env.close()


def test_scenario_envs_can_be_constructed_for_real_mahimahi() -> None:
    for env_id in ["MahiWiredBottleneck-v0", "MahiCellularBursty-v0", "MahiLeoSatellite-v0"]:
        env = gym.make(env_id, use_mahimahi=True, episode_steps=3)
        assert env.action_space.shape == (1,)
        assert env.observation_space.shape == (7,)
        env.close()

    env = gym.make("MahiMultiFlowFairness-v0", use_mahimahi=True, n_flows=3, episode_steps=2)
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (10,)
    env.close()
