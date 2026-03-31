"""
Tests for configuration management.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pympc.config import (
    MPCConfig,
    ConfigManager,
    create_default_config,
    load_config,
    PlannerConfig,
    SolverConfig,
)
from pympc.exceptions import ConfigValidationError, ConfigNotFoundError


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_default_constraint_type(self):
        """Default constraint type should be scenario."""
        config = create_default_config()
        assert config["obstacle_constraint_type"] == "scenario"

    def test_scenario_constraint_type(self):
        """Can create config with scenario constraints."""
        config = create_default_config("scenario")
        assert config["obstacle_constraint_type"] == "scenario"

    def test_linearized_constraint_type(self):
        """Can create config with linearized constraints."""
        config = create_default_config("linearized")
        assert config["obstacle_constraint_type"] == "linearized"

    def test_default_horizon(self):
        """Default horizon should be 20."""
        config = create_default_config()
        assert config["planner"]["horizon"] == 20

    def test_default_timestep(self):
        """Default timestep should be 0.1."""
        config = create_default_config()
        assert config["planner"]["timestep"] == 0.1


class TestMPCConfig:
    """Tests for MPCConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = MPCConfig()
        assert config.planner.horizon == 20
        assert config.planner.timestep == 0.1
        assert config.obstacle_constraint_type == "scenario"

    def test_to_dict(self):
        """Config should convert to dictionary."""
        config = MPCConfig()
        d = config.to_dict()
        assert "planner" in d
        assert "solver" in d
        assert d["planner"]["horizon"] == 20

    def test_from_dict(self):
        """Config should be created from dictionary."""
        data = {
            "planner": {"horizon": 30, "timestep": 0.05},
            "obstacle_constraint_type": "linearized",
        }
        config = MPCConfig.from_dict(data)
        assert config.planner.horizon == 30
        assert config.planner.timestep == 0.05
        assert config.obstacle_constraint_type == "linearized"

    def test_validation_passes_for_valid_config(self):
        """Validation should pass for valid config."""
        config = MPCConfig()
        config.validate()  # Should not raise

    def test_validation_fails_for_invalid_horizon(self):
        """Validation should fail for invalid horizon."""
        config = MPCConfig()
        config.planner.horizon = 0
        with pytest.raises(ConfigValidationError):
            config.validate()

    def test_validation_fails_for_invalid_timestep(self):
        """Validation should fail for invalid timestep."""
        config = MPCConfig()
        config.planner.timestep = -0.1
        with pytest.raises(ConfigValidationError):
            config.validate()


class TestPlannerConfig:
    """Tests for PlannerConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = PlannerConfig()
        assert config.horizon == 20
        assert config.timestep == 0.1

    def test_validation_passes(self):
        """Validation should pass for valid config."""
        config = PlannerConfig(horizon=10, timestep=0.2)
        config.validate()  # Should not raise

    def test_validation_fails_for_zero_horizon(self):
        """Validation should fail for zero horizon."""
        config = PlannerConfig(horizon=0)
        with pytest.raises(ConfigValidationError) as excinfo:
            config.validate()
        assert "horizon" in str(excinfo.value)


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_load_defaults(self):
        """Should load default configuration."""
        manager = ConfigManager()
        config = manager.load()
        assert config.planner.horizon == 20

    def test_load_from_file(self, temp_config_file):
        """Should load configuration from file."""
        manager = ConfigManager(temp_config_file)
        config = manager.load()
        assert config.planner.horizon == 15
        assert config.planner.timestep == 0.2

    def test_file_not_found(self, tmp_path):
        """Should raise ConfigNotFoundError for missing file."""
        manager = ConfigManager(tmp_path / "nonexistent.yml")
        with pytest.raises(ConfigNotFoundError):
            manager.load()

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("PYMPC_PLANNER_HORIZON", "50")
        manager = ConfigManager()
        manager.load()
        # Note: env parsing puts values in _raw_config
        assert manager.get("planner.horizon") == 50

    def test_get_nested_value(self):
        """Should get nested configuration values."""
        manager = ConfigManager()
        manager.load()
        assert manager.get("planner.horizon") == 20
        assert manager.get("planner.timestep") == 0.1

    def test_get_missing_value_with_default(self):
        """Should return default for missing keys."""
        manager = ConfigManager()
        manager.load()
        assert manager.get("nonexistent.key", "default") == "default"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_file(self, temp_config_file):
        """Should load valid configuration file."""
        config = load_config(temp_config_file)
        assert config["planner"]["horizon"] == 15

    def test_load_nonexistent_file(self, tmp_path):
        """Should raise error for nonexistent file."""
        with pytest.raises(ConfigNotFoundError):
            load_config(tmp_path / "nonexistent.yml")
