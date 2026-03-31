"""
Tests for exception hierarchy.
"""

from __future__ import annotations

import pytest

from pympc.exceptions import (
    PyMPCError,
    ConfigurationError,
    ConfigNotFoundError,
    ConfigValidationError,
    SolverError,
    SolverFailedError,
    InfeasibleProblemError,
    PlanningError,
    InvalidStateError,
    ModuleError,
    ConstraintError,
    DataError,
    InvalidObstacleError,
    RuntimeLimitError,
    TimeoutError,
    IterationLimitError,
)


class TestPyMPCError:
    """Tests for base PyMPCError."""

    def test_basic_message(self):
        """Should store and return message."""
        error = PyMPCError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_message_with_details(self):
        """Should include details in string representation."""
        error = PyMPCError("Test error", details={"key": "value"})
        assert "key=value" in str(error)
        assert error.details == {"key": "value"}

    def test_is_exception(self):
        """Should be an Exception subclass."""
        error = PyMPCError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised(self):
        """Should be raiseable."""
        with pytest.raises(PyMPCError):
            raise PyMPCError("Test error")


class TestConfigurationErrors:
    """Tests for configuration-related errors."""

    def test_config_not_found(self):
        """ConfigNotFoundError should include path."""
        error = ConfigNotFoundError("/path/to/config.yml")
        assert "/path/to/config.yml" in str(error)
        assert error.details["path"] == "/path/to/config.yml"

    def test_config_validation_error(self):
        """ConfigValidationError should include key and reason."""
        error = ConfigValidationError("horizon", "must be positive", value=-1)
        assert "horizon" in str(error)
        assert "must be positive" in str(error)
        assert error.details["key"] == "horizon"
        assert error.details["value"] == "-1"

    def test_inheritance(self):
        """Configuration errors should inherit from ConfigurationError."""
        assert issubclass(ConfigNotFoundError, ConfigurationError)
        assert issubclass(ConfigValidationError, ConfigurationError)
        assert issubclass(ConfigurationError, PyMPCError)


class TestSolverErrors:
    """Tests for solver-related errors."""

    def test_solver_failed_error(self):
        """SolverFailedError should include reason."""
        error = SolverFailedError("convergence failed", iterations=100)
        assert "convergence failed" in str(error)
        assert error.details["iterations"] == 100

    def test_infeasible_problem_error(self):
        """InfeasibleProblemError should list violated constraints."""
        error = InfeasibleProblemError(violated_constraints=["collision", "bounds"])
        assert "infeasible" in str(error).lower()
        assert error.details["violated_constraints"] == ["collision", "bounds"]

    def test_inheritance(self):
        """Solver errors should inherit from SolverError."""
        assert issubclass(SolverFailedError, SolverError)
        assert issubclass(InfeasibleProblemError, SolverError)
        assert issubclass(SolverError, PyMPCError)


class TestPlanningErrors:
    """Tests for planning-related errors."""

    def test_invalid_state_error(self):
        """InvalidStateError should include state name and reason."""
        error = InvalidStateError("velocity", "must be non-negative")
        assert "velocity" in str(error)
        assert "must be non-negative" in str(error)

    def test_inheritance(self):
        """Planning errors should inherit from PlanningError."""
        assert issubclass(InvalidStateError, PlanningError)
        assert issubclass(PlanningError, PyMPCError)


class TestModuleErrors:
    """Tests for module-related errors."""

    def test_constraint_error(self):
        """ConstraintError should include constraint name and reason."""
        error = ConstraintError("collision", "invalid obstacle data")
        assert "collision" in str(error)
        assert "invalid obstacle data" in str(error)

    def test_inheritance(self):
        """Module errors should inherit from ModuleError."""
        assert issubclass(ConstraintError, ModuleError)
        assert issubclass(ModuleError, PyMPCError)


class TestDataErrors:
    """Tests for data-related errors."""

    def test_invalid_obstacle_error(self):
        """InvalidObstacleError should include obstacle ID and reason."""
        error = InvalidObstacleError(42, "negative radius")
        assert "42" in str(error)
        assert "negative radius" in str(error)

    def test_inheritance(self):
        """Data errors should inherit from DataError."""
        assert issubclass(InvalidObstacleError, DataError)
        assert issubclass(DataError, PyMPCError)


class TestRuntimeErrors:
    """Tests for runtime limit errors."""

    def test_timeout_error(self):
        """TimeoutError should include timeout and elapsed time."""
        error = TimeoutError(timeout_seconds=10.0, elapsed_seconds=15.5)
        assert "10" in str(error) or "10.0" in str(error)
        assert "15" in str(error) or "15.5" in str(error)

    def test_iteration_limit_error(self):
        """IterationLimitError should include max and actual iterations."""
        error = IterationLimitError(max_iterations=100, actual_iterations=150)
        assert "100" in str(error)
        assert "150" in str(error)

    def test_inheritance(self):
        """Runtime errors should inherit from RuntimeLimitError."""
        assert issubclass(TimeoutError, RuntimeLimitError)
        assert issubclass(IterationLimitError, RuntimeLimitError)
        assert issubclass(RuntimeLimitError, PyMPCError)


class TestExceptionCatching:
    """Tests for exception hierarchy catching."""

    def test_catch_all_pympc_errors(self):
        """Should be able to catch all PyMPC errors with base class."""
        errors = [
            ConfigNotFoundError("/path"),
            SolverFailedError("test"),
            InvalidStateError("x", "test"),
            ConstraintError("test", "test"),
            InvalidObstacleError(0, "test"),
            TimeoutError(1.0, 2.0),
        ]

        for error in errors:
            try:
                raise error
            except PyMPCError:
                pass  # Expected
            else:
                pytest.fail(f"{type(error).__name__} not caught by PyMPCError")

    def test_specific_catching(self):
        """Should be able to catch specific error types."""
        try:
            raise ConfigNotFoundError("/path")
        except ConfigNotFoundError:
            pass  # Expected
        except Exception:
            pytest.fail("ConfigNotFoundError not caught specifically")
