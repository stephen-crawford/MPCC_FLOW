"""
Production logging system for PyMPC.

This module provides structured logging with:
- Configurable log levels via environment variables
- JSON formatting option for production
- Context-aware logging
- Performance profiling utilities
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Type variable for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Log Level Configuration
# =============================================================================

# Environment variable for log level
LOG_LEVEL_ENV = "PYMPC_LOG_LEVEL"
LOG_FORMAT_ENV = "PYMPC_LOG_FORMAT"
LOG_FILE_ENV = "PYMPC_LOG_FILE"

# Default log format
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
JSON_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'


def get_log_level() -> int:
    """Get log level from environment variable."""
    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_log_format() -> str:
    """Get log format from environment variable."""
    format_type = os.environ.get(LOG_FORMAT_ENV, "default").lower()
    if format_type == "json":
        return JSON_FORMAT
    return DEFAULT_FORMAT


def get_log_file() -> Optional[str]:
    """Get log file path from environment variable."""
    return os.environ.get(LOG_FILE_ENV)


# =============================================================================
# Logger Setup
# =============================================================================

# Root logger for PyMPC
_root_logger: Optional[logging.Logger] = None
_handlers: list = []


def setup_logging(
    level: Optional[int] = None,
    format_str: Optional[str] = None,
    log_file: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Setup the PyMPC logging system.

    Args:
        level: Log level (default: from env or INFO).
        format_str: Log format string (default: from env or DEFAULT_FORMAT).
        log_file: Optional file to write logs to.
        force: Force reconfiguration even if already setup.

    Returns:
        Configured root logger.
    """
    global _root_logger, _handlers

    if _root_logger is not None and not force:
        return _root_logger

    # Clean up existing handlers
    if _root_logger is not None:
        for handler in _handlers:
            _root_logger.removeHandler(handler)
    _handlers = []

    # Create logger
    _root_logger = logging.getLogger("pympc")
    _root_logger.setLevel(level or get_log_level())

    # Prevent propagation to root logger
    _root_logger.propagate = False

    # Formatter
    formatter = logging.Formatter(format_str or get_log_format())

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    _root_logger.addHandler(console_handler)
    _handlers.append(console_handler)

    # File handler (optional)
    file_path = log_file or get_log_file()
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        _root_logger.addHandler(file_handler)
        _handlers.append(file_handler)

    return _root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (will be prefixed with 'pympc.').
              If None, returns root pympc logger.

    Returns:
        Logger instance.
    """
    global _root_logger

    if _root_logger is None:
        setup_logging()

    if name:
        return logging.getLogger(f"pympc.{name}")
    return _root_logger


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================


def LOG_DEBUG(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    get_logger().debug(msg, *args, **kwargs)


def LOG_INFO(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    get_logger().info(msg, *args, **kwargs)


def LOG_WARN(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    get_logger().warning(msg, *args, **kwargs)


def LOG_WARNING(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message (alias)."""
    LOG_WARN(msg, *args, **kwargs)


def LOG_ERROR(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message."""
    get_logger().error(msg, *args, **kwargs)


def LOG_CRITICAL(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a critical message."""
    get_logger().critical(msg, *args, **kwargs)


# =============================================================================
# Context-Aware Logging
# =============================================================================


class LogContext:
    """Context manager for adding context to log messages."""

    _context: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        """Initialize with context key-value pairs."""
        self._local_context = kwargs
        self._previous_context: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        """Enter context, save previous values."""
        self._previous_context = {}
        for key, value in self._local_context.items():
            if key in LogContext._context:
                self._previous_context[key] = LogContext._context[key]
            LogContext._context[key] = value
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context, restore previous values."""
        for key in self._local_context:
            if key in self._previous_context:
                LogContext._context[key] = self._previous_context[key]
            else:
                del LogContext._context[key]

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return cls._context.get(key, default)

    @classmethod
    def format_context(cls) -> str:
        """Format current context as string."""
        if not cls._context:
            return ""
        return " ".join(f"{k}={v}" for k, v in cls._context.items())


def log_with_context(level: int, msg: str, *args: Any, **kwargs: Any) -> None:
    """Log message with current context."""
    context_str = LogContext.format_context()
    if context_str:
        msg = f"[{context_str}] {msg}"
    get_logger().log(level, msg, *args, **kwargs)


# =============================================================================
# Performance Profiling
# =============================================================================


@contextmanager
def profile_scope(name: str, log_level: int = logging.DEBUG):
    """Context manager for profiling code execution time.

    Args:
        name: Name of the profiled scope.
        log_level: Log level for the timing message.

    Yields:
        None

    Example:
        with profile_scope("optimization"):
            result = solver.solve()
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        get_logger().log(log_level, f"{name} took {elapsed:.4f}s")


# Alias for backward compatibility
PROFILE_SCOPE = profile_scope


def timed(func: F) -> F:
    """Decorator for timing function execution.

    Args:
        func: Function to time.

    Returns:
        Wrapped function that logs execution time.

    Example:
        @timed
        def solve_mpc():
            ...
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            get_logger().debug(f"{func.__name__} took {elapsed:.4f}s")

    return wrapper  # type: ignore


class TimeTracker:
    """Track execution times for performance analysis.

    Example:
        tracker = TimeTracker("solver")
        for step in range(100):
            with tracker.measure():
                solver.solve()
        tracker.print_stats()
    """

    def __init__(self, name: str):
        """Initialize the time tracker.

        Args:
            name: Name of the tracked operation.
        """
        self.name = name
        self._times: list[float] = []

    def add(self, timing_ms: float) -> None:
        """Add a timing measurement in milliseconds.

        Args:
            timing_ms: Time in milliseconds.
        """
        self._times.append(timing_ms)

    @contextmanager
    def measure(self):
        """Context manager for measuring execution time.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._times.append(elapsed_ms)

    def get_stats(self) -> tuple[float, float, int]:
        """Get timing statistics.

        Returns:
            Tuple of (mean_ms, max_ms, count).
        """
        if not self._times:
            return 0.0, 0.0, 0

        import numpy as np

        return float(np.mean(self._times)), float(np.max(self._times)), len(self._times)

    def print_stats(self) -> None:
        """Print timing statistics."""
        mean, max_val, count = self.get_stats()
        if count == 0:
            LOG_INFO(f"--- No timing data for {self.name} ---")
            return

        LOG_INFO(f"--- Computation Times: {self.name} ---")
        LOG_INFO(f"  Mean: {mean:.1f} ms")
        LOG_INFO(f"  Max: {max_val:.1f} ms")
        LOG_INFO(f"  Count: {count}")

    def reset(self) -> None:
        """Reset timing data."""
        self._times = []


# =============================================================================
# Assertion with Logging
# =============================================================================


def PYMPC_ASSERT(condition: bool, message: str) -> None:
    """Assert with detailed error logging.

    Args:
        condition: Condition to assert.
        message: Error message if assertion fails.

    Raises:
        AssertionError: If condition is False.
    """
    if not condition:
        import traceback

        # Get caller info
        stack = traceback.extract_stack()
        if len(stack) >= 2:
            caller = stack[-2]
            location = f"{caller.filename}:{caller.lineno}"
        else:
            location = "unknown"

        LOG_ERROR(f"Assertion failed at {location}: {message}")
        raise AssertionError(message)


# =============================================================================
# Module Initialization
# =============================================================================

# Setup logging on module import with defaults
setup_logging()
