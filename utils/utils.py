"""
PyMPC Utilities.

This module provides core utilities used throughout the codebase:
- Logging functions (LOG_INFO, LOG_DEBUG, etc.)
- Configuration management (load_yaml, CONFIG)
- Time profiling (PROFILE_SCOPE, TimeTracker)
- Display utilities (print_value, print_header, etc.)

Note: For new code, prefer importing from pympc directly:
    from pympc import LOG_INFO, LOG_DEBUG, TimeTracker
"""

import inspect
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import yaml

# Re-export from pympc.logging for new code that imports from here
try:
    from pympc.logging import (
        LOG_DEBUG as _LOG_DEBUG,
        LOG_INFO as _LOG_INFO,
        LOG_WARN as _LOG_WARN,
        LOG_ERROR as _LOG_ERROR,
        PYMPC_ASSERT as _PYMPC_ASSERT,
        TimeTracker as _TimeTracker,
        PROFILE_SCOPE as _PROFILE_SCOPE,
    )
    _USE_PYMPC_LOGGING = True
except ImportError:
    _USE_PYMPC_LOGGING = False


# ============================================================================
# Configuration Management
# ============================================================================

def load_yaml(path: str, target_dict: Dict) -> None:
    """Load YAML configuration file into the target dictionary.

    Args:
        path: Path to the YAML file.
        target_dict: Dictionary to update with loaded data.
    """
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            if data:
                target_dict.update(data)
    except Exception as e:
        LOG_DEBUG(f"Error loading config from {path}: {e}")


def read_config_file() -> Optional[Dict]:
    """Read the main configuration file.

    Returns:
        Configuration dictionary or None if reading fails.
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "CONFIG.yml")
    config_path = os.path.abspath(config_path)
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}


def write_to_config(key: str, value: Any) -> bool:
    """Write a key-value pair to the configuration file.

    Args:
        key: Configuration key.
        value: Value to set.

    Returns:
        True if successful, False otherwise.
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "CONFIG.yml")
    config_path = os.path.abspath(config_path)

    # Load current config
    try:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file) or {}
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return False

    # Update config
    config_data[key] = value

    # Write updated config
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config_data, file, default_flow_style=False)
        print(f"Updated {key} in config file.")
        return True
    except Exception as e:
        print(f"Error writing to config file: {e}")
        return False


def get_config_dotted(config: Optional[Dict], dotted_key: str, default: Any = None) -> Any:
    """Get a nested config value using dotted key notation.

    Args:
        config: The config dictionary.
        dotted_key: Dot-separated key path (e.g., 'planner.horizon').
        default: Default value if key not found.

    Returns:
        The config value or default if not found.
    """
    if config is None:
        return default

    keys = dotted_key.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# Initialize global configuration
CONFIG = read_config_file() or {}
SAVE_FOLDER = CONFIG.get("recording", {}).get("folder", "recordings")
SAVE_FILE = CONFIG.get("recording", {}).get("file", "data")


# ============================================================================
# Logging
# ============================================================================

# Use pympc.logging if available, otherwise fall back to local logger
if _USE_PYMPC_LOGGING:
    # Re-export from pympc.logging
    LOG_DEBUG = _LOG_DEBUG
    LOG_INFO = _LOG_INFO
    LOG_WARN = _LOG_WARN
    LOG_ERROR = _LOG_ERROR
    PYMPC_ASSERT = _PYMPC_ASSERT

    # Get the logger from pympc
    from pympc.logging import get_logger
    logger = get_logger()
else:
    # Fallback: Create and configure local logger
    logger = logging.getLogger("PyMPC")
    logger.setLevel(logging.INFO)

    # Console handler
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.DEBUG)
    _console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(_console_handler)

    # File handler
    try:
        _file_handler = logging.FileHandler('debug.log')
        _file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(_file_handler)
    except (IOError, OSError):
        pass  # Skip file handler if we can't write to debug.log

    def LOG_DEBUG(msg: str) -> None:
        """Log a debug message."""
        logger.debug(msg)

    def LOG_INFO(msg: str) -> None:
        """Log an info message."""
        logger.info(msg)

    def LOG_WARN(msg: str) -> None:
        """Log a warning message."""
        logger.warning(msg)

    def LOG_ERROR(msg: str) -> None:
        """Log an error message."""
        logger.error(msg)

    def PYMPC_ASSERT(expr: bool, msg: str) -> None:
        """Assert with detailed error logging.

        Args:
            expr: Expression to assert.
            msg: Error message if assertion fails.

        Raises:
            AssertionError: If expr is False.
        """
        if not expr:
            frame = inspect.currentframe().f_back
            file = frame.f_code.co_filename
            line = frame.f_lineno
            expr_str = frame.f_globals.get("__name__", "Unknown")

            logger.error(f"Assert failed:\t{msg}\n"
                         f"Expected:\t{expr_str}\n"
                         f"Source:\t\t{file}, line {line}\n")
            raise AssertionError(msg)


# ============================================================================
# Time Profiling
# ============================================================================

# Use pympc versions if available
if _USE_PYMPC_LOGGING:
    PROFILE_SCOPE = _PROFILE_SCOPE
    TimeTracker = _TimeTracker
else:
    @contextmanager
    def PROFILE_SCOPE(name: str):
        """Context manager for profiling code execution time.

        Args:
            name: Name of the profiled scope.

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            logging.debug(f"{name} took {elapsed_time:.6f} seconds")

    class TimeTracker:
        """Track execution times for performance analysis."""

        def __init__(self, name: str):
            """Initialize the time tracker.

            Args:
                name: Name of the tracked operation.
            """
            self.name = name
            self._times = []

        def add(self, timing: float) -> None:
            """Add a timing measurement.

            Args:
                timing: Time in milliseconds.
            """
            self._times.append(timing)

        def get_stats(self):
            """Get timing statistics.

            Returns:
                Tuple of (mean, max, count).
            """
            if not self._times:
                return 0.0, 0.0, 0
            return np.mean(self._times), np.max(self._times), len(self._times)

        def print_stats(self) -> None:
            """Print timing statistics."""
            if not self._times:
                print(f"--- No timing data for {self.name} ---")
                return
            print(f"--- Computation Times {self.name} ---")
            print_value("Mean", f"{np.mean(self._times):.1f} ms", tab=True)
            print_value("Max", f"{np.max(self._times):.1f} ms", tab=True)
            print_value("Number of calls", len(self._times), tab=True)


# ============================================================================
# Display Utilities
# ============================================================================

class bcolors:
    """ANSI color codes for terminal output."""
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    HEADER = BOLD


def print_value(name: str, value: Any, tab: bool = False, **kwargs) -> None:
    """Print a formatted name-value pair.

    Args:
        name: Name/label to display.
        value: Value to display.
        tab: Whether to indent with a space.
        **kwargs: Additional kwargs passed to print().
    """
    prefix = " " if tab else ""
    print(prefix + bcolors.BOLD + bcolors.UNDERLINE + f"{name}" + bcolors.ENDC + f": {value}", **kwargs)


def print_path(name: str, value: str, tab: bool = False, **kwargs) -> None:
    """Print a formatted path value.

    Args:
        name: Name/label to display.
        value: Path to display (will be converted to absolute).
        tab: Whether to indent with a space.
        **kwargs: Additional kwargs passed to print().
    """
    print_value(name, os.path.abspath(value), tab, **kwargs)


def print_success(msg: str) -> None:
    """Print a success message in green.

    Args:
        msg: Message to display.
    """
    print(bcolors.BOLD + bcolors.OKGREEN + f"{msg}" + bcolors.ENDC)


def print_warning(msg: str, no_tab: bool = False) -> None:
    """Print a warning message in yellow.

    Args:
        msg: Message to display.
        no_tab: If True, don't indent the message.
    """
    prefix = "" if no_tab else "\t"
    print(prefix + bcolors.BOLD + bcolors.WARNING + f"Warning: {msg}" + bcolors.ENDC)


def print_header(msg: str) -> None:
    """Print a formatted header.

    Args:
        msg: Header text to display.
    """
    print("==============================================")
    print("\t" + bcolors.HEADER + f"{msg}" + bcolors.ENDC)
    print("==============================================")


# ============================================================================
# Path Utilities
# ============================================================================

def get_base_path() -> str:
    """Get the base path for configuration files."""
    return os.path.dirname(os.path.realpath('../../utils'))


def save_config_path() -> str:
    """Get the path for saving configuration files."""
    config_path = os.path.join(get_base_path(), "utils/")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    return config_path


def model_map_path() -> str:
    """Get the path for the model map YAML file."""
    return os.path.join(save_config_path(), "model_map.yml")


def write_to_yaml(filename: str, data: Dict) -> None:
    """Write data to a YAML file.

    Args:
        filename: Path to the YAML file.
        data: Dictionary to write.
    """
    with open(filename, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


# ============================================================================
# Robot Area Definition (re-exported from planning.types)
# ============================================================================

if TYPE_CHECKING:
    from planning.types import Disc


def define_robot_area(length: float, width: float, n_discs: int):
    """Define robot area with discs.

    This function is re-exported from planning.types for backward compatibility.

    Args:
        length: Robot length.
        width: Robot width.
        n_discs: Number of collision discs.

    Returns:
        List of Disc objects.
    """
    from planning.types import define_robot_area as _define_robot_area
    return _define_robot_area(length, width, n_discs)
