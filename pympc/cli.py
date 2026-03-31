"""
Command-line interface for PyMPC.

This module provides the CLI entry point for running MPC planners
from the command line.

Usage:
    pympc run --constraint-type scenario
    pympc list-constraints
    pympc validate-config config.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pympc",
        description="PyMPC - Model Predictive Control for Motion Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pympc run                          Run MPC with default settings
  pympc run -c scenario              Run with scenario constraints
  pympc run --horizon 30 --timestep 0.05
  pympc list-constraints             List available constraint types
  pympc validate config.yml          Validate a configuration file
        """,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the MPC planner",
        description="Run the MPC planner with specified configuration",
    )
    _add_run_arguments(run_parser)

    # List constraints command
    list_parser = subparsers.add_parser(
        "list-constraints",
        help="List available constraint types",
        description="Display all registered constraint types",
    )

    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
        description="Validate a YAML configuration file",
    )
    validate_parser.add_argument(
        "config_file",
        type=Path,
        help="Path to configuration file",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display system and dependency information",
    )

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the run command."""
    # Configuration
    parser.add_argument(
        "--config", "-f",
        type=Path,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--constraint-type", "-c",
        choices=["scenario", "linearized"],
        default="scenario",
        help="Obstacle constraint type (default: scenario)",
    )

    # Planner settings
    parser.add_argument(
        "--horizon", "-H",
        type=int,
        default=20,
        help="Planning horizon (default: 20)",
    )

    parser.add_argument(
        "--timestep", "-t",
        type=float,
        default=0.1,
        help="Timestep in seconds (default: 0.1)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="Maximum MPC iterations (default: 80)",
    )

    # Goal settings
    parser.add_argument(
        "--goal",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=[18.0, 0.0],
        help="Goal position (default: 18.0 0.0)",
    )

    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "PSI"),
        default=[0.0, 0.0, 0.0],
        help="Start position and heading (default: 0.0 0.0 0.0)",
    )

    # Output settings
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for trajectory (JSON format)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable trajectory plotting",
    )


def setup_logging(verbose: int, quiet: bool) -> None:
    """Setup logging based on verbosity level."""
    import logging
    from pympc.logging import setup_logging as _setup_logging

    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    _setup_logging(level=level, force=True)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    from pympc.logging import LOG_INFO, LOG_ERROR

    try:
        # Import here to avoid slow startup for help commands
        from pympc import create_planner, run_mpc, create_default_config
        from planning.types import DynamicObstacle, Prediction, PredictionType
        from planning.types import generate_reference_path

        LOG_INFO("=" * 60)
        LOG_INFO("PyMPC - Model Predictive Control for Motion Planning")
        LOG_INFO("=" * 60)

        # Create configuration
        if args.config:
            from pympc.config import load_config
            config = load_config(args.config)
        else:
            config = create_default_config(args.constraint_type)

        # Override with CLI arguments
        config["planner"]["horizon"] = args.horizon
        config["planner"]["timestep"] = args.timestep

        LOG_INFO(f"Constraint type: {args.constraint_type}")
        LOG_INFO(f"Horizon: {args.horizon}, Timestep: {args.timestep}")

        # Create reference path
        start = args.start
        goal_pos = [args.goal[0], args.goal[1], 0.0]
        ref_path = generate_reference_path(
            start, goal_pos,
            path_type="straight",
            num_points=50,
        )
        LOG_INFO(f"Reference path: {len(ref_path.x)} points")

        # Create obstacles
        obstacles = []
        obs1 = DynamicObstacle(
            index=0,
            position=np.array([10.0, 0.5]),
            angle=np.pi,
            radius=0.5,
        )
        obs1.prediction = Prediction(PredictionType.GAUSSIAN)
        obs1.velocity = np.array([-0.3, 0.0])
        obstacles.append(obs1)

        obs2 = DynamicObstacle(
            index=1,
            position=np.array([8.0, -2.0]),
            angle=np.pi / 2,
            radius=0.5,
        )
        obs2.prediction = Prediction(PredictionType.GAUSSIAN)
        obs2.velocity = np.array([0.0, 0.2])
        obstacles.append(obs2)

        LOG_INFO(f"Created {len(obstacles)} obstacles")

        # Initial state
        initial_state = {
            "x": start[0],
            "y": start[1],
            "psi": start[2],
            "v": 0.5,
            "spline": 0.0,
        }
        LOG_INFO(f"Initial: x={initial_state['x']}, y={initial_state['y']}, v={initial_state['v']}")

        # Goal
        goal = np.array(args.goal)
        LOG_INFO(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")

        # Create planner
        LOG_INFO("Initializing planner...")
        planner = create_planner(initial_state, ref_path, obstacles, goal, config)

        # Run MPC
        LOG_INFO("Running MPC...")
        result = run_mpc(planner, max_steps=args.max_steps)

        # Print results
        LOG_INFO("=" * 60)
        LOG_INFO("Results:")
        LOG_INFO(f"  Steps: {result['steps']}")
        LOG_INFO(f"  Goal reached: {result['goal_reached']}")
        if result["trajectory"]:
            final_pos = result["trajectory"][-1]
            LOG_INFO(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
        LOG_INFO("=" * 60)

        # Save output if requested
        if args.output:
            import json
            output_data = {
                "steps": result["steps"],
                "goal_reached": result["goal_reached"],
                "trajectory": [(p[0], p[1]) for p in result["trajectory"]],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            LOG_INFO(f"Results saved to {args.output}")

        return 0 if result["goal_reached"] else 1

    except Exception as e:
        LOG_ERROR(f"Error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1


def cmd_list_constraints(args: argparse.Namespace) -> int:
    """Execute the list-constraints command."""
    from pympc import list_constraint_types

    print("Available constraint types:")
    for constraint_type in list_constraint_types():
        print(f"  - {constraint_type}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    from pympc.config import ConfigManager
    from pympc.exceptions import ConfigurationError

    try:
        manager = ConfigManager(args.config_file)
        config = manager.load(validate=True)
        print(f"Configuration file '{args.config_file}' is valid.")
        print(f"  Horizon: {config.planner.horizon}")
        print(f"  Timestep: {config.planner.timestep}")
        print(f"  Constraint type: {config.obstacle_constraint_type}")
        return 0

    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading configuration: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    import platform

    print("PyMPC System Information")
    print("=" * 40)
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

    print("Dependencies:")
    dependencies = ["numpy", "scipy", "casadi", "matplotlib", "pyyaml"]
    for dep in dependencies:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {dep}: {version}")
        except ImportError:
            print(f"  {dep}: NOT INSTALLED")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(args.verbose, args.quiet)

    # Execute command
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list-constraints":
        return cmd_list_constraints(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command is None:
        # Default to run with no arguments
        args.command = "run"
        args.config = None
        args.constraint_type = "scenario"
        args.horizon = 20
        args.timestep = 0.1
        args.max_steps = 80
        args.goal = [18.0, 0.0]
        args.start = [0.0, 0.0, 0.0]
        args.output = None
        args.no_plot = False
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
