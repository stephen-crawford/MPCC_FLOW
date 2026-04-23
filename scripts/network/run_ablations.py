"""MPCC ablation driver (paper §V.C).

Starts from a base MPCC weight YAML and re-runs the matrix with one weight
zeroed out at a time, so we can isolate the contribution of each MPCC term:

  no_contour    -- contour_weight = 0
  no_lag        -- contouring_lag_weight = 0
  no_fairness   -- fairness_weight = 0
  qp_vs_nlp     -- flip --mpcc-solver between qp and nlp (paper §V.D)

For each variant we invoke `scripts/network/run_cc_experiments.py run` with
`--ccas mpcc` and a generated override YAML. Output lives under
<output>/ablations/<variant>/. After the runs complete, paper_metrics.py
produces the per-variant tables.

Usage:

  sudo python -m scripts.network.run_ablations \\
      --scenarios wired cellular fairness \\
      --duration 30 --seeds 2 \\
      --output results/ablations/$(date +%Y%m%dT%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML required for ablations (pip install pyyaml)", file=sys.stderr)
    raise

logger = logging.getLogger("run_ablations")

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_DIR = REPO_ROOT / "scripts" / "network" / "configs" / "paper"

ABLATIONS: dict[str, dict] = {
    "baseline": {},  # no overrides
    "no_contour": {"weights": {"contour_weight": 0.0}},
    "no_lag": {"weights": {"contouring_lag_weight": 0.0}},
    "no_fairness": {"weights": {"fairness_weight": 0.0}},
    # QP vs NLP is handled via --solver, not a YAML override.
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = json.loads(json.dumps(base))  # simple deep copy
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _make_variant_config_dir(
    variant: str, override: dict, base_dir: Path, out_dir: Path,
) -> Path:
    """Clone base_dir YAMLs into out_dir, applying override to each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for yml in base_dir.glob("*.yml"):
        with open(yml) as f:
            base_cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(base_cfg, override)
        with open(out_dir / yml.name, "w") as f:
            yaml.safe_dump(merged, f, sort_keys=False)
    logger.info("variant=%s wrote %d config(s) to %s",
                variant, len(list(out_dir.glob('*.yml'))), out_dir)
    return out_dir


def _run_matrix(
    scenarios: list[str], duration_s: float, seeds: int,
    config_dir: Path, solver: str, output: Path,
) -> int:
    cmd = [
        sys.executable, "-m", "scripts.network.run_cc_experiments", "run",
        "--scenarios", *scenarios,
        "--ccas", "mpcc",
        "--duration", str(duration_s),
        "--seeds", str(seeds),
        "--output", str(output),
        "--mpcc-config-dir", str(config_dir),
        "--mpcc-solver", solver,
    ]
    logger.info("exec: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def run(
    scenarios: list[str], duration_s: float, seeds: int,
    output_root: Path, include_nlp: bool = False,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    # Weight ablations: each keeps solver=qp, overrides one weight.
    for variant, override in ABLATIONS.items():
        variant_cfg_dir = output_root / "_configs" / variant
        _make_variant_config_dir(variant, override, BASE_CONFIG_DIR, variant_cfg_dir)
        run_out = output_root / variant
        _run_matrix(scenarios, duration_s, seeds, variant_cfg_dir, "qp", run_out)

    # QP vs NLP (paper §V.D): requires pyportus (the Rust binary is QP-only).
    if include_nlp:
        variant_cfg_dir = output_root / "_configs" / "nlp"
        _make_variant_config_dir("nlp", {}, BASE_CONFIG_DIR, variant_cfg_dir)
        run_out = output_root / "qp_vs_nlp"
        _run_matrix(scenarios, duration_s, seeds, variant_cfg_dir, "nlp", run_out)

    # Post-process every variant directory with paper_metrics.
    for sub in output_root.iterdir():
        if sub.name.startswith("_"):
            continue
        if not sub.is_dir():
            continue
        subprocess.call(
            [sys.executable, "-m", "scripts.network.paper_metrics", str(sub)],
            cwd=REPO_ROOT,
        )

    # Emit a consolidated ablation summary.
    summary: dict = {"variants": {}}
    for sub in output_root.iterdir():
        if sub.name.startswith("_") or not sub.is_dir():
            continue
        agg = sub / "paper_summary.json"
        if agg.is_file():
            with open(agg) as f:
                summary["variants"][sub.name] = json.load(f)
    with open(output_root / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", output_root / "ablation_summary.json")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(prog="run_ablations")
    p.add_argument("--scenarios", nargs="+",
                   default=["wired", "cellular", "fairness"])
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--output", "-o", required=True,
                   help="Output root; each variant writes a subdirectory")
    p.add_argument("--include-nlp", action="store_true",
                   help="Also run QP-vs-NLP ablation (requires pyportus)")
    args = p.parse_args(argv)
    if shutil.which("iperf3") is None:
        logger.error("iperf3 not in PATH")
        return 1
    run(
        scenarios=args.scenarios, duration_s=args.duration,
        seeds=args.seeds, output_root=Path(args.output),
        include_nlp=args.include_nlp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
