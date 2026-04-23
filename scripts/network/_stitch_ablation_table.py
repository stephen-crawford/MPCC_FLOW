r"""Build the LaTeX ablation table from the output of run_ablations.py.

Emits <paper_root>/tables/ablation.tex, which is \input{...}'d by mpcc_flow.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def stitch(ablations_root: Path, out_path: Path) -> int:
    variants: dict[str, list[dict]] = {}
    for variant_dir in sorted(ablations_root.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
            continue
        summary = variant_dir / "paper_summary.json"
        if not summary.is_file():
            continue
        try:
            variants[variant_dir.name] = json.loads(summary.read_text())
        except json.JSONDecodeError:
            print(f"skip malformed {summary}", file=sys.stderr)

    if not variants:
        print(f"no variants found under {ablations_root}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(
            "\\begin{tabular}{llrrr}\n\\toprule\n"
            "Variant & Scenario & Throughput (Mbps) & p50 RTT (ms) & Rate CoV \\\\ \n"
            "\\midrule\n"
        )
        for vname, rows in variants.items():
            safe_name = vname.replace("_", "\\_")
            for r in rows:
                f.write(
                    f"{safe_name} & {r['scenario']} & "
                    f"{r.get('throughput_mean_mbps_mean', 0.0):.2f} & "
                    f"{r.get('rtt_p50_ms_mean', 0.0):.1f} & "
                    f"{r.get('rate_cov_mean', 0.0):.3f} \\\\ \n"
                )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stitch_ablation")
    p.add_argument("ablations_root")
    p.add_argument("out_path")
    args = p.parse_args(argv)
    return stitch(Path(args.ablations_root), Path(args.out_path))


if __name__ == "__main__":
    raise SystemExit(main())
