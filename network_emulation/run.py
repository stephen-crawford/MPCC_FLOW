"""
CLI: list cellular traces, print mm-link commands, check Mahimahi/Nimbus paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from network_emulation.cellular_traces import get_pair, list_cellular_pairs
from network_emulation.mahimahi import MmLinkSpec, check_mahimahi_tools, mm_link_from_pair
from network_emulation.nimbus_ccp import find_nimbus_binary, nimbus_usage_notes
from network_emulation.paths import mahimahi_root, nimbus_root


def _cmd_list_traces(args: argparse.Namespace) -> int:
    pairs = list_cellular_pairs()
    if not pairs:
        print(f"No trace pairs found under {mahimahi_root() / 'traces'}", file=sys.stderr)
        print("Set MAHIMAHI_ROOT or add a Mahimahi checkout.", file=sys.stderr)
        return 1
    for p in pairs:
        line = f"{p.name}\t{p.carrier}\t{p.uplink.name}"
        print(line)
    return 0


def _cmd_mm_link(args: argparse.Namespace) -> int:
    pair = get_pair(args.trace)
    if pair is None or not pair.exists():
        print(f"Unknown or missing trace pair: {args.trace}", file=sys.stderr)
        return 1
    inner = args.inner_parts or ["/bin/bash"]
    spec = mm_link_from_pair(
        pair,
        inner,
        uplink_log=Path(args.uplink_log) if args.uplink_log else None,
        downlink_log=Path(args.downlink_log) if args.downlink_log else None,
    )
    cmd = spec.argv()
    if args.json:
        print(json.dumps({"argv": cmd}))
    else:
        print(" ".join(f"'{c}'" if " " in c else c for c in cmd))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    tools = check_mahimahi_tools()
    nb = find_nimbus_binary()
    out = {
        "mahimahi": tools,
        "mahimahi_traces": str(mahimahi_root() / "traces"),
        "nimbus_root": str(nimbus_root()),
        "nimbus_binary": str(nb) if nb else None,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for sk, sv in v.items():
                    print(f"  {sk}: {sv}")
            else:
                print(f"{k}: {v}")
        if nb is None:
            print("\nNimbus binary not built; run: cd NIMBUS_ROOT && cargo build --release")
    return 0


def _cmd_nimbus_help(args: argparse.Namespace) -> int:
    print(nimbus_usage_notes())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mpcc-network", description="Mahimahi / Nimbus helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-traces", help="List cellular uplink/downlink trace pairs")
    sp.set_defaults(func=_cmd_list_traces)

    sp = sub.add_parser("mm-link-cmd", help="Print mm-link argv for a trace pair")
    sp.add_argument("trace", help="Trace stem, e.g. Verizon-LTE-short")
    sp.add_argument(
        "inner_parts",
        nargs="*",
        default=["/bin/bash"],
        help="Command to run inside mm-link (default: /bin/bash)",
    )
    sp.add_argument("--uplink-log", default=None)
    sp.add_argument("--downlink-log", default=None)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_mm_link)

    sp = sub.add_parser("check", help="Show resolved paths and tools")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_check)

    sp = sub.add_parser("nimbus-help", help="Print how to run Nimbus with CCP")
    sp.set_defaults(func=_cmd_nimbus_help)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
