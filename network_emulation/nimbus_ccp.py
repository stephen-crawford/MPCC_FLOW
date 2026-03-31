"""
Nimbus (CCP) — elasticity detection for congestion control; runs as a Portus algorithm.

This is **not** a Python library: the implementation lives in the Nimbus Rust crate and
communicates with the kernel CCP datapath. Use this module to locate the binary and
surface the documented invocation pattern.
"""

from __future__ import annotations

from pathlib import Path

from network_emulation.paths import nimbus_debug_binary, nimbus_release_binary, nimbus_root


def find_nimbus_binary(prefer_release: bool = True) -> Path | None:
    rel = nimbus_release_binary()
    dbg = nimbus_debug_binary()
    if prefer_release and rel.is_file():
        return rel
    if dbg.is_file():
        return dbg
    if rel.is_file():
        return rel
    return None


def nimbus_usage_notes() -> str:
    return """\
Nimbus (ccp_nimbus) runs as a CCP congestion-control algorithm:

  1. Kernel: load the CCP / datapath modules required by your Portus setup
     (see https://ccp-project.github.io/guide).

  2. Start Nimbus before or after flows, depending on your setup:

       {path_example} --ipc unix

  3. TCP flows using the CCP datapath will then use the Nimbus algorithm.

  Environment: NIMBUS_ROOT={root}
  Default binary: target/release/nimbus after `cargo build --release`
""".format(
        path_example=str(nimbus_release_binary() or "nimbus"),
        root=nimbus_root(),
    )
