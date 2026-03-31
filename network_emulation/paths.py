"""
Resolve locations of Mahimahi (emulation + traces + plotting) and Nimbus (CCP).

Resolution order:

1. ``MAHIMAHI_ROOT`` / ``NIMBUS_ROOT`` if set (non-empty).
2. ``<mpcc_flow>/third_party/mahimahi`` or ``third_party/nimbus`` if present (symlink or dir).
3. ``~/mahimahi`` and ``~/nimbus``.

``third_party`` links are created by ``third_party/setup_symlinks.sh`` (relative symlinks).
"""

from __future__ import annotations

import os
from pathlib import Path


def _expand(p: str | None, default: str) -> Path:
    base = p.strip() if p else default
    return Path(os.path.expanduser(base)).resolve()


def repo_root() -> Path:
    """Root of the mpcc_flow repository (parent of ``network_emulation``)."""
    return Path(__file__).resolve().parent.parent


def mahimahi_root() -> Path:
    """Root of the Mahimahi source tree (contains ``traces/``, ``scripts/``)."""
    env = os.environ.get("MAHIMAHI_ROOT")
    if env and env.strip():
        return _expand(env, "")
    bundled = repo_root() / "third_party" / "mahimahi"
    if bundled.is_dir():
        return bundled.resolve()
    return Path(os.path.expanduser("~/mahimahi")).resolve()


def nimbus_root() -> Path:
    """Root of the Nimbus (ccp_nimbus) Rust workspace."""
    env = os.environ.get("NIMBUS_ROOT")
    if env and env.strip():
        return _expand(env, "")
    bundled = repo_root() / "third_party" / "nimbus"
    if bundled.is_dir():
        return bundled.resolve()
    return Path(os.path.expanduser("~/nimbus")).resolve()


def mahimahi_traces_dir() -> Path:
    return mahimahi_root() / "traces"


def mahimahi_scripts_dir() -> Path:
    return mahimahi_root() / "scripts"


def nimbus_release_binary() -> Path:
    """Expected path after ``cargo build --release`` in the Nimbus repo."""
    return nimbus_root() / "target" / "release" / "nimbus"


def nimbus_debug_binary() -> Path:
    return nimbus_root() / "target" / "debug" / "nimbus"
