"""
End-to-end checks: mm-link runs a real inner command, optional iperf3/CCP diagnostics.

``third_party/`` symlinks: run ``third_party/setup_symlinks.sh`` so Mahimahi/Nimbus live under the repo.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from network_emulation.cellular_traces import get_pair, list_cellular_pairs
from network_emulation.mahimahi import find_mm_link
from network_emulation.paths import mahimahi_root, nimbus_root, repo_root


@pytest.mark.integration
@pytest.mark.mahimahi
def test_mm_link_runs_inner_true_with_cellular_traces() -> None:
    """Full mm-link path: uplink + downlink traces, inner ``/bin/true`` exits 0."""
    mm = find_mm_link()
    pair = get_pair("Verizon-LTE-short")
    if pair is None:
        pairs = list_cellular_pairs()
        pair = pairs[0] if pairs else None
    if not mm or pair is None:
        pytest.skip("mm-link or cellular traces unavailable")
    r = subprocess.run(
        [mm, str(pair.uplink), str(pair.downlink), "--", "/bin/true"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0 and "Operation not permitted" in (r.stderr or ""):
        pytest.skip("mm-link needs privileges (setuid install or root)")
    assert r.returncode == 0, (r.stderr, r.stdout)


@pytest.mark.integration
def test_default_mahimahi_root_uses_third_party_when_present() -> None:
    """Without ``MAHIMAHI_ROOT``, prefer ``<repo>/third_party/mahimahi`` if it exists."""
    link = repo_root() / "third_party" / "mahimahi"
    if not link.is_dir():
        pytest.skip("third_party/mahimahi missing — run third_party/setup_symlinks.sh")
    if os.environ.get("MAHIMAHI_ROOT", "").strip():
        pytest.skip("MAHIMAHI_ROOT is set; unsets third_party default")
    assert mahimahi_root() == link.resolve()


@pytest.mark.integration
def test_default_nimbus_root_uses_third_party_when_present() -> None:
    link = repo_root() / "third_party" / "nimbus"
    if not link.is_dir():
        pytest.skip("third_party/nimbus missing")
    if os.environ.get("NIMBUS_ROOT", "").strip():
        pytest.skip("NIMBUS_ROOT is set")
    assert nimbus_root() == link.resolve()


@pytest.mark.integration
@pytest.mark.optional_tooling
def test_iperf3_available_for_throughput_experiments() -> None:
    exe = shutil.which("iperf3")
    if not exe:
        pytest.skip("iperf3 not installed (sudo apt install iperf3)")
    r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5)
    assert r.returncode == 0
    assert "iperf" in (r.stdout + r.stderr).lower()


@pytest.mark.integration
@pytest.mark.optional_tooling
def test_ccp_kernel_module_for_nimbus_runtime() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux only")
    if not Path("/proc/modules").is_file():
        pytest.skip("/proc/modules missing")
    data = Path("/proc/modules").read_text(encoding="utf-8", errors="replace")
    if "\nccp " not in data and not data.startswith("ccp "):
        pytest.skip(
            "CCP kernel module not loaded — Nimbus binary works; live TCP control needs "
            "https://ccp-project.github.io/guide"
        )
    assert "ccp" in data


@pytest.mark.integration
@pytest.mark.nimbus
def test_nimbus_binary_under_resolved_root() -> None:
    root = nimbus_root()
    if not (root / "Cargo.toml").is_file():
        pytest.skip(f"nimbus not cloned at {root} — run scripts/network/setup_baselines.sh")
    bin_release = root / "target" / "release" / "nimbus"
    if not bin_release.is_file():
        pytest.skip("run: cargo build --release (under nimbus_root)")
    assert os.access(bin_release, os.X_OK)
