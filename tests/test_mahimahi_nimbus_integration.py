"""
Integration checks against real Mahimahi and Nimbus installs.

These tests **skip** when binaries or traces are missing (typical CI / minimal env).
When ``mm-link``, traces under ``MAHIMAHI_ROOT``, and/or ``nimbus`` exist locally,
they **run** and validate CLI behavior.

Set ``MAHIMAHI_ROOT``, ``NIMBUS_ROOT``, ``MAHIMAHI_MM_LINK`` as needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from network_emulation.cellular_traces import get_pair, list_cellular_pairs
from network_emulation.mahimahi import find_mm_graph, find_mm_link, mm_link_from_pair
from network_emulation.nimbus_ccp import find_nimbus_binary


@pytest.mark.integration
@pytest.mark.mahimahi
def test_mm_link_prints_usage_without_trace_arguments() -> None:
    mm = find_mm_link()
    if not mm:
        pytest.skip("mm-link not found: install Mahimahi (sudo make install) or set MAHIMAHI_MM_LINK")
    r = subprocess.run([mm], capture_output=True, text=True, timeout=15)
    assert r.returncode != 0
    combined = (r.stderr or "") + (r.stdout or "")
    assert "UPLINK-TRACE" in combined or "Usage" in combined


@pytest.mark.integration
@pytest.mark.mahimahi
def test_mm_graph_requires_log_arguments() -> None:
    g = find_mm_graph()
    if not g:
        pytest.skip("mm-graph not found; set MAHIMAHI_ROOT to Mahimahi source tree")
    script = Path(g)
    cmd = [sys.executable, str(script)] if script.is_file() else [g]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert r.returncode != 0
    combined = (r.stderr or "") + (r.stdout or "")
    assert "log" in combined.lower() or "ms_per_bin" in combined or "arguments" in combined.lower()


@pytest.mark.integration
@pytest.mark.mahimahi
def test_cellular_trace_pairs_on_disk() -> None:
    pairs = list_cellular_pairs()
    if not pairs:
        pytest.skip(
            "No .up/.down pairs under MAHIMAHI_ROOT/traces — set MAHIMAHI_ROOT to your Mahimahi checkout"
        )
    for p in pairs[:5]:
        assert p.uplink.is_file(), p.uplink
        assert p.downlink.is_file(), p.downlink


@pytest.mark.integration
@pytest.mark.mahimahi
def test_mm_link_argv_uses_verizon_short_or_first_pair() -> None:
    mm = find_mm_link()
    pair = get_pair("Verizon-LTE-short")
    if pair is None:
        all_p = list_cellular_pairs()
        pair = all_p[0] if all_p else None
    if not mm or pair is None:
        pytest.skip("need mm-link and at least one cellular trace pair")
    spec = mm_link_from_pair(pair, ["/bin/true"])
    argv = spec.argv()
    assert argv[0] == mm or argv[0].endswith("mm-link")
    assert argv[1] == str(pair.uplink)
    assert argv[2] == str(pair.downlink)
    assert "--" in argv
    assert argv[-2] == "--"
    assert argv[-1] == "/bin/true"


@pytest.mark.integration
@pytest.mark.nimbus
def test_nimbus_binary_accepts_help() -> None:
    if os.environ.get("SKIP_NIMBUS_TESTS"):
        pytest.skip("SKIP_NIMBUS_TESTS is set")
    nb = find_nimbus_binary()
    if not nb:
        pytest.skip("Nimbus binary not found: cd NIMBUS_ROOT && cargo build --release")
    r = subprocess.run([str(nb), "--help"], capture_output=True, text=True, timeout=15)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0 or "ipc" in out.lower() or "nimbus" in out.lower()


@pytest.mark.integration
@pytest.mark.nimbus
def test_nimbus_rust_sources_readable() -> None:
    """Lightweight check that NIMBUS_ROOT looks like the ccp_nimbus workspace (no binary required)."""
    from network_emulation.paths import nimbus_root

    root = nimbus_root()
    cargo = root / "Cargo.toml"
    lib = root / "src" / "lib.rs"
    if not cargo.is_file() or not lib.is_file():
        pytest.skip(f"NIMBUS_ROOT does not look like nimbus checkout: {root}")
    text = cargo.read_text(encoding="utf-8", errors="replace")
    assert "ccp_nimbus" in text or "nimbus" in text.lower()


@pytest.mark.integration
@pytest.mark.mahimahi
def test_paper_config_yamls_are_complete() -> None:
    """The paper matrix expects four per-scenario weight YAMLs under
    scripts/network/configs/paper/. Verify they exist and parse."""
    import yaml
    root = Path(__file__).resolve().parent.parent / "scripts" / "network" / "configs" / "paper"
    for scen in ("wired", "cellular", "leo", "fairness"):
        p = root / f"{scen}.yml"
        assert p.is_file(), f"missing {p}"
        cfg = yaml.safe_load(p.read_text())
        # Every scenario YAML must name every weight explicitly so paper
        # ablations can zero individual terms without surprise defaults.
        assert "weights" in cfg, p
        for w in (
            "contour_weight", "contouring_lag_weight", "delay_weight",
            "power_weight", "acceleration_weight", "fairness_weight",
        ):
            assert w in cfg["weights"], f"{p} missing weights.{w}"


@pytest.mark.integration
@pytest.mark.mahimahi
def test_portus_mpcc_binary_built() -> None:
    """The Rust portus-mpcc binary is the production datapath. It must exist
    before run_paper_matrix.sh fires, otherwise every MPCC run falls back to
    kernel CUBIC and the paper's §V.A numbers are meaningless."""
    from scripts.network.run_cc_experiments import _find_portus_mpcc
    bin_path = _find_portus_mpcc()
    if bin_path is None:
        pytest.skip("build it first: cd third_party/portus-mpcc && cargo build --release")
    # Sanity-check the binary announces itself as mpcc_cca.
    r = subprocess.run([bin_path, "--help"], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0
    assert "mpcc" in r.stdout.lower()


@pytest.mark.integration
@pytest.mark.mahimahi
def test_paper_metrics_regex_matches_log_format() -> None:
    """Guard against regressions in the mpcc_step log format that
    paper_metrics.py relies on to extract solve times."""
    from scripts.network.paper_metrics import MPCC_LOG_LINE
    sample = (
        "[2026-04-21T15:05:00Z INFO  portus_mpcc] "
        "mpcc_step solve_ms=0.812 rate_bps=9.5e6 tput_bps=8.4e6 "
        "rtt_us=41200 q_bytes=1500.5 success=1"
    )
    m = MPCC_LOG_LINE.search(sample)
    assert m is not None
    assert float(m.group("solve")) == pytest.approx(0.812)
    assert int(m.group("rtt")) == 41200
