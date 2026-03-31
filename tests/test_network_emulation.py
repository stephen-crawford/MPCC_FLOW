"""Tests for Mahimahi/Nimbus path helpers (no system tools required)."""

from __future__ import annotations

from pathlib import Path

from network_emulation.cellular_traces import CellularTracePair, get_pair, list_cellular_pairs
from network_emulation.mahimahi import MmLinkSpec


def test_iter_cellular_pairs_finds_pairs(tmp_path: Path) -> None:
    d = tmp_path / "traces"
    d.mkdir()
    (d / "Foo-LTE.up").write_text("0\n1\n")
    (d / "Foo-LTE.down").write_text("0\n1\n")
    pairs = list_cellular_pairs(d)
    assert len(pairs) == 1
    assert pairs[0].name == "Foo-LTE"
    assert pairs[0].carrier == "Foo"
    assert pairs[0].exists()


def test_get_pair_missing_returns_none(tmp_path: Path) -> None:
    assert get_pair("nope", tmp_path) is None


def test_mm_link_spec_builds_argv(tmp_path: Path) -> None:
    up = tmp_path / "a.up"
    down = tmp_path / "b.down"
    up.write_text("0\n")
    down.write_text("0\n")
    spec = MmLinkSpec(
        uplink_trace=up,
        downlink_trace=down,
        inner_command=["echo", "hi"],
        mm_link_binary="/bin/mm-link",
    )
    argv = spec.argv()
    assert argv[:4] == ["/bin/mm-link", str(up), str(down), "--"]
    assert argv[4:] == ["echo", "hi"]


def test_cellular_trace_pair_dataclass() -> None:
    p = CellularTracePair(
        name="x",
        uplink=Path("/a.up"),
        downlink=Path("/b.down"),
        carrier="c",
        kind="k",
    )
    assert p.name == "x"
