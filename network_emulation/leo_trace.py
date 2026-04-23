"""Generate LEO-inspired and wired-bottleneck mahimahi traces.

The mahimahi trace format is one timestamp (ms, one-based) per MTU-sized
delivery opportunity at that instant. For a constant C Mbps link:

    each 1500-byte packet delivery takes  1500 * 8 / (C * 1e6) seconds,
    so timestamps are  k * (12_000 / C_mbps)  microseconds, rounded to ms.

LEO-inspired traces here model the dominant pattern Starlink/Kuiper links
exhibit to end hosts: bulk capacity is relatively high but changes abruptly
on tens of seconds (handover between satellites), with occasional short
outages of a few hundred ms. Propagation delay is handled separately through
`mm-delay` and is *not* encoded in this trace — so the generator only models
time-varying capacity.

CLI:

    python -m network_emulation.leo_trace wired \\
        --rate-mbps 10 --duration-s 60 --out traces/Constant-10Mbps

    python -m network_emulation.leo_trace leo \\
        --baseline-mbps 50 --duration-s 60 --out traces/LEO-handover \\
        --handover-period-s 15 --handover-drop-mbps 5 --seed 0

Each command writes a matching `.up` / `.down` pair at the output stem.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable


_MTU_BITS = 1500 * 8


def _rate_to_interval_us(mbps: float) -> float:
    """Return microseconds per MTU delivery at `mbps`."""
    if mbps <= 0:
        return math.inf
    return _MTU_BITS / (mbps * 1e6) * 1e6


def constant_trace(rate_mbps: float, duration_s: float) -> list[int]:
    """Uniform MTU deliveries for `duration_s` at `rate_mbps`."""
    interval_us = _rate_to_interval_us(rate_mbps)
    n_pkts = int(duration_s * 1e6 / interval_us)
    return [max(1, int(round((k + 1) * interval_us / 1000))) for k in range(n_pkts)]


def variable_trace(segments: Iterable[tuple[float, float]]) -> list[int]:
    """Piecewise-constant capacity trace.

    segments: iterable of (duration_s, rate_mbps). Emits MTU deliveries for
    each segment in order.
    """
    ts_ms: list[int] = []
    t_us = 0.0
    for duration_s, rate_mbps in segments:
        interval_us = _rate_to_interval_us(rate_mbps)
        if math.isinf(interval_us):
            t_us += duration_s * 1e6
            continue
        n_pkts = int(duration_s * 1e6 / interval_us)
        for k in range(n_pkts):
            t_us += interval_us
            ts_ms.append(max(1, int(round(t_us / 1000))))
    return ts_ms


def leo_trace(
    baseline_mbps: float,
    duration_s: float,
    *,
    handover_period_s: float = 15.0,
    handover_drop_mbps: float = 5.0,
    handover_duration_s: float = 0.8,
    outage_prob: float = 0.05,
    outage_duration_s: float = 0.3,
    jitter_frac: float = 0.15,
    seed: int | None = 0,
) -> list[int]:
    """Synthesize a LEO-inspired time-varying capacity trace.

    Model: a baseline rate punctuated by periodic handover dips (satellite
    switch) and occasional brief outages. The intent is to stress controllers
    with abrupt capacity jumps while staying qualitatively similar to
    measurements reported in SaTCP / LeoCC.
    """
    rng = random.Random(seed)
    segments: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_s:
        # Regular cruise window with mild jitter around baseline.
        cruise = min(handover_period_s - handover_duration_s, duration_s - t)
        rate = baseline_mbps * (1 + rng.uniform(-jitter_frac, jitter_frac))
        if rng.random() < outage_prob:
            pre_outage = max(0.0, cruise - outage_duration_s) / 2
            segments.append((pre_outage, rate))
            segments.append((outage_duration_s, 0.01))  # near-zero, not zero
            segments.append((cruise - pre_outage - outage_duration_s, rate))
        else:
            segments.append((cruise, rate))
        t += cruise
        if t >= duration_s:
            break

        # Handover dip.
        dip = min(handover_duration_s, duration_s - t)
        segments.append((dip, max(handover_drop_mbps, 0.05)))
        t += dip

    return variable_trace(segments)


def write_trace_pair(ts_ms: list[int], stem: Path) -> tuple[Path, Path]:
    """Write the same timestamp list as `<stem>.up` and `<stem>.down`."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    up = stem.with_suffix(stem.suffix + ".up") if stem.suffix else stem.parent / f"{stem.name}.up"
    down = stem.with_suffix(stem.suffix + ".down") if stem.suffix else stem.parent / f"{stem.name}.down"
    body = "\n".join(str(x) for x in ts_ms) + "\n"
    up.write_text(body)
    down.write_text(body)
    return up, down


def _cmd_wired(args) -> int:
    ts = constant_trace(args.rate_mbps, args.duration_s)
    up, down = write_trace_pair(ts, Path(args.out))
    print(f"wrote {len(ts)} timestamps -> {up}, {down}")
    return 0


def _cmd_leo(args) -> int:
    ts = leo_trace(
        baseline_mbps=args.baseline_mbps,
        duration_s=args.duration_s,
        handover_period_s=args.handover_period_s,
        handover_drop_mbps=args.handover_drop_mbps,
        handover_duration_s=args.handover_duration_s,
        outage_prob=args.outage_prob,
        outage_duration_s=args.outage_duration_s,
        jitter_frac=args.jitter_frac,
        seed=args.seed,
    )
    up, down = write_trace_pair(ts, Path(args.out))
    print(f"wrote {len(ts)} timestamps -> {up}, {down}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="leo-trace")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("wired", help="Constant-rate bottleneck trace")
    sp.add_argument("--rate-mbps", type=float, required=True)
    sp.add_argument("--duration-s", type=float, default=60.0)
    sp.add_argument("--out", required=True,
                    help="Output stem (writes <stem>.up and <stem>.down)")
    sp.set_defaults(func=_cmd_wired)

    sp = sub.add_parser("leo", help="LEO-inspired bursty trace")
    sp.add_argument("--baseline-mbps", type=float, default=50.0)
    sp.add_argument("--duration-s", type=float, default=60.0)
    sp.add_argument("--handover-period-s", type=float, default=15.0)
    sp.add_argument("--handover-drop-mbps", type=float, default=5.0)
    sp.add_argument("--handover-duration-s", type=float, default=0.8)
    sp.add_argument("--outage-prob", type=float, default=0.05)
    sp.add_argument("--outage-duration-s", type=float, default=0.3)
    sp.add_argument("--jitter-frac", type=float, default=0.15)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", required=True,
                    help="Output stem (writes <stem>.up and <stem>.down)")
    sp.set_defaults(func=_cmd_leo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
