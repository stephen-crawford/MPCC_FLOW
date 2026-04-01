"""Trace-driven link emulator for congestion control testing.

Reads Mahimahi-format trace files (one delivery opportunity per line,
timestamp in ms) and simulates a bottleneck link with a finite queue.
Produces ACK and loss events that drive CCA implementations.

This is the Python-level equivalent of Mahimahi's mm-link, enabling
fast iteration without requiring system-level network namespaces.
"""

from __future__ import annotations

import bisect
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterator, List, Optional, Tuple

logger = logging.getLogger("cc.emulation")

# Mahimahi trace format: each line is a timestamp (ms) at which one
# MTU-sized packet (1500 bytes) can be delivered.  The trace repeats
# cyclically.

MTU = 1500  # bytes


@dataclass
class LinkConfig:
    """Configuration for the emulated link."""
    propagation_delay_ms: float = 20.0   # one-way base delay
    queue_size_bytes: int = 150_000      # bottleneck queue (100 pkts)
    queue_size_packets: Optional[int] = None  # alternative: packet count
    loss_rate: float = 0.0               # random loss probability [0,1]
    jitter_ms: float = 0.0              # random jitter (uniform ± jitter_ms)


@dataclass(order=True)
class Packet:
    """A packet in the emulation."""
    send_time_s: float
    seq_num: int = field(compare=False)
    size_bytes: int = field(compare=False, default=MTU)
    enqueue_time_s: float = field(compare=False, default=0.0)


@dataclass(frozen=True)
class DeliveryEvent:
    """A packet delivery (ACK) event."""
    ack_time_s: float       # time ACK arrives at sender
    send_time_s: float      # when packet was originally sent
    seq_num: int
    size_bytes: int
    rtt_s: float


@dataclass(frozen=True)
class DropEvent:
    """A packet drop event."""
    drop_time_s: float
    seq_num: int
    size_bytes: int
    reason: str             # "queue_full" or "random_loss"


class TraceSchedule:
    """Parses and replays a Mahimahi trace file cyclically."""

    def __init__(self, trace_path: Path):
        self.trace_path = trace_path
        self._timestamps_ms: List[int] = []
        self._cycle_duration_ms: int = 0
        self._load()

    def _load(self) -> None:
        with open(self.trace_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self._timestamps_ms.append(int(line))
        if not self._timestamps_ms:
            raise ValueError(f"Empty trace file: {self.trace_path}")
        self._timestamps_ms.sort()
        self._cycle_duration_ms = self._timestamps_ms[-1]
        logger.debug("Loaded trace %s: %d entries, cycle=%dms, avg_rate=%.1f Mbps",
                      self.trace_path.name, len(self._timestamps_ms),
                      self._cycle_duration_ms,
                      len(self._timestamps_ms) * MTU * 8 / (self._cycle_duration_ms / 1000) / 1e6)

    def delivery_opportunities(self, start_ms: int, end_ms: int) -> int:
        """Count how many MTU-sized packets can be delivered in [start_ms, end_ms)."""
        if self._cycle_duration_ms <= 0:
            return 0
        count = 0
        # Handle full cycles
        full_cycles = (end_ms - start_ms) // self._cycle_duration_ms
        count += full_cycles * len(self._timestamps_ms)
        # Remaining partial cycle
        rem_start = start_ms % self._cycle_duration_ms
        rem_end = rem_start + ((end_ms - start_ms) % self._cycle_duration_ms)
        if rem_end <= self._cycle_duration_ms:
            lo = bisect.bisect_left(self._timestamps_ms, rem_start)
            hi = bisect.bisect_right(self._timestamps_ms, rem_end)
            count += hi - lo
        else:
            # Wraps around
            lo = bisect.bisect_left(self._timestamps_ms, rem_start)
            count += len(self._timestamps_ms) - lo
            hi = bisect.bisect_right(self._timestamps_ms, rem_end - self._cycle_duration_ms)
            count += hi
        return count

    def avg_rate_mbps(self) -> float:
        if self._cycle_duration_ms <= 0:
            return 0.0
        return len(self._timestamps_ms) * MTU * 8 / (self._cycle_duration_ms / 1000) / 1e6

    def avg_rate_bps(self) -> float:
        return self.avg_rate_mbps() * 1e6


class LinkEmulator:
    """Simulates a bottleneck link with trace-driven capacity and a finite queue.

    Usage:
        link = LinkEmulator(trace_path, config)
        for t in simulation_ticks:
            drops = link.enqueue(packets)
            deliveries = link.dequeue(current_time_s)
    """

    def __init__(self, trace_path: Path, config: LinkConfig | None = None):
        self.config = config or LinkConfig()
        self.trace = TraceSchedule(trace_path)

        # Queue state
        max_q = self.config.queue_size_bytes
        if self.config.queue_size_packets is not None:
            max_q = self.config.queue_size_packets * MTU
        self._max_queue_bytes = max_q
        self._queue: Deque[Packet] = deque()
        self._queue_bytes: int = 0

        # Delivery tracking
        self._last_dequeue_time_ms: int = 0
        self._total_delivered: int = 0
        self._total_dropped: int = 0
        self._base_delay_s = self.config.propagation_delay_ms / 1000.0

        # Random state for loss
        self._loss_rng_state: int = 42

    def enqueue(self, packet: Packet) -> Optional[DropEvent]:
        """Try to enqueue a packet. Returns DropEvent if dropped."""
        # Random loss
        if self.config.loss_rate > 0:
            self._loss_rng_state = (self._loss_rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            if (self._loss_rng_state / 0x7FFFFFFF) < self.config.loss_rate:
                self._total_dropped += 1
                return DropEvent(packet.send_time_s, packet.seq_num, packet.size_bytes, "random_loss")

        # Queue overflow
        if self._queue_bytes + packet.size_bytes > self._max_queue_bytes:
            self._total_dropped += 1
            return DropEvent(packet.send_time_s, packet.seq_num, packet.size_bytes, "queue_full")

        packet.enqueue_time_s = packet.send_time_s
        self._queue.append(packet)
        self._queue_bytes += packet.size_bytes
        return None

    def dequeue(self, current_time_s: float) -> List[DeliveryEvent]:
        """Deliver packets that the trace schedule allows up to current_time_s."""
        if not self._queue:
            return []

        current_time_ms = int(current_time_s * 1000)
        opportunities = self.trace.delivery_opportunities(self._last_dequeue_time_ms, current_time_ms)
        self._last_dequeue_time_ms = current_time_ms

        deliveries = []
        for _ in range(opportunities):
            if not self._queue:
                break
            pkt = self._queue.popleft()
            self._queue_bytes -= pkt.size_bytes

            # Queue delay = time spent in queue
            queue_delay_s = current_time_s - pkt.enqueue_time_s
            # Total RTT = propagation (both ways) + queue delay
            rtt_s = 2 * self._base_delay_s + queue_delay_s
            ack_time_s = pkt.send_time_s + rtt_s

            self._total_delivered += 1
            deliveries.append(DeliveryEvent(
                ack_time_s=ack_time_s,
                send_time_s=pkt.send_time_s,
                seq_num=pkt.seq_num,
                size_bytes=pkt.size_bytes,
                rtt_s=rtt_s,
            ))

        return deliveries

    def queue_occupancy_bytes(self) -> int:
        return self._queue_bytes

    def queue_occupancy_packets(self) -> int:
        return len(self._queue)

    def queue_delay_estimate_s(self) -> float:
        """Estimated delay for a packet entering the queue now."""
        if not self._queue:
            return 0.0
        rate_bps = self.trace.avg_rate_bps()
        if rate_bps <= 0:
            return float("inf")
        return (self._queue_bytes * 8) / rate_bps

    @property
    def avg_capacity_mbps(self) -> float:
        return self.trace.avg_rate_mbps()

    @property
    def stats(self) -> dict:
        return {
            "delivered": self._total_delivered,
            "dropped": self._total_dropped,
            "queue_bytes": self._queue_bytes,
            "queue_pkts": len(self._queue),
        }
