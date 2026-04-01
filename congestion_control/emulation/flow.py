"""Flow abstraction: connects a CCA to a link emulator and drives the simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from congestion_control.base import AckInfo, CongestionController, LossInfo
from congestion_control.diagnostics import Diagnostics
from congestion_control.emulation.link_emulator import (
    DeliveryEvent,
    DropEvent,
    LinkEmulator,
    MTU,
    Packet,
)

logger = logging.getLogger("cc.flow")


@dataclass
class FlowConfig:
    """Configuration for a simulated flow."""
    duration_s: float = 60.0
    tick_ms: float = 1.0        # simulation granularity
    start_delay_s: float = 0.0  # offset before flow begins sending


class Flow:
    """Drives a CCA over an emulated link for a specified duration.

    The simulation loop:
        1. Send packets up to cwnd (respecting bytes_in_flight), capped per tick
        2. Advance time by tick_ms
        3. Dequeue delivered packets from the link
        4. Deliver ACK events to the CCA
        5. Deliver loss events to the CCA
        6. Repeat
    """

    # Max packets to enqueue per tick to prevent infinite loops
    MAX_SENDS_PER_TICK = 200

    def __init__(self, cca: CongestionController, link: LinkEmulator,
                 config: FlowConfig | None = None):
        self.cca = cca
        self.link = link
        self.config = config or FlowConfig()
        self.diag = Diagnostics(cca, link_capacity_bps=link.trace.avg_rate_bps())

        self._seq_num: int = 0
        self._bytes_in_flight: int = 0
        self._cumulative_delivered: int = 0
        self._current_time_s: float = 0.0
        self._pending_acks: List[DeliveryEvent] = []
        self._done: bool = False

    def run(self) -> Diagnostics:
        """Run the flow for the configured duration. Returns diagnostics."""
        tick_s = self.config.tick_ms / 1000.0
        end_time = self.config.duration_s + self.config.start_delay_s
        self._current_time_s = self.config.start_delay_s
        cca = self.cca
        link = self.link

        logger.info("Starting flow %s: duration=%.1fs, link_avg=%.1f Mbps",
                     cca.name, self.config.duration_s, link.avg_capacity_mbps)

        cca._start_time_s = self._current_time_s

        while self._current_time_s < end_time:
            # 1. Send packets up to cwnd, capped per tick
            cwnd = cca.get_cwnd()
            sends_this_tick = 0
            while (self._bytes_in_flight + MTU <= cwnd
                   and sends_this_tick < self.MAX_SENDS_PER_TICK):
                pkt = Packet(
                    send_time_s=self._current_time_s,
                    seq_num=self._seq_num,
                    size_bytes=MTU,
                )
                drop = link.enqueue(pkt)
                if drop is not None:
                    # Queue drop -> loss event, then stop sending this tick
                    cwnd_before = cca._cwnd
                    loss = LossInfo(
                        timestamp_s=self._current_time_s,
                        seq_num=drop.seq_num,
                        bytes_lost=drop.size_bytes,
                        is_timeout=False,
                    )
                    cca.on_loss(loss)
                    self.diag.record_loss(self._current_time_s, drop.size_bytes,
                                          cwnd_before, cca._cwnd)
                    self._seq_num += 1
                    break  # stop sending on drop
                else:
                    self._bytes_in_flight += MTU
                    self.diag.record_send(self._current_time_s, MTU)
                self._seq_num += 1
                sends_this_tick += 1
                # Re-check cwnd after potential loss-driven change
                cwnd = cca.get_cwnd()

            # 2. Advance time
            self._current_time_s += tick_s

            # 3. Dequeue delivered packets
            deliveries = link.dequeue(self._current_time_s)

            # 4. Process deliveries whose ACK time has passed
            for delivery in deliveries:
                if delivery.ack_time_s <= self._current_time_s:
                    self._process_delivery(delivery)
                else:
                    self._pending_acks.append(delivery)

            # 5. Process pending ACKs
            still_pending = []
            for pending in self._pending_acks:
                if pending.ack_time_s <= self._current_time_s:
                    self._process_delivery(pending)
                else:
                    still_pending.append(pending)
            self._pending_acks = still_pending

        self.diag.log_summary()
        return self.diag

    def _process_delivery(self, delivery: DeliveryEvent) -> None:
        self._bytes_in_flight = max(0, self._bytes_in_flight - delivery.size_bytes)
        self._cumulative_delivered += delivery.size_bytes

        ack = AckInfo(
            timestamp_s=delivery.ack_time_s,
            seq_num=delivery.seq_num,
            rtt_s=delivery.rtt_s,
            bytes_acked=delivery.size_bytes,
            send_timestamp_s=delivery.send_time_s,
            delivered_bytes=self._cumulative_delivered,
        )
        self.cca.on_ack(ack)
        self.diag.record_ack(
            delivery.ack_time_s, delivery.rtt_s,
            delivery.size_bytes, self.cca._cwnd,
        )
