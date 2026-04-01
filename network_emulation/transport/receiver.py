"""
UDP receiver for MPCC congestion control experiments.

Receives packets and echoes ACKs with receive timestamps.
Designed to run inside a mahimahi shell (mm-link).

Usage:
    python -m network_emulation.transport.receiver --port 9000
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HEADER_FMT = "!QdQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
ACK_FMT = "!Qdd"
ACK_SIZE = struct.calcsize(ACK_FMT)


@dataclass
class RecvStats:
    packets_received: int = 0
    bytes_received: int = 0
    acks_sent: int = 0
    start_time: float = 0.0
    out_of_order: int = 0


class UDPReceiver:

    def __init__(
        self,
        bind_addr: tuple[str, int] = ("0.0.0.0", 9000),
        log_path: str | None = None,
    ):
        self.bind_addr = bind_addr
        self.log_path = log_path
        self._running = False
        self._stats = RecvStats()
        self._sock: socket.socket | None = None
        self._log_file = None
        self._last_seq = -1

    def run(self, duration_s: float = 0.0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1.0)
        self._sock.bind(self.bind_addr)
        self._running = True
        self._stats = RecvStats(start_time=time.monotonic())

        if self.log_path:
            self._log_file = open(self.log_path, "w")
            self._log_file.write("seq,recv_time,size,owd\n")

        logger.info("UDPReceiver listening on %s:%d", *self.bind_addr)
        deadline = time.monotonic() + duration_s if duration_s > 0 else float("inf")

        try:
            while self._running and time.monotonic() < deadline:
                try:
                    data, sender_addr = self._sock.recvfrom(65536)
                except socket.timeout:
                    continue

                if len(data) < HEADER_SIZE:
                    continue

                recv_time = time.monotonic()
                seq_num, send_ts, payload_size = struct.unpack(
                    HEADER_FMT, data[:HEADER_SIZE]
                )

                self._stats.packets_received += 1
                self._stats.bytes_received += len(data)

                if seq_num <= self._last_seq:
                    self._stats.out_of_order += 1
                self._last_seq = max(self._last_seq, seq_num)

                if self._log_file:
                    owd = recv_time - send_ts
                    self._log_file.write(
                        f"{seq_num},{recv_time:.6f},{len(data)},{owd:.6f}\n"
                    )

                ack = struct.pack(ACK_FMT, seq_num, send_ts, recv_time)
                try:
                    self._sock.sendto(ack, sender_addr)
                    self._stats.acks_sent += 1
                except OSError as e:
                    logger.debug("ACK send error: %s", e)
        finally:
            if self._log_file:
                self._log_file.close()
            self._sock.close()
            logger.info(
                "UDPReceiver stopped: %d packets, %d ACKs sent",
                self._stats.packets_received,
                self._stats.acks_sent,
            )

    def stop(self):
        self._running = False

    @property
    def stats(self) -> RecvStats:
        return self._stats


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="mpcc-receiver",
        description="UDP receiver for MPCC congestion control experiments",
    )
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    receiver = UDPReceiver(bind_addr=(args.host, args.port), log_path=args.log)

    try:
        receiver.run(duration_s=args.duration)
    except KeyboardInterrupt:
        receiver.stop()


if __name__ == "__main__":
    main()
