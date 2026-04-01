"""
UDP rate-controlled sender for MPCC congestion control experiments.

Sends fixed-size UDP packets at a rate dictated by the MPCC controller.
Uses a token-bucket pacer. Receives ACKs to feed the bandwidth estimator.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

from planning.network_estimator import AckInfo

logger = logging.getLogger(__name__)

HEADER_FMT = "!QdQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
DEFAULT_PAYLOAD_SIZE = 1400
ACK_FMT = "!Qdd"
ACK_SIZE = struct.calcsize(ACK_FMT)


@dataclass
class SendStats:
    packets_sent: int = 0
    bytes_sent: int = 0
    acks_received: int = 0
    start_time: float = 0.0
    rtts: list[float] = field(default_factory=list)


class UDPSender:

    def __init__(
        self,
        receiver_addr: tuple[str, int] = ("127.0.0.1", 9000),
        payload_size: int = DEFAULT_PAYLOAD_SIZE,
        ack_callback=None,
        log_path: str | None = None,
    ):
        self.receiver_addr = receiver_addr
        self.payload_size = payload_size
        self.ack_callback = ack_callback
        self.log_path = log_path

        self._rate = 0.0
        self._seq = 0
        self._running = False
        self._stats = SendStats()

        self._sock: socket.socket | None = None
        self._send_thread: threading.Thread | None = None
        self._ack_thread: threading.Thread | None = None
        self._log_file = None

    def set_rate(self, rate_bytes_per_sec: float):
        self._rate = max(0.0, rate_bytes_per_sec)

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)
        self._sock.bind(("", 0))
        self._running = True
        self._stats = SendStats(start_time=time.monotonic())

        if self.log_path:
            self._log_file = open(self.log_path, "w")
            self._log_file.write("seq,send_time,size\n")

        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._ack_thread = threading.Thread(target=self._ack_loop, daemon=True)
        self._send_thread.start()
        self._ack_thread.start()
        logger.info(
            "UDPSender started → %s:%d (bound on port %d)",
            *self.receiver_addr,
            self._sock.getsockname()[1],
        )

    def stop(self):
        self._running = False
        if self._send_thread:
            self._send_thread.join(timeout=2.0)
        if self._ack_thread:
            self._ack_thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()
        if self._log_file:
            self._log_file.close()
        logger.info(
            "UDPSender stopped: %d packets sent, %d ACKs received",
            self._stats.packets_sent,
            self._stats.acks_received,
        )

    @property
    def stats(self) -> SendStats:
        return self._stats

    def _send_loop(self):
        payload = b"\x00" * self.payload_size
        tokens = 0.0
        last_time = time.monotonic()
        pkt_size = HEADER_SIZE + self.payload_size

        while self._running:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            tokens += self._rate * dt
            tokens = min(tokens, pkt_size * 10)

            while tokens >= pkt_size and self._running:
                self._send_one_packet(payload, now)
                tokens -= pkt_size

            if self._rate > 0:
                sleep_s = max(0.001, pkt_size / self._rate - 0.0005)
            else:
                sleep_s = 0.050
            time.sleep(sleep_s)

    def _send_one_packet(self, payload: bytes, now: float):
        header = struct.pack(HEADER_FMT, self._seq, now, self.payload_size)
        try:
            self._sock.sendto(header + payload, self.receiver_addr)
        except OSError as e:
            logger.debug("Send error: %s", e)
            return

        if self._log_file:
            self._log_file.write(f"{self._seq},{now:.6f},{HEADER_SIZE + self.payload_size}\n")

        self._stats.packets_sent += 1
        self._stats.bytes_sent += HEADER_SIZE + self.payload_size
        self._seq += 1

    def _ack_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(ACK_SIZE + 64)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < ACK_SIZE:
                continue

            seq_num, send_ts, recv_ts = struct.unpack(ACK_FMT, data[:ACK_SIZE])
            now = time.monotonic()

            ack = AckInfo(
                timestamp=now,
                send_timestamp=send_ts,
                seq_num=seq_num,
                bytes_acked=self.payload_size,
            )

            self._stats.acks_received += 1
            self._stats.rtts.append(now - send_ts)

            if self.ack_callback:
                self.ack_callback(ack)
