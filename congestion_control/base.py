"""Base congestion control interface and shared types.

All CCA implementations inherit from CongestionController and implement
on_ack(), on_loss(), and get_cwnd()/get_pacing_rate().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Shared event / state types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AckInfo:
    """Information delivered to CCA when an ACK arrives."""
    timestamp_s: float          # wall-clock time of ACK receipt
    seq_num: int                # sequence number being acknowledged
    rtt_s: float                # measured round-trip time (seconds)
    bytes_acked: int            # newly acknowledged bytes
    send_timestamp_s: float     # when the acked packet was sent
    delivered_bytes: int = 0    # cumulative bytes delivered so far
    ecn_marked: bool = False    # explicit congestion notification


@dataclass(frozen=True)
class LossInfo:
    """Information delivered to CCA on a detected loss event."""
    timestamp_s: float
    seq_num: int
    bytes_lost: int
    is_timeout: bool = False    # True if loss detected via timeout


class CCAState(Enum):
    """Operating state of a congestion controller."""
    STARTUP = auto()
    SLOW_START = auto()
    STEADY = auto()
    RECOVERY = auto()
    DRAIN = auto()


# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------

class CongestionController(ABC):
    """Abstract base for all congestion control algorithms.

    Subclasses must implement:
        on_ack      -- react to an acknowledged packet
        on_loss     -- react to a detected loss
        get_cwnd    -- return current congestion window (bytes)

    Optional overrides:
        on_timeout  -- react to an RTO
        get_pacing_rate -- return sending rate (bytes/sec)
        reset       -- reinitialise state
    """

    def __init__(self, name: str, mtu: int = 1500, initial_cwnd_pkts: int = 10):
        self.name = name
        self.mtu = mtu
        self.initial_cwnd = initial_cwnd_pkts * mtu
        self._cwnd: float = float(self.initial_cwnd)
        self._ssthresh: float = float("inf")
        self._state = CCAState.SLOW_START
        self._min_rtt_s: float = float("inf")
        self._latest_rtt_s: float = 0.0
        self._srtt_s: float = 0.0         # smoothed RTT
        self._rttvar_s: float = 0.0        # RTT variance
        self._bytes_in_flight: int = 0
        self._bytes_delivered: int = 0
        self._bytes_lost: int = 0
        self._start_time_s: float = 0.0
        self._last_event_time_s: float = 0.0
        # Diagnostic hooks (set by Diagnostics)
        self._diag_on_ack = None
        self._diag_on_loss = None
        self._diag_on_cwnd_change = None

    # ---- abstract interface ----

    @abstractmethod
    def on_ack(self, ack: AckInfo) -> None:
        """Process an incoming ACK."""

    @abstractmethod
    def on_loss(self, loss: LossInfo) -> None:
        """Process a loss event."""

    @abstractmethod
    def get_cwnd(self) -> int:
        """Return the current congestion window in bytes."""

    # ---- optional overrides ----

    def on_timeout(self) -> None:
        """Handle an RTO timeout.  Default: halve cwnd, enter recovery."""
        self._ssthresh = max(self._cwnd / 2, 2 * self.mtu)
        self._cwnd = float(self.mtu)
        self._state = CCAState.SLOW_START

    def get_pacing_rate(self) -> Optional[float]:
        """Return pacing rate in bytes/sec, or None for window-only."""
        return None

    def get_state(self) -> CCAState:
        return self._state

    def reset(self) -> None:
        """Reset the controller to initial state."""
        self._cwnd = float(self.initial_cwnd)
        self._ssthresh = float("inf")
        self._state = CCAState.SLOW_START
        self._min_rtt_s = float("inf")
        self._latest_rtt_s = 0.0
        self._srtt_s = 0.0
        self._rttvar_s = 0.0
        self._bytes_in_flight = 0
        self._bytes_delivered = 0
        self._bytes_lost = 0

    # ---- helpers for subclasses ----

    def _update_rtt(self, rtt_s: float) -> None:
        """Update smoothed RTT (RFC 6298)."""
        self._latest_rtt_s = rtt_s
        if rtt_s < self._min_rtt_s:
            self._min_rtt_s = rtt_s
        if self._srtt_s == 0.0:
            self._srtt_s = rtt_s
            self._rttvar_s = rtt_s / 2
        else:
            alpha, beta = 0.125, 0.25
            self._rttvar_s = (1 - beta) * self._rttvar_s + beta * abs(self._srtt_s - rtt_s)
            self._srtt_s = (1 - alpha) * self._srtt_s + alpha * rtt_s

    def _rto_s(self) -> float:
        """Retransmission timeout (RFC 6298)."""
        rto = self._srtt_s + max(0.010, 4 * self._rttvar_s)
        return max(rto, 1.0)  # minimum 1s

    def _notify_cwnd_change(self, old_cwnd: float, new_cwnd: float,
                            timestamp_s: float, reason: str) -> None:
        """Fire diagnostic callback on cwnd change."""
        if self._diag_on_cwnd_change and old_cwnd != new_cwnd:
            self._diag_on_cwnd_change(timestamp_s, old_cwnd, new_cwnd, reason)

    def __repr__(self) -> str:
        return (f"{self.name}(cwnd={self._cwnd:.0f}, "
                f"ssthresh={self._ssthresh:.0f}, "
                f"state={self._state.name}, "
                f"srtt={self._srtt_s*1000:.1f}ms)")
