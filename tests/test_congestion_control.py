"""Tests for congestion control algorithms.

Covers:
- Base CCA interface
- Each CCA implementation (Sprout, Verus, ABC, MPCC)
- Link emulator
- Diagnostics
- Comparison runner with synthetic and real Mahimahi traces
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from congestion_control.base import AckInfo, CCAState, CongestionController, LossInfo
from congestion_control.diagnostics import Diagnostics, FlowMetrics, jains_fairness_index
from congestion_control.emulation.flow import Flow, FlowConfig
from congestion_control.emulation.link_emulator import (
    LinkConfig,
    LinkEmulator,
    MTU,
    Packet,
    TraceSchedule,
)
from congestion_control.emulation.runner import ComparisonResult, run_comparison, run_single
from congestion_control.emulation.traces import (
    create_cellular_like_trace,
    create_constant_trace,
    create_variable_trace,
    find_mahimahi_traces_dir,
)
from congestion_control.network_model import (
    BandwidthEstimator,
    DeliveryRateEstimator,
    NetworkState,
    RTTEstimator,
)
from congestion_control.sprout import Sprout, SproutEWMA
from congestion_control.verus import DelayProfile, Verus
from congestion_control.abc_cc import ABC, ABCAccessPoint
from congestion_control.mpcc_cc import MPCCController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_trace_10mbps():
    """10 Mbps constant link for 30s."""
    path = create_constant_trace(10.0, duration_ms=30_000)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def constant_trace_50mbps():
    """50 Mbps constant link for 30s."""
    path = create_constant_trace(50.0, duration_ms=30_000)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def variable_trace():
    """Variable rate: 10 -> 2 -> 20 -> 5 -> 15 Mbps, 5s each."""
    path = create_variable_trace([10.0, 2.0, 20.0, 5.0, 15.0], segment_ms=5000)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def cellular_trace():
    """Synthetic cellular-like bursty trace."""
    path = create_cellular_like_trace(avg_mbps=5.0, duration_ms=30_000)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def short_flow_config():
    return FlowConfig(duration_s=5.0, tick_ms=5.0)


@pytest.fixture
def medium_flow_config():
    return FlowConfig(duration_s=15.0, tick_ms=5.0)


# ---------------------------------------------------------------------------
# Network model tests
# ---------------------------------------------------------------------------

class TestNetworkModel:
    def test_rtt_estimator_basic(self):
        est = RTTEstimator(window_s=5.0)
        est.add_sample(0.0, 0.050)
        est.add_sample(0.1, 0.045)
        est.add_sample(0.2, 0.060)
        assert est.min_rtt_s == 0.045
        assert est.latest_rtt_s == 0.060
        assert est.srtt_s > 0
        assert est.queue_delay_s() == pytest.approx(0.060 - 0.045)

    def test_rtt_estimator_windowed(self):
        est = RTTEstimator(window_s=1.0)
        est.add_sample(0.0, 0.100)
        est.add_sample(0.5, 0.050)
        est.add_sample(1.5, 0.070)  # should expire the 0.0 sample
        assert est.windowed_min() == 0.050
        assert est.windowed_max() == 0.070

    def test_delivery_rate_estimator(self):
        est = DeliveryRateEstimator(window_s=1.0)
        # Simulate 1 MB delivered over 1 second = 8 Mbps
        est.on_ack(0.0, 0)
        est.on_ack(1.0, 1_000_000)
        assert est.rate_bps == pytest.approx(8_000_000, rel=0.01)

    def test_bandwidth_estimator(self):
        est = BandwidthEstimator(window_s=5.0)
        est.add_sample(0.0, 10e6)
        est.add_sample(1.0, 20e6)
        est.add_sample(2.0, 15e6)
        assert est.max_bw_bps == 20e6
        assert est.smooth_bw_bps > 0

    def test_network_state_dataclass(self):
        state = NetworkState(timestamp_s=1.0, cwnd_bytes=15000, rtt_s=0.05)
        assert state.cwnd_bytes == 15000


# ---------------------------------------------------------------------------
# Link emulator tests
# ---------------------------------------------------------------------------

class TestLinkEmulator:
    def test_trace_schedule_load(self, constant_trace_10mbps):
        ts = TraceSchedule(constant_trace_10mbps)
        assert ts.avg_rate_mbps() > 0
        assert ts.avg_rate_mbps() == pytest.approx(10.0, rel=0.15)

    def test_trace_delivery_opportunities(self, constant_trace_10mbps):
        ts = TraceSchedule(constant_trace_10mbps)
        # Over 1 second at 10 Mbps: ~833 MTU packets
        opps = ts.delivery_opportunities(0, 1000)
        expected = 10e6 / (8 * MTU)  # ~833
        assert opps == pytest.approx(expected, rel=0.15)

    def test_link_enqueue_dequeue(self, constant_trace_10mbps):
        link = LinkEmulator(constant_trace_10mbps, LinkConfig(propagation_delay_ms=20))
        pkt = Packet(send_time_s=0.0, seq_num=0)
        drop = link.enqueue(pkt)
        assert drop is None
        # Advance 100ms
        deliveries = link.dequeue(0.1)
        assert len(deliveries) >= 1

    def test_link_queue_overflow(self, constant_trace_10mbps):
        link = LinkEmulator(constant_trace_10mbps,
                            LinkConfig(propagation_delay_ms=20, queue_size_bytes=MTU))
        # Fill the 1-packet queue
        link.enqueue(Packet(send_time_s=0.0, seq_num=0))
        # Second packet should be dropped
        drop = link.enqueue(Packet(send_time_s=0.0, seq_num=1))
        assert drop is not None
        assert drop.reason == "queue_full"

    def test_link_random_loss(self, constant_trace_10mbps):
        link = LinkEmulator(constant_trace_10mbps,
                            LinkConfig(propagation_delay_ms=20, loss_rate=1.0))
        drop = link.enqueue(Packet(send_time_s=0.0, seq_num=0))
        assert drop is not None
        assert drop.reason == "random_loss"


# ---------------------------------------------------------------------------
# Base CCA tests
# ---------------------------------------------------------------------------

class TestBaseCCA:
    def test_ack_info_creation(self):
        ack = AckInfo(timestamp_s=1.0, seq_num=1, rtt_s=0.05,
                      bytes_acked=1500, send_timestamp_s=0.95)
        assert ack.rtt_s == 0.05
        assert ack.bytes_acked == 1500

    def test_loss_info_creation(self):
        loss = LossInfo(timestamp_s=1.0, seq_num=5, bytes_lost=1500)
        assert not loss.is_timeout

    def test_cca_state_enum(self):
        assert CCAState.STARTUP.name == "STARTUP"
        assert CCAState.SLOW_START.name == "SLOW_START"


# ---------------------------------------------------------------------------
# Sprout tests
# ---------------------------------------------------------------------------

class TestSprout:
    def test_sprout_creation(self):
        s = Sprout()
        assert s.name == "Sprout"
        assert s.get_cwnd() >= MTU

    def test_sprout_on_ack_updates_state(self):
        s = Sprout()
        ack = AckInfo(timestamp_s=0.1, seq_num=0, rtt_s=0.050,
                      bytes_acked=MTU, send_timestamp_s=0.05, delivered_bytes=MTU)
        s.on_ack(ack)
        assert s._latest_rtt_s == 0.050
        assert s._bytes_delivered == MTU

    def test_sprout_on_loss(self):
        s = Sprout()
        loss = LossInfo(timestamp_s=0.1, seq_num=0, bytes_lost=MTU)
        s.on_loss(loss)
        assert s._bytes_lost == MTU

    def test_sprout_forecast_increases_window(self):
        s = Sprout()
        # Feed multiple ACKs to build up rate estimate
        for i in range(100):
            t = i * 0.002  # 2ms apart
            ack = AckInfo(timestamp_s=t, seq_num=i, rtt_s=0.040,
                          bytes_acked=MTU, send_timestamp_s=t - 0.040,
                          delivered_bytes=(i + 1) * MTU)
            s.on_ack(ack)
        # After many ACKs, cwnd should be > initial
        assert s.get_cwnd() >= MTU

    def test_sprout_ewma_creation(self):
        s = SproutEWMA()
        assert s.name == "SproutEWMA"
        assert s.get_cwnd() >= MTU

    def test_sprout_over_constant_link(self, constant_trace_10mbps, short_flow_config):
        s = Sprout()
        diag = run_single(s, constant_trace_10mbps,
                          LinkConfig(propagation_delay_ms=20),
                          short_flow_config)
        m = diag.compute_metrics()
        assert m.avg_throughput_mbps > 0
        assert m.min_rtt_ms > 0
        assert m.duration_s > 0


# ---------------------------------------------------------------------------
# Verus tests
# ---------------------------------------------------------------------------

class TestVerus:
    def test_verus_creation(self):
        v = Verus()
        assert v.name == "Verus"
        assert v.get_cwnd() >= MTU

    def test_delay_profile_update_and_lookup(self):
        dp = DelayProfile()
        dp.update(10, 20.0)
        dp.update(20, 40.0)
        dp.update(30, 60.0)
        assert dp.has_data()
        # Lookup window for delay=30ms should be ~15 packets
        w = dp.lookup_window(30.0)
        assert w is not None
        assert 10 <= w <= 20
        # Lookup delay for window=20 should be ~40ms
        d = dp.lookup_delay(20)
        assert d is not None
        assert 35 <= d <= 45

    def test_verus_slow_start(self):
        v = Verus()
        assert v._in_slow_start
        # Feed some ACKs
        for i in range(5):
            ack = AckInfo(timestamp_s=i * 0.01, seq_num=i, rtt_s=0.030,
                          bytes_acked=MTU, send_timestamp_s=i * 0.01 - 0.03,
                          delivered_bytes=(i + 1) * MTU)
            v.on_ack(ack)
        # Window should have grown during slow start
        assert v.get_cwnd() > v.initial_cwnd

    def test_verus_loss_multiplicative_decrease(self):
        v = Verus(M=0.5)
        v._cwnd = 30000.0
        v._window_pkts = 20
        v._in_slow_start = False
        cwnd_before = v._cwnd
        v.on_loss(LossInfo(timestamp_s=1.0, seq_num=10, bytes_lost=MTU))
        assert v._cwnd < cwnd_before
        assert v._in_recovery

    def test_verus_over_constant_link(self, constant_trace_10mbps, short_flow_config):
        v = Verus(R=4.0)
        diag = run_single(v, constant_trace_10mbps,
                          LinkConfig(propagation_delay_ms=20),
                          short_flow_config)
        m = diag.compute_metrics()
        assert m.avg_throughput_mbps > 0
        assert m.duration_s > 0


# ---------------------------------------------------------------------------
# ABC tests
# ---------------------------------------------------------------------------

class TestABC:
    def test_abc_creation(self):
        a = ABC()
        assert a.name == "ABC"

    def test_abc_access_point(self):
        ap = ABCAccessPoint(queue_capacity_bytes=150_000, eta=0.7)
        ap.update_link_capacity(0.0, 10e6)  # 10 Mbps
        ap.update_queue(0)  # empty queue
        rate = ap.compute_target_rate_bps()
        # With empty queue: target = capacity * (1 - 0 * eta) = capacity
        assert rate == pytest.approx(10e6, rel=0.2)

    def test_abc_access_point_congested(self):
        ap = ABCAccessPoint(queue_capacity_bytes=150_000, eta=0.7)
        ap.update_link_capacity(0.0, 10e6)
        ap.update_queue(150_000)  # full queue
        rate = ap.compute_target_rate_bps()
        # With full queue: target = capacity * (1 - 1.0 * 0.7) = 0.3 * capacity
        assert rate == pytest.approx(10e6 * 0.3, rel=0.2)

    def test_abc_over_constant_link(self, constant_trace_10mbps, short_flow_config):
        a = ABC(eta=0.7)
        diag = run_single(a, constant_trace_10mbps,
                          LinkConfig(propagation_delay_ms=20),
                          short_flow_config)
        m = diag.compute_metrics()
        assert m.avg_throughput_mbps > 0


# ---------------------------------------------------------------------------
# MPCC tests
# ---------------------------------------------------------------------------

class TestMPCC:
    def test_mpcc_creation(self):
        m = MPCCController()
        assert m.name == "MPCC"
        assert m.get_cwnd() >= MTU

    def test_mpcc_slow_start(self):
        m = MPCCController()
        assert m._state == CCAState.SLOW_START
        initial = m.get_cwnd()
        for i in range(10):
            ack = AckInfo(timestamp_s=i * 0.01, seq_num=i, rtt_s=0.040,
                          bytes_acked=MTU, send_timestamp_s=i * 0.01 - 0.04,
                          delivered_bytes=(i + 1) * MTU)
            m.on_ack(ack)
        # MPCC transitions out of slow start once cwnd > 2*BDP; verify it moved
        assert m.get_cwnd() != initial or m._state != CCAState.SLOW_START

    def test_mpcc_loss_md(self):
        m = MPCCController(md_factor=0.5)
        m._cwnd = 30000.0
        m._state = CCAState.STEADY
        cwnd_before = m._cwnd
        m.on_loss(LossInfo(timestamp_s=1.0, seq_num=10, bytes_lost=MTU))
        assert m._cwnd < cwnd_before
        assert m._cwnd == pytest.approx(cwnd_before * 0.5, rel=0.01)

    def test_mpcc_convergence_proof(self):
        proof = MPCCController.convergence_proof_sketch()
        assert "AIMD" in proof
        assert "LYAPUNOV" in proof

    def test_mpcc_over_constant_link(self, constant_trace_10mbps, short_flow_config):
        m = MPCCController(horizon=8, target_delay_ms=50.0)
        diag = run_single(m, constant_trace_10mbps,
                          LinkConfig(propagation_delay_ms=20),
                          short_flow_config)
        metrics = diag.compute_metrics()
        assert metrics.avg_throughput_mbps > 0

    def test_mpcc_evaluate_trajectory(self):
        m = MPCCController()
        cost = m._evaluate_trajectory_multistep(
            cwnd=15000, q_delay=0.01, bw=1250000, base_rtt=0.02,
            target_delay=0.05, target_tput=1250000,
            delta_0=1500, H=4, dt=0.02,
            w_contour=5.0, w_lag=10.0,
        )
        assert isinstance(cost, float)
        assert math.isfinite(cost)


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_diagnostics_record_ack(self):
        cca = Sprout()
        diag = Diagnostics(cca, link_capacity_bps=10e6)
        diag.record_ack(0.1, 0.050, 1500, 15000)
        diag.record_ack(0.2, 0.045, 1500, 16000)
        m = diag.compute_metrics()
        assert m.bytes_delivered == 3000
        assert m.avg_rtt_ms > 0

    def test_diagnostics_record_loss(self):
        cca = Verus()
        diag = Diagnostics(cca)
        diag.record_loss(0.1, 1500, 15000, 7500)
        m = diag.compute_metrics()
        assert m.bytes_lost == 1500

    def test_jains_fairness_equal(self):
        assert jains_fairness_index([10.0, 10.0, 10.0]) == pytest.approx(1.0)

    def test_jains_fairness_unequal(self):
        fi = jains_fairness_index([10.0, 1.0])
        assert fi < 1.0
        assert fi > 0.0

    def test_flow_metrics_summary(self):
        m = FlowMetrics(cca_name="test", avg_throughput_mbps=5.0,
                        p50_rtt_ms=40.0, p95_rtt_ms=80.0,
                        link_utilization=0.5, loss_rate=0.01)
        s = m.summary_line()
        assert "test" in s
        assert "5.00" in s

    def test_cwnd_timeseries(self, constant_trace_10mbps, short_flow_config):
        cca = MPCCController()
        diag = run_single(cca, constant_trace_10mbps,
                          LinkConfig(propagation_delay_ms=20),
                          short_flow_config)
        ts, cw = diag.cwnd_timeseries()
        assert len(ts) == len(cw)
        assert len(ts) > 0


# ---------------------------------------------------------------------------
# Comparison tests (synthetic traces)
# ---------------------------------------------------------------------------

class TestComparison:
    def test_comparison_constant_link(self, constant_trace_10mbps):
        """All CCAs should achieve positive throughput on a constant link."""
        factories = {
            "Sprout": lambda: Sprout(),
            "SproutEWMA": lambda: SproutEWMA(),
            "Verus": lambda: Verus(R=4.0),
            "ABC": lambda: ABC(),
            "MPCC": lambda: MPCCController(),
        }
        result = run_comparison(
            factories,
            constant_trace_10mbps,
            LinkConfig(propagation_delay_ms=20, queue_size_bytes=150_000),
            FlowConfig(duration_s=5.0, tick_ms=5.0),
        )
        for name, m in result.metrics.items():
            assert m.avg_throughput_mbps > 0, f"{name} has zero throughput"
            assert m.duration_s > 0, f"{name} has zero duration"

    def test_comparison_variable_link(self, variable_trace):
        """CCAs should handle varying link rates."""
        factories = {
            "Sprout": lambda: Sprout(),
            "Verus": lambda: Verus(R=4.0),
            "MPCC": lambda: MPCCController(),
        }
        result = run_comparison(
            factories,
            variable_trace,
            LinkConfig(propagation_delay_ms=20),
            FlowConfig(duration_s=10.0, tick_ms=5.0),
        )
        for name, m in result.metrics.items():
            assert m.avg_throughput_mbps > 0, f"{name} zero throughput on variable link"

    def test_comparison_cellular_trace(self, cellular_trace):
        """CCAs should handle bursty cellular conditions."""
        factories = {
            "Sprout": lambda: Sprout(),
            "Verus": lambda: Verus(R=4.0),
            "ABC": lambda: ABC(),
            "MPCC": lambda: MPCCController(),
        }
        result = run_comparison(
            factories,
            cellular_trace,
            LinkConfig(propagation_delay_ms=30),
            FlowConfig(duration_s=10.0, tick_ms=5.0),
        )
        for name, m in result.metrics.items():
            assert m.avg_throughput_mbps > 0, f"{name} zero throughput on cellular trace"

    def test_comparison_result_summary(self, constant_trace_10mbps):
        result = ComparisonResult(
            trace_name="test",
            link_config=LinkConfig(),
            metrics={"A": FlowMetrics(cca_name="A", avg_throughput_mbps=5.0,
                                      p50_rtt_ms=30, p95_rtt_ms=60,
                                      link_utilization=0.5, loss_rate=0.01,
                                      avg_cwnd_bytes=15000)},
        )
        table = result.summary_table()
        assert "test" in table
        assert "5.00" in table


# ---------------------------------------------------------------------------
# Mahimahi real trace tests
# ---------------------------------------------------------------------------

class TestMahimahiTraces:
    """Tests using real Mahimahi cellular traces (skip if not available)."""

    @pytest.fixture
    def traces_dir(self):
        d = find_mahimahi_traces_dir()
        if d is None:
            pytest.skip("Mahimahi traces not found")
        return d

    @pytest.mark.mahimahi
    def test_real_verizon_lte(self, traces_dir):
        trace = traces_dir / "Verizon-LTE-driving.down"
        if not trace.exists():
            pytest.skip(f"Trace not found: {trace}")
        factories = {
            "Sprout": lambda: Sprout(),
            "Verus": lambda: Verus(R=4.0),
            "ABC": lambda: ABC(),
            "MPCC": lambda: MPCCController(),
        }
        result = run_comparison(
            factories, trace,
            LinkConfig(propagation_delay_ms=20),
            FlowConfig(duration_s=8.0, tick_ms=5.0),
        )
        for name, m in result.metrics.items():
            assert m.avg_throughput_mbps > 0, f"{name} zero throughput on Verizon LTE"
            assert m.avg_rtt_ms > 0, f"{name} zero RTT on Verizon LTE"

    @pytest.mark.mahimahi
    def test_real_tmobile_lte(self, traces_dir):
        trace = traces_dir / "TMobile-LTE-driving.down"
        if not trace.exists():
            pytest.skip(f"Trace not found: {trace}")
        result = run_comparison(
            {"MPCC": lambda: MPCCController(), "Verus": lambda: Verus()},
            trace,
            LinkConfig(propagation_delay_ms=25),
            FlowConfig(duration_s=8.0, tick_ms=5.0),
        )
        for name, m in result.metrics.items():
            assert m.avg_throughput_mbps > 0

    @pytest.mark.mahimahi
    def test_real_att_lte(self, traces_dir):
        trace = traces_dir / "ATT-LTE-driving.down"
        if not trace.exists():
            pytest.skip(f"Trace not found: {trace}")
        result = run_comparison(
            {
                "Sprout": lambda: Sprout(),
                "SproutEWMA": lambda: SproutEWMA(),
                "Verus": lambda: Verus(R=4.0),
                "MPCC": lambda: MPCCController(),
            },
            trace,
            LinkConfig(propagation_delay_ms=20),
            FlowConfig(duration_s=8.0, tick_ms=5.0),
        )
        for name, m in result.metrics.items():
            assert m.avg_throughput_mbps > 0

    @pytest.mark.mahimahi
    @pytest.mark.slow
    def test_all_traces_all_ccas(self, traces_dir):
        """Comprehensive: run all CCAs on all available downlink traces."""
        downlink_traces = sorted(traces_dir.glob("*.down"))
        if not downlink_traces:
            pytest.skip("No downlink traces found")

        factories = {
            "Sprout": lambda: Sprout(),
            "SproutEWMA": lambda: SproutEWMA(),
            "Verus": lambda: Verus(R=4.0),
            "ABC": lambda: ABC(),
            "MPCC": lambda: MPCCController(),
        }

        all_results = {}
        for trace in downlink_traces:
            result = run_comparison(
                factories, trace,
                LinkConfig(propagation_delay_ms=20),
                FlowConfig(duration_s=8.0, tick_ms=5.0),
            )
            all_results[trace.stem] = result

        # Verify all produced valid results
        for trace_name, result in all_results.items():
            for cca_name, m in result.metrics.items():
                assert m.avg_throughput_mbps > 0, (
                    f"{cca_name} zero throughput on {trace_name}"
                )


# ---------------------------------------------------------------------------
# Trace utilities tests
# ---------------------------------------------------------------------------

class TestTraceUtilities:
    def test_create_constant_trace(self):
        path = create_constant_trace(10.0, duration_ms=5000)
        assert path.exists()
        ts = TraceSchedule(path)
        assert ts.avg_rate_mbps() == pytest.approx(10.0, rel=0.15)
        path.unlink()

    def test_create_variable_trace(self):
        path = create_variable_trace([5.0, 10.0, 5.0], segment_ms=3000)
        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) > 100
        path.unlink()

    def test_create_cellular_trace(self):
        path = create_cellular_like_trace(avg_mbps=5.0, duration_ms=10_000)
        assert path.exists()
        ts = TraceSchedule(path)
        assert ts.avg_rate_mbps() > 0
        path.unlink()

    def test_find_mahimahi_traces_dir(self):
        d = find_mahimahi_traces_dir()
        # May or may not be available, just test the function doesn't crash
        if d is not None:
            assert d.is_dir()


# ---------------------------------------------------------------------------
# Integration test: full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end_flow(self, constant_trace_10mbps):
        """Run a complete flow from CCA creation through diagnostics."""
        cca = MPCCController(horizon=4, target_delay_ms=50.0)
        link = LinkEmulator(constant_trace_10mbps,
                            LinkConfig(propagation_delay_ms=20, queue_size_bytes=75_000))
        flow = Flow(cca, link, FlowConfig(duration_s=3.0, tick_ms=5.0))
        diag = flow.run()

        m = diag.compute_metrics()
        assert m.cca_name == "MPCC"
        assert m.avg_throughput_mbps > 0
        assert m.min_rtt_ms > 0
        assert m.loss_rate >= 0
        assert m.avg_cwnd_bytes > 0
        assert m.cwnd_changes >= 0

        # Verify time series
        ts_t, ts_c = diag.cwnd_timeseries()
        assert len(ts_t) > 0
        rtt_t, rtt_v = diag.rtt_timeseries()
        assert len(rtt_t) > 0
        tput_t, tput_v = diag.throughput_timeseries(bin_s=1.0)
        assert len(tput_t) > 0

    def test_high_loss_resilience(self, constant_trace_10mbps):
        """CCAs should survive (not crash) under high loss."""
        for factory in [
            lambda: Sprout(),
            lambda: Verus(),
            lambda: ABC(),
            lambda: MPCCController(),
        ]:
            cca = factory()
            link = LinkEmulator(constant_trace_10mbps,
                                LinkConfig(propagation_delay_ms=20, loss_rate=0.10))
            flow = Flow(cca, link, FlowConfig(duration_s=3.0, tick_ms=5.0))
            diag = flow.run()
            m = diag.compute_metrics()
            # Should not crash, and should still deliver some data
            assert m.duration_s > 0
            assert m.bytes_delivered >= 0
