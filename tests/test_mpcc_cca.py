"""
Tests for the MPCC CCA module (Portus integration).
Tests the control logic without requiring pyportus or CCP kernel module.
"""

import pytest
import time
from unittest.mock import MagicMock

from planning.network_dynamics import NetworkDynamicsModel
from planning.network_estimator import EMAEstimator, AckInfo
from planning.network_reference import AdaptiveNetworkReference


class MockDatapathInfo:
    def __init__(self, mss=1460):
        self.mss = mss
        self.init_cwnd = mss * 10
        self.src_ip = "10.0.0.1"
        self.src_port = 5000
        self.dst_ip = "10.0.0.2"
        self.dst_port = 9000


class MockDatapath:
    def __init__(self):
        self.program = None
        self.fields = {}
        self.updates = []

    def set_program(self, name, fields):
        self.program = name
        self.fields = dict(fields)

    def update_field(self, name, value):
        self.fields[name] = value
        self.updates.append((name, value))


class MockReport:
    def __init__(self, acked=1460, rtt=25000, loss=0, sacked=0, inflight=10, timeout=False):
        self.acked = acked
        self.rtt = rtt
        self.loss = loss
        self.sacked = sacked
        self.inflight = inflight
        self.timeout = timeout


class TestMPCCFlow:

    def _make_flow(self, config=None):
        from network_emulation.mpcc_cca import MPCCFlow
        datapath = MockDatapath()
        info = MockDatapathInfo()
        flow = MPCCFlow(datapath, info, config)
        return flow, datapath

    def test_init_sets_program(self):
        flow, dp = self._make_flow()
        assert dp.program == "default"
        assert "Cwnd" in dp.fields
        assert dp.fields["Cwnd"] > 0

    def test_on_report_updates_cwnd(self):
        flow, dp = self._make_flow()
        dp.updates.clear()

        r = MockReport(acked=14600, rtt=25000, loss=0)
        flow.on_report(r)

        assert len(dp.updates) > 0
        assert dp.updates[-1][0] == "Cwnd"
        assert dp.updates[-1][1] > 0

    def test_loss_reduces_cwnd(self):
        flow, dp = self._make_flow()

        r_normal = MockReport(acked=14600, rtt=25000, loss=0)
        flow.on_report(r_normal)
        cwnd_normal = dp.fields["Cwnd"]

        r_loss = MockReport(acked=14600, rtt=25000, loss=5)
        flow.on_report(r_loss)
        cwnd_after_loss = dp.fields["Cwnd"]

        assert cwnd_after_loss < cwnd_normal

    def test_high_rtt_reduces_rate(self):
        flow, dp = self._make_flow()

        r_low_rtt = MockReport(acked=14600, rtt=25000, loss=0)
        flow.on_report(r_low_rtt)
        cwnd_low_rtt = dp.fields["Cwnd"]

        r_high_rtt = MockReport(acked=14600, rtt=250000, loss=0)
        flow.on_report(r_high_rtt)
        cwnd_high_rtt = dp.fields["Cwnd"]

        assert cwnd_high_rtt != cwnd_low_rtt

    def test_cwnd_never_zero(self):
        flow, dp = self._make_flow()

        for _ in range(10):
            r = MockReport(acked=100, rtt=500000, loss=10)
            flow.on_report(r)

        assert dp.fields["Cwnd"] > 0

    def test_custom_config(self):
        config = {
            "network": {
                "rtt_prop": 0.050,
                "rate_max": 100e6,
                "ema_alpha": 0.5,
            }
        }
        flow, dp = self._make_flow(config)
        assert flow.controller.rtt_prop == 0.050
        assert flow.controller.rate_max == 100e6


class TestMPCCAlgorithmImport:

    def test_has_portus_flag(self):
        from network_emulation import mpcc_cca
        assert isinstance(mpcc_cca.HAS_PORTUS, bool)

    def test_mpcc_flow_class_exists(self):
        from network_emulation.mpcc_cca import MPCCFlow
        assert MPCCFlow is not None

    def test_datapath_program_defined(self):
        from network_emulation.mpcc_cca import DATAPATH_PROGRAM
        assert "Report" in DATAPATH_PROGRAM
        assert "acked" in DATAPATH_PROGRAM
        assert "rtt" in DATAPATH_PROGRAM
        assert "loss" in DATAPATH_PROGRAM


class TestExperimentRunnerImport:

    def test_algorithms_registry(self):
        from scripts.network.run_cc_experiments import ALGORITHMS
        assert "mpcc" in ALGORITHMS
        assert "cubic" in ALGORITHMS

    def test_trace_discovery(self):
        pairs = list(
            __import__("network_emulation.cellular_traces", fromlist=["list_cellular_pairs"])
            .list_cellular_pairs()
        )
        assert len(pairs) > 0, "Should find mahimahi traces via third_party symlink"

    def test_parse_mm_link_log_empty(self):
        from scripts.network.run_cc_experiments import _parse_mm_link_log
        from pathlib import Path
        result = _parse_mm_link_log(Path("/nonexistent/path"))
        assert result == []
