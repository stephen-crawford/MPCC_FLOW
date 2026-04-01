"""
Unit tests for the network reference trajectory and adaptive reference.
"""

import numpy as np
import pytest

from planning.network_reference import (
    AdaptiveNetworkReference,
    generate_network_reference,
)


class TestGenerateNetworkReference:
    """Tests for generate_network_reference."""

    def test_basic_generation(self):
        ref = generate_network_reference(bw_est=10e6)
        assert not ref.empty()
        assert len(ref.x) == 51  # default num_points
        assert len(ref.y) == 51
        assert len(ref.s) == 51
        assert len(ref.psi) == 51

    def test_x_is_throughput(self):
        """x-coordinates should be throughput values from 0 to bw_est."""
        bw = 10e6
        ref = generate_network_reference(bw_est=bw)
        assert ref.x[0] == pytest.approx(0.0)
        assert ref.x[-1] == pytest.approx(bw)

    def test_y_is_rtt(self):
        """y-coordinates should be RTT values starting at rtt_prop."""
        rtt_prop = 0.025
        alpha = 0.150
        ref = generate_network_reference(bw_est=10e6, rtt_prop=rtt_prop, alpha=alpha)
        assert ref.y[0] == pytest.approx(rtt_prop)
        assert ref.y[-1] == pytest.approx(rtt_prop + alpha)

    def test_arc_length_monotonic(self):
        """Arc length should be monotonically increasing."""
        ref = generate_network_reference(bw_est=10e6)
        s = np.array(ref.s)
        assert np.all(np.diff(s) > 0), "Arc length should be strictly increasing"

    def test_arc_length_starts_at_zero(self):
        ref = generate_network_reference(bw_est=10e6)
        assert ref.s[0] == 0.0

    def test_has_splines(self):
        """Splines should be created for solver evaluation."""
        ref = generate_network_reference(bw_est=10e6)
        assert ref.x_spline is not None
        assert ref.y_spline is not None
        assert ref.v_spline is not None

    def test_different_bandwidths(self):
        """Higher bandwidth should produce longer reference in x."""
        ref_low = generate_network_reference(bw_est=1e6)
        ref_high = generate_network_reference(bw_est=100e6)
        assert ref_high.x[-1] > ref_low.x[-1]

    def test_different_alpha(self):
        """Higher alpha should produce more RTT growth."""
        ref_low = generate_network_reference(bw_est=10e6, alpha=0.050)
        ref_high = generate_network_reference(bw_est=10e6, alpha=0.500)
        assert ref_high.y[-1] > ref_low.y[-1]

    def test_custom_num_points(self):
        ref = generate_network_reference(bw_est=10e6, num_points=101)
        assert len(ref.x) == 101

    def test_reference_path_interface(self):
        """Should satisfy the ReferencePath interface used by ContouringObjective."""
        ref = generate_network_reference(bw_est=10e6)
        assert ref.has_distance()
        assert ref.has_velocity()
        arc_len = ref.get_arc_length()
        assert arc_len > 0


class TestAdaptiveNetworkReference:
    """Tests for AdaptiveNetworkReference."""

    def test_initial_generation(self):
        adaptive = AdaptiveNetworkReference(rtt_prop=0.025)
        ref = adaptive.get_reference(bw_est=10e6)
        assert ref is not None
        assert not ref.empty()

    def test_caching(self):
        """Same bw_est should return the same reference object."""
        adaptive = AdaptiveNetworkReference()
        ref1 = adaptive.get_reference(bw_est=10e6)
        ref2 = adaptive.get_reference(bw_est=10e6)
        assert ref1 is ref2

    def test_regeneration_on_large_change(self):
        """Should regenerate when bw_est changes by more than threshold."""
        adaptive = AdaptiveNetworkReference(update_threshold=0.10)
        ref1 = adaptive.get_reference(bw_est=10e6)
        ref2 = adaptive.get_reference(bw_est=12e6)  # 20% change > 10% threshold
        assert ref2 is not ref1
        # New reference should have higher max throughput
        assert ref2.x[-1] > ref1.x[-1]

    def test_no_regeneration_on_small_change(self):
        """Should NOT regenerate when bw_est changes by less than threshold."""
        adaptive = AdaptiveNetworkReference(update_threshold=0.10)
        ref1 = adaptive.get_reference(bw_est=10e6)
        ref2 = adaptive.get_reference(bw_est=10.5e6)  # 5% change < 10% threshold
        assert ref2 is ref1

    def test_reference_property(self):
        adaptive = AdaptiveNetworkReference()
        assert adaptive.reference is None
        adaptive.get_reference(bw_est=10e6)
        assert adaptive.reference is not None
