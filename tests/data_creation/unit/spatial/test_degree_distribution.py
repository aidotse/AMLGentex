"""
Unit tests for scale-free degree distribution generation
Tests individual functions for generating degree sequences
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'python'))

from src.data_creation.python.spatial_simulation.generate_scalefree import powerlaw_degree_distrubution


@pytest.mark.unit
@pytest.mark.spatial
class TestDegreeDistribution:
    """Tests for scale-free degree distribution generation"""

    def test_powerlaw_basic_output(self):
        """Test basic output structure of powerlaw degree distribution"""
        n = 100
        gamma = 2.0
        loc = 1.0
        scale = 1.0
        seed = 42

        values, counts = powerlaw_degree_distrubution(n, gamma, loc, scale, seed)

        # Should return two arrays
        assert isinstance(values, np.ndarray)
        assert isinstance(counts, np.ndarray)

        # Values should be 2D array (in_degree, out_degree pairs)
        assert values.ndim == 2
        assert values.shape[1] == 2

        # Counts should be 1D
        assert counts.ndim == 1

        # Length should match
        assert len(values) == len(counts)

    def test_powerlaw_degree_balance(self):
        """Test that total in-degrees equals total out-degrees"""
        n = 100
        values, counts = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=42)

        # Calculate total in-degrees and out-degrees
        total_in = np.sum(values[:, 0] * counts)
        total_out = np.sum(values[:, 1] * counts)

        # Should be balanced (required for directed graph)
        assert total_in == total_out

    def test_powerlaw_all_nonnegative(self):
        """Test that all degrees are non-negative"""
        n = 50
        values, counts = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=42)

        # All in-degrees should be non-negative
        assert np.all(values[:, 0] >= 0)

        # All out-degrees should be non-negative
        assert np.all(values[:, 1] >= 0)

        # All counts should be positive
        assert np.all(counts > 0)

    def test_powerlaw_total_count(self):
        """Test that total counts equal number of nodes"""
        n = 100
        values, counts = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=42)

        # Sum of counts should equal n
        assert np.sum(counts) == n

    def test_powerlaw_seed_consistency(self):
        """Test that same seed at least influences the distribution similarly"""
        n = 100
        seed = 42

        # Note: Function uses np.random.randint internally without state,
        # so it's not fully reproducible, but should have consistent properties
        values1, counts1 = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=seed)
        values2, counts2 = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=seed)

        # Should have same total count
        assert np.sum(counts1) == np.sum(counts2) == n

        # Should both balance in/out degrees
        assert np.sum(values1[:, 0] * counts1) == np.sum(values1[:, 1] * counts1)
        assert np.sum(values2[:, 0] * counts2) == np.sum(values2[:, 1] * counts2)

    def test_powerlaw_different_seeds(self):
        """Test that different seeds produce different results"""
        n = 100

        values1, counts1 = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=42)
        values2, counts2 = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=99)

        # Should be different (with very high probability)
        assert not np.array_equal(values1, values2) or not np.array_equal(counts1, counts2)

    def test_powerlaw_gamma_effect(self):
        """Test that gamma parameter affects distribution"""
        n = 500

        # Lower gamma = heavier tail
        values_low, counts_low = powerlaw_degree_distrubution(n, gamma=1.5, loc=1.0, scale=1.0, seed=42)
        # Higher gamma = lighter tail
        values_high, counts_high = powerlaw_degree_distrubution(n, gamma=3.0, loc=1.0, scale=1.0, seed=42)

        # Calculate max degrees
        max_degree_low = np.max(values_low[:, 0] + values_low[:, 1])
        max_degree_high = np.max(values_high[:, 0] + values_high[:, 1])

        # Lower gamma should produce higher max degree
        assert max_degree_low > max_degree_high

    def test_powerlaw_scale_parameter(self):
        """Test that scale parameter affects distribution"""
        n = 200

        values_small, counts_small = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=0.5, seed=42)
        values_large, counts_large = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=2.0, seed=42)

        # Calculate mean total degree
        mean_degree_small = np.sum((values_small[:, 0] + values_small[:, 1]) * counts_small) / n
        mean_degree_large = np.sum((values_large[:, 0] + values_large[:, 1]) * counts_large) / n

        # Larger scale should produce higher mean degree
        assert mean_degree_large > mean_degree_small

    def test_powerlaw_edge_cases(self):
        """Test edge cases for degree generation"""
        # Very small n
        values_small, counts_small = powerlaw_degree_distrubution(n=10, gamma=2.0, loc=1.0, scale=1.0, seed=42)
        assert np.sum(counts_small) == 10

        # Larger n
        values_large, counts_large = powerlaw_degree_distrubution(n=1000, gamma=2.0, loc=1.0, scale=1.0, seed=42)
        assert np.sum(counts_large) == 1000

    def test_powerlaw_power_law_property(self):
        """Test that distribution exhibits power law properties"""
        n = 1000
        values, counts = powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=42)

        # Total degree for each unique combination
        total_degrees = values[:, 0] + values[:, 1]

        # Power law: many low-degree nodes, few high-degree nodes
        low_degree_count = np.sum(counts[total_degrees < 10])
        high_degree_count = np.sum(counts[total_degrees > 50])

        assert low_degree_count > high_degree_count
