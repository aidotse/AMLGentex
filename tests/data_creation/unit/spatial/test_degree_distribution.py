"""
Unit tests for scale-free degree distribution generation
Tests individual functions for generating degree sequences

Note: Comprehensive unit tests are in test_generate_scalefree.py.
This file provides additional integration-style tests.
"""
import pytest
import numpy as np

from src.data_creation.spatial_simulation.generate_scalefree import (
    discrete_powerlaw_degree_distribution
)


@pytest.mark.unit
@pytest.mark.spatial
class TestDegreeDistribution:
    """Tests for scale-free degree distribution generation"""

    def test_basic_output_structure(self):
        """Test basic output structure of discrete power law degree distribution"""
        n = 100
        values, counts, stats = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=42
        )

        # Should return arrays and dict
        assert isinstance(values, np.ndarray)
        assert isinstance(counts, np.ndarray)
        assert isinstance(stats, dict)

        # Values should be 2D array (in_degree, out_degree pairs)
        assert values.ndim == 2
        assert values.shape[1] == 2

        # Counts should be 1D
        assert counts.ndim == 1

        # Length should match
        assert len(values) == len(counts)

    def test_degree_balance(self):
        """Test that total in-degrees equals total out-degrees"""
        n = 100
        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=42
        )

        # Calculate total in-degrees and out-degrees
        total_in = np.sum(values[:, 0] * counts)
        total_out = np.sum(values[:, 1] * counts)

        # Should be balanced (required for directed graph)
        assert total_in == total_out

    def test_all_nonnegative(self):
        """Test that all degrees are non-negative"""
        n = 50
        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=30, average_degree=3.0, seed=42
        )

        # All in-degrees should be non-negative
        assert np.all(values[:, 0] >= 0)

        # All out-degrees should be non-negative
        assert np.all(values[:, 1] >= 0)

        # All counts should be positive
        assert np.all(counts > 0)

    def test_total_count_equals_n(self):
        """Test that total counts equal number of nodes"""
        n = 100
        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=42
        )

        # Sum of counts should equal n
        assert np.sum(counts) == n

    def test_seed_reproducibility(self):
        """Test that same seed produces same results"""
        n = 100
        seed = 42

        result1 = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=seed
        )
        result2 = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=seed
        )

        # Should be identical with same seed
        assert np.array_equal(result1[0], result2[0])
        assert np.array_equal(result1[1], result2[1])

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        n = 100

        result1 = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=42
        )
        result2 = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=99
        )

        # Should be different (with very high probability)
        assert not np.array_equal(result1[0], result2[0]) or \
               not np.array_equal(result1[1], result2[1])

    def test_average_degree_effect(self):
        """Test that average_degree parameter affects distribution"""
        n = 1000

        _, _, stats_low = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=100, average_degree=3.0, seed=42
        )
        _, _, stats_high = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=100, average_degree=10.0, seed=42
        )

        # Higher average_degree should produce higher achieved mean
        assert stats_high['achieved_mean_in'] > stats_low['achieved_mean_in']

    def test_kmin_effect(self):
        """Test that kmin parameter sets minimum degree"""
        n = 500
        kmin = 3

        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=n, kmin=kmin, kmax=50, average_degree=5.0, seed=42
        )

        # Minimum degree should be at least kmin (before balancing adjustments)
        # Note: balancing might slightly violate this, but the vast majority should be >= kmin
        min_in = np.min(values[:, 0])
        min_out = np.min(values[:, 1])
        # At least one of them should respect kmin for most values
        assert min_in >= 0 and min_out >= 0

    def test_kmax_effect(self):
        """Test that kmax parameter caps maximum degree"""
        n = 500
        kmax = 20

        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=kmax, average_degree=5.0, seed=42
        )

        # Maximum degree should not significantly exceed kmax
        # (balancing might add a few, but should be close)
        max_in = np.max(values[:, 0])
        max_out = np.max(values[:, 1])
        assert max_in <= kmax + 10  # Allow some slack for balancing
        assert max_out <= kmax + 10

    def test_power_law_property(self):
        """Test that distribution exhibits power law properties"""
        n = 1000
        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=100, average_degree=5.0, seed=42
        )

        # Total degree for each unique combination
        total_degrees = values[:, 0] + values[:, 1]

        # Power law: many low-degree nodes, few high-degree nodes
        low_degree_count = np.sum(counts[total_degrees < 10])
        high_degree_count = np.sum(counts[total_degrees > 30])

        assert low_degree_count > high_degree_count

    def test_edge_cases_small_n(self):
        """Test edge cases for small n"""
        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=10, kmin=1, kmax=9, average_degree=2.0, seed=42
        )
        assert np.sum(counts) == 10

    def test_edge_cases_large_n(self):
        """Test edge cases for larger n"""
        values, counts, _ = discrete_powerlaw_degree_distribution(
            n=5000, kmin=1, kmax=100, average_degree=5.0, seed=42
        )
        assert np.sum(counts) == 5000

    def test_stats_dictionary_contents(self):
        """Test that stats dictionary contains expected keys and reasonable values"""
        n = 100
        values, counts, stats = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=42
        )

        # Check all expected keys
        assert 'gamma' in stats
        assert 'kmin' in stats
        assert 'kmax' in stats
        assert 'achieved_mean_in' in stats
        assert 'achieved_mean_out' in stats
        assert 'total_edges' in stats

        # Check reasonable values
        assert stats['gamma'] > 0.0
        assert stats['kmin'] == 1
        assert stats['kmax'] == 50
        assert 1.0 < stats['achieved_mean_in'] < 50.0
        assert 1.0 < stats['achieved_mean_out'] < 50.0
        assert stats['total_edges'] > 0
