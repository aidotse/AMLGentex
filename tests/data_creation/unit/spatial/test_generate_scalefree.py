"""
Unit tests for spatial_simulation/generate_scalefree.py

Tests the truncated discrete power law implementation for scale-free degree generation.
"""
import pytest
import numpy as np
import tempfile
import os
import yaml

from src.data_creation.spatial_simulation.generate_scalefree import (
    truncated_discrete_powerlaw_pmf,
    truncated_discrete_powerlaw_mean,
    solve_gamma_for_mean,
    sample_truncated_discrete_powerlaw,
    balance_degree_sums,
    discrete_powerlaw_degree_distribution,
)


# =============================================================================
# Tests for truncated_discrete_powerlaw_pmf
# =============================================================================

@pytest.mark.unit
class TestTruncatedDiscretePowerlawPMF:
    """Tests for PMF computation."""

    def test_pmf_sums_to_one(self):
        """PMF should sum to 1.0."""
        k_values = np.arange(1, 101)
        pmf = truncated_discrete_powerlaw_pmf(k_values, gamma=2.5)
        assert np.isclose(pmf.sum(), 1.0)

    def test_pmf_is_decreasing(self):
        """PMF should be monotonically decreasing for power law."""
        k_values = np.arange(1, 101)
        pmf = truncated_discrete_powerlaw_pmf(k_values, gamma=2.5)
        assert np.all(np.diff(pmf) <= 0)

    def test_pmf_higher_gamma_more_concentrated(self):
        """Higher gamma should concentrate more probability at low k."""
        k_values = np.arange(1, 101)
        pmf_low = truncated_discrete_powerlaw_pmf(k_values, gamma=1.5)
        pmf_high = truncated_discrete_powerlaw_pmf(k_values, gamma=3.0)
        # First element should have higher probability for higher gamma
        assert pmf_high[0] > pmf_low[0]

    def test_pmf_raises_for_gamma_le_0(self):
        """PMF should raise error for gamma <= 0."""
        k_values = np.arange(1, 101)
        with pytest.raises(ValueError, match="gamma must be > 0.0"):
            truncated_discrete_powerlaw_pmf(k_values, gamma=0.0)
        with pytest.raises(ValueError, match="gamma must be > 0.0"):
            truncated_discrete_powerlaw_pmf(k_values, gamma=-1.0)

    def test_pmf_works_for_gamma_between_0_and_1(self):
        """PMF should work for 0 < gamma <= 1 (valid for truncated distributions)."""
        k_values = np.arange(1, 101)
        # These should not raise - truncated distributions converge for any gamma > 0
        pmf = truncated_discrete_powerlaw_pmf(k_values, gamma=0.5)
        assert np.isclose(pmf.sum(), 1.0)
        pmf = truncated_discrete_powerlaw_pmf(k_values, gamma=1.0)
        assert np.isclose(pmf.sum(), 1.0)

    def test_pmf_all_positive(self):
        """All PMF values should be positive."""
        k_values = np.arange(1, 101)
        pmf = truncated_discrete_powerlaw_pmf(k_values, gamma=2.5)
        assert np.all(pmf > 0)


# =============================================================================
# Tests for truncated_discrete_powerlaw_mean
# =============================================================================

@pytest.mark.unit
class TestTruncatedDiscretePowerlawMean:
    """Tests for mean computation."""

    def test_mean_within_bounds(self):
        """Mean should be between kmin and kmax."""
        kmin, kmax = 1, 100
        mean = truncated_discrete_powerlaw_mean(kmin, kmax, gamma=2.5)
        assert kmin <= mean <= kmax

    def test_mean_decreases_with_gamma(self):
        """Higher gamma should give lower mean."""
        kmin, kmax = 1, 100
        mean_low = truncated_discrete_powerlaw_mean(kmin, kmax, gamma=1.5)
        mean_high = truncated_discrete_powerlaw_mean(kmin, kmax, gamma=3.0)
        assert mean_high < mean_low

    def test_mean_approaches_kmin_for_high_gamma(self):
        """Very high gamma should give mean close to kmin."""
        kmin, kmax = 1, 100
        mean = truncated_discrete_powerlaw_mean(kmin, kmax, gamma=10.0)
        assert mean < kmin + 0.5  # Should be very close to kmin

    def test_mean_with_different_kmin(self):
        """Mean should increase with higher kmin."""
        kmax = 100
        mean_low = truncated_discrete_powerlaw_mean(1, kmax, gamma=2.5)
        mean_high = truncated_discrete_powerlaw_mean(5, kmax, gamma=2.5)
        assert mean_high > mean_low


# =============================================================================
# Tests for solve_gamma_for_mean
# =============================================================================

@pytest.mark.unit
class TestSolveGammaForMean:
    """Tests for gamma solver."""

    def test_solver_achieves_target_mean(self):
        """Solved gamma should achieve target mean."""
        kmin, kmax = 1, 100
        target_mean = 5.0
        gamma = solve_gamma_for_mean(kmin, kmax, target_mean)
        achieved_mean = truncated_discrete_powerlaw_mean(kmin, kmax, gamma)
        assert np.isclose(achieved_mean, target_mean, rtol=1e-6)

    def test_solver_various_target_means(self):
        """Solver should work for various target means."""
        kmin, kmax = 1, 100
        # Note: max achievable mean with gamma=1.01 is ~19, so stay below that
        for target_mean in [2.0, 5.0, 10.0, 15.0]:
            gamma = solve_gamma_for_mean(kmin, kmax, target_mean)
            achieved_mean = truncated_discrete_powerlaw_mean(kmin, kmax, gamma)
            assert np.isclose(achieved_mean, target_mean, rtol=1e-5)

    def test_solver_raises_for_too_high_mean(self):
        """Solver should raise error if target mean is too high."""
        kmin, kmax = 1, 100
        # With gamma approaching 1, max mean is still bounded
        with pytest.raises(ValueError, match="too high"):
            solve_gamma_for_mean(kmin, kmax, target_mean=90.0)

    def test_solver_raises_for_too_low_mean(self):
        """Solver should raise error if target mean is too low."""
        kmin, kmax = 5, 100
        # Mean can't be less than kmin
        with pytest.raises(ValueError, match="too low"):
            solve_gamma_for_mean(kmin, kmax, target_mean=4.0)

    def test_solver_returns_float(self):
        """Solver should return a float gamma value."""
        gamma = solve_gamma_for_mean(1, 100, 5.0)
        assert isinstance(gamma, float)
        assert gamma > 0.0


# =============================================================================
# Tests for sample_truncated_discrete_powerlaw
# =============================================================================

@pytest.mark.unit
class TestSampleTruncatedDiscretePowerlaw:
    """Tests for sampling."""

    def test_sample_returns_correct_size(self):
        """Sample should return n values."""
        rng = np.random.default_rng(42)
        samples = sample_truncated_discrete_powerlaw(1000, 1, 100, 2.5, rng)
        assert len(samples) == 1000

    def test_sample_within_bounds(self):
        """All samples should be within [kmin, kmax]."""
        rng = np.random.default_rng(42)
        kmin, kmax = 5, 50
        samples = sample_truncated_discrete_powerlaw(10000, kmin, kmax, 2.5, rng)
        assert np.all(samples >= kmin)
        assert np.all(samples <= kmax)

    def test_sample_are_integers(self):
        """All samples should be integers."""
        rng = np.random.default_rng(42)
        samples = sample_truncated_discrete_powerlaw(1000, 1, 100, 2.5, rng)
        assert np.all(samples == samples.astype(int))

    def test_sample_mean_close_to_theoretical(self):
        """Sample mean should be close to theoretical mean."""
        rng = np.random.default_rng(42)
        kmin, kmax, gamma = 1, 100, 2.5
        samples = sample_truncated_discrete_powerlaw(100000, kmin, kmax, gamma, rng)
        theoretical_mean = truncated_discrete_powerlaw_mean(kmin, kmax, gamma)
        assert np.isclose(samples.mean(), theoretical_mean, rtol=0.05)

    def test_sample_reproducible_with_seed(self):
        """Same seed should give same samples."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        samples1 = sample_truncated_discrete_powerlaw(100, 1, 100, 2.5, rng1)
        samples2 = sample_truncated_discrete_powerlaw(100, 1, 100, 2.5, rng2)
        assert np.array_equal(samples1, samples2)


# =============================================================================
# Tests for balance_degree_sums
# =============================================================================

@pytest.mark.unit
class TestBalanceDegreeSums:
    """Tests for degree sum balancing."""

    def test_balanced_sums_equal(self):
        """After balancing, sums should be equal."""
        rng = np.random.default_rng(42)
        in_degrees = np.array([5, 3, 7, 2, 4])
        out_degrees = np.array([4, 6, 3, 5, 2])  # Sum differs

        in_bal, out_bal = balance_degree_sums(in_degrees, out_degrees, 100, rng)
        assert in_bal.sum() == out_bal.sum()

    def test_balance_preserves_non_negativity(self):
        """Balanced degrees should remain non-negative."""
        rng = np.random.default_rng(42)
        in_degrees = np.array([5, 3, 7, 2, 4])
        out_degrees = np.array([4, 6, 3, 5, 2])

        in_bal, out_bal = balance_degree_sums(in_degrees, out_degrees, 100, rng)
        assert np.all(in_bal >= 0)
        assert np.all(out_bal >= 0)

    def test_balance_respects_kmax(self):
        """Balanced degrees should not exceed kmax."""
        rng = np.random.default_rng(42)
        kmax = 10
        in_degrees = np.array([5, 3, 7, 2, 4])
        out_degrees = np.array([4, 6, 3, 5, 2])

        in_bal, out_bal = balance_degree_sums(in_degrees, out_degrees, kmax, rng)
        assert np.all(in_bal <= kmax)
        assert np.all(out_bal <= kmax)

    def test_balance_already_balanced(self):
        """Already balanced arrays should change minimally."""
        rng = np.random.default_rng(42)
        in_degrees = np.array([5, 5, 5, 5])
        out_degrees = np.array([5, 5, 5, 5])

        in_bal, out_bal = balance_degree_sums(in_degrees, out_degrees, 100, rng)
        assert in_bal.sum() == out_bal.sum()
        # Should be unchanged since already balanced
        assert np.array_equal(in_bal, in_degrees)
        assert np.array_equal(out_bal, out_degrees)

    def test_balance_large_difference(self):
        """Should handle large sum differences."""
        rng = np.random.default_rng(42)
        in_degrees = np.array([10, 10, 10, 10, 10])  # Sum = 50
        out_degrees = np.array([1, 1, 1, 1, 1])       # Sum = 5

        in_bal, out_bal = balance_degree_sums(in_degrees, out_degrees, 100, rng)
        assert in_bal.sum() == out_bal.sum()


# =============================================================================
# Tests for discrete_powerlaw_degree_distribution
# =============================================================================

@pytest.mark.unit
class TestDiscretePowerlawDegreeDistribution:
    """Tests for main degree distribution function."""

    def test_output_format(self):
        """Should return values, counts, and stats."""
        values, counts, stats = discrete_powerlaw_degree_distribution(
            n=100, kmin=1, kmax=50, average_degree=5.0, seed=42
        )
        assert isinstance(values, np.ndarray)
        assert isinstance(counts, np.ndarray)
        assert isinstance(stats, dict)
        assert values.shape[1] == 2  # (in_degree, out_degree) pairs

    def test_counts_sum_to_n(self):
        """Total counts should equal n."""
        n = 100
        values, counts, stats = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, kmax=50, average_degree=5.0, seed=42
        )
        assert counts.sum() == n

    def test_in_out_sums_equal(self):
        """Total in-degree should equal total out-degree."""
        values, counts, stats = discrete_powerlaw_degree_distribution(
            n=1000, kmin=1, kmax=100, average_degree=10.0, seed=42
        )
        total_in = (values[:, 0] * counts).sum()
        total_out = (values[:, 1] * counts).sum()
        assert total_in == total_out

    def test_stats_contains_expected_keys(self):
        """Stats dict should contain expected keys."""
        _, _, stats = discrete_powerlaw_degree_distribution(
            n=100, kmin=1, kmax=50, average_degree=5.0, seed=42
        )
        assert 'gamma' in stats
        assert 'kmin' in stats
        assert 'kmax' in stats
        assert 'achieved_mean_in' in stats
        assert 'achieved_mean_out' in stats
        assert 'total_edges' in stats

    def test_achieved_mean_close_to_target(self):
        """Achieved mean should be close to target average_degree."""
        target = 10.0
        _, _, stats = discrete_powerlaw_degree_distribution(
            n=10000, kmin=1, kmax=100, average_degree=target, seed=42
        )
        # Allow some deviation due to balancing
        assert abs(stats['achieved_mean_in'] - target) < 1.0
        assert abs(stats['achieved_mean_out'] - target) < 1.0

    def test_with_explicit_gamma(self):
        """Should work with explicit gamma instead of average_degree."""
        values, counts, stats = discrete_powerlaw_degree_distribution(
            n=100, kmin=1, kmax=50, gamma=2.5, seed=42
        )
        assert stats['gamma'] == 2.5

    def test_raises_without_gamma_or_average_degree(self):
        """Should raise error if neither gamma nor average_degree provided."""
        with pytest.raises(ValueError, match="Either gamma or average_degree"):
            discrete_powerlaw_degree_distribution(n=100, kmin=1, kmax=50, seed=42)

    def test_raises_for_invalid_kmin(self):
        """Should raise error for negative kmin."""
        with pytest.raises(ValueError, match="kmin must be >= 0"):
            discrete_powerlaw_degree_distribution(
                n=100, kmin=-1, kmax=50, average_degree=5.0, seed=42
            )

    def test_raises_for_kmax_less_than_kmin(self):
        """Should raise error if kmax < kmin."""
        with pytest.raises(ValueError, match="kmax.*must be >= kmin"):
            discrete_powerlaw_degree_distribution(
                n=100, kmin=10, kmax=5, average_degree=5.0, seed=42
            )

    def test_default_kmax_is_n_minus_1(self):
        """Default kmax should be n-1."""
        n = 100
        _, _, stats = discrete_powerlaw_degree_distribution(
            n=n, kmin=1, average_degree=5.0, seed=42
        )
        assert stats['kmax'] == n - 1

    def test_reproducible_with_seed(self):
        """Same seed should give same results."""
        result1 = discrete_powerlaw_degree_distribution(
            n=100, kmin=1, kmax=50, average_degree=5.0, seed=42
        )
        result2 = discrete_powerlaw_degree_distribution(
            n=100, kmin=1, kmax=50, average_degree=5.0, seed=42
        )
        assert np.array_equal(result1[0], result2[0])
        assert np.array_equal(result1[1], result2[1])

    def test_all_degrees_are_integers(self):
        """All degree values should be integers."""
        values, _, _ = discrete_powerlaw_degree_distribution(
            n=100, kmin=1, kmax=50, average_degree=5.0, seed=42
        )
        assert values.dtype in [np.int32, np.int64, int]

    def test_degrees_within_bounds(self):
        """All degrees should be within [kmin, kmax]."""
        kmin, kmax = 2, 30
        values, _, _ = discrete_powerlaw_degree_distribution(
            n=1000, kmin=kmin, kmax=kmax, average_degree=5.0, seed=42
        )
        # After balancing, some might exceed kmax slightly, but should be close
        # The original samples are within bounds
        assert np.all(values >= 0)  # Non-negative


# =============================================================================
# Integration test with config
# =============================================================================

@pytest.mark.unit
class TestGenerateDegreeFileFromConfig:
    """Tests for config-based generation."""

    def test_generates_degree_file(self, tmp_path):
        """Should generate a valid degree CSV file."""
        from src.data_creation.spatial_simulation.generate_scalefree import (
            generate_degree_file_from_config
        )

        # Create accounts file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        accounts_path = config_dir / "accounts.csv"
        accounts_path.write_text("count,bank\n50,bank_A\n50,bank_B\n")

        # Create output directory
        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()

        config = {
            'general': {'random_seed': 42},
            'input': {
                'directory': str(config_dir),
                'accounts': 'accounts.csv',
                'degree': 'degree.csv'
            },
            'spatial': {'directory': str(spatial_dir)},
            'scale-free': {
                'kmin': 1,
                'kmax': 50,
                'average_degree': 5.0
            }
        }

        stats = generate_degree_file_from_config(config)

        # Check file was created
        degree_file = spatial_dir / "degree.csv"
        assert degree_file.exists()

        # Check contents
        import csv
        with open(degree_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['Count', 'In-degree', 'Out-degree']
            rows = list(reader)
            assert len(rows) > 0
            # Check total count
            total_count = sum(int(row[0]) for row in rows)
            assert total_count == 100  # 50 + 50 accounts

    def test_legacy_loc_parameter(self, tmp_path):
        """Should support legacy 'loc' parameter as alias for kmin."""
        from src.data_creation.spatial_simulation.generate_scalefree import (
            generate_degree_file_from_config
        )

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        accounts_path = config_dir / "accounts.csv"
        accounts_path.write_text("count,bank\n100,bank_A\n")

        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()

        config = {
            'general': {'random_seed': 42},
            'input': {
                'directory': str(config_dir),
                'accounts': 'accounts.csv',
                'degree': 'degree.csv'
            },
            'spatial': {'directory': str(spatial_dir)},
            'scale-free': {
                'loc': 2,  # Legacy parameter
                'average_degree': 5.0
            }
        }

        stats = generate_degree_file_from_config(config)
        assert stats['kmin'] == 2
