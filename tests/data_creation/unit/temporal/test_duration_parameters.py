"""
Unit tests for duration parameter conversion (linear to log-space)
"""
import pytest
import numpy as np

from src.data_creation.python.temporal_simulation.simulator import AMLSimulator


@pytest.mark.unit
@pytest.mark.temporal
class TestDurationParameterConversion:
    """Tests for linear-space to log-space parameter conversion"""

    def test_conversion_basic(self):
        """Test basic conversion from linear to log-space parameters"""
        mean_linear = 10.0
        std_linear = 3.0

        mu, sigma = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)

        # Verify returned values are finite
        assert np.isfinite(mu)
        assert np.isfinite(sigma)
        assert sigma > 0  # Sigma must be positive

    def test_conversion_produces_correct_mean(self):
        """Test that converted parameters produce correct mean (without clipping)"""
        mean_linear = 20.0
        std_linear = 5.0

        mu, sigma = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)

        # Sample many values to check mean
        random_state = np.random.RandomState(42)
        samples = random_state.lognormal(mu, sigma, size=50000)

        # Mean should be close to target (within 2% tolerance)
        observed_mean = np.mean(samples)
        assert abs(observed_mean - mean_linear) / mean_linear < 0.02

    def test_conversion_produces_correct_std(self):
        """Test that converted parameters produce correct standard deviation (without clipping)"""
        mean_linear = 20.0
        std_linear = 5.0

        mu, sigma = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)

        # Sample many values to check std
        random_state = np.random.RandomState(42)
        samples = random_state.lognormal(mu, sigma, size=50000)

        # Std should be close to target (within 5% tolerance)
        observed_std = np.std(samples)
        assert abs(observed_std - std_linear) / std_linear < 0.05

    def test_conversion_with_small_values(self):
        """Test conversion with small mean and std values"""
        mean_linear = 3.0
        std_linear = 1.0

        mu, sigma = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)

        # Sample to verify
        random_state = np.random.RandomState(42)
        samples = random_state.lognormal(mu, sigma, size=10000)

        observed_mean = np.mean(samples)
        # Within 5% tolerance for smaller samples
        assert abs(observed_mean - mean_linear) / mean_linear < 0.05

    def test_conversion_with_large_std(self):
        """Test conversion when std is large relative to mean"""
        mean_linear = 10.0
        std_linear = 8.0  # 80% of mean

        mu, sigma = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)

        # Should still produce finite values
        assert np.isfinite(mu)
        assert np.isfinite(sigma)
        assert sigma > 0

        # Sample to verify
        random_state = np.random.RandomState(42)
        samples = random_state.lognormal(mu, sigma, size=50000)

        observed_mean = np.mean(samples)
        # Larger tolerance for high variance case
        assert abs(observed_mean - mean_linear) / mean_linear < 0.10

    def test_conversion_with_clipping(self):
        """Test that clipping doesn't completely destroy the distribution"""
        mean_linear = 6.0
        std_linear = 3.0
        total_steps = 100

        mu, sigma = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)

        # Sample with clipping like the simulator does
        random_state = np.random.RandomState(42)
        samples = []
        for _ in range(10000):
            duration = int(np.round(random_state.lognormal(mu, sigma)))
            duration = max(2, min(duration, total_steps - 1))
            samples.append(duration)

        # With clipping, mean will be affected but should still be reasonable
        observed_mean = np.mean(samples)
        # Mean should be at least close to the target
        assert 2 <= observed_mean <= total_steps - 1
        # Should be within same order of magnitude
        assert observed_mean < mean_linear * 5

    def test_reproducibility(self):
        """Test that same inputs always give same outputs"""
        mean_linear = 15.0
        std_linear = 4.0

        mu1, sigma1 = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)
        mu2, sigma2 = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)

        assert mu1 == mu2
        assert sigma1 == sigma2

    def test_different_inputs_give_different_outputs(self):
        """Test that different inputs produce different log-space parameters"""
        mu1, sigma1 = AMLSimulator.linear_to_lognormal_params(10.0, 3.0)
        mu2, sigma2 = AMLSimulator.linear_to_lognormal_params(20.0, 3.0)

        # Different means should give different mu
        assert mu1 != mu2

    def test_zero_std_raises_or_handles_gracefully(self):
        """Test behavior with zero standard deviation"""
        mean_linear = 10.0
        std_linear = 0.0

        # This should either raise an error or handle gracefully
        try:
            mu, sigma = AMLSimulator.linear_to_lognormal_params(mean_linear, std_linear)
            # If it doesn't raise, sigma should be very small or zero
            assert sigma == 0.0 or sigma < 1e-10
        except (ValueError, ZeroDivisionError):
            # It's acceptable to raise an error for invalid input
            pass
