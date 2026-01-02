"""
Unit tests for utility functions
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'python'))

from src.data_creation.python.temporal_simulation.utils import sigmoid, TruncatedNormal


@pytest.mark.unit
@pytest.mark.temporal
class TestSigmoid:
    """Tests for sigmoid function"""

    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5"""
        assert sigmoid(0) == pytest.approx(0.5)

    def test_sigmoid_positive(self):
        """Test sigmoid of positive values"""
        result = sigmoid(2.0)
        assert 0.5 < result < 1.0

    def test_sigmoid_negative(self):
        """Test sigmoid of negative values"""
        result = sigmoid(-2.0)
        assert 0.0 < result < 0.5

    def test_sigmoid_large_positive(self):
        """Test sigmoid handles large positive values"""
        result = sigmoid(1000)
        assert result == pytest.approx(1.0)

    def test_sigmoid_large_negative(self):
        """Test sigmoid handles large negative values"""
        result = sigmoid(-1000)
        assert result == pytest.approx(0.0)

    def test_sigmoid_range(self):
        """Test sigmoid output is always in [0, 1]"""
        test_values = [-100, -10, -1, 0, 1, 10, 100]
        for x in test_values:
            result = sigmoid(x)
            assert 0.0 <= result <= 1.0


@pytest.mark.unit
@pytest.mark.temporal
class TestTruncatedNormal:
    """Tests for TruncatedNormal distribution"""

    def test_initialization(self):
        """Test TruncatedNormal initialization"""
        tn = TruncatedNormal(mean=100, std=10, lower_bound=50, upper_bound=150)
        assert tn.mean == 100
        assert tn.std == 10
        assert tn.lower_bound == 50
        assert tn.upper_bound == 150

    def test_sample_within_bounds(self):
        """Test samples are within specified bounds"""
        tn = TruncatedNormal(mean=100, std=10, lower_bound=80, upper_bound=120)
        rng = np.random.RandomState(42)

        # Generate many samples
        samples = [tn.sample(rng) for _ in range(1000)]

        # All samples should be within bounds
        assert all(80 <= s <= 120 for s in samples)

    def test_sample_mean_approximation(self):
        """Test sample mean approximates specified mean (for large samples)"""
        tn = TruncatedNormal(mean=100, std=20, lower_bound=0, upper_bound=200)
        rng = np.random.RandomState(42)

        samples = [tn.sample(rng) for _ in range(10000)]
        sample_mean = np.mean(samples)

        # Should be roughly close to 100 (within 5%)
        assert 95 < sample_mean < 105

    def test_sample_reproducibility(self):
        """Test samples are reproducible with same random state"""
        tn = TruncatedNormal(mean=100, std=10, lower_bound=50, upper_bound=150)

        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        sample1 = tn.sample(rng1)
        sample2 = tn.sample(rng2)

        assert sample1 == sample2

    def test_tight_bounds(self):
        """Test behavior with very tight bounds"""
        tn = TruncatedNormal(mean=100, std=10, lower_bound=99, upper_bound=101)
        rng = np.random.RandomState(42)

        sample = tn.sample(rng)
        assert 99 <= sample <= 101

    def test_zero_std(self):
        """Test behavior with zero standard deviation"""
        tn = TruncatedNormal(mean=100, std=0, lower_bound=50, upper_bound=150)
        rng = np.random.RandomState(42)

        sample = tn.sample(rng)
        # With std=0, should always return the mean
        assert sample == pytest.approx(100)

    def test_bounds_same_as_mean(self):
        """Test when bounds equal the mean"""
        tn = TruncatedNormal(mean=100, std=10, lower_bound=100, upper_bound=100)
        rng = np.random.RandomState(42)

        sample = tn.sample(rng)
        assert sample == pytest.approx(100)
