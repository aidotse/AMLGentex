"""
Unit tests for spatial simulation utilities
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'python'))

from src.data_creation.python.spatial_simulation.utils.random_amount import RandomAmount
from src.data_creation.python.spatial_simulation.utils.rounded_amount import RoundedAmount


@pytest.mark.unit
@pytest.mark.spatial
class TestRandomAmount:
    """Tests for RandomAmount class"""

    def test_initialization(self):
        """Test RandomAmount initialization"""
        ra = RandomAmount(min=100, max=1000)
        assert ra.min == 100
        assert ra.max == 1000

    def test_get_amount_within_range(self):
        """Test generated amount is within range"""
        ra = RandomAmount(min=100, max=1000)

        # Generate many amounts
        for _ in range(100):
            amount = ra.getAmount()
            assert 100 <= amount <= 1000

    def test_get_amount_equal_bounds(self):
        """Test when min equals max"""
        ra = RandomAmount(min=500, max=500)
        amount = ra.getAmount()
        assert amount == pytest.approx(500)

    def test_get_amount_distribution(self):
        """Test amount distribution is uniform"""
        ra = RandomAmount(min=1, max=10000)

        amounts = [ra.getAmount() for _ in range(1000)]

        # Should have values across the range
        assert min(amounts) >= 1
        assert max(amounts) <= 10000

        # Mean should be roughly in the middle for uniform distribution
        mean_amount = sum(amounts) / len(amounts)
        assert 3000 <= mean_amount <= 7000


@pytest.mark.unit
@pytest.mark.spatial
class TestRoundedAmount:
    """Tests for RoundedAmount class"""

    def test_initialization(self):
        """Test RoundedAmount initialization"""
        ra = RoundedAmount(min=100, max=1000)
        assert ra.min == 100
        assert ra.max == 1000

    def test_get_amount_within_range(self):
        """Test generated amount is within range"""
        ra = RoundedAmount(min=100, max=1000)

        for _ in range(100):
            amount = ra.getAmount()
            assert 100 <= amount <= 1000

    def test_get_amount_rounding(self):
        """Test amount is properly rounded"""
        ra = RoundedAmount(min=100, max=1000)

        amount = ra.getAmount()

        # Amount should be rounded to integer with no decimal places
        assert amount == float(int(amount))

    def test_get_amount_small_range(self):
        """Test with a small but valid range"""
        ra = RoundedAmount(min=500, max=600)

        for _ in range(50):
            amount = ra.getAmount()
            assert 500 <= amount <= 600
            # Should be rounded to integer
            assert amount == float(int(amount))
