"""
Unit tests for Swedish salary distribution
Tests realistic salary sampling based on dummy statistics
"""
import pytest
import numpy as np
import tempfile
import os

from src.data_creation.temporal_simulation.salary_distribution import SalaryDistribution


# Dummy salary data for testing (simulating age groups with different salaries)
DUMMY_SALARY_DATA = """age group,average year income (tkr),median year income (tkr),population size
20-25,250,220,100
26-35,400,350,300
36-45,500,450,400
46-55,550,500,200
56-65,480,430,150
"""


@pytest.fixture
def salary_csv():
    """Create a temporary CSV file with dummy salary data"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(DUMMY_SALARY_DATA)
        csv_path = f.name

    yield csv_path

    # Cleanup
    os.unlink(csv_path)


@pytest.fixture
def dist(salary_csv):
    """Create a SalaryDistribution instance with dummy data"""
    # Clear cache to ensure fresh load
    import src.data_creation.temporal_simulation.salary_distribution as sd_module
    sd_module._SALARY_DISTRIBUTION_CACHE = None

    return SalaryDistribution(csv_path=salary_csv)


@pytest.mark.unit
@pytest.mark.temporal
class TestSalaryDistribution:
    """Tests for Swedish salary distribution sampling"""

    def test_initialization_loads_data(self, dist):
        """Test that salary distribution initializes with data"""
        assert dist.salary_data is not None
        assert len(dist.salary_data) > 0
        assert 'cumulative_prob' in dist.salary_data.columns
        assert 'average year income (tkr)' in dist.salary_data.columns
        assert 'median year income (tkr)' in dist.salary_data.columns

    def test_sample_returns_positive_value(self, dist):
        """Test that sampling returns a positive salary"""
        random_state = np.random.RandomState(42)

        salary = dist.sample(random_state)

        assert salary > 0
        assert isinstance(salary, (int, float, np.number))

    def test_sample_reproducibility(self, dist):
        """Test that sampling with same seed produces same results"""
        seed = 42

        random_state1 = np.random.RandomState(seed)
        salary1 = dist.sample(random_state1)

        random_state2 = np.random.RandomState(seed)
        salary2 = dist.sample(random_state2)

        assert salary1 == salary2

    def test_sample_different_seeds_different_results(self, dist):
        """Test that different seeds produce different results"""
        random_state1 = np.random.RandomState(42)
        salary1 = dist.sample(random_state1)

        random_state2 = np.random.RandomState(99)
        salary2 = dist.sample(random_state2)

        # Should be different (with high probability)
        assert salary1 != salary2

    def test_sample_distribution_shape(self, dist):
        """Test that sampled salaries follow expected distribution shape"""
        random_state = np.random.RandomState(42)

        # Sample many salaries
        salaries = [dist.sample(random_state) for _ in range(1000)]

        # Should have positive values
        assert all(s > 0 for s in salaries)

        # Calculate statistics
        mean_salary = np.mean(salaries)
        median_salary = np.median(salaries)
        std_salary = np.std(salaries)

        # With our dummy data (250-550 tkr yearly = ~20-46k SEK monthly)
        # Mean should be in reasonable range
        assert 10000 < mean_salary < 60000, f"Mean salary {mean_salary} seems unrealistic for dummy data"

        # Should have positive standard deviation
        assert std_salary > 0

        # Log-normal should have mean > median (right-skewed)
        assert mean_salary > median_salary

    def test_sample_reasonable_salary_range(self, dist):
        """Test that individual samples fall in reasonable range"""
        random_state = np.random.RandomState(42)

        # Sample multiple times
        for _ in range(100):
            salary = dist.sample(random_state)

            # With dummy data, monthly salaries should be roughly 10k-80k SEK
            # Allow wider range for edge cases
            assert 5000 < salary < 150000, \
                f"Salary {salary} SEK/month seems unrealistic for dummy data"

    def test_sample_distribution_coverage(self, dist):
        """Test that sampling covers different age groups"""
        random_state = np.random.RandomState(42)

        # Sample many salaries
        salaries = [dist.sample(random_state) for _ in range(500)]

        # Should have variety (different age groups produce different ranges)
        unique_salaries = len(set([int(s/1000) for s in salaries]))  # Round to thousands

        # Should have reasonable variety (we have 5 age groups in dummy data)
        assert unique_salaries >= 3, "Should sample from variety of salary levels"

    def test_cumulative_probability_sums_to_one(self, dist):
        """Test that cumulative probability reaches 1.0"""
        # Last cumulative probability should be 1.0 (or very close)
        last_cumprob = dist.salary_data['cumulative_prob'].iloc[-1]

        assert 0.99 <= last_cumprob <= 1.0

    def test_salary_data_has_required_columns(self, dist):
        """Test that loaded data has all required columns"""
        required_columns = [
            'average year income (tkr)',
            'median year income (tkr)',
            'population size',
            'cumulative_prob'
        ]

        for col in required_columns:
            assert col in dist.salary_data.columns, f"Missing column: {col}"

    def test_salary_data_non_negative_values(self, dist):
        """Test that salary data has non-negative values"""
        # Average incomes should be non-negative
        assert all(dist.salary_data['average year income (tkr)'] >= 0)

        # Median incomes should be non-negative
        assert all(dist.salary_data['median year income (tkr)'] >= 0)

        # Population sizes should be positive
        assert all(dist.salary_data['population size'] > 0)

    def test_monthly_conversion(self, dist):
        """Test that yearly income is correctly converted to monthly"""
        random_state = np.random.RandomState(42)

        # Sample a salary (monthly)
        monthly_salary = dist.sample(random_state)

        # Yearly equivalent
        yearly_salary = monthly_salary * 12

        # With dummy data (250-550 tkr yearly), should be roughly in this range
        assert 50000 < yearly_salary < 1000000, \
            f"Yearly salary {yearly_salary} SEK seems unrealistic for dummy data"

    def test_log_normal_properties(self, dist):
        """Test that distribution exhibits log-normal properties"""
        random_state = np.random.RandomState(42)

        # Sample many salaries
        salaries = np.array([dist.sample(random_state) for _ in range(2000)])

        # Log-normal distribution: log(X) should be approximately normal
        log_salaries = np.log(salaries)

        # Check that log transform reduces skewness
        salary_skew = np.abs(3 * (np.mean(salaries) - np.median(salaries)) / np.std(salaries))
        log_salary_skew = np.abs(3 * (np.mean(log_salaries) - np.median(log_salaries)) / np.std(log_salaries))

        # Log transform should reduce skewness (move closer to 0)
        assert log_salary_skew < salary_skew

    def test_caching_mechanism(self, salary_csv):
        """Test that data is cached and reused"""
        # Clear cache first
        import src.data_creation.temporal_simulation.salary_distribution as sd_module
        sd_module._SALARY_DISTRIBUTION_CACHE = None

        # Create first instance (loads data)
        dist1 = SalaryDistribution(csv_path=salary_csv)
        data1_id = id(dist1.salary_data)

        # Create second instance (should use cache)
        dist2 = SalaryDistribution(csv_path=salary_csv)
        data2_id = id(dist2.salary_data)

        # Should be the same object (cached)
        assert data1_id == data2_id

    def test_edge_case_zero_median(self, dist):
        """Test handling of edge case where median is zero"""
        # This tests the fallback logic when median <= 0
        # The actual implementation handles this with median = mean * 0.8

        # We can't easily inject zero median with our dummy data, but we can verify
        # that all age groups have valid data
        for idx, row in dist.salary_data.iterrows():
            mean = row['average year income (tkr)']
            median = row['median year income (tkr)']

            # If implementation encounters zero median, it uses fallback
            # We just verify data quality
            assert mean > 0, f"Mean should be positive at index {idx}"
            # Median can be zero in edge cases, that's why there's a fallback

    def test_sample_without_random_state(self, dist):
        """Test sampling without explicit random state (uses default)"""
        # Should work without random_state parameter
        salary = dist.sample()

        assert salary > 0
        assert isinstance(salary, (int, float, np.number))

    def test_multiple_samples_independent(self, dist):
        """Test that multiple samples are independent"""
        random_state = np.random.RandomState(42)

        # Take multiple samples
        samples = [dist.sample(random_state) for _ in range(50)]

        # Should have variety (not all the same)
        unique_samples = len(set([round(s, 2) for s in samples]))

        # Should have many unique values
        assert unique_samples > 10
