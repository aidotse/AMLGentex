"""
Salary distribution based on Swedish salary statistics
Matches the Java implementation in SalaryDistribution.java
"""
import numpy as np
import pandas as pd
import os


# Global cache for salary distribution data (optimization: load CSV once)
_SALARY_DISTRIBUTION_CACHE = None


class SalaryDistribution:
    """
    Samples monthly salaries from Swedish salary statistics using log-normal distribution
    """

    def __init__(self, csv_path=None):
        global _SALARY_DISTRIBUTION_CACHE

        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), 'scb_statistics_2021.csv')

        # Use cached data if available (optimization: load CSV once)
        if _SALARY_DISTRIBUTION_CACHE is None:
            # Load salary data
            salary_data = pd.read_csv(csv_path)
            # Strip whitespace from column names
            salary_data.columns = salary_data.columns.str.strip()

            # Compute cumulative probabilities based on population sizes
            total_population = salary_data['population size'].sum()
            salary_data['cumulative_prob'] = salary_data['population size'].cumsum() / total_population

            _SALARY_DISTRIBUTION_CACHE = salary_data

        self.salary_data = _SALARY_DISTRIBUTION_CACHE

    def sample(self, random_state=None):
        """
        Sample a monthly salary from the distribution

        Returns:
            float: Monthly salary in SEK (thousands)
        """
        if random_state is None:
            random_state = np.random

        # Sample an age group based on population distribution
        random_value = random_state.random()

        # Find the age group
        age_index = np.searchsorted(self.salary_data['cumulative_prob'].values, random_value)

        # Get mean and median for this age group
        mean = self.salary_data.iloc[age_index]['average year income (tkr)']
        median = self.salary_data.iloc[age_index]['median year income (tkr)']

        # Handle edge case where median is 0
        if median <= 0:
            median = mean * 0.8  # Approximate fallback

        # Calculate log-normal distribution parameters
        # Following Java implementation:
        # mu = log(median)
        # sigma = sqrt(2 * abs(log(mean) - mu))
        mu = np.log(median)
        sigma = np.sqrt(2 * np.abs(np.log(mean) - mu))

        # Sample from log-normal distribution
        salary = random_state.lognormal(mu, sigma)

        # Convert from yearly (thousands SEK) to monthly
        # Java: salary * 1000 / 12
        monthly_salary = salary * 1000 / 12

        return monthly_salary
