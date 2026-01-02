"""
Utility functions for the AML simulator
"""
import numpy as np


class TruncatedNormal:
    """Fast truncated normal distribution using rejection sampling"""

    def __init__(self, mean, std, lower_bound, upper_bound):
        self.mean = mean
        self.std = std
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self, random_state=None):
        """Sample from truncated normal distribution using rejection sampling"""
        if self.std <= 0 or self.lower_bound >= self.upper_bound:
            return np.clip(self.mean, self.lower_bound, self.upper_bound)

        # Use rejection sampling - much faster than scipy for typical cases
        max_tries = 100
        for _ in range(max_tries):
            if random_state is not None:
                sample = random_state.normal(self.mean, self.std)
            else:
                sample = np.random.normal(self.mean, self.std)

            if self.lower_bound <= sample <= self.upper_bound:
                return sample

        # Fallback: if rejection sampling fails, just clamp to bounds
        if random_state is not None:
            sample = random_state.normal(self.mean, self.std)
        else:
            sample = np.random.normal(self.mean, self.std)
        return np.clip(sample, self.lower_bound, self.upper_bound)


def sigmoid(x):
    """Sigmoid function for spending probability (optimized for scalar)"""
    # Avoid overflow for large negative x
    if x < -500:
        return 0.0
    elif x > 500:
        return 1.0
    # Use math.exp for scalar values (faster than numpy for single values)
    import math
    return 1.0 / (1.0 + math.exp(-x))
