"""
Generate scale-free degree distribution using truncated discrete power law.

This module implements proper discrete power-law sampling for directed graphs,
following the specification for truncated discrete power law on k = kmin..kmax.

P(K=k) ∝ k^(-gamma), k ∈ {kmin, ..., kmax}
"""

import numpy as np
import csv
import sys
import yaml
import os
from scipy.optimize import brentq
from typing import Tuple, Optional, Dict, Any


# =============================================================================
# Core Distribution Functions
# =============================================================================

def truncated_discrete_powerlaw_pmf(k_values: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute PMF for truncated discrete power law: P(K=k) ∝ k^(-gamma).

    Args:
        k_values: Array of integer degree values (support of distribution)
        gamma: Power law exponent (tail parameter)

    Returns:
        Normalized probability mass function
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0.0, got {gamma}")

    unnormalized = np.power(k_values.astype(float), -gamma)
    return unnormalized / unnormalized.sum()


def truncated_discrete_powerlaw_mean(kmin: int, kmax: int, gamma: float) -> float:
    """
    Compute mean of truncated discrete power law.

    Args:
        kmin: Minimum degree (inclusive)
        kmax: Maximum degree (inclusive)
        gamma: Power law exponent

    Returns:
        Expected value E[K]
    """
    k_values = np.arange(kmin, kmax + 1)
    pmf = truncated_discrete_powerlaw_pmf(k_values, gamma)
    return float(np.sum(k_values * pmf))


def solve_gamma_for_mean(
    kmin: int,
    kmax: int,
    target_mean: float,
    gamma_bounds: Tuple[float, float] = (1.01, 20.0)
) -> float:
    """
    Solve for gamma such that truncated discrete power law has target mean.

    Uses Brent's method for robust root finding.

    Args:
        kmin: Minimum degree
        kmax: Maximum degree
        target_mean: Desired mean degree
        gamma_bounds: Search bounds for gamma (default: 1.01 to 20.0)

    Returns:
        gamma value that achieves target_mean

    Raises:
        ValueError: If target_mean is outside achievable range
    """
    # Validate target_mean is achievable
    # Note: larger gamma -> steeper decay (lighter tail) -> mass concentrates at kmin -> smaller mean
    mean_at_low_gamma = truncated_discrete_powerlaw_mean(kmin, kmax, gamma_bounds[0])
    mean_at_high_gamma = truncated_discrete_powerlaw_mean(kmin, kmax, gamma_bounds[1])

    if target_mean > mean_at_low_gamma:
        raise ValueError(
            f"target_mean={target_mean} is too high. "
            f"Maximum achievable mean with gamma={gamma_bounds[0]} is {mean_at_low_gamma:.2f}. "
            f"Try increasing kmax or decreasing target_mean."
        )
    if target_mean < mean_at_high_gamma:
        raise ValueError(
            f"target_mean={target_mean} is too low. "
            f"Minimum achievable mean with gamma={gamma_bounds[1]} is {mean_at_high_gamma:.2f}. "
            f"Try increasing kmin or increasing target_mean."
        )

    def objective(gamma):
        return truncated_discrete_powerlaw_mean(kmin, kmax, gamma) - target_mean

    return brentq(objective, gamma_bounds[0], gamma_bounds[1])


def sample_truncated_discrete_powerlaw(
    n: int,
    kmin: int,
    kmax: int,
    gamma: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample n values from truncated discrete power law.

    Args:
        n: Number of samples
        kmin: Minimum degree
        kmax: Maximum degree
        gamma: Power law exponent
        rng: NumPy random generator

    Returns:
        Array of n integer degree values
    """
    k_values = np.arange(kmin, kmax + 1)
    pmf = truncated_discrete_powerlaw_pmf(k_values, gamma)
    return rng.choice(k_values, size=n, p=pmf)


# =============================================================================
# Degree Balancing
# =============================================================================

def balance_degree_sums(
    in_degrees: np.ndarray,
    out_degrees: np.ndarray,
    kmax: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust degree arrays so sum(in) == sum(out) with integer-only changes.

    Strategy: Add 1 to random nodes on the smaller-sum side.
    This preserves the heavy-tail character while ensuring graph feasibility.

    Args:
        in_degrees: Array of in-degrees
        out_degrees: Array of out-degrees
        kmax: Maximum degree (cap for adjustments)
        rng: NumPy random generator

    Returns:
        Tuple of (adjusted_in_degrees, adjusted_out_degrees)

    Raises:
        ValueError: If balancing is impossible (all degrees at kmax)
    """
    in_degrees = in_degrees.copy()
    out_degrees = out_degrees.copy()

    delta = int(out_degrees.sum() - in_degrees.sum())

    n = len(in_degrees)

    if delta > 0:
        # out_sum > in_sum: need to increase in_degrees
        candidates = np.where(in_degrees < kmax)[0]
        if len(candidates) == 0:
            raise ValueError("Cannot balance: all in_degrees at kmax")

        # Add to random candidates (vectorized with bincount)
        to_adjust = rng.choice(candidates, size=abs(delta), replace=True)
        increments = np.bincount(to_adjust, minlength=n)
        in_degrees = np.minimum(in_degrees + increments, kmax)

    elif delta < 0:
        # in_sum > out_sum: need to increase out_degrees
        candidates = np.where(out_degrees < kmax)[0]
        if len(candidates) == 0:
            raise ValueError("Cannot balance: all out_degrees at kmax")

        to_adjust = rng.choice(candidates, size=abs(delta), replace=True)
        increments = np.bincount(to_adjust, minlength=n)
        out_degrees = np.minimum(out_degrees + increments, kmax)

    # Final verification loop if still not balanced (due to kmax caps)
    max_iters = 10000
    iters = 0
    while in_degrees.sum() != out_degrees.sum() and iters < max_iters:
        if in_degrees.sum() < out_degrees.sum():
            candidates = np.where(in_degrees < kmax)[0]
            if len(candidates) == 0:
                raise ValueError(
                    f"Cannot balance: all in_degrees at kmax={kmax}. "
                    f"Remaining delta={out_degrees.sum() - in_degrees.sum()}"
                )
            in_degrees[rng.choice(candidates)] += 1
        else:
            candidates = np.where(out_degrees < kmax)[0]
            if len(candidates) == 0:
                raise ValueError(
                    f"Cannot balance: all out_degrees at kmax={kmax}. "
                    f"Remaining delta={in_degrees.sum() - out_degrees.sum()}"
                )
            out_degrees[rng.choice(candidates)] += 1
        iters += 1

    # Final assertion - should never fail if loop completed
    if in_degrees.sum() != out_degrees.sum():
        raise ValueError(
            f"Failed to balance after {max_iters} iterations. "
            f"in_sum={in_degrees.sum()}, out_sum={out_degrees.sum()}"
        )

    return in_degrees, out_degrees


# =============================================================================
# Main Degree Generation
# =============================================================================

def discrete_powerlaw_degree_distribution(
    n: int,
    kmin: int = 1,
    kmax: Optional[int] = None,
    gamma: Optional[float] = None,
    average_degree: Optional[float] = None,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate in/out degree distribution using truncated discrete power law.

    Samples in-degrees and out-degrees separately from the same distribution,
    then balances sums with integer-only adjustments.

    Args:
        n: Number of nodes
        kmin: Minimum degree (default: 1)
        kmax: Maximum degree (default: n-1)
        gamma: Power law exponent. If None, solved from average_degree.
        average_degree: Target mean degree. Required if gamma is None.
        seed: Random seed

    Returns:
        Tuple of (values, counts, stats) where:
            - values: Nx2 array of unique (in_degree, out_degree) pairs
            - counts: Array of counts for each unique pair
            - stats: Dict with gamma, kmin, kmax, achieved_mean_in, achieved_mean_out, total_edges

    Raises:
        ValueError: If parameters are invalid or inconsistent
    """
    # ===================
    # Input Validation
    # ===================
    if kmin < 0:
        raise ValueError(f"kmin must be >= 0, got {kmin}")
    if kmax is None:
        kmax = n - 1
    if kmax < kmin:
        raise ValueError(f"kmax ({kmax}) must be >= kmin ({kmin})")
    if kmax > n - 1:
        raise ValueError(
            f"kmax ({kmax}) cannot exceed n-1 ({n-1}) for simple directed graphs "
            f"(no self-loops or multi-edges)"
        )

    # Determine gamma
    if gamma is None:
        if average_degree is None:
            raise ValueError("Either gamma or average_degree must be provided")

        # Validate average_degree is within bounds
        if average_degree < kmin:
            raise ValueError(
                f"average_degree ({average_degree}) cannot be less than kmin ({kmin})"
            )
        if average_degree > kmax:
            raise ValueError(
                f"average_degree ({average_degree}) cannot be greater than kmax ({kmax})"
            )

        # Solve for gamma
        gamma = solve_gamma_for_mean(kmin, kmax, average_degree)

    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0.0, got {gamma}")

    # ===================
    # Sampling
    # ===================
    rng = np.random.default_rng(seed)

    # Sample in-degrees and out-degrees separately
    in_degrees = sample_truncated_discrete_powerlaw(n, kmin, kmax, gamma, rng)
    out_degrees = sample_truncated_discrete_powerlaw(n, kmin, kmax, gamma, rng)

    # ===================
    # Balance sums
    # ===================
    in_degrees, out_degrees = balance_degree_sums(in_degrees, out_degrees, kmax, rng)

    # ===================
    # Final Validation
    # ===================
    if in_degrees.sum() != out_degrees.sum():
        raise ValueError(f"Sum mismatch: in={in_degrees.sum()}, out={out_degrees.sum()}")
    if not np.all(in_degrees >= 0):
        raise ValueError("Negative in_degrees found")
    if not np.all(out_degrees >= 0):
        raise ValueError("Negative out_degrees found")
    if not np.all(in_degrees == in_degrees.astype(int)):
        raise ValueError("Non-integer in_degrees found")
    if not np.all(out_degrees == out_degrees.astype(int)):
        raise ValueError("Non-integer out_degrees found")

    # ===================
    # Format output
    # ===================
    degrees = np.column_stack((in_degrees.astype(int), out_degrees.astype(int)))
    values, counts = np.unique(degrees, return_counts=True, axis=0)

    # Report achieved statistics
    achieved_mean_in = in_degrees.mean()
    achieved_mean_out = out_degrees.mean()

    return values, counts, {
        'gamma': gamma,
        'kmin': kmin,
        'kmax': kmax,
        'achieved_mean_in': achieved_mean_in,
        'achieved_mean_out': achieved_mean_out,
        'total_edges': int(in_degrees.sum())
    }


# =============================================================================
# Legacy Compatibility / Config-Based Generation
# =============================================================================

def get_n(conf_file: str) -> int:
    """Get total number of accounts from config file."""
    with open(conf_file, "r") as rf:
        conf = yaml.safe_load(rf)
        directory = conf["input"]["directory"]
        accounts_file = conf["input"]["accounts"]

    accounts_path = os.path.join(directory, accounts_file)

    with open(accounts_path, "r") as rf:
        next(rf)  # skip header
        n = sum([int(line.split(',')[0]) for line in rf])
    return n


def generate_degree_file_from_config(config: dict) -> dict:
    """
    Generate degree file from config dictionary (with absolute paths).

    Args:
        config: Configuration dictionary with absolute paths

    Returns:
        Statistics dict with achieved parameters

    Note:
        Degree file is written to spatial output directory.
    """
    # Get number of accounts from input directory
    input_dir = config["input"]["directory"]
    accounts_file = config["input"]["accounts"]
    accounts_path = os.path.join(input_dir, accounts_file)

    with open(accounts_path, "r") as rf:
        next(rf)  # skip header
        n = sum([int(line.split(',')[0]) for line in rf])

    # Get scale-free parameters
    scale_free_params = config["scale-free"]
    gamma = scale_free_params.get("gamma", None)
    kmin = int(scale_free_params.get("kmin", scale_free_params.get("loc", 1)))
    kmax = scale_free_params.get("kmax", n - 1)
    if kmax is not None:
        kmax = int(kmax)
    average_degree = scale_free_params.get("average_degree", None)

    # Get output path and seed
    output_dir = config["spatial"]["directory"]
    deg_file = config["input"]["degree"]
    seed = config["general"]["random_seed"]
    deg_file_path = os.path.join(output_dir, deg_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate degree distribution
    values, counts, stats = discrete_powerlaw_degree_distribution(
        n=n,
        kmin=kmin,
        kmax=kmax,
        gamma=gamma,
        average_degree=average_degree,
        seed=seed
    )

    # Write degree distribution to file
    with open(deg_file_path, "w") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Count", "In-degree", "Out-degree"])
        for value, count in zip(values, counts):
            writer.writerow([count, int(value[0]), int(value[1])])

    print(f"Generated degree file: {deg_file_path}")
    print(f"  n={n}, kmin={stats['kmin']}, kmax={stats['kmax']}")
    print(f"  gamma={stats['gamma']:.4f}")
    print(f"  achieved_mean_in={stats['achieved_mean_in']:.2f}")
    print(f"  achieved_mean_out={stats['achieved_mean_out']:.2f}")
    print(f"  total_edges={stats['total_edges']}")

    return stats


def generate_degree_file(conf_file: str) -> dict:
    """Generate degree file from config file path (for CLI usage)."""
    with open(conf_file, "r") as rf:
        conf = yaml.safe_load(rf)

    # For legacy compatibility, build absolute paths if needed
    if not os.path.isabs(conf["input"]["directory"]):
        conf["input"]["directory"] = os.path.abspath(conf["input"]["directory"])
    if not os.path.isabs(conf.get("spatial", {}).get("directory", "")):
        # If no spatial directory, use input directory
        if "spatial" not in conf:
            conf["spatial"] = {}
        if "directory" not in conf["spatial"]:
            conf["spatial"]["directory"] = conf["input"]["directory"]

    return generate_degree_file_from_config(conf)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for generating scale-free degree distribution."""
    argv = sys.argv

    if len(argv) < 2:
        print("Error: Configuration file is required")
        print("Usage: python generate_scalefree.py <config.yaml>")
        print("Example: python generate_scalefree.py experiments/template/config/data.yaml")
        sys.exit(1)

    conf_file = argv[1]
    generate_degree_file(conf_file)


if __name__ == "__main__":
    main()
