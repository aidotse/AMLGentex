"""
Generate scale-free graph and output degree-distribution CSV file
"""

import numpy as np
import networkx as nx
import csv
import sys
import yaml
import os
from scipy import stats
from scipy.special import zeta

def get_n(conf_file):
    # open config file and get accounts file
    with open(conf_file, "r") as rf:
        conf = yaml.safe_load(rf)
        directory = conf["input"]["directory"]
        accounts_file = conf["input"]["accounts"]
    # build accounts file path
    accounts_path = directory + '/' + accounts_file
    # open accounts file and get number of accounts
    with open(accounts_path, "r") as rf:
        # skip header
        next(rf)
        # sum first int on each line
        n = sum([int(line.split(',')[0]) for line in rf])
    return n

def get_edge_factor(conf_file):
    with open(conf_file, "r") as rf:
        conf = yaml.safe_load(rf)
        edge_factor = conf["default"]["edge_factor"]
    return edge_factor

def get_scale_free_params(conf_file):
    """
    Get scale-free distribution parameters from config.

    Computes scale from average_degree using the Riemann zeta function:
        scale = (average_degree - loc) / (zeta(gamma) - 1)

    Based on E[deg] = loc + scale (zeta(gamma) - 1) from Appendix E.1 of
    AMLgentex paper (arxiv:2506.13989).

    Args:
        conf_file: Path to configuration file

    Returns:
        tuple: (gamma, loc, scale)
    """
    with open(conf_file, "r") as rf:
        conf = yaml.safe_load(rf)
        scale_free_params = conf["scale-free"]

        # Get gamma (tail exponent) - controls distribution heaviness
        gamma = scale_free_params.get("gamma", 2.0)

        # Get loc (minimum degree) - sets minimum degree
        loc = scale_free_params.get("loc", 1.0)

        # Get average_degree (target mean degree)
        average_degree = scale_free_params["average_degree"]

        # Compute scale using Riemann zeta function
        # Formula from Appendix E.1: scale = (d - loc) / (zeta(gamma) - 1)
        scale = (average_degree - loc) / (zeta(gamma) - 1)

    return gamma, loc, scale

def plot_powerlaw_degree_distrubution(n, gamma=2, edge_factor=20, scale=1.0, min_degree=1, values=None, counts=None, alp=None, bet=None, gam=None):
    
    import matplotlib.pyplot as plt
    
    def func(x, scale, gamma):
        return scale * np.power(x, -gamma)
    
    plt.figure(figsize=(10, 10))
    x = np.linspace(1, 1000, 1000)
    
    plt.plot(x, func(x, scale, gamma), label=f'reference\n  gamma={gamma:.2f}\n  scale={scale:.2f}', color='C0')
    
    if values is not None and counts is not None:
        probs = counts / n
        log_values = np.log(values)
        log_probs = np.log(probs)
        coeffs = np.polyfit(log_values, log_probs, 1)
        gamma, scale = coeffs
        print(f'pareto sampling: gamma={gamma}, scale={np.exp(scale)}')
        plt.plot(x, func(x, np.exp(scale), -gamma), label=f'pareto sampling fit\n  gamma={-gamma:.2f}\n  scale={np.exp(scale):.2f}, min_deg={min_degree}', color='C1')
        plt.scatter(values, probs, label='original', color='C1')
    else:
        degrees = (min_degree + scale * np.random.pareto(gamma, n)).round()
        pareto_values, pareto_counts = np.unique(degrees, return_counts=True)
        pareto_probs = pareto_counts / n
        pareto_log_values = np.log(pareto_values)
        pareto_log_probs = np.log(pareto_probs)
        pareto_coeffs = np.polyfit(pareto_log_values, pareto_log_probs, 1)
        pareto_gamma, pareto_scale = pareto_coeffs
        print(f'pareto sampling: gamma={pareto_gamma}, scale={np.exp(pareto_scale)}')
        plt.plot(x, func(x, np.exp(pareto_scale), -pareto_gamma), label=f'pareto sampling fit\n  gamma={-pareto_gamma:.2f}\n  scale={np.exp(pareto_scale):.2f}\n  min_deg={min_degree}', color='C1')
        plt.scatter(pareto_values, pareto_probs, label='pareto sampling', color='C1')
    
    if edge_factor is not None:
        g = nx.barabasi_albert_graph(n, edge_factor)
        degrees = np.array(list(dict(g.degree()).values()))
        baralb_values, baralb_counts = np.unique(degrees, return_counts=True)
        baralb_probs = baralb_counts / n
        baralb_log_values = np.log(baralb_values)
        baralb_log_probs = np.log(baralb_probs)
        baralb_coeffs = np.polyfit(baralb_log_values, baralb_log_probs, 1)
        baralb_gamma, baralb_scale = baralb_coeffs
        print(f'barabasi-albert: gamma={baralb_coeffs[0]}, scale={np.exp(baralb_coeffs[1])}')
        plt.plot(x, func(x, np.exp(baralb_scale), -baralb_gamma), label=f'barabasi-albert fit\n  gamma={-baralb_gamma:.2f}\n  scale={np.exp(baralb_scale):.2f}', color='C2')
        plt.scatter(baralb_values, baralb_probs, label='barabasi-albert', color='C2')
    
    if alp is not None and bet is not None and gam is not None:
        g = nx.scale_free_graph(n, alpha=alp, beta=bet, gamma=gam)
        degrees = np.array(list(dict(g.degree()).values()))
        sf_values, sf_counts = np.unique(degrees, return_counts=True)
        sf_probs = sf_counts / n
        sf_log_values = np.log(sf_values)
        sf_log_probs = np.log(sf_probs)
        sf_coeffs = np.polyfit(sf_log_values, sf_log_probs, 1)
        sf_gamma, sf_scale = sf_coeffs
        print(f'scale-free: gamma={sf_coeffs[0]}, scale={np.exp(sf_coeffs[1])}')
        plt.plot(x, func(x, np.exp(sf_scale), -sf_gamma), label=f'scale-free fit\n  gamma={-sf_gamma:.2f}\n  scale={np.exp(sf_scale):.2f}', color='C3')
        plt.scatter(sf_values, sf_probs, label='scale-free', color='C3')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('log(degree)')
    plt.ylabel('log(probability)')
    plt.legend()
    plt.grid()
    
    # save plot
    plt.savefig('degree_distributions.png')

def powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=0):
    
    degrees = stats.pareto.rvs(gamma, loc=loc, scale=scale, size=n, random_state=seed).round()
    
    # if degree sum is odd, add one to a random degree
    if degrees.sum() % 2 == 1:
        degrees[np.random.randint(n)] += 1
    split = np.random.rand(n)
    
    in_degrees = (degrees*split).round()
    out_degrees = (degrees*(1-split)).round()
    
    iters = 0
    while in_degrees.sum() != out_degrees.sum() and iters < 10000:
        if in_degrees.sum() > out_degrees.sum():
            idx = np.random.choice(np.where(in_degrees > 1.0)[0])
            in_degrees[idx] -= 1
            out_degrees[np.random.randint(n)] += 1
        else:
            idx = np.random.choice(np.where(out_degrees > 1.0)[0])
            in_degrees[np.random.randint(n)] += 1
            out_degrees[idx] -= 1
        iters += 1
    if in_degrees.sum() > out_degrees.sum():
        diff = in_degrees.sum() - out_degrees.sum()
        assert diff % 2 == 0
        in_degrees[np.argmax(in_degrees)] -= diff / 2
        out_degrees[np.argmax(out_degrees)] += diff / 2
    elif in_degrees.sum() < out_degrees.sum():
        diff = out_degrees.sum() - in_degrees.sum()
        assert diff % 2 == 0
        in_degrees[np.argmax(in_degrees)] += diff / 2
        out_degrees[np.argmax(out_degrees)] -= diff / 2
    
    degrees = np.column_stack((in_degrees,out_degrees))
    
    values, counts = np.unique(degrees, return_counts=True, axis=0)
    
    return values, counts
    
def generate_degree_file_from_config(config):
    """
    Generate degree file from config dictionary (with absolute paths).

    Args:
        config: Configuration dictionary with absolute paths
    """
    # Get number of accounts
    directory = config["input"]["directory"]
    accounts_file = config["input"]["accounts"]
    accounts_path = os.path.join(directory, accounts_file)

    with open(accounts_path, "r") as rf:
        next(rf)  # skip header
        n = sum([int(line.split(',')[0]) for line in rf])

    # Get scale-free parameters
    scale_free_params = config["scale-free"]
    gamma = scale_free_params.get("gamma", 2.0)
    loc = scale_free_params.get("loc", 1.0)
    average_degree = scale_free_params["average_degree"]
    scale = (average_degree - loc) / (zeta(gamma) - 1)

    # Get output path and seed
    deg_file = config["input"]["degree"]
    seed = config["general"]["random_seed"]
    deg_file_path = os.path.join(directory, deg_file)

    # Generate degree distribution
    values, counts = powerlaw_degree_distrubution(n, gamma, loc, scale, seed)

    # Write degree distribution to file
    with open(deg_file_path, "w") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Count", "In-degree", "Out-degree"])
        for value, count in zip(values, counts):
            writer.writerow([count, int(value[0]), int(value[1])])

def generate_degree_file(conf_file):
    """Generate degree file from config file path (for CLI usage)."""
    n = get_n(conf_file)
    # get edge factor from config file
    gamma, loc, scale = get_scale_free_params(conf_file)
    # get directory from config file
    with open(conf_file, "r") as rf:
        conf = yaml.safe_load(rf)
        directory = conf["input"]["directory"]
        deg_file = conf["input"]["degree"]
        seed = conf["general"]["random_seed"]
    # build degree file path
    deg_file_path = os.path.join(directory, deg_file)

    # generate degree distribution
    values, counts = powerlaw_degree_distrubution(n, gamma, loc, scale, seed)

    # write degree distribution to file
    with open(deg_file_path, "w") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Count", "In-degree", "Out-degree"])
        for value, count in zip(values, counts):
            writer.writerow([count, int(value[0]), int(value[1])])
        
def main():
    """Main entry point for generating scale-free degree distribution"""
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


