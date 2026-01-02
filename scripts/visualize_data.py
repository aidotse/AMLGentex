"""
Visualization script for transaction data and network properties.

Convention over configuration:
- Auto-discovers raw transaction data from experiment structure
- Raw data at: {experiment_root}/temporal/tx_log.csv (or .parquet)
- Plots saved to: {experiment_root}/visualizations/data/

Usage examples:
    # Visualize transaction data for an experiment
    python scripts/visualize_data.py --experiment 10k_accounts

    # Specify custom output directory
    python scripts/visualize_data.py --experiment 10k_accounts --output custom_dir
"""
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.visualize.data import plot


def main():
    parser = argparse.ArgumentParser(
        description='Visualize transaction data and network properties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Experiment name (e.g., 10k_accounts)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for plots. Default: {experiment_root}/visualizations/data/',
        default=None
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility',
        default=42
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Construct experiment root path
    experiment_root = Path('experiments') / args.experiment

    if not experiment_root.exists():
        print(f"Error: Experiment directory not found: {experiment_root}")
        return

    # Auto-discover transaction log
    tx_log_parquet = experiment_root / 'temporal' / 'tx_log.parquet'
    tx_log_csv = experiment_root / 'temporal' / 'tx_log.csv'

    if tx_log_parquet.exists():
        tx_log = tx_log_parquet
        print(f"Found transaction log: {tx_log}")
        df = pd.read_parquet(tx_log)
    elif tx_log_csv.exists():
        tx_log = tx_log_csv
        print(f"Found transaction log: {tx_log}")
        df = pd.read_csv(tx_log)
    else:
        print(f"Error: No transaction log found in {experiment_root / 'temporal'}")
        print("Expected: tx_log.parquet or tx_log.csv")
        return

    # Determine output directory
    if args.output:
        plot_dir = Path(args.output)
    else:
        plot_dir = experiment_root / 'visualizations' / 'data'

    os.makedirs(plot_dir, exist_ok=True)

    print(f"Generating visualizations...")
    print(f"  Output directory: {plot_dir}")

    # Generate all plots
    plot(df, str(plot_dir))

    print(f"\n✓ All visualizations saved to {plot_dir}")
    print(f"\nGenerated plots:")
    print(f"  - sar_pattern_account_hist.png")
    print(f"  - sar_pattern_txs_hist.png")
    print(f"  - sar_account_participation_hist.png")
    print(f"  - sar_over_n_banks_hist.png")
    print(f"  - edge_label_hist.png")
    print(f"  - node_label_hist.png")
    print(f"  - homophily.csv")
    print(f"  - Per-bank subdirectories with additional plots")


if __name__ == '__main__':
    main()
