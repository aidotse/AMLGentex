"""
Visualization script for feature space using dimensionality reduction.

Convention over configuration:
- Auto-discovers preprocessed data from experiment structure
- Data at: {experiment_root}/preprocessed/centralized/trainset_nodes.parquet
- Data at: {experiment_root}/preprocessed/clients/{bank}/trainset_nodes.parquet
- Plots saved to: {experiment_root}/visualizations/features/

Requirements:
- umap-learn package must be installed: pip install umap-learn

Usage examples:
    # Visualize centralized features
    python scripts/visualize_features.py --experiment 10k_accounts

    # Visualize per-client features
    python scripts/visualize_features.py --experiment 10k_accounts --clients nordea swedbank

    # Specify custom output directory
    python scripts/visualize_features.py --experiment 10k_accounts --output custom_dir
"""
import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from src.visualize.features import plot_umap
except ImportError:
    print("Error: umap-learn is required for feature visualization")
    print("Install with: pip install umap-learn")
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize feature space using UMAP dimensionality reduction',
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
        '--clients',
        nargs='+',
        help='Specific clients to visualize. If not specified, visualizes centralized data.',
        default=None
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for plots. Default: {experiment_root}/visualizations/features/',
        default=None
    )

    args = parser.parse_args()

    # Construct experiment root path
    experiment_root = Path('experiments') / args.experiment

    if not experiment_root.exists():
        print(f"Error: Experiment directory not found: {experiment_root}")
        return

    # Determine output directory
    if args.output:
        plot_dir = Path(args.output)
    else:
        plot_dir = experiment_root / 'visualizations' / 'features'

    os.makedirs(plot_dir, exist_ok=True)

    print(f"Generating UMAP visualizations...")
    print(f"  Output directory: {plot_dir}")

    if args.clients:
        # Visualize per-client features
        for client in args.clients:
            client_file = experiment_root / 'preprocessed' / 'clients' / client / 'trainset_nodes.parquet'

            if not client_file.exists():
                print(f"  Warning: Data not found for client '{client}': {client_file}")
                continue

            print(f"  Processing client: {client}")
            df = pd.read_parquet(client_file)

            # Extract features (all columns except metadata and label)
            # Assuming: first columns are metadata (account, bank), last is label (is_sar)
            X = df.iloc[:, 2:-1].to_numpy()
            y = df.iloc[:, -1].to_numpy()

            fig = plot_umap(X, y)

            # Create client subdirectory
            client_dir = plot_dir / client
            os.makedirs(client_dir, exist_ok=True)
            output_file = client_dir / 'umap.png'
            fig.savefig(output_file)
            print(f"    Saved: {output_file}")

    else:
        # Visualize centralized features
        centralized_file = experiment_root / 'preprocessed' / 'centralized' / 'trainset_nodes.parquet'

        if not centralized_file.exists():
            print(f"Error: Centralized data not found: {centralized_file}")
            return

        print(f"  Processing centralized data")
        df = pd.read_parquet(centralized_file)

        # Extract features
        X = df.iloc[:, 2:-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

        fig = plot_umap(X, y)
        output_file = plot_dir / 'umap.png'
        fig.savefig(output_file)
        print(f"    Saved: {output_file}")

    print(f"\n✓ All feature visualizations saved to {plot_dir}")


if __name__ == '__main__':
    main()
