"""
Visualization script for ML training results.

Convention over configuration:
- Auto-discovers results files from experiment structure
- Results at: {experiment_root}/results/{setting}/{model}/results.pkl
- Plots saved to same directory as results

Usage examples:
    # Plot all results for an experiment
    python scripts/plot.py --experiment 10k_accounts

    # Plot specific settings
    python scripts/plot.py --experiment 10k_accounts --settings centralized federated

    # Plot specific models
    python scripts/plot.py --experiment 10k_accounts --models GraphSAGE GCN

    # Plot specific metrics
    python scripts/plot.py --experiment 10k_accounts --metrics average_precision f1

    # Control aggregation
    python scripts/plot.py --experiment 10k_accounts --reduction mean  # Average across clients
    python scripts/plot.py --experiment 10k_accounts --reduction none  # Plot each client separately
"""
import argparse
from pathlib import Path
from src.visualize import plot_metrics
from src.visualize.utils import discover_results, load_results


def main():
    parser = argparse.ArgumentParser(
        description='Plot ML training results with auto-discovery',
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
        '--settings',
        nargs='+',
        help='Settings to plot (centralized, federated, isolated). Default: all available',
        default=None
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Models to plot (e.g., GraphSAGE GCN MLP). Default: all available',
        default=None
    )
    parser.add_argument(
        '--clients',
        nargs='+',
        help='Clients to include. Default: all clients',
        default=None
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Datasets to include',
        default=['trainset', 'valset', 'testset']
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        help='Metrics to plot',
        default=[
            'loss',
            'accuracy',
            'balanced_accuracy',
            'recall',
            'precision',
            'average_precision',
            'f1',
            'roc_curve',
            'precision_recall_curve'
        ]
    )
    parser.add_argument(
        '--reduction',
        help='Type of reduction for multiple clients (mean or none)',
        default='mean'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        help='Output formats',
        default=['png', 'csv']
    )

    args = parser.parse_args()

    # Construct experiment root path
    experiment_root = Path('experiments') / args.experiment

    if not experiment_root.exists():
        print(f"Error: Experiment directory not found: {experiment_root}")
        return

    # Auto-discover results files
    results_files = discover_results(
        experiment_root,
        settings=args.settings,
        models=args.models
    )

    if not results_files:
        print(f"No results files found in {experiment_root / 'results'}")
        print(f"Expected structure: experiments/{args.experiment}/results/{{setting}}/{{model}}/results.pkl")
        return

    print(f"Found {len(results_files)} results file(s):")
    for key, path in results_files.items():
        print(f"  - {key}")

    # Plot each results file
    for key, results_file in results_files.items():
        print(f"\nPlotting {key}...")

        # Load results
        data = load_results(results_file)

        # Output directory is same as results file location
        output_dir = results_file.parent

        # Generate plots
        plot_metrics(
            data,
            str(output_dir),
            args.metrics,
            args.clients,
            args.datasets,
            args.reduction,
            args.formats
        )

        print(f"  Saved plots to {output_dir}")

    print("\n✓ All plots generated successfully!")


if __name__ == '__main__':
    main()
