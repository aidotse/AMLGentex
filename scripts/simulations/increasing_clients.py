"""
Experiment: Increasing Number of Clients

Tests federated learning performance as the number of collaborating banks increases.
Shows the benefit of data collaboration vs isolated learning.

Usage:
    uv run python scripts/simulations/increasing_clients.py \
        --experiment_dir experiments/10k_accounts \
        --model_types GCN GAT GraphSAGE \
        --seeds 42 43 44

Output:
    experiments/<name>/results/federated/<model>/<n>_clients/<seed>/results.pkl
    experiments/<name>/results/federated/<model>_increasing_clients.png
"""

import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pickle
import yaml
from pathlib import Path

from src.ml import servers, clients, models
from src.ml.training import federated
from src.utils import get_optimal_params


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Test effect of increasing number of federated clients')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--model_types', nargs='+',
                        default=['DecisionTreeClassifier', 'RandomForestClassifier', 'GCN'],
                        help='Models to train')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='Random seeds for reproducibility')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers for multiprocessing')
    parser.add_argument('--use_optimal_params', action='store_true',
                        help='Use optimized hyperparameters from Optuna')
    args = parser.parse_args()

    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    print()

    # Load model configs
    experiment_dir = Path(args.experiment_dir)
    config_dir = experiment_dir / 'config'

    with open(config_dir / 'models.yaml', 'r') as f:
        models_config = yaml.safe_load(f)

    results_dir = experiment_dir / 'results' / 'federated'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover clients from preprocessed data
    preprocessed_dir = experiment_dir / 'preprocessed'
    clients_dir = preprocessed_dir / 'clients'
    all_client_ids = sorted([d.name for d in clients_dir.iterdir() if d.is_dir()])

    print(f'Found {len(all_client_ids)} clients: {all_client_ids}\n')

    for model_type in args.model_types:
        print(f'\n{"="*60}')
        print(f'Model: {model_type}')
        print(f'{"="*60}')

        train_avg_precisions = []
        val_avg_precisions = []
        test_avg_precisions = []

        # Progressively add clients
        for i in range(len(all_client_ids)):
            n_clients = i + 1
            client_subset = all_client_ids[:n_clients]
            print(f'\n  Testing with {n_clients} client(s): {client_subset}')

            # Get model config
            kwargs = models_config[model_type]['default'].copy()

            # Build client config
            kwargs['clients'] = {}
            for client_id in client_subset:
                client_path = clients_dir / client_id
                kwargs['clients'][client_id] = {
                    'trainset': str(client_path / 'trainset_nodes.parquet'),
                    'testset': str(client_path / 'testset_nodes.parquet'),
                    'trainset_nodes': str(client_path / 'trainset_nodes.parquet'),
                    'trainset_edges': str(client_path / 'trainset_edges.parquet'),
                    'testset_nodes': str(client_path / 'testset_nodes.parquet'),
                    'testset_edges': str(client_path / 'testset_edges.parquet'),
                }

            # Get optimal params if requested
            if args.use_optimal_params:
                optuna_dir = results_dir / model_type
                kwargs = get_optimal_params(kwargs, str(optuna_dir))

            train_avg_precision = 0.0
            val_avg_precision = 0.0
            test_avg_precision = 0.0

            for seed in args.seeds:
                print(f'    Seed {seed}...')

                results = federated(
                    seed=seed,
                    Server=getattr(servers, kwargs['server_type']),
                    Client=getattr(clients, kwargs['client_type']),
                    Model=getattr(models, model_type),
                    n_workers=args.n_workers,
                    **kwargs
                )

                # Save results
                seed_results_dir = results_dir / model_type / f'{n_clients}_clients' / f'seed_{seed}'
                seed_results_dir.mkdir(parents=True, exist_ok=True)
                with open(seed_results_dir / 'results.pkl', 'wb') as f:
                    pickle.dump(results, f)
                print(f'    Saved to {seed_results_dir}/results.pkl')

                # Aggregate metrics across clients
                for id in results:
                    train_avg_precision += results[id]['trainset']['average_precision'][-1] / len(results) / len(args.seeds)
                    val_avg_precision += results[id]['valset']['average_precision'][-1] / len(results) / len(args.seeds)
                    test_avg_precision += results[id]['testset']['average_precision'][-1] / len(results) / len(args.seeds)

            print(f'    Results - Train: {train_avg_precision:.4f}, Val: {val_avg_precision:.4f}, Test: {test_avg_precision:.4f}')

            train_avg_precisions.append(train_avg_precision)
            val_avg_precisions.append(val_avg_precision)
            test_avg_precisions.append(test_avg_precision)

        # Plot results for this model
        fig = plt.figure(figsize=(10, 6))
        x = list(range(1, len(all_client_ids) + 1))
        plt.plot(x, train_avg_precisions, marker='o', label='Train', linewidth=2)
        plt.plot(x, val_avg_precisions, marker='s', label='Validation', linewidth=2)
        plt.plot(x, test_avg_precisions, marker='^', label='Test', linewidth=2)
        plt.xlabel('Number of Clients', fontsize=12)
        plt.ylabel('Average Precision', fontsize=12)
        plt.title(f'{model_type}: Effect of Increasing Clients', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        plot_path = results_dir / f'{model_type}_increasing_clients.png'
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f'\n  Saved plot to {plot_path}')

    print(f'\n{"="*60}')
    print(f'Results saved to {results_dir}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
