"""
Experiment: Decreasing Number of Labels

Tests model performance with progressively smaller fractions of labeled training data.
Simulates scenarios with limited labeled examples (e.g., few confirmed SAR cases).

Usage:
    uv run python scripts/simulations/decreasing_n_labels.py \
        --experiment_dir experiments/10k_accounts \
        --trainset_sizes 0.6 0.06 0.006 0.0006 \
        --model_types DecisionTreeClassifier GCN \
        --training_regime centralized \
        --seeds 42 43 44

Output:
    experiments/<name>/results/<regime>/trainset_size_<fraction>/<model>/<seed>/results.pkl
    experiments/<name>/results/<regime>/decreasing_trainsize_train_avg_precision.csv
    experiments/<name>/results/<regime>/decreasing_trainsize_val_avg_precision.csv
    experiments/<name>/results/<regime>/decreasing_trainsize_test_avg_precision.csv
"""

import argparse
import multiprocessing as mp
import os
import pickle
import yaml
from pathlib import Path

from src.ml import servers, clients, models
from src.ml.training import centralized, federated, isolated
from src.utils import get_optimal_params


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Test effect of reducing labeled training data')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--trainset_sizes', nargs='+', type=float,
                        default=[0.6, 0.06, 0.006, 0.0006],
                        help='Fractions of trainset to use')
    parser.add_argument('--model_types', nargs='+',
                        default=['DecisionTreeClassifier', 'RandomForestClassifier', 'GCN'],
                        help='Models to train')
    parser.add_argument('--training_regime', type=str, default='centralized',
                        choices=['centralized', 'federated', 'isolated'],
                        help='Training setting')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='Random seeds for reproducibility')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers for multiprocessing')
    parser.add_argument('--use_optimal_params', action='store_true',
                        help='Use optimized hyperparameters from Optuna')
    parser.add_argument('--save_seed_results', action='store_true',
                        help='Save individual pickle files for each seed/trainset_size')
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

    results_dir = experiment_dir / 'results' / args.training_regime
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CSV files
    with open(results_dir / 'decreasing_trainsize_train_avg_precision.csv', 'w') as f:
        f.write('fraction,' + ','.join(args.model_types) + '\n')
    with open(results_dir / 'decreasing_trainsize_val_avg_precision.csv', 'w') as f:
        f.write('fraction,' + ','.join(args.model_types) + '\n')
    with open(results_dir / 'decreasing_trainsize_test_avg_precision.csv', 'w') as f:
        f.write('fraction,' + ','.join(args.model_types) + '\n')

    for trainset_size in args.trainset_sizes:
        print(f'\n{"="*60}')
        print(f'Trainset size: {trainset_size:.4f}')
        print(f'{"="*60}')

        train_avg_precisions = []
        val_avg_precisions = []
        test_avg_precisions = []

        for model_type in args.model_types:
            print(f'\n  Training {model_type}...')

            # Get model config
            kwargs = models_config[model_type]['default'].copy()
            kwargs['trainset_size'] = trainset_size

            # Get optimal params if requested
            if args.use_optimal_params:
                optuna_dir = results_dir / model_type
                kwargs = get_optimal_params(kwargs, str(optuna_dir))

            # Auto-discover data paths (convention over configuration)
            preprocessed_dir = experiment_dir / 'preprocessed'
            if args.training_regime == 'centralized':
                kwargs['trainset'] = str(preprocessed_dir / 'centralized' / 'trainset_nodes.parquet')
                kwargs['testset'] = str(preprocessed_dir / 'centralized' / 'testset_nodes.parquet')
                kwargs['trainset_nodes'] = str(preprocessed_dir / 'centralized' / 'trainset_nodes.parquet')
                kwargs['trainset_edges'] = str(preprocessed_dir / 'centralized' / 'trainset_edges.parquet')
                kwargs['testset_nodes'] = str(preprocessed_dir / 'centralized' / 'testset_nodes.parquet')
                kwargs['testset_edges'] = str(preprocessed_dir / 'centralized' / 'testset_edges.parquet')
            else:
                # For federated/isolated, auto-discover clients from directory
                clients_dir = preprocessed_dir / 'clients'
                client_ids = [d.name for d in clients_dir.iterdir() if d.is_dir()]
                kwargs['clients'] = {}
                for client_id in client_ids:
                    client_path = clients_dir / client_id
                    kwargs['clients'][client_id] = {
                        'trainset': str(client_path / 'trainset_nodes.parquet'),
                        'testset': str(client_path / 'testset_nodes.parquet'),
                        'trainset_nodes': str(client_path / 'trainset_nodes.parquet'),
                        'trainset_edges': str(client_path / 'trainset_edges.parquet'),
                        'testset_nodes': str(client_path / 'testset_nodes.parquet'),
                        'testset_edges': str(client_path / 'testset_edges.parquet'),
                    }

            train_avg_precision = 0.0
            val_avg_precision = 0.0
            test_avg_precision = 0.0

            for seed in args.seeds:
                print(f'    Seed {seed}...')

                # Run training based on regime
                if args.training_regime == 'centralized':
                    results = centralized(
                        seed=seed,
                        Client=getattr(clients, kwargs['client_type']),
                        Model=getattr(models, model_type),
                        **kwargs
                    )
                elif args.training_regime == 'federated':
                    results = federated(
                        seed=seed,
                        Server=getattr(servers, kwargs['server_type']),
                        Client=getattr(clients, kwargs['client_type']),
                        Model=getattr(models, model_type),
                        n_workers=args.n_workers,
                        **kwargs
                    )
                elif args.training_regime == 'isolated':
                    results = isolated(
                        seed=seed,
                        Server=getattr(servers, kwargs['server_type']),
                        Client=getattr(clients, kwargs['client_type']),
                        Model=getattr(models, model_type),
                        n_workers=args.n_workers,
                        **kwargs
                    )

                # Save seed results if requested
                if args.save_seed_results:
                    seed_results_dir = results_dir / model_type / f'trainset_size_{trainset_size}' / f'seed_{seed}'
                    seed_results_dir.mkdir(parents=True, exist_ok=True)
                    with open(seed_results_dir / 'results.pkl', 'wb') as f:
                        pickle.dump(results, f)
                    print(f'    Saved to {seed_results_dir}/results.pkl')

                # Aggregate metrics across clients/seeds
                for id in results:
                    train_avg_precision += results[id]['trainset']['average_precision'][-1] / len(results) / len(args.seeds)
                    val_avg_precision += results[id]['valset']['average_precision'][-1] / len(results) / len(args.seeds)
                    test_avg_precision += results[id]['testset']['average_precision'][-1] / len(results) / len(args.seeds)

            train_avg_precisions.append(train_avg_precision)
            val_avg_precisions.append(val_avg_precision)
            test_avg_precisions.append(test_avg_precision)

        # Write results for this trainset size
        with open(results_dir / 'decreasing_trainsize_train_avg_precision.csv', 'a') as f:
            row = f'{trainset_size},' + ','.join([str(x) for x in train_avg_precisions]) + '\n'
            f.write(row)
        with open(results_dir / 'decreasing_trainsize_val_avg_precision.csv', 'a') as f:
            row = f'{trainset_size},' + ','.join([str(x) for x in val_avg_precisions]) + '\n'
            f.write(row)
        with open(results_dir / 'decreasing_trainsize_test_avg_precision.csv', 'a') as f:
            row = f'{trainset_size},' + ','.join([str(x) for x in test_avg_precisions]) + '\n'
            f.write(row)

    print(f'\n{"="*60}')
    print(f'Results saved to {results_dir}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
