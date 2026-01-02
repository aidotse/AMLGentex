"""
Experiment: Changing Train/Test Overlap

Tests the effect of varying train/test temporal overlap on model performance.
This simulates transductive (high overlap) vs inductive (low overlap) learning.

Usage:
    uv run python scripts/simulations/changing_overlap.py \
        --experiment_dir experiments/10k_accounts \
        --overlap_steps 1 7 14 21 28 35 42 49 56 \
        --model_types DecisionTreeClassifier GCN \
        --seeds 42 43 44

Output:
    experiments/<name>/results/centralized/<overlap>_overlap/<model>/<seed>/results.pkl
    experiments/<name>/results/centralized/<overlap>_overlap/train_avg_precision.csv
    experiments/<name>/results/centralized/<overlap>_overlap/val_avg_precision.csv
    experiments/<name>/results/centralized/<overlap>_overlap/test_avg_precision.csv
"""

import argparse
import numpy as np
import os
import pickle
import yaml
from pathlib import Path

from src.ml import clients, models
from src.feature_engineering import DataPreprocessor
from src.ml.training import centralized


def main():
    parser = argparse.ArgumentParser(description='Test effect of train/test temporal overlap')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--overlap_steps', nargs='+', type=int, default=[1, 7, 14, 28, 56],
                        help='Step offsets for test set (0=full overlap, 56=no overlap for 112-day window)')
    parser.add_argument('--model_types', nargs='+',
                        default=['DecisionTreeClassifier', 'RandomForestClassifier', 'GCN'],
                        help='Models to train')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                        help='Random seeds for reproducibility')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers for multiprocessing')
    args = parser.parse_args()

    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    print()

    # Load configs
    experiment_dir = Path(args.experiment_dir)
    config_dir = experiment_dir / 'config'

    with open(config_dir / 'preprocessing.yaml', 'r') as f:
        preprocess_config = yaml.safe_load(f)

    with open(config_dir / 'models.yaml', 'r') as f:
        models_config = yaml.safe_load(f)

    results_dir = experiment_dir / 'results' / 'centralized'

    # Total time window
    total_steps = preprocess_config['train_end_step'] - preprocess_config['train_start_step']

    for step in args.overlap_steps:
        overlap_fraction = 1 - step / total_steps
        print(f'\nTrain/test overlap: {overlap_fraction:.3f} ({step} step offset)')

        # Preprocess with new test window
        preprocessed_dir = experiment_dir / 'preprocessed' / f'centralized_{overlap_fraction:.3f}_overlap'

        if not preprocessed_dir.exists():
            print(f'  Preprocessing data with {step} step offset...')
            temp_config = preprocess_config.copy()
            temp_config['test_start_step'] = temp_config['train_start_step'] + step
            temp_config['test_end_step'] = temp_config['train_end_step'] + step
            temp_config['preprocessed_data_dir'] = str(preprocessed_dir)

            preprocessor = DataPreprocessor(temp_config)
            datasets = preprocessor(temp_config['raw_data_file'])

            # Save datasets
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            for name, dataset in datasets.items():
                output_path = preprocessed_dir / f'{name}.parquet'
                dataset.to_parquet(output_path, index=False)
            print(f'  Saved preprocessed data to {preprocessed_dir}')
        else:
            print(f'  Using existing preprocessed data: {preprocessed_dir}')

        # Create results directory
        overlap_results_dir = results_dir / f'{overlap_fraction:.3f}_overlap'
        overlap_results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV files
        with open(overlap_results_dir / 'train_avg_precision.csv', 'w') as f:
            f.write('round,' + ','.join(args.model_types) + '\n')
        with open(overlap_results_dir / 'val_avg_precision.csv', 'w') as f:
            f.write('round,' + ','.join(args.model_types) + '\n')
        with open(overlap_results_dir / 'test_avg_precision.csv', 'w') as f:
            f.write('round,' + ','.join(args.model_types) + '\n')

        train_avg_precisions = []
        val_avg_precisions = []
        test_avg_precisions = []

        for model_type in args.model_types:
            print(f'\n  Training {model_type}...')

            # Get model config
            kwargs = models_config[model_type]['default'].copy()

            # Update paths to use overlap-specific preprocessed data
            kwargs['trainset'] = str(preprocessed_dir / 'trainset_nodes.parquet')
            kwargs['testset'] = str(preprocessed_dir / 'testset_nodes.parquet')
            kwargs['trainset_nodes'] = str(preprocessed_dir / 'trainset_nodes.parquet')
            kwargs['trainset_edges'] = str(preprocessed_dir / 'trainset_edges.parquet')
            kwargs['testset_nodes'] = str(preprocessed_dir / 'testset_nodes.parquet')
            kwargs['testset_edges'] = str(preprocessed_dir / 'testset_edges.parquet')

            train_avg_precision = []
            val_avg_precision = []
            test_avg_precision = []

            for seed in args.seeds:
                print(f'    Seed {seed}...')
                results = centralized(
                    seed=seed,
                    Client=getattr(clients, kwargs['client_type']),
                    Model=getattr(models, model_type),
                    **kwargs
                )

                # Save individual seed results
                seed_results_dir = overlap_results_dir / model_type / f'seed_{seed}'
                seed_results_dir.mkdir(parents=True, exist_ok=True)
                with open(seed_results_dir / 'results.pkl', 'wb') as f:
                    pickle.dump(results, f)
                print(f'    Saved to {seed_results_dir}/results.pkl')

                # Collect metrics
                train_values = []
                val_values = []
                test_values = []
                for id in results:
                    train_values.append(results[id]['trainset']['average_precision'])
                    val_values.append(results[id]['valset']['average_precision'])
                    test_values.append(results[id]['testset']['average_precision'])

                train_avg_precision.append(np.array(train_values).mean(axis=0))
                val_avg_precision.append(np.array(val_values).mean(axis=0))
                test_avg_precision.append(np.array(test_values).mean(axis=0))

            train_avg_precisions.append(np.array(train_avg_precision).mean(axis=0))
            val_avg_precisions.append(np.array(val_avg_precision).mean(axis=0))
            test_avg_precisions.append(np.array(test_avg_precision).mean(axis=0))

        # Convert to arrays
        train_avg_precisions = np.array(train_avg_precisions)
        val_avg_precisions = np.array(val_avg_precisions)
        test_avg_precisions = np.array(test_avg_precisions)

        # Write aggregated results
        with open(overlap_results_dir / 'train_avg_precision.csv', 'a') as f:
            rounds = np.arange(0, 301, 1)
            for round_num, train_avg_precision in zip(rounds, train_avg_precisions.T):
                row = f'{round_num},' + ','.join([str(x) for x in train_avg_precision]) + '\n'
                f.write(row)

        with open(overlap_results_dir / 'val_avg_precision.csv', 'a') as f:
            rounds = np.arange(0, 301, 5)
            for round_num, val_avg_precision in zip(rounds, val_avg_precisions.T):
                row = f'{round_num},' + ','.join([str(x) for x in val_avg_precision]) + '\n'
                f.write(row)

        with open(overlap_results_dir / 'test_avg_precision.csv', 'a') as f:
            rounds = [300]
            for round_num, test_avg_precision in zip(rounds, test_avg_precisions.T):
                row = f'{round_num},' + ','.join([str(x) for x in test_avg_precision]) + '\n'
                f.write(row)

        print(f'\n  Results saved to {overlap_results_dir}/')


if __name__ == '__main__':
    main()
