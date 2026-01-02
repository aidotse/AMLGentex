import argparse
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_creation import DataGenerator
from src.feature_engineering import DataPreprocessor
from src.data_tuning import DataTuner
from time import time

def main():

    EXPERIMENT = 'template_experiment'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=False, default=f'experiments/{EXPERIMENT}')
    parser.add_argument('--num_trials_data', type=int, default=10, help='Number of data optimization trials')
    parser.add_argument('--num_trials_model', type=int, default=15, help='Number of model hyperparameter trials per data trial')
    parser.add_argument('--constraint_type', type=str, default='K', choices=['K', 'fpr', 'recall'],
                        help='Type of constraint: K (alert budget), fpr (max FPR), recall (min recall)')
    parser.add_argument('--constraint_value', type=float, default=100,
                        help='Value of constraint (e.g., K=100, alpha=0.01, min_recall=0.7)')
    parser.add_argument('--utility_metric', type=str, default='precision', choices=['precision', 'recall'],
                        help='Metric to optimize: precision or recall')
    parser.add_argument('--target', type=float, default=0.8,
                        help='Target value for utility metric')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-update-config', action='store_true', help='Skip updating data.yaml with best parameters after optimization')
    args = parser.parse_args()

    # Update paths to match new structure
    args.data_conf_file = f'{args.experiment_dir}/config/data.yaml'
    args.preprocessing_config = f'{args.experiment_dir}/config/preprocessing.yaml'
    args.models_config = f'{args.experiment_dir}/config/models.yaml'
    args.bo_dir = f'{args.experiment_dir}/data_tuning'

    # Convert to absolute paths
    if not os.path.isabs(args.data_conf_file):
        args.data_conf_file = os.path.abspath(args.data_conf_file)
    if not os.path.isabs(args.preprocessing_config):
        args.preprocessing_config = os.path.abspath(args.preprocessing_config)
    if not os.path.isabs(args.models_config):
        args.models_config = os.path.abspath(args.models_config)
    if not os.path.isabs(args.bo_dir):
        args.bo_dir = os.path.abspath(args.bo_dir)

    # Load configs with auto-discovery
    from src.utils.config import load_data_config, load_preprocessing_config

    # Load and process data config with auto-discovery
    data_config = load_data_config(args.data_conf_file)

    # Write back the processed config for generator to use
    with open(args.data_conf_file, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    generator = DataGenerator(args.data_conf_file)

    # Run spatial simulation once upfront (will be reused across all trials)
    print("Running spatial simulation (graph generation) once...")
    print("This graph will be reused across all optimization trials.\n")
    generator.run_spatial(force=False)  # Use existing if available

    preprocessing_config = load_preprocessing_config(args.preprocessing_config)
    preprocessor = DataPreprocessor(preprocessing_config)

    with open(args.models_config, 'r') as f:
        models_config = yaml.safe_load(f)

    # Combine configs for optimizer (it expects both preprocessing and model configs)
    combined_config = {
        'preprocess': preprocessing_config.copy(),
        **models_config  # Unpack model configs at top level
    }

    # Add preprocessed_data_dir for the optimizer to use
    combined_config['preprocess']['preprocessed_data_dir'] = os.path.join(args.bo_dir, 'preprocessed')

    # Print optimization setup
    print("="*60)
    print("Data Tuning Configuration")
    print("="*60)
    if args.constraint_type == 'K':
        print(f"Objective: Optimize {args.utility_metric} in top {int(args.constraint_value)} alerts")
        print(f"Target: {args.target:.1%} {args.utility_metric}")
    elif args.constraint_type == 'fpr':
        print(f"Objective: Optimize {args.utility_metric} at FPR ≤ {args.constraint_value}")
        print(f"Target: {args.target:.1%} {args.utility_metric}")
    elif args.constraint_type == 'recall':
        print(f"Objective: Optimize {args.utility_metric} at Recall ≥ {args.constraint_value}")
        print(f"Target: {args.target:.1%} {args.utility_metric}")
    print(f"Data trials: {args.num_trials_data}")
    print(f"Model trials per data trial: {args.num_trials_model}")
    print(f"Random seed: {args.seed}")
    print("="*60 + "\n")

    tuner = DataTuner(
        data_conf_file=args.data_conf_file,
        config=combined_config,
        generator=generator,
        preprocessor=preprocessor,
        target=args.target,
        constraint_type=args.constraint_type,
        constraint_value=args.constraint_value,
        utility_metric=args.utility_metric,
        model='DecisionTreeClassifier',
        bo_dir=args.bo_dir,
        seed=args.seed,
        num_trials_model=args.num_trials_model
    )

    # Tune the temporal sar parameters
    t = time()
    best_trials = tuner(args.num_trials_data)
    t = time() - t
    print(f'\nTotal execution time: {t/60:.1f} minutes\n')

    # Update config with best parameters (closest to origin in objective space)
    if not args.no_update_config and best_trials:
        import math
        # Find trial closest to origin (balances all objectives)
        best_trial = min(best_trials, key=lambda t: math.sqrt(sum(v**2 for v in t.values)))
        tuner.optimizer.update_config_with_trial(best_trial.number)

if __name__ == '__main__':
    main()