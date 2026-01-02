import optuna
from src.ml import clients, models
from src.ml.training.hyperparameter_tuning import HyperparamTuner
from src.ml.training.centralized import centralized
import matplotlib.pyplot as plt
import yaml
import os
import warnings
import logging
from tqdm.auto import tqdm as tqdm_auto

class Optimizer():
    def __init__(self, data_conf_file, config, generator, preprocessor, target:float,
                 constraint_type:str, constraint_value:float, utility_metric:str,
                 model:str='DecisionTreeClassifier', bank=None, bo_dir:str='tmp',
                 seed:int=0, num_trials_model:int=1, verbose:bool=False):
        """
        Data tuning optimizer.

        Args:
            data_conf_file: Path to data configuration file
            config: Combined preprocessing and model configuration
            generator: DataGenerator instance
            preprocessor: DataPreprocessor instance
            target: Target value for the utility metric
            constraint_type: Type of constraint - 'K' (alert budget), 'fpr' (max FPR), or 'recall' (min recall)
            constraint_value: Value of the constraint (e.g., K=100, alpha=0.01, min_recall=0.7)
            utility_metric: Metric to optimize - 'precision' or 'recall'
            model: Model type to use for evaluation
            bank: Specific bank to tune on (if multiple banks)
            bo_dir: Directory for Bayesian optimization results
            seed: Random seed
            num_trials_model: Number of hyperparameter optimization trials per data trial
            verbose: Whether to print detailed progress

        Examples:
            # Precision@K: Optimize data to achieve 80% precision in top 100 alerts
            Optimizer(..., target=0.8, constraint_type='K', constraint_value=100, utility_metric='precision')

            # Recall at FPR: Optimize data to achieve 70% recall at FPR≤1%
            Optimizer(..., target=0.7, constraint_type='fpr', constraint_value=0.01, utility_metric='recall')

            # Precision at Recall: Optimize data to achieve 50% precision at 70% recall
            Optimizer(..., target=0.5, constraint_type='recall', constraint_value=0.7, utility_metric='precision')
        """
        self.data_conf_file = data_conf_file
        self.config = config
        self.generator = generator
        self.preprocessor = preprocessor
        self.target = target
        self.constraint_type = constraint_type
        self.constraint_value = constraint_value
        self.utility_metric = utility_metric
        self.model = model
        self.bank = bank
        self.bo_dir = bo_dir
        self.seed = seed
        self.num_trials_model = num_trials_model
        self.verbose = verbose

        # Suppress warnings and optuna logging unless verbose
        if not verbose:
            warnings.filterwarnings('ignore')
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            # Set verbose=False on generator and preprocessor
            if hasattr(self.generator, 'verbose'):
                self.generator.verbose = False
            if hasattr(self.preprocessor, 'verbose'):
                self.preprocessor.verbose = False
    
    def objective(self, trial:optuna.Trial):
        with open(self.data_conf_file, 'r') as f:
            data_config = yaml.safe_load(f)

        # Update data generation parameters from optimization bounds
        updated_params = {}
        for k, v in data_config['optimisation_bounds'].items():
            lower = v[0]
            upper = v[1]
            if type(lower) is int:
                data_config['default'][k] = trial.suggest_int(k, lower, upper)
            elif type(lower) is float:
                data_config['default'][k] = trial.suggest_float(k, lower, upper)
            else:
                raise ValueError(f'Type {type(lower)} in optimisation bounds not recognised, use int or float.')
            updated_params[k] = data_config['default'][k]

        # Keep random seed fixed across trials for fair parameter comparison
        data_config['general']['random_seed'] = self.seed

        # Print trial parameters if verbose
        if self.verbose:
            print(f"\n[Trial {trial.number}] Testing parameters:")
            for k, v in updated_params.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        with open(self.data_conf_file, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

        # Clean up preprocessed files from previous trial to force regeneration
        self._cleanup_intermediate_files()

        # Run only temporal simulation (spatial graph is reused across trials)
        # The generator will reload config from self.conf_file in run_temporal()
        tx_log_file = self.generator(spatial=False)

        datasets = self.preprocessor(tx_log_file)

        banks = datasets['trainset_nodes']['bank'].unique()

        # Determine which bank to use for tuning
        if len(banks) > 1:
            # Multiple banks: pick one for tuning
            if self.bank is not None:
                # Use specified bank
                target_bank = self.bank
                if target_bank not in banks:
                    raise ValueError(f"Specified bank '{self.bank}' not found in generated data. Available banks: {list(banks)}")
            else:
                # Pick first bank by default
                target_bank = banks[0]
            if self.verbose:
                print(f"Multiple banks detected. Tuning on bank: {target_bank}")
        else:
            # Single bank: use all data (that one bank)
            target_bank = banks[0]
            if self.verbose:
                print(f"Single bank detected. Tuning on all data (bank: {target_bank})")

        # Save preprocessed data for the selected bank
        os.makedirs(self.config['preprocess']['preprocessed_data_dir'], exist_ok=True)
        df_nodes = datasets['trainset_nodes']
        df_nodes[df_nodes['bank'] == target_bank].to_parquet(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'trainset_nodes.parquet'), index=False)
        unique_nodes = df_nodes[df_nodes['bank'] == target_bank]['account'].unique()
        df_edges = datasets['trainset_edges']
        df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_parquet(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'trainset_edges.parquet'), index=False)
        df_nodes = datasets['valset_nodes']
        df_nodes[df_nodes['bank'] == target_bank].to_parquet(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'valset_nodes.parquet'), index=False)
        unique_nodes = df_nodes[df_nodes['bank'] == target_bank]['account'].unique()
        df_edges = datasets['valset_edges']
        df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_parquet(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'valset_edges.parquet'), index=False)
        df_nodes = datasets['testset_nodes']
        df_nodes[df_nodes['bank'] == target_bank].to_parquet(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'testset_nodes.parquet'), index=False)
        unique_nodes = df_nodes[df_nodes['bank'] == target_bank]['account'].unique()
        df_edges = datasets['testset_edges']
        df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_parquet(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'testset_edges.parquet'), index=False)

        # Update config with preprocessed data paths and utility metric parameters
        params = self.config[self.model]['default'].copy()
        params['trainset'] = os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'trainset_nodes.parquet')
        params['save_utility_metric'] = True
        params['constraint_type'] = self.constraint_type
        params['constraint_value'] = self.constraint_value
        params['utility_metric'] = self.utility_metric
        params['save_feature_importances_error'] = True

        # Run hyperparameter tuning on the selected bank (use same seed for fair comparison)
        storage = None
        search_space = self.config[self.model]['search_space']
        hyperparamtuner = HyperparamTuner(
            study_name = 'hp_study',
            obj_fn = centralized,
            params = params,
            search_space = search_space,
            Client = getattr(clients, self.config[self.model]['default']['client_type']),
            Model = getattr(models, self.model),
            seed = self.seed,  # Same seed across trials for controlled comparison
            n_workers = 1,
            storage = storage,
            verbose = self.verbose  # Pass through verbose flag to suppress nested progress bars
        )
        best_trials = hyperparamtuner.optimize(n_trials=self.num_trials_model)

        # Extract utility metric and calculate loss
        avg_utility = hyperparamtuner.utility_metric
        utility_loss = abs(avg_utility - self.target)

        avg_feature_importances_error = hyperparamtuner.feature_importances_error

        # Store results in trial for callback access
        trial.set_user_attr('utility_metric', avg_utility)
        trial.set_user_attr('utility_loss', utility_loss)

        if self.verbose:
            print(f"  → Achieved {self.utility_metric}: {avg_utility:.4f} (target: {self.target:.4f}, loss: {utility_loss:.4f})")
            print(f"  → Feature importance variance: {avg_feature_importances_error:.4f}\n")

        return utility_loss, avg_feature_importances_error
    
    def _cleanup_intermediate_files(self, verbose=False):
        """Remove intermediate preprocessed files created during optimization."""
        # Get preprocessed_data_dir - might be in preprocess config or needs to be constructed
        if 'preprocessed_data_dir' in self.config['preprocess']:
            preprocessed_dir = self.config['preprocess']['preprocessed_data_dir']
        else:
            # If not provided, skip cleanup (file was likely not created yet)
            return
        intermediate_files = [
            'trainset_nodes.parquet',
            'trainset_edges.parquet',
            'valset_nodes.parquet',
            'valset_edges.parquet',
            'testset_nodes.parquet',
            'testset_edges.parquet'
        ]

        removed_files = []
        for filename in intermediate_files:
            filepath = os.path.join(preprocessed_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                removed_files.append(filename)

        if removed_files and verbose:
            print(f'\nCleaned up {len(removed_files)} intermediate files from {preprocessed_dir}')

    def optimize(self, n_trials:int=10):
        # Store all BO results in bo_dir (e.g., experiments/{name}/data_tuning/)
        os.makedirs(self.bo_dir, exist_ok=True)
        storage = 'sqlite:///' + os.path.join(self.bo_dir, 'data_tuning_study.db')
        study = optuna.create_study(storage=storage, sampler=optuna.samplers.TPESampler(multivariate=True), study_name='data_tuning_study', directions=['minimize', 'minimize'], load_if_exists=True, pruner=optuna.pruners.HyperbandPruner())

        # Print trial results as they complete
        if not self.verbose:
            trial_count = [0]  # Use list to allow modification in nested function

            def callback(study, trial):
                trial_count[0] += 1
                utility = trial.user_attrs.get('utility_metric', 0.0)
                utility_loss = trial.user_attrs.get('utility_loss', 0.0)
                print(f"Trial {trial_count[0]}/{n_trials}: {self.utility_metric}={utility:.3f}, loss={utility_loss:.3f}")

            study.optimize(self.objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
        else:
            study.optimize(self.objective, n_trials=n_trials, show_progress_bar=False)

        # Clean up intermediate preprocessed files (final cleanup with message)
        self._cleanup_intermediate_files(verbose=True)

        # Save pareto front plot
        # Format constraint string for plot
        if self.constraint_type == 'K':
            constraint_str = f"K={int(self.constraint_value)}"
        elif self.constraint_type == 'fpr':
            constraint_str = f"FPR≤{self.constraint_value}"
        elif self.constraint_type == 'recall':
            constraint_str = f"Recall≥{self.constraint_value}"
        else:
            constraint_str = self.constraint_type

        ax = optuna.visualization.matplotlib.plot_pareto_front(study, target_names=['utility_loss', 'importance_loss'])
        ax.set_xlabel(f'{self.utility_metric.capitalize()} Loss (|{self.utility_metric} - {self.target}|) at {constraint_str}', fontsize=12)
        ax.set_ylabel('Feature Importance Variance', fontsize=12)
        ax.set_title('Data Tuning: Pareto Front', fontsize=14)
        fig_path = os.path.join(self.bo_dir, 'pareto_front.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # Save best trials log
        log_path = os.path.join(self.bo_dir, 'best_trials.txt')
        with open(log_path, 'w') as f:
            for trial in study.best_trials:
                f.write(f'\ntrial: {trial.number}\n')
                f.write(f'values: {trial.values}\n')
                for param in trial.params:
                    f.write(f'{param}: {trial.params[param]}\n')

        print(f'\nData tuning results saved to: {self.bo_dir}')
        print(f'  - Database: data_tuning_study.db')
        print(f'  - Pareto front: pareto_front.png')
        print(f'  - Best trials: best_trials.txt')

        return study.best_trials

    def update_config_with_trial(self, trial_number):
        """
        Update data.yaml with parameters from a specific trial.

        Args:
            trial_number: Trial number to use for parameters
        """
        # Load study to get trial
        storage = 'sqlite:///' + os.path.join(self.bo_dir, 'data_tuning_study.db')
        study = optuna.load_study(storage=storage, study_name='data_tuning_study')

        # Find trial
        trial = None
        for t in study.trials:
            if t.number == trial_number:
                trial = t
                break

        if trial is None:
            raise ValueError(f"Trial {trial_number} not found in study")

        # Load current config and save old values
        with open(self.data_conf_file, 'r') as f:
            data_config = yaml.safe_load(f)

        old_values = {}
        for param in trial.params.keys():
            old_values[param] = data_config['default'].get(param)

        # Update with trial parameters
        for param, value in trial.params.items():
            data_config['default'][param] = value

        # Write back to file
        with open(self.data_conf_file, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

        print(f'\n{"="*60}')
        print(f'Updated {self.data_conf_file}')
        print(f'Using trial {trial_number}:')
        print(f'  FPR loss: {trial.values[0]:.4f}')
        print(f'  Feature importance loss: {trial.values[1]:.4f}')
        print(f'\nParameter changes:')
        for param, new_value in trial.params.items():
            old_value = old_values[param]
            if isinstance(new_value, float):
                print(f'  {param}: {old_value:.4f} → {new_value:.4f}')
            else:
                print(f'  {param}: {old_value} → {new_value}')
        print(f'{"="*60}\n')
    





