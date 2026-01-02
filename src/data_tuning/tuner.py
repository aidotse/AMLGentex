from src.data_tuning.optimizer import Optimizer

class DataTuner:
    def __init__(self, data_conf_file, config, generator, preprocessor, target,
                 constraint_type, constraint_value, utility_metric,
                 model, bo_dir, seed, num_trials_model, verbose=False):
        """
        Data tuning wrapper.

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
            bo_dir: Directory for Bayesian optimization results
            seed: Random seed
            num_trials_model: Number of hyperparameter optimization trials per data trial
            verbose: Whether to print detailed progress

        Examples:
            # Precision@K: Optimize data to achieve 80% precision in top 100 alerts
            DataTuner(..., target=0.8, constraint_type='K', constraint_value=100, utility_metric='precision')

            # Recall at FPR: Optimize data to achieve 70% recall at FPR≤1%
            DataTuner(..., target=0.7, constraint_type='fpr', constraint_value=0.01, utility_metric='recall')

            # Precision at Recall: Optimize data to achieve 50% precision at 70% recall
            DataTuner(..., target=0.5, constraint_type='recall', constraint_value=0.7, utility_metric='precision')
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
        self.bo_dir = bo_dir
        self.seed = seed
        self.num_trials_model = num_trials_model
        self.verbose = verbose

    def __call__(self, n_trials):

        self.optimizer = Optimizer(
            data_conf_file=self.data_conf_file,
            config=self.config,
            generator=self.generator,
            preprocessor=self.preprocessor,
            target=self.target,
            constraint_type=self.constraint_type,
            constraint_value=self.constraint_value,
            utility_metric=self.utility_metric,
            model=self.model,
            bo_dir=self.bo_dir,
            seed=self.seed,
            num_trials_model=self.num_trials_model,
            verbose=self.verbose
        )
        best_trials = self.optimizer.optimize(n_trials=n_trials)

        if self.verbose:
            for trial in best_trials:
                print(f'\ntrial: {trial.number}')
                print(f'values: {trial.values}')
                for param in trial.params:
                    print(f'{param}: {trial.params[param]}')

        return best_trials