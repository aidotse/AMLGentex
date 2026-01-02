import optuna
from src.utils import set_random_seed 
from typing import Any, Dict, List

class HyperparamTuner():
    def __init__(self, study_name: str, obj_fn: Any, params: Dict, search_space: Dict, Client: Any, Model: Any, Server: Any = None, seed: int = 42, n_workers: int = None, storage: str = None, metrics: List[str] = ['average_precision'], verbose: bool = True):
        self.study_name = study_name
        self.obj_fn = obj_fn
        self.seed = seed
        self.n_workers = n_workers
        self.storage = storage
        self.Server = Server
        self.Client = Client
        self.Model = Model
        self.params = params
        self.search_space = search_space
        self.metrics = metrics
        self.verbose = verbose

    def objective(self, trial: optuna.Trial):
        kwargs = {}
        for param in self.search_space:
            if self.search_space[param]['type'] == 'categorical':
                kwargs[param] = trial.suggest_categorical(param, self.search_space[param]['values'])
            elif self.search_space[param]['type'] == 'int':
                kwargs[param] = trial.suggest_int(param, self.search_space[param]['low'], self.search_space[param]['high'])
            elif self.search_space[param]['type'] == 'float':
                kwargs[param] = trial.suggest_float(param, self.search_space[param]['low'], self.search_space[param]['high'], log=self.search_space[param].get('log', False))
            else:
                kwargs[param] = self.search_space[param]
        kwargs = self.params | kwargs
        results = self.obj_fn(seed=self.seed, Server=self.Server, Client=self.Client, Model=self.Model, n_workers=self.n_workers, **kwargs)
        
        # Extract constrained utility metric for data tuning
        if self.params.get('save_utility_metric', False) is True:
            utility_metric = 0.0
            for client in results:
                utility_metric += results[client].get('utility_metric') / len(results)
            self.utility_metric = utility_metric

        if self.params.get('save_feature_importances_error', False) is True:
            feature_importances_error = 0.0
            for client in results:
                feature_importances_error += results[client].get('feature_importances_error') / len(results)
            self.feature_importances_error = feature_importances_error
        
        rets = [0.0] * len(self.metrics)
        for client in results:
            for i, metric in enumerate(self.metrics):
                rets[i] += results[client]['valset'][metric][-1] / len(results)
        return tuple(rets)

    def optimize(self, n_trials=10):
        set_random_seed(self.seed)
        study = optuna.create_study(storage=self.storage, sampler=optuna.samplers.TPESampler(seed=self.seed, multivariate=True), study_name=self.study_name, direction='maximize', load_if_exists=True, pruner=optuna.pruners.HyperbandPruner())
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=self.verbose)
        return study.best_trials

if __name__ == '__main__':
    
    import argparse
    from src.ml import training
    import multiprocessing as mp
    import os
    import time
    import yaml
    from pathlib import Path
    from src.ml import servers, clients, models
    from src.utils.config import load_training_config

    mp.set_start_method('spawn', force=True)

    EXPERIMENT = '3_banks_homo_easy'
    SETTING = 'federated' # 'centralized', 'federated', 'isolated'
    SERVER_TYPE = 'TorchServer'
    CLIENT_TYPE = 'TorchGeometricClient' # 'TorchClient', 'TorchGeometricClient'
    MODEL_TYPE = 'GCN' # 'LogisticRegressor', 'MLP', 'GCN'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to models config file.', default=f'experiments/{EXPERIMENT}/config/models.yaml')
    parser.add_argument('--study_name', type=str, help='Name of study.', default='hp_study')
    parser.add_argument('--setting', type=str, help='Name of ojective function. Can be "centralized", "federated" or "isolated".', default=SETTING)
    parser.add_argument('--client_type', type=str, help='Client class.', default=CLIENT_TYPE)
    parser.add_argument('--model_type', type=str, help='Model class.', default=MODEL_TYPE)
    parser.add_argument('--server_type', type=str, help='Server class.', default=SERVER_TYPE)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=None)
    parser.add_argument('--n_trials', type=int, help='Number of trials.', default=2)
    parser.add_argument('--results_dir', type=str, help='Path to directory for storage and result.', default=None)
    parser.add_argument('--device', type=str, help='Device to use (cpu, cuda:0, etc.)', default=None)
    args = parser.parse_args()

    # Derive experiment name from config path if results_dir not specified
    if args.results_dir is None:
        config_path = Path(args.config)
        experiment_name = config_path.parent.parent.name  # Extract experiment name from config path
        args.results_dir = f'experiments/{experiment_name}/results/{args.setting}/{args.model_type}'

    t = time.time()

    # Load config with auto-discovery of paths and clients
    params = load_training_config(
        args.config,
        args.model_type,
        setting=args.setting,
        client_type=args.client_type
    )
    if args.device is not None:
        params['device'] = args.device

    # Load search space separately
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    search_space = config[args.model_type]['search_space']
    os.makedirs(args.results_dir, exist_ok=True)
    storage = 'sqlite:///' + os.path.join(args.results_dir, 'hp_study.db')
    
    hyperparamtuner = HyperparamTuner(
        study_name = args.study_name,
        obj_fn = getattr(training, args.setting),
        params = params,
        search_space = search_space,
        Client = getattr(clients, args.client_type),
        Model = getattr(models, args.model_type),
        Server = getattr(servers, args.server_type),
        seed = args.seed,
        n_workers = args.n_workers,
        storage = storage
    )
    best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
    
    t = time.time() - t
    
    print('Done')
    print(f'Exec time: {t:.2f}s')
    best_trials_file = os.path.join(args.results_dir, 'best_trials.txt')
    with open(best_trials_file, 'w') as f:
        for trial in best_trials:
            print(f'\ntrial: {trial.number}')
            f.write(f'\ntrial: {trial.number}\n')
            print(f'values: {trial.values}')
            f.write(f'values: {trial.values}\n')
            for param in trial.params:
                f.write(f'{param}: {trial.params[param]}\n')
                print(f'{param}: {trial.params[param]}')
    print(f'Saved results to {best_trials_file}\n')
