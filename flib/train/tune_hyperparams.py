import optuna
from flib.utils import set_random_seed 
from typing import Any, Dict, List

class HyperparamTuner():
    def __init__(self, study_name: str, obj_fn: Any, params: Dict, search_space: Dict, Client: Any, Model: Any, Server: Any = None, seed: int = 42, n_workers: int = None, storage: str = None, metrics: List[str] = ['average_precision']):
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
        
        if self.params.get('save_fpr', False) is True:
            fpr = 0.0
            for client in results:
                fpr += results[client].get('fpr') / len(results)
            self.fpr = fpr
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
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_trials

if __name__ == '__main__':
    
    import argparse
    import flib.train
    import multiprocessing as mp
    import os
    import time
    import yaml
    from flib import servers, clients, models
    
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '3_banks_homo_easy'
    SETTING = 'federated' # 'centralized', 'federated', 'isolated'
    SERVER_TYPE = 'TorchServer'
    CLIENT_TYPE = 'TorchGeometricClient' # 'TorchClient', 'TorchGeometricClient'
    MODEL_TYPE = 'GCN' # 'LogisticRegressor', 'MLP', 'GCN'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--study_name', type=str, help='Name of study.', default='hp_study')
    parser.add_argument('--setting', type=str, help='Name of ojective function. Can be "centralized", "federated" or "isolated".', default=SETTING)
    parser.add_argument('--client_type', type=str, help='Client class.', default=CLIENT_TYPE)
    parser.add_argument('--model_type', type=str, help='Model class.', default=MODEL_TYPE)
    parser.add_argument('--server_type', type=str, help='Server class.', default=SERVER_TYPE)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=None)
    parser.add_argument('--n_trials', type=int, help='Number of trials.', default=2)
    parser.add_argument('--results_dir', type=str, help='Path to directory for storage and result.', default=f'experiments/{EXPERIMENT}/results/{SETTING}/{MODEL_TYPE}')
    args = parser.parse_args()
    
    t = time.time()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    params = config[args.model_type]['default'] | config[args.model_type][args.setting]
    search_space = config[args.model_type]['search_space']
    os.makedirs(args.results_dir, exist_ok=True)
    storage = 'sqlite:///' + os.path.join(args.results_dir, 'hp_study.db')
    
    hyperparamtuner = HyperparamTuner(
        study_name = args.study_name,
        obj_fn = getattr(flib.train, args.setting),
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
