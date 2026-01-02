import argparse
import multiprocessing as mp
import os
import time
import yaml
from src.ml import servers, clients, models
from src.ml.training import centralized, federated, HyperparamTuner

def main():
    
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '12_banks_homo_easy'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--model_types', nargs='+', help='Types of models to train.', default=['GCN']) # 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    parser.add_argument('--n_trials', type=int, help='Number of trials.', default=90)
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=4)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results_incomplet_labels')
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized']) # 'centralized', 'federated', 'isolated'
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for model_type in args.model_types:
        if 'centralized' in args.settings:
            print(f'\nTuning hyperparameters for {model_type} in a centralized setting.')
            t = time.time()
            os.makedirs(os.path.join(args.results_dir, f'centralized/{model_type}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'centralized/{model_type}/hp_study.db')
            params = config[model_type]['default'] | config[model_type]['centralized']
            search_space = config[model_type]['search_space']
            hyperparamtuner = HyperparamTuner(
                study_name = 'hp_study',
                obj_fn = centralized,
                params = params,
                search_space = search_space,
                Client = getattr(clients, config[model_type]['default']['client_type']),
                Model = getattr(models, model_type),
                seed = args.seed,
                n_workers = args.n_workers,
                storage = storage
            )
            best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            best_trials_file = os.path.join(args.results_dir, f'centralized/{model_type}/best_trials.txt')
            with open(best_trials_file, 'w') as f:
                for trial in best_trials:
                    print(f'\ntrial: {trial.number}')
                    f.write(f'\ntrial: {trial.number}\n')
                    print(f'values: {trial.values}')
                    f.write(f'values: {trial.values}\n')
                    for param in trial.params:
                        f.write(f'{param}: {trial.params[param]}\n')
                        print(f'{param}: {trial.params[param]}')
            print()
        
        if 'federated' in args.settings and 'federated' in config[model_type]:
            print(f'\nTuning hyperparameters for {model_type} in a federated setting.')
            t = time.time()
            os.makedirs(os.path.join(args.results_dir, f'federated/{model_type}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'federated/{model_type}/hp_study.db')
            params = config[model_type]['default'] | config[model_type]['federated']
            search_space = config[model_type]['search_space']
            hyperparamtuner = HyperparamTuner(
                study_name = 'hp_study',
                obj_fn = federated,
                params = params,
                search_space = search_space,
                Server = getattr(servers, config[model_type]['default']['server_type']),
                Client = getattr(clients, config[model_type]['default']['client_type']),
                Model = getattr(models, model_type),
                seed = args.seed,
                n_workers = args.n_workers,
                storage = storage
            )
            best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            best_trials_file = os.path.join(args.results_dir, f'federated/{model_type}/best_trials.txt')
            with open(best_trials_file, 'w') as f:
                for trial in best_trials:
                    print(f'\ntrial: {trial.number}')
                    f.write(f'\ntrial: {trial.number}\n')
                    print(f'values: {trial.values}')
                    f.write(f'values: {trial.values}\n')
                    for param in trial.params:
                        f.write(f'{param}: {trial.params[param]}\n')
                        print(f'{param}: {trial.params[param]}')
            print()
            
        if 'isolated' in args.settings:
            print(f'\nTuning hyperparameters for {model_type} in a isolated setting.')
            t = time.time()
            for client in config[model_type]['isolated']['clients']:
                os.makedirs(os.path.join(args.results_dir, f'isolated/{model_type}/clients/{client}'), exist_ok=True)
                storage = 'sqlite:///' + os.path.join(args.results_dir, f'isolated/{model_type}/clients/{client}/hp_study.db')
                params = config[model_type]['default'] | config[model_type]['isolated']['clients'][client]
                search_space = config[model_type]['search_space']
                hyperparamtuner = HyperparamTuner(
                    study_name = 'hp_study',
                    obj_fn = centralized, # OBS: using centralised here but only with data from one client
                    params = params,
                    search_space = search_space,
                    Client = getattr(clients, config[model_type]['default']['client_type']),
                    Model = getattr(models, model_type),
                    seed = args.seed,
                    n_workers = args.n_workers,
                    storage = storage
                )
                best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
                best_trials_file = os.path.join(args.results_dir, f'isolated/{model_type}/clients/{client}/best_trials.txt')
                with open(best_trials_file, 'w') as f:
                    for trial in best_trials:
                        print(f'\ntrial: {trial.number}')
                        f.write(f'\ntrial: {trial.number}\n')
                        print(f'values: {trial.values}')
                        f.write(f'values: {trial.values}\n')
                        for param in trial.params:
                            f.write(f'{param}: {trial.params[param]}\n')
                            print(f'{param}: {trial.params[param]}')
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s\n')

if __name__ == '__main__':
    main()

