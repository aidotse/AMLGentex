import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import yaml
from src.ml import servers, clients, models
from src.ml.training import centralized, federated, isolated
from src.utils import get_optimal_params


def main():
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '12_banks_difficult'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--model_types', nargs='+', help='Types of models to train.', default=['LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE']) # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=4)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seeds', nargs='+', type=int, help='Seeding, supports multiple seeds.', default=[42])
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized']) # 'centralized', 'federated', 'isolated'
    parser.add_argument('--use_optimal_params', type=bool, help='Read the parameters from Optuna.', default=False)
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    print()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for setting in args.settings:
        
        with open(f'{args.results_dir}/{setting}/train_avg_precision.csv', 'w') as f:
            header = 'round,'+','.join(args.model_types)+'\n'
            f.write(header)
        with open(f'{args.results_dir}/{setting}/val_avg_precision.csv', 'w') as f:
            header = 'round,'+','.join(args.model_types)+'\n'
            f.write(header)
        with open(f'{args.results_dir}/{setting}/test_avg_precision.csv', 'w') as f:
            header = 'round,'+','.join(args.model_types)+'\n'
            f.write(header)
        
        train_avg_precisions = [] 
        val_avg_precisions = [] 
        test_avg_precisions = [] 
        for model_type in args.model_types:
            kwargs = config[model_type]['default'] | config[model_type][setting]
            kwargs = get_optimal_params(kwargs, f"{args.results_dir}/federated/{model_type}") if args.use_optimal_params else kwargs
            train_avg_precision = []
            val_avg_precision = []
            test_avg_precision = []
            for seed in args.seeds:
                if setting == 'centralized':
                    results = centralized(
                        seed = seed, 
                        Client = getattr(clients, config[model_type]['default']['client_type']),
                        Model = getattr(models, model_type), 
                        **kwargs
                    )
                elif setting == 'federated':
                    results = federated(
                        seed = seed, 
                        Server = getattr(servers, config[model_type]['default']['server_type']),
                        Client = getattr(clients, config[model_type]['default']['client_type']),
                        Model = getattr(models, model_type), 
                        n_workers = args.n_workers, 
                        **kwargs
                    )
                elif setting == 'isolated':
                    results = isolated(
                        seed = seed, 
                        Server = getattr(servers, config[model_type]['default']['server_type']),
                        Client = getattr(clients, config[model_type]['default']['client_type']),
                        Model = getattr(models, model_type), 
                        n_workers = args.n_workers, 
                        **kwargs
                    )
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
        train_avg_precisions = np.array(train_avg_precisions)
        val_avg_precisions = np.array(val_avg_precisions)
        test_avg_precisions = np.array(test_avg_precisions)
            
        with open(f'{args.results_dir}/{setting}/train_avg_precision.csv', 'a') as f:
            rounds = np.arange(0, 301, 1)
            for round, train_avg_precision in zip(rounds, train_avg_precisions.T):
                row = f'{round},' + ','.join([str(x) for x in train_avg_precision]) + '\n'    
                f.write(row)
        with open(f'{args.results_dir}/{setting}/val_avg_precision.csv', 'a') as f:
            rounds = np.arange(0, 301, 5)
            for round, val_avg_precision in zip(rounds, val_avg_precisions.T):
                row = f'{round},' + ','.join([str(x) for x in val_avg_precision]) + '\n'    
                f.write(row)
        with open(f'{args.results_dir}/{setting}/test_avg_precision.csv', 'a') as f:
            rounds = [300]
            for round, test_avg_precision in zip(rounds, test_avg_precisions.T):
                row = f'{round},' + ','.join([str(x) for x in test_avg_precision]) + '\n'    
                f.write(row)

if __name__ == '__main__':
    main()

