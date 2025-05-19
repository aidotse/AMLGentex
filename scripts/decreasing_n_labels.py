import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import yaml
from flib import servers, clients, models
from flib.train import centralized, federated, isolated
from flib.utils import get_optimal_params


def main():
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '12_banks_homo_easy'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--model_types', nargs='+', help='Types of models to train.', default=['LogisticRegressor', 'GCN']) # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=4)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seeds', nargs='+', help='Seeds.', default=[42]) # 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized']) # 'centralized', 'federated', 'isolated'
    parser.add_argument('--trainset_sizes', nargs='+', help='Fractions of trainset to use.', default=[0.6, 0.06, 0.006, 0.0006]) # 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01
    parser.add_argument('--use_optimal_params', type=bool, help='Read the parameters from Optuna.', default=False)
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    print()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for setting in args.settings:
        
        with open(f'{args.results_dir}/{setting}/decreasing_trainsize_train_avg_precision.csv', 'w') as f:
            header = 'fraction,'+','.join(args.model_types)+'\n'
            f.write(header)
        with open(f'{args.results_dir}/{setting}/decreasing_trainsize_val_avg_precision.csv', 'w') as f:
            header = 'fraction,'+','.join(args.model_types)+'\n'
            f.write(header)
        with open(f'{args.results_dir}/{setting}/decreasing_trainsize_test_avg_precision.csv', 'w') as f:
            header = 'fraction,'+','.join(args.model_types)+'\n'
            f.write(header)
        
        for trainset_size in args.trainset_sizes:
            train_avg_precisions = [] 
            val_avg_precisions = [] 
            test_avg_precisions = [] 
            for model_type in args.model_types:
                kwargs = config[model_type]['default'] | config[model_type][setting]
                kwargs['trainset_size'] = trainset_size
                kwargs = get_optimal_params(kwargs, f"{args.results_dir}/federated/{model_type}") if args.use_optimal_params else kwargs
                train_avg_precision = 0.0
                val_avg_precision = 0.0
                test_avg_precision = 0.0
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
                    for id in results:
                        train_avg_precision += results[id]['trainset']['average_precision'][-1] / len(results) / len(args.seeds)
                        val_avg_precision += results[id]['valset']['average_precision'][-1] / len(results) / len(args.seeds)
                        test_avg_precision += results[id]['testset']['average_precision'][-1] / len(results) / len(args.seeds)
                train_avg_precisions.append(train_avg_precision) 
                val_avg_precisions.append(val_avg_precision) 
                test_avg_precisions.append(test_avg_precision) 
            with open(f'{args.results_dir}/{setting}/decreasing_trainsize_train_avg_precision.csv', 'a') as f:
                row = f'{trainset_size},' + ','.join([str(x) for x in train_avg_precisions]) + '\n'
                f.write(row)
            with open(f'{args.results_dir}/{setting}/decreasing_trainsize_val_avg_precision.csv', 'a') as f:
                row = f'{trainset_size},' + ','.join([str(x) for x in val_avg_precisions]) + '\n'
                f.write(row)
            with open(f'{args.results_dir}/{setting}/decreasing_trainsize_test_avg_precision.csv', 'a') as f:
                row = f'{trainset_size},' + ','.join([str(x) for x in test_avg_precisions]) + '\n'
                f.write(row)

if __name__ == '__main__':
    main()

