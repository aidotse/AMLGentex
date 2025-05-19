import argparse
import multiprocessing as mp
import os
import pickle
import yaml
from flib import servers, clients, models
from flib.train import centralized, federated, isolated, HyperparamTuner

def main():
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '12_banks'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--model_types', nargs='+', help='Types of models to train.', default=['LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE']) # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=4)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seeds', nargs='+', help='Seeds.', default=[42]) # 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized']) # 'centralized', 'federated', 'isolated'
    parser.add_argument('--trainset_sizes', nargs='+', help='Fractions of trainset to use.', default=[0.6, 0.18, 0.054, 0.0162, 0.00486, 0.001458]) # 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    print()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for setting in args.settings:
        
        os.makedirs(f'{args.results_dir}/{setting}', exist_ok=True)
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
                train_avg_precision = 0.0
                val_avg_precision = 0.0
                test_avg_precision = 0.0
                
                for seed in args.seeds:
                    
                    print(f'\n---------\nSEED = {seed}\n---------\n')
                    # params = config[model_type]['default'] | config[model_type]['centralized']
                    # params['trainset_size'] = trainset_size
                    # search_space = config[model_type]['search_space']
                    # hyperparamtuner = HyperparamTuner(
                    #     study_name = 'hp_study',
                    #     obj_fn = centralized,
                    #     params = params,
                    #     search_space = search_space,
                    #     Client = getattr(clients, config[model_type]['default']['client_type']),
                    #     Model = getattr(models, model_type),
                    #     seed = seed,
                    #     n_workers = args.n_workers,
                    # )
                    # best_trials = hyperparamtuner.optimize(n_trials=30)
                    # for param in best_trials[-1].params:
                    #     config[param] = best_trials[-1].params[param]
                    
                    kwargs = config[model_type]['default'] | config[model_type][setting]
                    kwargs['trainset_size'] = trainset_size
                
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
                    results_dir = os.path.join(args.results_dir, setting, model_type, f'trainset_size_{trainset_size}', f'seed_{seed}')
                    os.makedirs(results_dir, exist_ok=True)
                    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                        pickle.dump(results, f)
                    print(f'Saved results to {results_dir}/results.pkl\n')
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

