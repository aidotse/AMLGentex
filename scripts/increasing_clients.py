import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pickle
import yaml
from flib import servers, clients, models
from flib.train import federated
from flib.utils import get_optimal_params


def main():
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '12_banks'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--model_types', nargs='+', help='Types of models to train.', default=['GCN', 'GAT', 'GraphSAGE']) # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=4)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seeds', nargs='+', help='Seeds.', default=[42]) # 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
    parser.add_argument('--use_optimal_params', type=bool, help='Read the parameters from Optuna.', default=False)
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    print()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for model_type in args.model_types:
        client_ids = list(config[model_type]['federated']['clients'].keys())
        train_avg_precisions = [] 
        val_avg_precisions = [] 
        test_avg_precisions = [] 
        for i in range(len(client_ids)):
            kwargs = config[model_type]['default'] | config[model_type]['federated']
            kwargs['clients'] = {k: v for k, v in kwargs['clients'].items() if k in client_ids[:i+1]}
            # kwargs = get_optimal_params(kwargs, f"{args.results_dir}/federated/{model_type}") if args.use_optimal_params else kwargs
            train_avg_precision = 0.0
            val_avg_precision = 0.0
            test_avg_precision = 0.0
            for seed in args.seeds:
                results = federated(
                    seed = seed, 
                    Server = getattr(servers, config[model_type]['default']['server_type']),
                    Client = getattr(clients, config[model_type]['default']['client_type']),
                    Model = getattr(models, model_type), 
                    n_workers = args.n_workers, 
                    **kwargs
                )
                results_dir = os.path.join(args.results_dir, 'federated', model_type, f'{i+1}_clients', f'seed_{seed}')
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                    pickle.dump(results, f)
                print(f'Saved results to {results_dir}/results.pkl\n')
                for id in results:
                    train_avg_precision += results[id]['trainset']['average_precision'][-1] / len(results) / len(args.seeds)
                    val_avg_precision += results[id]['valset']['average_precision'][-1] / len(results) / len(args.seeds)
                    test_avg_precision += results[id]['testset']['average_precision'][-1] / len(results) / len(args.seeds)
            print(f'n clients: {i+1}, avg precision: trainset: {train_avg_precision}, valset: {val_avg_precision}, testset: {test_avg_precision}')
            train_avg_precisions.append(train_avg_precision) 
            val_avg_precisions.append(val_avg_precision) 
            test_avg_precisions.append(test_avg_precision) 
        fig = plt.figure()
        plt.plot(client_ids, train_avg_precisions, label='trainset')
        plt.plot(client_ids, val_avg_precisions, label='valset')
        plt.plot(client_ids, test_avg_precisions, label='testset')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.xlabel('n clients')
        plt.ylabel('avg precision')
        plt.savefig(os.path.join(args.results_dir, f'{model_type}_increasing_clients.png'))
        

if __name__ == '__main__':
    main()

