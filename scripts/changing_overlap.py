import argparse
import numpy as np
import pandas as pd
import pickle
import os
import yaml

from typing import Dict

from flib import clients, models
from flib.preprocess import DataPreprocessor
from flib.train import centralized

def main():
    
    EXPERIMENT = '12_banks_homo_mid_dist_change'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--model_types', nargs='+', help='Types of models to train.', default=['LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE']) # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=4)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seeds', nargs='+', help='Seeds.', default=[43, 44, 45, 46, 47, 48, 49, 50, 51]) # 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for step in [1]: # range(7, 57, 7):
        
        print(f'\nTrain and test overlap: {1-step/56:.3f} %')
        
        if not os.path.exists(f'experiments/{EXPERIMENT}/data/preprocessed/centralized_{1-step/56:.3f}_overlap'):
            config['preprocess']['test_start_step'] = config['preprocess']['train_start_step'] + step
            config['preprocess']['test_end_step'] = config['preprocess']['train_end_step'] + step
            preprocessor = DataPreprocessor(config['preprocess'])
            datasets = preprocessor(config['preprocess']['raw_data_file'])
            os.makedirs(os.path.join(config['preprocess']['preprocessed_data_dir'], f'centralized_{1-step/56:.3f}_overlap'), exist_ok=True)
            for name, dataset in datasets.items():
                dataset.to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], f'centralized_{1-step/56:.3f}_overlap', name+'.csv'), index=False)
        os.makedirs(f'{args.results_dir}/centralized/{1-step/56:.3f}_overlap', exist_ok=True)
        with open(f'{args.results_dir}/centralized/{1-step/56:.3f}_overlap/train_avg_precision.csv', 'w') as f:
            header = 'round,'+','.join(args.model_types)+'\n'
            f.write(header)
        with open(f'{args.results_dir}/centralized/{1-step/56:.3f}_overlap/val_avg_precision.csv', 'w') as f:
            header = 'round,'+','.join(args.model_types)+'\n'
            f.write(header)
        with open(f'{args.results_dir}/centralized/{1-step/56:.3f}_overlap/test_avg_precision.csv', 'w') as f:
            header = 'round,'+','.join(args.model_types)+'\n'
            f.write(header)
        
        train_avg_precisions = [] 
        val_avg_precisions = [] 
        test_avg_precisions = [] 
        
        for model_type in args.model_types:
            
            config[model_type]['centralized']['trainset'] = f'experiments/{EXPERIMENT}/data/preprocessed/centralized_{1-step/56:.3f}_overlap/trainset_nodes.csv'
            config[model_type]['centralized']['testset'] = f'experiments/{EXPERIMENT}/data/preprocessed/centralized_{1-step/56:.3f}_overlap/testset_nodes.csv'
            config[model_type]['centralized']['trainset_nodes'] = f'experiments/{EXPERIMENT}/data/preprocessed/centralized_{1-step/56:.3f}_overlap/trainset_nodes.csv'
            config[model_type]['centralized']['trainset_edges'] = f'experiments/{EXPERIMENT}/data/preprocessed/centralized_{1-step/56:.3f}_overlap/trainset_edges.csv'
            config[model_type]['centralized']['testset_nodes'] = f'experiments/{EXPERIMENT}/data/preprocessed/centralized_{1-step/56:.3f}_overlap/testset_nodes.csv'
            config[model_type]['centralized']['testset_edges'] = f'experiments/{EXPERIMENT}/data/preprocessed/centralized_{1-step/56:.3f}_overlap/testset_edges.csv'
            
            kwargs = config[model_type]['default'] | config[model_type]['centralized']
            train_avg_precision = []
            val_avg_precision = []
            test_avg_precision = []
            for seed in args.seeds:
                results = centralized(
                    seed = seed, 
                    Client = getattr(clients, config[model_type]['default']['client_type']),
                    Model = getattr(models, model_type), 
                    **kwargs
                )
                results_dir = os.path.join(args.results_dir, 'centralized', model_type, f'{1-step/56:.3f}_overlap', f'seed_{seed}')
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                    pickle.dump(results, f)
                print(f'Saved results to {results_dir}/results.pkl\n')
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
        
        with open(f'{args.results_dir}/centralized/{1-step/56:.3f}_overlap/train_avg_precision.csv', 'a') as f:
            rounds = np.arange(0, 301, 1)
            for round, train_avg_precision in zip(rounds, train_avg_precisions.T):
                row = f'{round},' + ','.join([str(x) for x in train_avg_precision]) + '\n'    
                f.write(row)
        with open(f'{args.results_dir}/centralized/{1-step/56:.3f}_overlap/val_avg_precision.csv', 'a') as f:
            rounds = np.arange(0, 301, 5)
            for round, val_avg_precision in zip(rounds, val_avg_precisions.T):
                row = f'{round},' + ','.join([str(x) for x in val_avg_precision]) + '\n'    
                f.write(row)
        with open(f'{args.results_dir}/centralized/{1-step/56:.3f}_overlap/test_avg_precision.csv', 'a') as f:
            rounds = [300]
            for round, test_avg_precision in zip(rounds, test_avg_precisions.T):
                row = f'{round},' + ','.join([str(x) for x in test_avg_precision]) + '\n'    
                f.write(row)

if __name__ == '__main__':
    main()
