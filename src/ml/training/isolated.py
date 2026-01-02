import multiprocessing as mp
import numpy as np
from typing import Any, Dict, List

def run_clients(clients: List[Any], params: List[Dict]):
    ids = []
    results = []
    for client, param in zip(clients, params):
        ids.append(client.id)
        results.append(client.run(**param))
    return ids, results

def isolated(seed: int, Client: Any, Model: Any, n_workers: int = None, **kwargs):
    
    client_configs = kwargs.pop('clients')
    clients = []
    client_params = []
    for id, config in client_configs.items():
        client_params.append(kwargs | config)
        clients.append(Client(id=id, seed=seed, Model=Model, **client_params[-1]))
    
    if n_workers is None:
        n_workers = len(clients)

    with mp.Pool(n_workers) as p:
        client_splits = np.array_split(clients, n_workers)
        param_splits = np.array_split(client_params, n_workers)
        results = p.starmap(run_clients, [(client_split, param_split) for client_split, param_split in zip(client_splits, param_splits)]) 
    results = {id: res for result in results for id, res in zip(result[0], result[1])}
    
    return results

if __name__ == '__main__':
    
    import argparse
    from src.ml import clients, models
    from src.utils.config import load_training_config, get_client_config
    import os
    import pickle
    import time
    from pathlib import Path

    mp.set_start_method('spawn', force=True)

    EXPERIMENT = '3_banks_homo_mid'
    CLIENT_TYPE = 'TorchGeometricClient' # 'TorchClient', 'TorchGeometricClient'
    MODEL_TYPE = 'GCN' # 'LogisticRegressor', 'MLP', 'GCN'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to models config file.', default=f'experiments/{EXPERIMENT}/config/models.yaml')
    parser.add_argument('--results', type=str, help='Path to results file.', default=None)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of workers. Default is number of clients.', default=None)
    parser.add_argument('--client_type', type=str, help='Client class.', default=CLIENT_TYPE)
    parser.add_argument('--model_type', type=str, help='Model class.', default=MODEL_TYPE)
    parser.add_argument('--device', type=str, help='Device to use (cpu, cuda:0, etc.)', default=None)
    args = parser.parse_args()

    # Derive experiment name from config path if results not specified
    if args.results is None:
        config_path = Path(args.config)
        experiment_name = config_path.parent.parent.name  # Extract experiment name from config path
        args.results = f'experiments/{experiment_name}/results/isolated/{args.model_type}/results.pkl'

    t = time.time()

    # Load config with auto-discovery of paths and clients
    kwargs = load_training_config(
        args.config,
        args.model_type,
        setting='isolated',
        client_type=args.client_type
    )
    if args.device is not None:
        kwargs['device'] = args.device

    # Apply per-client overrides if present
    if '_client_overrides' in kwargs:
        base_config = kwargs.copy()
        for client_id in kwargs['clients'].keys():
            client_config = get_client_config(base_config, client_id)
            kwargs['clients'][client_id].update(client_config)
    Client = getattr(clients, args.client_type)
    Model = getattr(models, args.model_type)
    results = isolated(seed=args.seed, Client=Client, Model=Model, n_workers=None, **kwargs)
    
    t = time.time() - t
    
    print('Done')
    print(f'Exec time: {t:.2f}s')
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {args.results}\n')
