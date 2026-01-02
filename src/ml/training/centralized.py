from typing import Any, Dict

def centralized(seed: int, Client: Any, Model: Any, **kwargs) -> Dict[str, Dict]:
    
    client = Client(id='cen', seed=seed, Model=Model, **kwargs)
    results = {client.id:  client.run(**kwargs)}
    
    return results

if __name__ == '__main__':

    import argparse
    import os
    import pickle
    import time
    from pathlib import Path
    from src.ml import clients, models
    from src.utils.config import load_training_config

    EXPERIMENT = '3_banks_homo_mid'
    CLIENT_TYPE = 'TorchGeometricClient' # 'TorchClient', 'TorchGeometricClient'
    MODEL_TYPE = 'GraphSAGE' # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to models config file.', default=f'experiments/{EXPERIMENT}/config/models.yaml')
    parser.add_argument('--results', type=str, help='Path to results file.', default=None)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--client_type', type=str, help='Client class.', default=CLIENT_TYPE)
    parser.add_argument('--model_type', type=str, help='Model class.', default=MODEL_TYPE)
    parser.add_argument('--device', type=str, help='Device to use (cpu, cuda:0, etc.)', default=None)
    args = parser.parse_args()

    # Derive experiment name from config path if results not specified
    if args.results is None:
        config_path = Path(args.config)
        experiment_name = config_path.parent.parent.name  # Extract experiment name from config path
        args.results = f'experiments/{experiment_name}/results/centralized/{args.model_type}/results.pkl'

    t = time.time()

    # Load config with auto-discovery of paths
    kwargs = load_training_config(
        args.config,
        args.model_type,
        setting='centralized',
        client_type=args.client_type
    )
    if args.device is not None:
        kwargs['device'] = args.device
    Client = getattr(clients, args.client_type)
    Model = getattr(models, args.model_type)
    results = centralized(seed=args.seed, Client=Client, Model=Model, **kwargs)

    t = time.time() - t

    print('Done')
    print(f'Exec time: {t:.2f}s')
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {args.results}\n')
