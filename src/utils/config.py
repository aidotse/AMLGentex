"""
Configuration loader with path auto-discovery and convention over configuration.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml


def discover_clients(preprocessed_dir: Path) -> List[str]:
    """
    Auto-discover client names from preprocessed directory structure.

    Args:
        preprocessed_dir: Path to preprocessed data directory

    Returns:
        List of client names found in preprocessed/clients/
    """
    clients_dir = preprocessed_dir / "clients"
    if not clients_dir.exists():
        return []

    clients = [
        d.name for d in clients_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]
    return sorted(clients)


def build_data_paths(
    experiment_root: Path,
    client_type: str,
    setting: str = "centralized",
    clients: Optional[List[str]] = None
) -> Dict:
    """
    Build data paths based on conventions.

    Args:
        experiment_root: Root directory of experiment
        client_type: Type of client (determines if edges are needed)
        setting: Training setting (centralized, federated, isolated)
        clients: List of client names (auto-discovered if None)

    Returns:
        Dictionary with data paths
    """
    preprocessed_dir = experiment_root / "preprocessed"

    # Auto-discover clients if not provided
    if clients is None:
        clients = discover_clients(preprocessed_dir)

    # Determine if we need edges (for GNN models)
    needs_edges = client_type == "TorchGeometricClient"

    paths = {}

    if setting == "centralized":
        paths["trainset_nodes"] = str(preprocessed_dir / "centralized" / "trainset_nodes.parquet")
        if needs_edges:
            paths["trainset_edges"] = str(preprocessed_dir / "centralized" / "trainset_edges.parquet")

    elif setting in ["federated", "isolated"]:
        paths["clients"] = {}
        for client in clients:
            client_dir = preprocessed_dir / "clients" / client
            paths["clients"][client] = {
                "trainset_nodes": str(client_dir / "trainset_nodes.parquet")
            }
            if needs_edges:
                paths["clients"][client]["trainset_edges"] = str(client_dir / "trainset_edges.parquet")

    return paths


def load_training_config(
    config_path: str,
    model_type: str,
    setting: str,
    client_type: str
) -> Dict:
    """
    Load training configuration with auto-discovery and path construction.

    Convention over configuration:
    - Experiment root: Auto-detected from config path (../../ from config file)
    - Clients: Auto-discovered from {experiment_root}/preprocessed/clients/
    - Data paths: Auto-constructed based on setting and client_type

    Args:
        config_path: Path to config YAML file (typically experiments/{name}/config/models.yaml)
        model_type: Name of model (e.g., "GraphSAGE")
        setting: Training setting (centralized, federated, isolated)
        client_type: Type of client (TorchClient, TorchGeometricClient, SklearnClient)

    Returns:
        Complete configuration dictionary with auto-constructed paths
    """
    config_path = Path(config_path)

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Auto-detect experiment root from config path
    # Config at: experiments/{name}/config/models.yaml
    # Root is: experiments/{name}/
    if 'experiment' in config and 'root' in config['experiment']:
        # Optional override for non-standard directory structures
        experiment_root = Path(config['experiment']['root'])
    else:
        # Standard convention: config is in experiments/{exp_name}/config/
        experiment_root = config_path.parent.parent

    # Get or discover clients
    clients = config.get('clients', None)
    if clients is None:
        clients = discover_clients(experiment_root / "preprocessed")

    # Get model config
    if model_type not in config:
        raise ValueError(f"Model '{model_type}' not found in config")

    model_config = config[model_type]

    # Build configuration by merging default -> setting
    kwargs = model_config.get('default', {}).copy()

    if setting in model_config:
        setting_config = model_config[setting]

        # Handle per-client overrides in isolated setting
        if setting == "isolated" and "clients" in setting_config:
            # For isolated, we need to handle per-client configs differently
            # We'll merge them when creating individual clients
            client_overrides = setting_config.pop("clients", {})
            kwargs.update(setting_config)
            kwargs["_client_overrides"] = client_overrides
        else:
            kwargs.update(setting_config)

    # Build and add data paths
    data_paths = build_data_paths(experiment_root, client_type, setting, clients)
    kwargs.update(data_paths)

    return kwargs


def get_client_config(base_config: Dict, client_id: str) -> Dict:
    """
    Get configuration for a specific client in isolated setting.

    Args:
        base_config: Base configuration with potential overrides
        client_id: Client identifier

    Returns:
        Configuration for specific client
    """
    config = base_config.copy()

    # Apply client-specific overrides if present
    if "_client_overrides" in config:
        overrides = config.pop("_client_overrides")
        if client_id in overrides:
            config.update(overrides[client_id])

    return config


def load_data_config(config_path: str) -> Dict:
    """
    Load data generation configuration with auto-discovery and path construction.

    Convention over configuration:
    - Experiment root: Auto-detected from config path (../../ from config file)
    - All paths: Auto-constructed based on experiment root

    Args:
        config_path: Path to data config YAML file (typically experiments/{name}/config/data.yaml)

    Returns:
        Complete configuration dictionary with auto-constructed paths
    """
    config_path = Path(config_path)

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Auto-detect experiment root from config path
    # Config at: experiments/{name}/config/data.yaml
    # Root is: experiments/{name}/
    experiment_root = config_path.parent.parent

    # Auto-construct paths based on experiment root
    config['input']['directory'] = str(experiment_root / 'config')
    config['temporal']['directory'] = str(experiment_root / 'spatial')
    config['output']['directory'] = str(experiment_root / 'temporal')

    return config


def load_preprocessing_config(config_path: str) -> Dict:
    """
    Load preprocessing configuration with auto-discovery and path construction.

    Convention over configuration:
    - Experiment root: Auto-detected from config path (../../ from config file)
    - All paths: Auto-constructed based on experiment root

    Args:
        config_path: Path to preprocessing config YAML file (typically experiments/{name}/config/preprocessing.yaml)

    Returns:
        Complete configuration dictionary with auto-constructed paths
    """
    config_path = Path(config_path)

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Auto-detect experiment root from config path
    # Config at: experiments/{name}/config/preprocessing.yaml
    # Root is: experiments/{name}/
    experiment_root = config_path.parent.parent

    # Auto-construct paths based on experiment root
    config['raw_data_file'] = str(experiment_root / 'temporal' / 'tx_log.parquet')
    config['preprocessed_data_dir'] = str(experiment_root / 'preprocessed')

    return config
