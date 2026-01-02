# Machine Learning Module

This module provides a flexible framework for training AML detection models in different settings: centralized, federated, and isolated learning.

## Overview

The ML module supports:
- **Multiple model architectures**: Neural networks (MLP, GNN) and tree-based models (Decision Trees, Random Forest, Gradient Boosting)
- **Three training paradigms**: Centralized, Federated, and Isolated learning
- **Graph neural networks**: GCN, GAT, and GraphSAGE for transaction network analysis
- **Hyperparameter optimization**: Bayesian optimization using Optuna
- **Flexible client-server architecture**: Modular design for different learning scenarios

## Directory Structure

```
src/ml/
├── clients/              # Client implementations for different model types
│   ├── torch_client.py           # PyTorch tabular models
│   ├── torch_geometric_client.py # PyTorch Geometric GNN models
│   └── sklearn_client.py         # Scikit-learn models
├── models/               # Model architectures
│   ├── torch_models.py           # LogisticRegressor, MLP
│   ├── gnn_models.py             # GCN, GAT, GraphSAGE
│   ├── sklearn_models.py         # DecisionTree, RandomForest, GradientBoosting
│   └── losses.py                 # Custom loss functions
├── servers/              # Server implementations for federated learning
│   └── torch_server.py           # FedAvg server for PyTorch models
└── training/             # Training scripts
    ├── centralized.py            # Centralized training
    ├── federated.py              # Federated training
    ├── isolated.py               # Isolated training
    └── hyperparameter_tuning.py  # Bayesian hyperparameter optimization
```

## Components

### Clients

Clients handle model training and evaluation. Each client type corresponds to a specific model framework:

- **TorchClient**: For tabular neural networks (LogisticRegressor, MLP)
- **TorchGeometricClient**: For graph neural networks (GCN, GAT, GraphSAGE)
- **SklearnClient**: For scikit-learn models (DecisionTree, RandomForest, GradientBoosting)

### Models

#### Neural Networks
- **LogisticRegressor**: Simple linear model for binary classification
- **MLP**: Multi-layer perceptron with configurable hidden layers

#### Graph Neural Networks
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **GraphSAGE**: Graph Sample and Aggregate

#### Tree-Based Models
- **DecisionTreeClassifier**: Single decision tree
- **RandomForestClassifier**: Ensemble of decision trees
- **GradientBoostingClassifier**: Gradient boosting ensemble

### Servers

- **TorchServer**: Implements Federated Averaging (FedAvg) for PyTorch models

## Training Modes

### 1. Centralized Training

All data is combined and trained on a single model.

```bash
uv run python -m src.ml.training.centralized \
  --config experiments/10k_accounts/config/models.yaml \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --seed 42 \
  --device cpu
```

**Use case**: Baseline performance when data can be shared

### 2. Federated Training

Multiple clients train collaboratively with a central server coordinating model aggregation.

```bash
uv run python -m src.ml.training.federated \
  --config experiments/10k_accounts/config/models.yaml \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --server_type TorchServer \
  --seed 42 \
  --device cpu \
  --n_workers 2
```

**Use case**: Privacy-preserving learning across multiple institutions (e.g., banks)

### 3. Isolated Training

Each client trains independently on their own data without sharing.

```bash
uv run python -m src.ml.training.isolated \
  --config experiments/10k_accounts/config/models.yaml \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --seed 42 \
  --device cpu
```

**Use case**: Baseline for comparison when no collaboration is possible

### 4. Hyperparameter Optimization

Bayesian optimization using Optuna to find optimal hyperparameters.

```bash
uv run python -m src.ml.training.hyperparameter_tuning \
  --config experiments/10k_accounts/config/models.yaml \
  --setting centralized \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --server_type TorchServer \
  --n_trials 10 \
  --seed 42 \
  --device cpu
```

**Settings**: `centralized`, `federated`, or `isolated`

### Extending the Search Space

The Bayesian optimization system supports optimizing **any** hyperparameter from the `default` section. Simply add the parameter to the `search_space` with appropriate type and bounds.

**Supported parameter types:**

1. **float**: Continuous values (e.g., learning rate, dropout)
   ```yaml
   lr: {type: float, low: 0.0001, high: 0.1, log: true}
   dropout: {type: float, low: 0.0, high: 0.5}
   ```

2. **int**: Integer values (e.g., layer counts, hidden dimensions)
   ```yaml
   n_conv_layers: {type: int, low: 1, high: 5}
   hidden_dim: {type: int, low: 100, high: 300}
   ```

3. **categorical**: Discrete choices (e.g., optimizers, loss functions)
   ```yaml
   optimizer: {type: categorical, values: [Adam, SGD, AdamW]}
   criterion: {type: categorical, values: [BCEWithLogitsLoss, FocalLoss]}
   ```

**Example - Extending beyond current search space:**

```yaml
GraphSAGE:
  default:
    lr: 0.0001
    dropout: 0.9
    n_conv_layers: 3
    hidden_dim: 100
    batch_size: 512         # Currently fixed
    n_rounds: 300           # Currently fixed
    lr_patience: 10         # Currently fixed
    optimizer: Adam         # Currently fixed

  search_space:
    # Currently optimized parameters:
    lr: {type: float, low: 0.0001, high: 0.1, log: true}
    n_conv_layers: {type: int, low: 1, high: 5}
    hidden_dim: {type: int, low: 100, high: 300}
    dropout: {type: float, low: 0.2, high: 0.9}

    # Add any additional parameters to optimize:
    batch_size: {type: int, low: 128, high: 1024}
    lr_patience: {type: int, low: 5, high: 20}
    n_warmup_rounds: {type: int, low: 50, high: 200}
    optimizer: {type: categorical, values: [Adam, SGD, AdamW]}
```

**Note**: More parameters = larger search space = longer optimization time. Start with the most impactful hyperparameters (typically learning rate, architecture dimensions, and dropout).

## Configuration

Model configurations are stored in `experiments/{experiment_name}/config/models.yaml`.

### Convention Over Configuration

The framework uses **auto-discovery** to eliminate hardcoded paths:
- **Data paths** are automatically constructed from experiment root
- **Clients** are auto-discovered from `preprocessed/clients/` directory
- **Results** paths are derived from config location

Each model configuration contains **only hyperparameters**:

```yaml
# Everything is auto-discovered by convention:
# - Experiment root: Detected from config path (experiments/{name}/config/models.yaml)
# - Clients: Discovered from preprocessed/clients/ directory

# Optional: Explicitly list clients (if not specified, auto-discover from preprocessed/clients/)
# clients:
#   - swedbank
#   - nordea

GraphSAGE:
  default:
    # Base hyperparameters used across all settings (centralized, federated, isolated)
    client_type: TorchGeometricClient
    server_type: TorchServer
    device: cuda:0
    valset_size: 0.2
    testset_size: 0.2
    lr_patience: 10
    es_patience: 300
    n_rounds: 300
    n_warmup_rounds: 100
    eval_every: 5
    optimizer: Adam
    criterion: BCEWithLogitsLoss
    lr: 0.0001
    dropout: 0.9
    input_dim: 94
    n_conv_layers: 3
    hidden_dim: 100
    output_dim: 1

  search_space:
    # Define which hyperparameters to optimize and their search ranges
    # You can add ANY parameter from the default section here
    lr: {type: float, low: 0.0001, high: 0.1, log: true}
    n_conv_layers: {type: int, low: 1, high: 5}
    hidden_dim: {type: int, low: 100, high: 300}
    dropout: {type: float, low: 0.2, high: 0.9}
```

### Path Conventions

Data paths follow standard conventions. The `{experiment_root}` is auto-detected from the config file location:
- Config at: `experiments/10k_accounts/config/models.yaml`
- Experiment root: `experiments/10k_accounts/`

**Centralized:**
- Nodes: `{experiment_root}/preprocessed/centralized/trainset_nodes.parquet`
- Edges: `{experiment_root}/preprocessed/centralized/trainset_edges.parquet` (GNNs only)

**Federated/Isolated:**
- Nodes: `{experiment_root}/preprocessed/clients/{client}/trainset_nodes.parquet`
- Edges: `{experiment_root}/preprocessed/clients/{client}/trainset_edges.parquet` (GNNs only)

**Results:**
- `{experiment_root}/results/{setting}/{model}/results.pkl`
- `{experiment_root}/results/{setting}/{model}/hp_study.db` (hyperparameter tuning)

### Client Auto-Discovery

Clients are automatically discovered by scanning `{experiment_root}/preprocessed/clients/`:

```bash
experiments/10k_accounts/preprocessed/clients/
├── nordea/
│   ├── trainset_nodes.parquet
│   └── trainset_edges.parquet
└── swedbank/
    ├── trainset_nodes.parquet
    └── trainset_edges.parquet
```

This automatically enables both `nordea` and `swedbank` for federated/isolated training.

### Per-Client Overrides

In isolated training, you can override hyperparameters per client:

```yaml
isolated:
  # Base hyperparameters for all clients
  lr: 0.0001
  n_conv_layers: 2

  # Per-client overrides (optional)
  clients:
    swedbank:
      lr: 0.0002  # Override for swedbank only
    # nordea inherits base settings
```

## Command Line Arguments

All training scripts support these arguments:

- `--config`: Path to models config file (required)
- `--model_type`: Model class name (e.g., GraphSAGE, MLP, DecisionTree)
- `--client_type`: Client class (TorchClient, TorchGeometricClient, SklearnClient)
- `--server_type`: Server class for federated learning (TorchServer)
- `--seed`: Random seed for reproducibility
- `--device`: Device to use (cpu, cuda:0, etc.)
- `--n_workers`: Number of parallel workers (federated/isolated only)
- `--results`: Path to save results (auto-generated if not specified)

## Results

Training results are saved as pickle files containing metrics for each client:

```python
{
  'client_id': {
    'trainset': {
      'average_precision': [0.95, 0.96, ...],  # per round
      'loss': [0.5, 0.4, ...],
      # ... other metrics
    },
    'valset': { ... },
    'testset': { ... }
  }
}
```

Results are automatically saved to:
- Centralized: `experiments/{exp}/results/centralized/{model}/results.pkl`
- Federated: `experiments/{exp}/results/federated/{model}/results.pkl`
- Isolated: `experiments/{exp}/results/isolated/{model}/results.pkl`

Hyperparameter optimization results:
- Best trials: `experiments/{exp}/results/{setting}/{model}/best_trials.txt`
- Optuna database: `experiments/{exp}/results/{setting}/{model}/hp_study.db`

## Model-Client Compatibility

| Model | Client Type |
|-------|-------------|
| LogisticRegressor | TorchClient |
| MLP | TorchClient |
| GCN | TorchGeometricClient |
| GAT | TorchGeometricClient |
| GraphSAGE | TorchGeometricClient |
| DecisionTreeClassifier | SklearnClient |
| RandomForestClassifier | SklearnClient |
| GradientBoostingClassifier | SklearnClient |

## Example Workflow

1. **Generate and preprocess data**:
```bash
# Generate transaction data
uv run python scripts/generate.py --config experiments/10k_accounts/config/data.yaml

# Preprocess features
uv run python scripts/preprocess.py --config experiments/10k_accounts/config/preprocessing.yaml
```

2. **Train models in different settings**:
```bash
# Centralized baseline
uv run python -m src.ml.training.centralized \
  --config experiments/10k_accounts/config/models.yaml \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --device cpu

# Federated learning (2 banks)
uv run python -m src.ml.training.federated \
  --config experiments/10k_accounts/config/models.yaml \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --server_type TorchServer \
  --device cpu \
  --n_workers 2

# Isolated learning
uv run python -m src.ml.training.isolated \
  --config experiments/10k_accounts/config/models.yaml \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --device cpu
```

3. **Optimize hyperparameters**:
```bash
uv run python -m src.ml.training.hyperparameter_tuning \
  --config experiments/10k_accounts/config/models.yaml \
  --setting centralized \
  --model_type GraphSAGE \
  --client_type TorchGeometricClient \
  --server_type TorchServer \
  --n_trials 10 \
  --device cpu
```

4. **Analyze results**:
```python
import pickle

with open('experiments/10k_accounts/results/federated/GraphSAGE/results.pkl', 'rb') as f:
    results = pickle.load(f)

for client_id, metrics in results.items():
    print(f'{client_id}: Test AP = {metrics["testset"]["average_precision"][-1]:.4f}')
```

## Notes

- GNN models require both node and edge data files
- Tabular models only need node data files
- The framework automatically handles train/val/test splits based on mask columns in the data
- Early stopping is implemented based on validation performance
- Learning rate scheduling with patience is supported
