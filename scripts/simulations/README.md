# Simulation Scripts

This directory contains specialized experiment scripts for the AMLGentex paper. These scripts run systematic studies to evaluate model robustness under different conditions.

---

## Overview

All simulation scripts follow the same pattern:
1. Load experiment configuration from `experiments/<name>/config/`
2. Vary a specific parameter (overlap, training data size, number of clients)
3. Train multiple models with multiple seeds
4. Save detailed results and aggregate statistics

---

## Available Simulations

### 1. `changing_overlap.py` - Transductive vs Inductive Learning

Tests the effect of train/test temporal overlap on model performance.

**What it varies:** Temporal offset between training and test windows
- **High overlap** (transductive): Test accounts were seen during training
- **Low overlap** (inductive): Test accounts are completely new

**Usage:**
```bash
uv run python scripts/simulations/changing_overlap.py \
    --experiment_dir experiments/10k_accounts \
    --overlap_steps 1 7 14 21 28 35 42 49 56 \
    --model_types DecisionTreeClassifier GCN \
    --seeds 42 43 44
```

**Parameters:**
- `--overlap_steps`: Step offsets for test window (0=full overlap, 56=no overlap for 112-day window)
- `--model_types`: Models to train
- `--seeds`: Random seeds for reproducibility

**Output:**
```
experiments/<name>/results/centralized/
├── 0.982_overlap/  # High overlap (1 step offset)
│   ├── train_avg_precision.csv
│   ├── val_avg_precision.csv
│   ├── test_avg_precision.csv
│   ├── DecisionTreeClassifier/
│   │   ├── seed_42/results.pkl
│   │   ├── seed_43/results.pkl
│   │   └── seed_44/results.pkl
│   └── GCN/
│       └── ...
├── 0.875_overlap/  # Medium overlap (7 step offset)
└── 0.500_overlap/  # Low overlap (28 step offset)
```

---

### 2. `decreasing_n_labels.py` - Limited Labeled Data

Tests model performance with progressively smaller fractions of labeled training data.

**What it varies:** Amount of labeled training data
- Simulates scenarios with few confirmed SAR cases
- Tests model robustness to label scarcity

**Usage:**
```bash
uv run python scripts/simulations/decreasing_n_labels.py \
    --experiment_dir experiments/10k_accounts \
    --trainset_sizes 0.6 0.06 0.006 0.0006 \
    --model_types DecisionTreeClassifier GCN \
    --training_regime centralized \
    --seeds 42 43 44
```

**Parameters:**
- `--trainset_sizes`: Fractions of training data to use (e.g., 0.6 = 60%, 0.006 = 0.6%)
- `--training_regime`: `centralized`, `federated`, or `isolated`
- `--use_optimal_params`: Use hyperparameters from Bayesian optimization
- `--save_seed_results`: Save individual pickle files per seed

**Output:**
```
experiments/<name>/results/<regime>/
├── decreasing_trainsize_train_avg_precision.csv
├── decreasing_trainsize_val_avg_precision.csv
├── decreasing_trainsize_test_avg_precision.csv
└── DecisionTreeClassifier/
    ├── trainset_size_0.6/
    │   ├── seed_42/results.pkl
    │   └── ...
    ├── trainset_size_0.06/
    └── trainset_size_0.006/
```

---

### 3. `increasing_clients.py` - Federated Collaboration Benefits

Tests federated learning performance as the number of collaborating banks increases.

**What it varies:** Number of clients participating in federated learning
- Shows benefit of data collaboration vs isolated learning
- Demonstrates federated learning scalability

**Usage:**
```bash
uv run python scripts/simulations/increasing_clients.py \
    --experiment_dir experiments/10k_accounts \
    --model_types GCN GAT GraphSAGE \
    --seeds 42 43 44
```

**Parameters:**
- `--model_types`: Models to train (typically GNNs for federated learning)
- `--seeds`: Random seeds for reproducibility
- `--n_workers`: Number of parallel workers
- `--use_optimal_params`: Use hyperparameters from Bayesian optimization

**Output:**
```
experiments/<name>/results/federated/
├── GCN_increasing_clients.png  # Plot showing performance vs n_clients
├── GAT_increasing_clients.png
├── GraphSAGE_increasing_clients.png
└── GCN/
    ├── 1_clients/
    │   ├── seed_42/results.pkl
    │   └── ...
    ├── 2_clients/
    ├── 3_clients/
    └── ...
```

---

## Common Workflow

### Running All Simulations for an Experiment

```bash
# 1. Generate and preprocess data
uv run python scripts/generate.py --conf_file experiments/my_experiment/config/data.yaml
uv run python scripts/preprocess.py --conf_file experiments/my_experiment/config/preprocessing.yaml

# 2. Run simulations
uv run python scripts/simulations/changing_overlap.py \
    --experiment_dir experiments/my_experiment \
    --model_types DecisionTreeClassifier GCN

uv run python scripts/simulations/decreasing_n_labels.py \
    --experiment_dir experiments/my_experiment \
    --trainset_sizes 0.6 0.06 0.006 \
    --training_regime centralized

uv run python scripts/simulations/increasing_clients.py \
    --experiment_dir experiments/my_experiment \
    --model_types GCN GAT GraphSAGE
```

---

## Output Format

All simulations produce:

1. **Individual results**: `results.pkl` files per model/seed/configuration
   - Full training history (metrics per round)
   - Final predictions
   - Model checkpoints (if enabled)

2. **Aggregated CSV files**: Average metrics across seeds/clients
   - Easy to load into plotting tools
   - Format: `<parameter_value>,<model1>,<model2>,...`

3. **Plots** (for `increasing_clients.py`): Performance vs parameter curves

---

## Notes

- **Reproducibility**: Always specify `--seeds` for consistent results
- **Parallelization**: Use `--n_workers` to speed up federated/isolated training
- **Hyperparameters**: Use `--use_optimal_params` to load tuned hyperparameters
- **Storage**: Results can be large - each simulation may generate 100s of MB
- **Convention over Configuration**: Paths are auto-discovered from experiment structure

---

## Extending

To create a new simulation:

1. Copy an existing script as a template
2. Modify the parameter sweep logic
3. Update output paths to avoid conflicts
4. Document in this README

Example parameter sweeps:
- Varying noise levels in features
- Different train/val/test splits
- Changing network topologies
- Testing with concept drift
