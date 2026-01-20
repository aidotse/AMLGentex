# Scripts Documentation

This directory contains scripts for the complete AML detection pipeline, from synthetic data generation to model training and results analysis.

## Table of Contents

- [Workflow Overview](#workflow-overview)
- [Core Pipeline Scripts](#core-pipeline-scripts)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Visualization Scripts](#visualization-scripts)
- [Specialized Simulations](#specialized-simulations)

---

## Workflow Overview

The typical workflow follows these steps:

```
1. generate.py       → Generate synthetic AML transaction data
2. preprocess.py     → Feature engineering and data splitting
3. train.py          → Train models (centralized/federated/isolated)
4. plot.py           → Visualize results
```

**Optional steps:**
- `tune_hyperparams.py` - Optimize model hyperparameters for a fixed dataset
- `tune_data.py` - Optimize data generation parameters (runs full pipeline internally)

---

## Core Pipeline Scripts

### 1. `generate.py` - Synthetic Data Generation

Generates synthetic AML transaction data including both spatial graph generation and temporal transaction simulation with configurable normal and suspicious (SAR) transaction patterns.

**Usage:**
```bash
uv run python scripts/generate.py --conf_file experiments/10k_accounts/config/data.yaml
```

**Arguments:**
- `--conf_file`: Path to data configuration file (default: `experiments/10k_accounts/config/data.yaml`)

**What it does:**
1. Generates degree distribution from `scale-free` parameters (if `degree.csv` doesn't exist)
2. Creates spatial transaction graph with normal patterns
3. Injects alert patterns (SAR)
4. Simulates temporal transactions over specified time steps

**Output:**
- `experiments/<name>/spatial/` - Spatial graph files
  - `accounts.csv` - Account attributes
  - `transactions.csv` - Graph edges
  - `alert_members.csv` - SAR pattern membership
- `experiments/<name>/temporal/tx_log.parquet` - Transaction log (Parquet format)

---

### 2. `preprocess.py` - Feature Engineering

Preprocesses raw transaction data into ML-ready datasets with windowed temporal aggregation.

**Usage:**
```bash
uv run python scripts/preprocess.py --conf_file experiments/10k_accounts/config/preprocessing.yaml
```

**Arguments:**
- `--conf_file`: Path to preprocessing configuration file

**What it does:**
1. Reads transaction log from `temporal/tx_log.parquet`
2. Aggregates features over time windows (e.g., 4 windows of 28 days)
3. Creates train/val/test splits (supports both transductive and inductive)
4. Generates both centralized and per-client datasets

**Output:**
- `experiments/<name>/preprocessed/centralized/`
  - `trainset_nodes.parquet`, `valset_nodes.parquet`, `testset_nodes.parquet`
  - `trainset_edges.parquet`, `valset_edges.parquet`, `testset_edges.parquet` (if `include_edge_features: true`)
- `experiments/<name>/preprocessed/clients/<bank_id>/`
  - Same structure, but filtered by bank for federated learning

---

### 3. `train.py` - Model Training

Main training script supporting centralized, federated, and isolated learning settings.

**Usage:**
```bash
# Centralized training
uv run python scripts/train.py \
  --experiment_dir experiments/10k_accounts \
  --model DecisionTreeClassifier \
  --training_regime centralized \
  --seed 42

# Federated training
uv run python scripts/train.py \
  --experiment_dir experiments/10k_accounts \
  --model GraphSAGE \
  --training_regime federated \
  --seed 42

# Isolated training (each bank independently)
uv run python scripts/train.py \
  --experiment_dir experiments/10k_accounts \
  --model RandomForestClassifier \
  --training_regime isolated \
  --seed 42
```

**Arguments:**
- `--experiment_dir`: Path to experiment directory
- `--model`: Model to train (see available models in `config/models.yaml`)
- `--training_regime`: Training setting (`centralized`, `federated`, `isolated`)
- `--seed`: Random seed for reproducibility (default: `42`)

**Available Models:**
- **Tabular:** `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`, `MLP`
- **GNNs:** `GCN`, `GAT`, `GraphSAGE`

**Training Regimes:**
- **Centralized**: Single model trained on all banks' data combined
- **Federated**: Banks collaborate via federated learning (privacy-preserving)
- **Isolated**: Each bank trains independently on local data only

**Output:**
- `experiments/<name>/results/<regime>/<model>/`
  - `results.pkl` - Full results with metrics over training rounds
  - Training/validation/test metrics (accuracy, average precision, balanced accuracy, F1, etc.)

---

## Hyperparameter Tuning

### 4. `tune_hyperparams.py` - Model Hyperparameter Optimization

Uses Optuna for Bayesian hyperparameter optimization for a given model and training regime.

**Usage:**
```bash
# Tune model for centralized training
uv run python scripts/tune_hyperparams.py \
  --experiment_dir experiments/10k_accounts \
  --model RandomForestClassifier \
  --training_regime centralized \
  --num_trials 100 \
  --seed 42

# Tune for specific client in isolated setting
uv run python scripts/tune_hyperparams.py \
  --experiment_dir experiments/10k_accounts \
  --model GCN \
  --training_regime isolated \
  --client_id bank_a \
  --num_trials 50
```

**Arguments:**
- `--experiment_dir`: Path to experiment directory
- `--model`: Model to tune
- `--training_regime`: Training setting to tune for
- `--client_id`: Client ID for isolated setting (optional)
- `--num_trials`: Number of optimization trials (default: `100`)
- `--seed`: Random seed (default: `42`)

**Output:**
- SQLite database: `experiments/<name>/results/BO/<regime>/<model>/hp_study.db`
- Best hyperparameters are automatically saved to `models.yaml` under `tuned` section

---

### 5. `tune_data.py` - Data Generation Parameter Tuning

Optimizes data generation parameters using two-level Bayesian optimization to achieve target metrics (e.g., FPR = 0.01).

**Usage:**
```bash
uv run python scripts/tune_data.py \
  --experiment_dir experiments/10k_accounts \
  --num_trials_data 50 \
  --num_trials_model 100 \
  --model DecisionTreeClassifier \
  --utility fpr \
  --seed 0
```

**Arguments:**
- `--experiment_dir`: Path to experiment directory
- `--num_trials_data`: Number of data optimization trials (default: `50`)
- `--num_trials_model`: Number of model tuning trials per data configuration (default: `100`)
- `--model`: Model to use for evaluation (default: `DecisionTreeClassifier`)
- `--utility`: Optimization objective (default: `fpr`)
- `--seed`: Random seed (default: `0`)

**How it works:**
```
For each of num_trials_data iterations:
  1. Sample new data parameters (mean_amount_sar, prob_spend_cash, etc.)
  2. Generate synthetic data with those parameters
  3. Preprocess data
  4. For each of num_trials_model iterations:
     - Sample model hyperparameters
     - Train model
     - Evaluate on validation set
  5. Compute data quality score (FPR + feature importance error)
```

**Output:**
- `experiments/<name>/results/BO/data_tuning/`
  - `data_tuning_study.db` - Optuna database with all trials
  - `pareto_front.png` - Multi-objective optimization visualization
  - `best_trials.txt` - Best parameter combinations
- Best parameters are automatically saved to `data.yaml`

**Note:** The optimization framework is flexible - you can modify the objective function to optimize for different metrics or add new parameters to optimize.

---

## Visualization Scripts

### 6. `plot.py` - Results Visualization

Generates plots from training results.

**Usage:**
```bash
uv run python scripts/plot.py \
  --experiment_dir experiments/10k_accounts \
  --training_regime centralized
```

**Arguments:**
- `--experiment_dir`: Path to experiment directory
- `--training_regime`: Training regime to visualize (optional, if not specified visualizes all)

**Output:**
- Plots saved in `experiments/<name>/results/<regime>/<model>/`:
  - Training curves (loss, metrics over rounds)
  - ROC curves
  - Precision-recall curves

---

### 7. `visualize_data.py` - Data Visualization

Visualizes properties of generated synthetic data.

**Usage:**
```bash
uv run python scripts/visualize_data.py \
  --experiment_dir experiments/10k_accounts
```

**Visualizations:**
- Transaction volume over time
- SAR vs normal transaction distributions
- Degree distributions
- Network statistics

---

### 8. `visualize_features.py` - Feature Visualization

Visualizes engineered features for ML models.

**Usage:**
```bash
uv run python scripts/visualize_features.py \
  --experiment_dir experiments/10k_accounts
```

**Visualizations:**
- Feature distributions
- Feature correlations
- Class imbalance
- Temporal patterns

---

## Common Workflows

### Complete Pipeline from Scratch

```bash
# 1. Generate synthetic data
uv run python scripts/generate.py \
  --conf_file experiments/my_experiment/config/data.yaml

# 2. Preprocess for ML
uv run python scripts/preprocess.py \
  --conf_file experiments/my_experiment/config/preprocessing.yaml

# 3. Train models in all three regimes
for regime in centralized federated isolated
do
  uv run python scripts/train.py \
    --experiment_dir experiments/my_experiment \
    --model DecisionTreeClassifier \
    --training_regime $regime \
    --seed 42
done

# 4. Visualize results
uv run python scripts/plot.py \
  --experiment_dir experiments/my_experiment
```

---

### Hyperparameter Optimization Workflow

```bash
# 1. Generate and preprocess data
uv run python scripts/generate.py --conf_file experiments/my_experiment/config/data.yaml
uv run python scripts/preprocess.py --conf_file experiments/my_experiment/config/preprocessing.yaml

# 2. Tune hyperparameters
uv run python scripts/tune_hyperparams.py \
  --experiment_dir experiments/my_experiment \
  --model RandomForestClassifier \
  --training_regime centralized \
  --num_trials 200 \
  --seed 42

# 3. Train with optimized hyperparameters (automatically uses tuned params from models.yaml)
uv run python scripts/train.py \
  --experiment_dir experiments/my_experiment \
  --model RandomForestClassifier \
  --training_regime centralized \
  --seed 42
```

---

### Data-Level Optimization Workflow

```bash
# This script runs the full pipeline internally
uv run python scripts/tune_data.py \
  --experiment_dir experiments/my_experiment \
  --num_trials_data 50 \
  --num_trials_model 100 \
  --model DecisionTreeClassifier \
  --utility fpr \
  --seed 0

# After optimization, train with optimized data parameters
uv run python scripts/generate.py --conf_file experiments/my_experiment/config/data.yaml
uv run python scripts/preprocess.py --conf_file experiments/my_experiment/config/preprocessing.yaml
uv run python scripts/train.py \
  --experiment_dir experiments/my_experiment \
  --model DecisionTreeClassifier \
  --training_regime centralized
```

---

## Specialized Simulations

The `scripts/simulations/` directory contains scripts for systematic experiments used in the AMLGentex paper. These scripts test model robustness under different conditions:

### `changing_overlap.py` - Transductive vs Inductive Learning
Tests train/test temporal overlap (0-56 step offsets). High overlap = transductive learning, low overlap = inductive learning.

### `decreasing_n_labels.py` - Limited Labeled Data
Tests model performance with progressively smaller fractions of labeled training data (e.g., 60%, 6%, 0.6%, 0.06%).

### `increasing_clients.py` - Federated Collaboration Benefits
Tests federated learning performance as the number of collaborating banks increases (1 to N clients).

**See [`scripts/simulations/README.md`](simulations/README.md) for detailed documentation and usage examples.**

---

## Notes

- **File Format**: All data is stored in Parquet format (4x smaller than CSV, 74% faster I/O)
- **Multiprocessing**: Many scripts support parallel processing via multiprocessing
- **Reproducibility**: Always use `--seed` argument for reproducible results
- **Configuration**: All parameters are defined in YAML files in `experiments/<name>/config/`
- **Auto-discovery**: Scripts use convention-over-configuration to find data automatically
- **Optimization**: Optuna databases stored in `experiments/<name>/results/BO/` for reproducibility

---

## File Organization

After running the full pipeline, your experiment directory will look like:

```
experiments/my_experiment/
├── config/
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── models.yaml
│   └── *.csv
├── spatial/
├── temporal/
│   └── tx_log.parquet
├── preprocessed/
│   ├── centralized/
│   └── clients/
└── results/
    ├── centralized/
    ├── federated/
    ├── isolated/
    └── BO/
```
