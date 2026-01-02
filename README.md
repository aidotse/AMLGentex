# AMLGentex

**Mobilizing Data-Driven Research to Combat Money Laundering**

AMLGentex is a comprehensive benchmarking framework for anti-money laundering (AML) research, developed by AI Sweden in collaboration with Handelsbanken and Swedbank. It enables generation of realistic synthetic transaction data, training of machine learning models, and application of explainability techniques to advance AML detection systems.

[![arXiv](https://img.shields.io/badge/arXiv-2506.13989-b31b1b.svg)](https://arxiv.org/abs/2506.13989)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Terminology

- **SAR**: Suspicious Activity Report - accounts/transactions flagged as suspicious
- **SWISH**: Swedish Instant Payment System - mobile payment system
- **AML**: Anti-Money Laundering - detecting and preventing money laundering
- **Transaction**: payment between two accounts
- **Income**: Money entering account from external source (salary)
- **Outcome**: Money leaving account to external sink (spending)
- **Normal Pattern**: Regular transaction behavior (fan-in, fan-out, mutual, forward, periodical, single)
- **Alert Pattern**: Suspicious transaction behavior (cycle, bipartite, stack, random, gather-scatter, scatter-gather)
- **Spatial**: Network topology (who can transact with whom)
- **Temporal**: Time-series of transactions

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Generation](#data-generation)
  - [Spatial Graph Generation](#spatial-graph-generation)
  - [Temporal Transaction Simulation](#temporal-transaction-simulation)
  - [Bayesian Optimization](#bayesian-optimization)
- [Feature Engineering](#feature-engineering)
- [Machine Learning](#machine-learning)
  - [Training Regimes](#training-regimes)
  - [Supported Models](#supported-models)
  - [Adding New Models](#adding-new-models)
- [Configuration Reference](#configuration-reference)
- [Usage Examples](#usage-examples)
- [Citation](#citation)

---

## Overview

AMLGentex provides a complete pipeline for generating, training, and evaluating machine learning models for AML detection. The framework is loosely based on the Swedish mobile payment system SWISH but is easily extensible to other payment systems. For a detailed description of the framework and its components, see the [AMLGentex paper](https://arxiv.org/abs/2506.13989) and its appendix.

### Real-World Challenges Addressed

AMLGentex captures a range of real-world data complexities identified by AML experts from Swedbank and Handelsbanken:

<p align="center">
  <img src="assets/images/radar_plot.png" alt="Severity of challenges addressed by AMLGentex" width="500">
  <br>
  <em>Figure 1: Expert-assessed severity of key challenges in AML transaction monitoring</em>
</p>

The framework models multiple sources of noise and complexity:

<p align="center">
  <img src="assets/images/noise.png" alt="Types of noise in transaction data" width="600">
  <br>
  <em>Figure 2: Different noise types affecting AML detection (label noise, feature drift, etc.)</em>
</p>

### Key Capabilities

- **Data Generation**: Create realistic synthetic transaction networks with controllable complexity
- **Pattern Injection**: Insert both normal and suspicious (SAR) transaction patterns
- **Training Flexibility**: Train models in three settings (centralized, federated, isolated)
- **Optimization**: Two-level Bayesian optimization for data generation and model hyperparameters
- **Model Support**: 8 ML models (Decision Trees, Random Forests, GBM, Logistic Regression, MLP, GCN, GAT, GraphSAGE) - easily extensible
- **Visualization**: Interactive tools for exploring transaction networks

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone repository
git clone https://github.com/aidotse/AMLGentex.git
cd AMLGentex

# Install dependencies using uv (recommended - fast!)
pip install uv
uv sync

# Or use pip
pip install -e .

# Optional: Install visualization tools
pip install -e ".[viz]"
pip install -e ".[network-explorer]"
```

**Key Dependencies:**
- `pandas`, `numpy`, `scikit-learn` - Data processing and ML
- `torch`, `torch_geometric` - Graph neural networks
- `optuna` - Bayesian optimization
- `pyarrow` - Parquet file support (4x smaller than CSV)
- `pyyaml` - Configuration management
- `panel`, `holoviews`, `datashader` - Interactive visualization

---

## Quick Start

### 📓 Interactive Tutorial (Recommended)

The fastest way to get started is with our comprehensive Jupyter notebook:

```bash
jupyter notebook tutorial.ipynb
```

**Tutorial covers:** Creating experiments, generating data, preprocessing, training models, and visualization.

---

### Command-Line Workflow

**Step 1: Generate synthetic data**
```bash
uv run python scripts/generate.py --conf_file experiments/10k_accounts/config/data.yaml
```

**Step 2 (Optional): Optimize data generation**
```bash
uv run python scripts/tune_data.py \
    --experiment_dir experiments/10k_accounts \
    --num_trials_data 50 \
    --num_trials_model 100 \
    --model DecisionTreeClassifier
```

**Step 3: Engineer features**
```bash
uv run python scripts/preprocess.py --conf_file experiments/10k_accounts/config/preprocessing.yaml
```

**Step 4: Train models**
```bash
uv run python scripts/train.py \
    --experiment_dir experiments/10k_accounts \
    --model DecisionTreeClassifier \
    --training_regime centralized
```

---

## Data Generation

Data generation follows a three-stage process: **spatial graph generation**, **temporal transaction simulation**, and **Bayesian optimization** for parameter tuning.

### Spatial Graph Generation

The spatial stage creates the transaction network topology. This determines which accounts can transact with each other.

#### 1. Scale-Free Network Blueprint

AMLGentex generates scale-free networks where node degree follows a power-law distribution. Three parameters control the topology:
- **`gamma`**: Power-law exponent (typically 2.0-3.0)
- **`loc`**: Minimum degree (offset for small degrees)
- **`average_degree`**: Target mean degree of the network

#### 2. Pattern Injection

**Normal Patterns:** Regular transaction behaviors inserted first, respecting network constraints.

<p align="center">
  <img src="assets/images/normal_patterns.png" alt="Normal transaction patterns" width="700">
  <br>
  <em>Figure 3: Normal transaction patterns - single, fan-in, fan-out, forward, mutual, periodical</em>
</p>

**Alert Patterns:** Suspicious activities (SAR patterns) inserted on top of the normal network.

<p align="center">
  <img src="assets/images/alert_patterns.png" alt="Alert transaction patterns" width="700">
  <br>
  <em>Figure 4: Alert patterns - fan-in, fan-out, cycle, bipartite, stack, random, gather-scatter, scatter-gather</em>
</p>

#### 3. Complete Network Creation Pipeline

<p align="center">
  <img src="assets/images/data_generation_procedure.png" alt="Network creation process" width="700">
  <br>
  <em>Figure 5: From degree distribution blueprint to final spatial graph with injected patterns</em>
</p>

**Configuration:** Spatial graph generation is controlled by `experiments/<name>/config/data.yaml` and CSV files defining:
- `accounts.csv` - Account properties (balance, bank, country)
- `degree.csv` - Degree distribution blueprint (auto-generated from `scale-free` parameters if not present)
- `normalModels.csv` - Normal transaction patterns
- `alertPatterns.csv` - Suspicious transaction patterns

---

### Temporal Transaction Simulation

Once the spatial graph is created, temporal simulation generates transaction sequences over time.

#### 1. Transaction Amount Modeling

Transaction amounts are sampled from truncated Gaussian distributions with separate parameters for normal and SAR transactions:

<p align="center">
  <img src="assets/images/truncated_gaussians.png" alt="Truncated Gaussian distributions" width="600">
  <br>
  <em>Figure 6: Transaction amount distributions - normal (left) vs SAR (right) transactions</em>
</p>

#### 2. Transaction Timing System

**Dynamic Duration Sampling:** Each pattern's duration is sampled from a lognormal distribution:
- `mean_duration_normal/alert` - Controls typical pattern length (in steps, linear space)
- `std_duration_normal/alert` - Controls duration variability (in steps, linear space)
- Parameters are automatically converted internally to log-space for lognormal sampling
- Start time randomly selected within valid range [0, T - duration]

**Note:** The configured duration is the time **window** in which transactions can occur. The actual observed span (first to last transaction) will typically be shorter because transactions are randomly placed within this window.

**Burstiness Control:** Four-level system using beta distributions controls transaction clustering:
- **Level 1** (Beta(1,1) - Uniform): Near-constant transaction gaps
- **Level 2** (Beta(2,2) - Symmetric): Regular spacing with some variation
- **Level 3** (Beta(0.5,3) - Right-skewed): Transactions cluster early in period
- **Level 4** (Beta(0.3,0.3) - Bimodal): Tight clusters with large gaps between

The `burstiness_bias_normal/alert` parameter provides smooth control over level probabilities using exponential weighting, favoring lower levels (uniform) for negative bias and higher levels (clustered) for positive bias.

#### 3. Account Behavior Modeling

Accounts exhibit realistic spending behavior based on balance history as shown below.

<p align="center">
  <img src="assets/images/in-out-flows.png" alt="In-flows and out-flows over time" width="600">
  <br>
  <em>Figure 7: Temporal dynamics - in-flows (salary) and out-flows (spending) over simulation period</em>
</p>

#### 4. Money Laundering Typologies

AMLGentex supports two main laundering approaches:

<p align="center">
  <img src="assets/images/laundering_stages.png" alt="Laundering methods" width="600">
  <br>
  <em>Figure 8: (Left) Transfer-based laundering through network, (Right) Cash-based with placement and integration</em>
</p>

**Transfer-based:** Money flows through the network via account-to-account transfers
- Placement: Initial deposit
- Layering: Complex transfers through multiple accounts
- Integration: Final extraction

**Cash-based:** SAR accounts can inject and extract cash
- `prob_spend_cash` controls cash usage probability
- Harder to trace than network transfers as it is invisible to banks

**Configuration:** Temporal simulation is controlled by parameters in `data.yaml`.

**Output:** Transaction log saved as `experiments/<name>/temporal/tx_log.parquet`

---

### Bayesian Optimization

AMLGentex uses **two-level Bayesian optimization** to find optimal data generation parameters and model hyperparameters.

<p align="center">
  <img src="assets/images/autotuning.png" alt="Two-level Bayesian optimization" width="600">
  <br>
  <em>Figure 9: Data-informed optimization finds better data configurations than model-only tuning</em>
</p>

#### How It Works

```
For num_trials_data iterations:
    1. Sample new alert data parameters (mean_amount_sar, prob_spend_cash, etc.)
    2. Generate synthetic data with those parameters
    3. Preprocess data
    4. For num_trials_model iterations:
        a. Sample model hyperparameters
        b. Train model
        c. Evaluate on validation set
    5. Record best model performance for this data configuration
    6. Compute data quality score (FPR + feature importance error)
```

**Objective:** Find data parameters that enable models to:
1. Achieve a target FPR = 0.01
2. Minimize feature importance error (correct features should be important)

**Note:** The optimization framework is flexible and easily adapted to:
- Different objectives (e.g., maximize recall, minimize false negatives, balance multiple metrics)
- Different parameters (any parameter in `data.yaml` can be added to `optimisation_bounds`)
- Custom utility functions for domain-specific requirements

**Configuration:** Search spaces defined in `data.yaml` (`optimisation_bounds`) and `models.yaml` (`optimization.search_space`)

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

---

## Feature Engineering

Raw transaction logs are transformed into ML-ready features through windowed temporal aggregation. The framework supports both **transductive** learning (full graph visible, test labels hidden) and **inductive** learning (test nodes completely unseen during training).

<p align="center">
  <img src="assets/images/blueprint.png" alt="Feature engineering pipeline" width="700">
  <br>
  <em>Figure 10: From spatial graph and transactions to windowed node features for ML models</em>
</p>

### Process

1. **Window Definition:** Divide simulation period into overlapping time windows
   - `window_len` - Window size in days (e.g., 28 days)
   - `num_windows` - Number of windows (e.g., 4)

2. **Feature Aggregation:** For each window, compute per-account features:
   - Transaction counts (sent, received)
   - Transaction volumes (total amount sent/received)
   - Network features (in-degree, out-degree)
   - Balance statistics (mean, std, min, max)
   - Phone/bank change frequency
   - Cash usage indicators

3. **Train/Val/Test Splits:** Supports both **transductive** and **inductive** learning
   - **Transductive**: All accounts seen during training (full graph visible, but test labels hidden)
   - **Inductive**: Test accounts completely unseen during training (realistic for new accounts)
   - Time windows can overlap (accounts can appear in multiple splits)
   - Separate files per bank for federated learning

**Configuration:** `experiments/<name>/config/preprocessing.yaml`

**Output:** Preprocessed features saved to `experiments/<name>/preprocessed/`

---

## Machine Learning

AMLGentex supports training in three regimes with 8 different model types.

### Training Regimes

| Regime | Description | Use Case |
|--------|-------------|----------|
| **Centralized** | All banks pool data, train single global model | Maximum performance, no privacy constraints |
| **Federated** | Banks collaborate without sharing raw data | Privacy-preserving, regulatory compliance |
| **Isolated** | Each bank trains independently on local data | Full privacy, simple deployment |

**Usage:**
```bash
# Centralized
uv run python scripts/train.py \
    --experiment_dir experiments/10k_accounts \
    --model DecisionTreeClassifier \
    --training_regime centralized

# Federated
uv run python scripts/train.py \
    --experiment_dir experiments/10k_accounts \
    --model GraphSAGE \
    --training_regime federated

# Isolated
uv run python scripts/train.py \
    --experiment_dir experiments/10k_accounts \
    --model RandomForestClassifier \
    --training_regime isolated
```

---

### Supported Models

#### Tabular Models
- **DecisionTreeClassifier** - Single decision tree
- **RandomForestClassifier** - Ensemble of decision trees
- **GradientBoostingClassifier** - Boosted decision trees
- **LogisticRegression** - Linear classifier
- **MLP** - Multi-layer perceptron (neural network)

#### Graph Neural Networks
- **GCN** - Graph Convolutional Network
- **GAT** - Graph Attention Network
- **GraphSAGE** - Inductive graph representation learning

**All models support:**
- Hyperparameter optimization with Optuna
- Training in all three regimes
- Custom metrics (average precision @ high recall)
- Automatic class imbalance handling

---

### Adding New Models

AMLGentex is designed for easy extensibility. To add a new model:

#### PyTorch Models

1. **Create the model class** in `src/ml/models/torch_models.py` (or `gnn_models.py` for GNNs):

```python
from src.ml.models.base import TorchBaseModel
import torch

class MyNewModel(TorchBaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MyNewModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x.squeeze()
```

2. **Export in** `src/ml/models/__init__.py`:

```python
from src.ml.models.torch_models import MyNewModel

__all__ = [..., 'MyNewModel']
```

3. **Configure in** `experiments/<name>/config/models.yaml`:

```yaml
MyNewModel:
  default:
    client_type: TorchClient
    server_type: TorchServer  # For federated learning
    device: cpu
    input_dim: 128
    hidden_dim: 64
    output_dim: 1
    lr: 0.001
    batch_size: 512

  optimization:
    search_space:
      hidden_dim:
        type: int
        low: 32
        high: 256
      lr:
        type: float
        low: 0.0001
        high: 0.01
        log: true
```

4. **Train**: Use the same training commands with `--model MyNewModel`

#### Scikit-Learn Models

1. **Import in** `src/ml/models/sklearn_models.py`:

```python
from sklearn.svm import SVC

__all__ = [..., 'SVC']
```

2. **Export in** `src/ml/models/__init__.py`:

```python
from src.ml.models.sklearn_models import SVC

__all__ = [..., 'SVC']
```

3. **Configure in** `models.yaml`:

```yaml
SVC:
  default:
    client_type: SklearnClient
    C: 1.0
    kernel: rbf
    class_weight: balanced

  optimization:
    search_space:
      C:
        type: float
        low: 0.001
        high: 100
        log: true
      kernel:
        type: categorical
        values: [linear, rbf, poly]
```

**Note:** Models inheriting from `TorchBaseModel` or `SklearnBaseModel` automatically support:
- Federated learning (get/set parameters)
- Hyperparameter optimization
- All three training regimes

---

### Custom Metrics

AMLGentex includes custom metrics for high-recall scenarios:

- **Average Precision @ High Recall**: Focuses on recall range [0.6, 1.0]
  - Critical for AML where missing suspicious activities is costly
- **Balanced Accuracy**: Handles class imbalance
- **Confusion Matrix**: Custom implementation with correct FP/FN definitions

**Implementation:** `src/ml/metrics/`

---

### Model Configuration

Models are configured in `experiments/<name>/config/models.yaml`:

```yaml
DecisionTreeClassifier:
  default:
    client_type: SklearnClient
    criterion: gini
    max_depth: ~
    class_weight: balanced

  optimization:
    search_space:
      criterion:
        type: categorical
        values: [gini, entropy, log_loss]
      max_depth:
        type: int
        low: 10
        high: 1000

GraphSAGE:
  default:
    client_type: TorchClient
    server_type: TorchServer
    device: cpu
    hidden_dim: 64
    num_layers: 2
    dropout: 0.5
    lr: 0.001
    batch_size: 512
```

---

## Configuration Reference

Experiments are organized under `experiments/<experiment_name>/` with three YAML configuration files.

### Directory Structure

```
experiments/<experiment_name>/
├── config/                   # ✅ Committed to Git
│   ├── data.yaml              # Data generation parameters
│   ├── preprocessing.yaml     # Feature engineering settings
│   ├── models.yaml           # Model configurations
│   ├── accounts.csv          # Account specifications
│   ├── degree.csv            # Network degree distribution (auto-generated)
│   ├── normalModels.csv      # Normal pattern definitions
│   └── alertPatterns.csv     # SAR pattern definitions
├── spatial/                  # ❌ Generated (ignored by Git)
├── temporal/                 # ❌ Generated (ignored by Git)
├── preprocessed/             # ❌ Generated (ignored by Git)
└── results/                  # ❌ Generated (ignored by Git)
```

**What gets committed:**
- ✅ `config/` - All configuration files (YAML and CSV) that define your experiment
- ❌ `spatial/`, `temporal/`, `preprocessed/`, `results/` - Generated outputs (can be reproduced from config)

This keeps your repository lean while ensuring reproducibility. Anyone can clone the repo and regenerate all outputs by running the pipeline with your committed config files.

**Creating new experiments:** See [`experiments/README.md`](experiments/README.md) for detailed instructions on setting up new experiments.

---

### Convention Over Configuration

AMLGentex uses auto-discovery to minimize manual configuration:

```python
from src.utils import find_experiment_root, find_clients

# Automatically find experiment directory
experiment_root = find_experiment_root("10k_accounts")

# Auto-discover client data
clients = find_clients(experiment_root / "preprocessed" / "clients")
```

Just organize files following the standard structure, and AMLGentex handles the rest!

---

## Project Structure

```
AMLGentex/
├── src/                        # Core framework code
│   ├── sim/                    # Transaction simulation
│   ├── ml/                     # ML models, clients, servers
│   │   ├── models/            # Model implementations
│   │   ├── clients/           # TorchClient, SklearnClient
│   │   ├── servers/           # TorchServer for federated learning
│   │   └── metrics/           # Custom metrics
│   ├── preprocessing/          # Feature engineering
│   ├── tuning/                # Bayesian optimization
│   ├── visualize/             # Plotting and visualization
│   │   └── transaction_network_explorer/  # Interactive dashboard
│   └── utils/                 # Configuration, helpers
├── experiments/               # Experiment configurations and results
│   └── <experiment_name>/
│       ├── config/           # YAML configs and CSV specifications
│       ├── spatial/          # Generated spatial graphs
│       ├── temporal/         # Transaction logs (Parquet)
│       ├── preprocessed/     # ML-ready features
│       └── results/          # Training results and plots
├── scripts/                   # Executable scripts
│   ├── generate.py           # Generate synthetic data
│   ├── preprocess.py         # Feature engineering
│   ├── train.py              # Train ML models
│   ├── tune_data.py          # Two-level Bayesian optimization
│   ├── tune_hyperparams.py   # Model hyperparameter tuning
│   └── plot.py               # Generate visualizations
├── tests/                     # Test suite
├── tutorial.ipynb            # Comprehensive tutorial notebook
└── pyproject.toml            # Project dependencies and config
```

---

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test markers
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m e2e         # End-to-end tests
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Citation

If you use AMLGentex in your research, please cite:

```bibtex
@misc{ostman2025amlgentexmobilizingdatadrivenresearch,
  title     = {AMLgentex: Mobilizing Data-Driven Research to Combat Money Laundering},
  author    = {Johan \"Ostman and Edvin Callisen and Anton Chen and Kristiina Ausmees and
               Emanuel G\aardh and Jovan Zamac and Jolanta Goldsteine and Hugo Wefer and
               Simon Whelan and Markus Reimeg\aard},
  year      = {2025},
  eprint    = {2506.13989},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI},
  url       = {https://arxiv.org/abs/2506.13989}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Developed by **AI Sweden** in collaboration with:
- **Handelsbanken**
- **Swedbank**

---

## Contact

For questions or issues:
- Open an issue on [GitHub](https://github.com/aidotse/AMLGentex/issues)
- Contact: [AI Sweden](https://www.ai.se/)

---
