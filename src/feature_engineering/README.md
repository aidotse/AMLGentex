# Preprocessing Module

This module handles feature engineering from raw transaction data to create ML-ready datasets for AML detection.

## Overview

The preprocessing pipeline transforms temporal transaction logs into aggregated node features (account-level) and optional edge features (transaction relationships) over sliding time windows.

**Input**: Raw transaction log (Parquet format)
**Output**: Train/validation/test splits with engineered features

## Files

### `preprocessor.py`
Main preprocessing class that generates features from transaction data.

**Key Features:**
- **Time window aggregation**: Aggregates transaction statistics over configurable sliding windows
- **Node features**: Account-level features (spending patterns, incoming/outgoing transactions, balance behavior)
- **Edge features**: Transaction relationship features (optional, for GNN models)
- **Train/val/test splits**: Supports both transductive and inductive learning

### `noise.py`
Utilities for injecting label noise into datasets for robustness experiments.

**Functions:**
- `flip_labels()`: Randomly flip labels of specified classes
- `missing_labels()`: Simulate missing labels by setting them to -1
- `flip_neighbours()`: Flip labels of normal accounts connected to SAR accounts
- `topology_noise()`: Flip labels based on alert pattern topologies

## Usage

### Basic Preprocessing

```python
from src.preprocess import DataPreprocessor
import yaml

# Load config
with open('experiments/10k_accounts/config/preprocessing.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize preprocessor
preprocessor = DataPreprocessor(config)

# Process raw transaction log
raw_data_file = 'experiments/10k_accounts/temporal/tx_log.parquet'
datasets = preprocessor(raw_data_file)

# Access splits
train_nodes = datasets['trainset_nodes']
val_nodes = datasets['valset_nodes']
test_nodes = datasets['testset_nodes']

# If include_edge_features=True in config
train_edges = datasets['trainset_edges']  # Optional
```

### Configuration

Example `preprocessing.yaml`:

```yaml
# Time window aggregation
num_windows: 4        # Number of sliding windows
window_len: 28        # Days per window

# Learning mode: transductive or inductive
learning_mode: transductive

# Transductive settings (only used when learning_mode: transductive)
time_start: 0
time_end: 112
transductive_train_fraction: 0.6
transductive_val_fraction: 0.2
transductive_test_fraction: 0.2
split_by_pattern: true  # Keep SAR patterns together to prevent leakage

# Include edge features (for GNN models)
include_edge_features: false
```

**Learning Paradigms:**

1. **Transductive Learning** (`learning_mode: transductive`):
   - Same graph for all splits, labels split into train/val/test
   - Single time window specified with `time_start` and `time_end`
   - Configure label fractions with `transductive_*_fraction`
   ```yaml
   learning_mode: transductive
   time_start: 0
   time_end: 112
   transductive_train_fraction: 0.6
   transductive_val_fraction: 0.2
   transductive_test_fraction: 0.2
   split_by_pattern: true
   ```

2. **Inductive Learning** (`learning_mode: inductive`):
   - Different time windows for train/val/test (temporal separation)
   - Test accounts completely unseen during training
   ```yaml
   learning_mode: inductive
   train_start_step: 0
   train_end_step: 50
   val_start_step: 51
   val_end_step: 80
   test_start_step: 81
   test_end_step: 112
   ```

## Node Features

For each time window, the following features are computed per account:

**Spending (to sink):**
- `sums_spending_{start}_{end}`: Total amount spent
- `means_spending_{start}_{end}`: Average spending amount
- `medians_spending_{start}_{end}`: Median spending amount
- `stds_spending_{start}_{end}`: Standard deviation of spending
- `maxs_spending_{start}_{end}`: Maximum spending transaction
- `mins_spending_{start}_{end}`: Minimum spending transaction
- `counts_spending_{start}_{end}`: Number of spending transactions

**Incoming transactions:**
- `sum_in_{start}_{end}`: Total amount received
- `mean_in_{start}_{end}`: Average incoming amount
- `median_in_{start}_{end}`: Median incoming amount
- `std_in_{start}_{end}`: Standard deviation
- `max_in_{start}_{end}`: Maximum incoming transaction
- `min_in_{start}_{end}`: Minimum incoming transaction
- `count_in_{start}_{end}`: Number of incoming transactions
- `count_unique_in_{start}_{end}`: Number of unique senders

**Outgoing transactions:**
- `sum_out_{start}_{end}`: Total amount sent
- `mean_out_{start}_{end}`: Average outgoing amount
- `median_out_{start}_{end}`: Median outgoing amount
- `std_out_{start}_{end}`: Standard deviation
- `max_out_{start}_{end}`: Maximum outgoing transaction
- `min_out_{start}_{end}`: Minimum outgoing transaction
- `count_out_{start}_{end}`: Number of outgoing transactions
- `count_unique_out_{start}_{end}`: Number of unique receivers

**Account metadata:**
- `counts_days_in_bank`: Days since bank registration
- `counts_phone_changes`: Number of phone number changes
- `is_sar`: Label (1 = suspicious activity report, 0 = normal)

## Edge Features

When `include_edge_features=True`, transaction relationships between accounts are captured:

For each time window, per (src, dst) pair:
- `sums_{start}_{end}`: Total transaction amount
- `means_{start}_{end}`: Average transaction amount
- `medians_{start}_{end}`: Median transaction amount
- `stds_{start}_{end}`: Standard deviation
- `maxs_{start}_{end}`: Maximum transaction
- `mins_{start}_{end}`: Minimum transaction
- `counts_{start}_{end}`: Number of transactions
- `is_sar`: Whether edge involves SAR account

## Time Windows

The preprocessor uses **overlapping sliding windows** to capture temporal patterns:

```
Time:     0 -------- 28 ------- 56 ------- 84 ------ 112
Window 1: [--------28--------]
Window 2:      [--------28--------]
Window 3:           [--------28--------]
Window 4:                [--------28--------]
```

Window overlap is automatically calculated based on `num_windows`, `window_len`, and dataset duration.

## Label Noise (Research)

The `noise.py` module supports experimental studies on label noise robustness:

```python
from src.preprocess.noise import flip_labels, topology_noise

# Flip 1% of normal accounts and 10% of SAR accounts
noisy_nodes = flip_labels(
    nodes=train_nodes,
    labels=[0, 1],
    fracs=[0.01, 0.1],
    seed=42
)

# Flip labels by topology type
noisy_nodes = topology_noise(
    nodes=train_nodes,
    alert_members=alert_members_df,
    topologies=['fan_in', 'fan_out'],
    fracs=0.5,
    seed=42
)
```

## Testing

Run tests for the preprocessing module:

```bash
cd src/preprocess
pytest tests/
```

## Design Notes

1. **Transductive vs Inductive Learning**:
   - **Transductive**: Train/val/test use the same time period but different labeled nodes
   - **Inductive**: Train/val/test can use different (possibly overlapping) time periods

2. **Bank Filtering**: The preprocessor can filter to specific banks by setting `self.bank` attribute

3. **Missing Values**: All NaN values are filled with 0.0, representing accounts with no activity in that window

4. **Source/Sink**: Transactions from "source" (income) and to "sink" (spending) are handled separately from network transactions

5. **Parquet Format**: All I/O uses Parquet for efficient storage and fast loading

6. **Edge Directionality**: Edges are computed directionally (src → dst) to capture transaction flow patterns
