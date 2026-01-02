# Experiments

This directory contains experiment configurations for AML detection research. Each experiment defines the data generation, feature engineering, and model training parameters.

---

## Directory Structure

Each experiment follows this structure:

```
experiment_name/
├── config/                   # ✅ Committed to Git
│   ├── data.yaml              # Data generation parameters
│   ├── preprocessing.yaml     # Feature engineering settings
│   ├── models.yaml           # Model configurations
│   ├── accounts.csv          # Account specifications
│   ├── degree.csv            # Network degree distribution (auto-generated)
│   ├── normalModels.csv      # Normal transaction patterns
│   ├── alertPatterns.csv     # SAR patterns
│   └── transactionType.csv   # Transaction types
├── spatial/                  # ❌ Generated (ignored by Git)
├── temporal/                 # ❌ Generated (ignored by Git)
├── preprocessed/             # ❌ Generated (ignored by Git)
└── results/                  # ❌ Generated (ignored by Git)
```

**What gets committed:**
- ✅ `config/` - All configuration files (YAML and CSV)
- ❌ Everything else - Generated outputs that can be reproduced

---

## Creating a New Experiment

### Option 1: Copy Existing Experiment (Recommended)

```bash
# Copy from 10k_accounts as a template
cp -r experiments/10k_accounts experiments/my_experiment

# Edit configuration files
cd experiments/my_experiment/config
# Modify data.yaml, preprocessing.yaml, models.yaml, and CSV files
```

### Option 2: Create from Scratch

```bash
# Create directory structure
mkdir -p experiments/my_experiment/config

# Create required configuration files in config/
cd experiments/my_experiment/config
```

**Required files:**

1. **`data.yaml`** - Data generation parameters:
   ```yaml
   general:
     simulation_name: my_experiment
     random_seed: 0
     total_steps: 367
     base_date: '2023-01-01'

   default:
     mean_amount: 100
     std_amount: 10
     mean_amount_sar: 643
     std_amount_sar: 320
     # ... (see 10k_accounts/config/data.yaml for full template)

   scale-free:
     gamma: 2.0
     loc: 1.0
     average_degree: 2.0
   ```

2. **`preprocessing.yaml`** - Feature engineering settings:
   ```yaml
   raw_data_file: experiments/my_experiment/temporal/tx_log.parquet
   preprocessed_data_dir: experiments/my_experiment/preprocessed

   num_windows: 4
   window_len: 28

   train_start_step: 0
   train_end_step: 112
   val_start_step: 0
   val_end_step: 112
   test_start_step: 0
   test_end_step: 112

   include_edges: true
   ```

3. **`models.yaml`** - Model configurations (copy from `10k_accounts/config/models.yaml`)

4. **CSV files** (copy from `10k_accounts/config/`):
   - `accounts.csv` - Define number of accounts, initial balances, banks
   - `normalModels.csv` - Normal transaction patterns
   - `alertPatterns.csv` - SAR patterns
   - `transactionType.csv` - Transaction types

---

## Running an Experiment

Once configuration files are in place:

```bash
# 1. Generate synthetic data
uv run python scripts/generate.py --conf_file experiments/my_experiment/config/data.yaml

# 2. Preprocess data for ML
uv run python scripts/preprocess.py --conf_file experiments/my_experiment/config/preprocessing.yaml

# 3. Train models
uv run python scripts/train.py \
    --experiment_dir experiments/my_experiment \
    --model DecisionTreeClassifier \
    --training_regime centralized

# 4. (Optional) Optimize data generation
uv run python scripts/tune_data.py \
    --experiment_dir experiments/my_experiment \
    --num_trials_data 50 \
    --num_trials_model 100 \
    --model DecisionTreeClassifier
```

---

## Available Experiments

### `10k_accounts/`
Example experiment with 10,000 accounts. Use this as a template for new experiments.

### `coarsening_easy/`
Experiment with easier detection settings (coarsening).

---

## Version Control

Only commit configuration files:

```bash
# Add new experiment config
git add experiments/my_experiment/config/
git commit -m "Add my_experiment configuration"

# Generated outputs are automatically ignored
```

The `.gitignore` ensures that only `config/` directories are tracked, while all generated outputs (`spatial/`, `temporal/`, `preprocessed/`, `results/`) are ignored.

---

## Tips

- **Start small**: Begin with a small experiment (e.g., 1000 accounts) to test configurations quickly
- **Use templates**: Copy from `10k_accounts/config/` rather than creating from scratch
- **Parameter tuning**: Use `scripts/tune_data.py` to optimize data generation parameters
- **Documentation**: Document your experiment's purpose and key parameters in comments within YAML files
