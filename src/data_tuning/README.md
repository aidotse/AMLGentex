# Data Tuning

Optimizes synthetic data generation parameters to achieve target model performance under operational constraints.

## Overview

Data tuning uses **nested Bayesian optimization** to find optimal parameters for generating synthetic AML data. Instead of tuning model hyperparameters for fixed data, it tunes the data generation process itself.

**What it optimizes**: Parameters in `data.yaml` under `optimisation_bounds` (SAR amounts, spending patterns, behavioral features, alert overlap).

**How it optimizes**: Multi-objective optimization balancing:
1. Utility metric loss: `|achieved_metric - target|`
2. Feature importance variance: Lower variance = more stable features

Results in a **Pareto front** of optimal trade-offs.

## Three Optimization Objectives

### 1. Precision@K (Alert Budget)

**Use Case**: "We can review 100 alerts daily. Optimize for 80% precision in top 100."

```bash
uv run python scripts/tune_data.py \
  --experiment_dir experiments/my_exp \
  --constraint_type K \
  --constraint_value 100 \
  --utility_metric precision \
  --target 0.8 \
  --num_trials_data 20
```

```python
tuner = DataTuner(
    data_conf_file='experiments/my_exp/config/data.yaml',
    config=combined_config,
    generator=generator,
    preprocessor=preprocessor,
    target=0.8,
    constraint_type='K',
    constraint_value=100,
    utility_metric='precision',
    model='DecisionTreeClassifier',
    bo_dir='experiments/my_exp/data_tuning',
    seed=42,
    num_trials_model=15
)
best_trials = tuner(n_trials=20)
```

### 2. Recall at FPR≤α (Regulatory Constraint)

**Use Case**: "Compliance requires FPR ≤ 1%. Optimize for 70% recall at this limit."

```bash
uv run python scripts/tune_data.py \
  --experiment_dir experiments/my_exp \
  --constraint_type fpr \
  --constraint_value 0.01 \
  --utility_metric recall \
  --target 0.7 \
  --num_trials_data 20
```

```python
tuner = DataTuner(
    data_conf_file='experiments/my_exp/config/data.yaml',
    config=combined_config,
    generator=generator,
    preprocessor=preprocessor,
    target=0.7,
    constraint_type='fpr',
    constraint_value=0.01,
    utility_metric='recall',
    model='DecisionTreeClassifier',
    bo_dir='experiments/my_exp/data_tuning',
    seed=42,
    num_trials_model=15
)
best_trials = tuner(n_trials=20)
```

### 3. Precision at Recall≥target (Coverage Constraint)

**Use Case**: "Must detect 70% of SARs. Optimize for 60% precision at this recall."

```bash
uv run python scripts/tune_data.py \
  --experiment_dir experiments/my_exp \
  --constraint_type recall \
  --constraint_value 0.7 \
  --utility_metric precision \
  --target 0.6 \
  --num_trials_data 20
```

```python
tuner = DataTuner(
    data_conf_file='experiments/my_exp/config/data.yaml',
    config=combined_config,
    generator=generator,
    preprocessor=preprocessor,
    target=0.6,
    constraint_type='recall',
    constraint_value=0.7,
    utility_metric='precision',
    model='DecisionTreeClassifier',
    bo_dir='experiments/my_exp/data_tuning',
    seed=42,
    num_trials_model=15
)
best_trials = tuner(n_trials=20)
```

## Workflow

### 1. Run Optimization

```bash
uv run python scripts/tune_data.py \
  --experiment_dir experiments/my_experiment \
  --constraint_type K \
  --constraint_value 100 \
  --utility_metric precision \
  --target 0.8 \
  --num_trials_data 20 \
  --num_trials_model 15 \
  --seed 42
```

**Key Arguments**:
- `--constraint_type`: K, fpr, or recall
- `--constraint_value`: Numeric constraint value
- `--utility_metric`: precision or recall
- `--target`: Target metric value
- `--num_trials_data`: Data optimization trials (outer loop)
- `--num_trials_model`: Model hyperparameter trials per data trial (inner loop)

### 2. Review Results

Results saved to `experiments/my_experiment/data_tuning/`:
- `data_tuning_study.db`: Optuna database
- `pareto_front.png`: Pareto front visualization
- `best_trials.txt`: Best parameter configurations

```bash
# View Pareto front
open experiments/my_experiment/data_tuning/pareto_front.png

# View best trials
cat experiments/my_experiment/data_tuning/best_trials.txt
```

### 3. Apply Best Parameters

Script automatically updates `data.yaml` with best trial (closest to origin). To select specific trial:

```python
from src.data_tuning import Optimizer

optimizer.update_config_with_trial(trial_number=17)
```

### 4. Regenerate Data

```bash
uv run python scripts/generate.py \
  --experiment_dir experiments/my_experiment \
  --force
```

## API Reference

### DataTuner

```python
class DataTuner:
    def __init__(
        data_conf_file: str,
        config: dict,                   # Combined preprocessing + model config
        generator: DataGenerator,
        preprocessor: DataPreprocessor,
        target: float,
        constraint_type: str,           # 'K', 'fpr', 'recall'
        constraint_value: float,
        utility_metric: str,            # 'precision', 'recall'
        model: str = 'DecisionTreeClassifier',
        bo_dir: str = 'tmp',
        seed: int = 0,
        num_trials_model: int = 1,
        verbose: bool = False
    )

    def __call__(self, n_trials: int) -> List[optuna.Trial]:
        """Run data tuning optimization."""
```

### Optimizer

```python
class Optimizer:
    def __init__(
        data_conf_file: str,
        config: dict,
        generator: DataGenerator,
        preprocessor: DataPreprocessor,
        target: float,
        constraint_type: str,
        constraint_value: float,
        utility_metric: str,
        model: str = 'DecisionTreeClassifier',
        bank: str = None,
        bo_dir: str = 'tmp',
        seed: int = 0,
        num_trials_model: int = 1,
        verbose: bool = False
    )

    def optimize(self, n_trials: int = 10) -> List[optuna.Trial]:
        """Run Bayesian optimization."""

    def update_config_with_trial(self, trial_number: int):
        """Update data.yaml with specific trial parameters."""
```

### Constrained Utility Metric

```python
from src.ml.metrics import constrained_utility_metric

constrained_utility_metric(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    constraint_type: str,        # 'K', 'fpr', 'recall'
    constraint_value: float,
    utility_metric: str          # 'precision', 'recall'
) -> float
```

## Optimization Search Space

Defined in `data.yaml` under `optimisation_bounds`:

```yaml
optimisation_bounds:
  mean_amount_sar: [600, 900]
  std_amount_sar: [200, 400]
  mean_outcome_sar: [300, 600]
  std_outcome_sar: [100, 200]
  prob_spend_cash: [0.1, 0.5]
  mean_phone_change_frequency_sar: [1260, 1460]
  std_phone_change_frequency_sar: [265, 465]
  mean_bank_change_frequency_sar: [1260, 1460]
  std_bank_change_frequency_sar: [265, 465]
  prob_participate_in_multiple_sars: [0.0, 0.5]
```

## Notes

- **Continue studies**: Uses Optuna's `load_if_exists=True` - run multiple times to add more trials
- **Multiple banks**: Automatically uses first bank found in data
- **Skip auto-update**: Use `--no-update-config` to manually select trial later
- **Realistic targets**: With imbalanced data, expect Precision@K: 0.6-0.8, Recall@FPR: 0.5-0.7

## Related

- [Data Creation](../data_creation/README.md)
- [Feature Engineering](../feature_engineering/README.md)
- [ML Training](../ml/README.md)
