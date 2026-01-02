# AMLSim - Anti-Money Laundering Simulator

This project is a multi-agent simulator for anti-money laundering (AML) research, generating synthetic transaction data for algorithm development and testing. This implementation is part of the **AMLgentex** project, which extends the original AMLSim framework with enhanced capabilities for data-driven AML research.

## Citation

Please cite the following papers if you use this simulator:

**AMLgentex (this implementation):**
```bibtex
@article{AMLgentex2025,
  title = {AMLgentex: Mobilizing Data-Driven Research to Combat Money Laundering},
  author = {Östman, Johan and Callisen, Edvin and Chen, Anton and Ausmees, Kristiina and Gårdh, Emanuel and Zamac, Jovan and Goldsteine, Jolanta and Wefer, Hugo and Whelan, Simon and Reimegård, Markus},
  journal = {arXiv preprint arXiv:2506.13989},
  year = {2025}
}
```

**Original AMLSim:**
```bibtex
@misc{AMLSim,
  author = {Toyotaro Suzumura and Hiroki Kanezashi},
  title = {{Anti-Money Laundering Datasets}: {InPlusLab} Anti-Money Laundering Datasets},
  howpublished = {\url{http://github.com/IBM/AMLSim/}},
  year = 2021
}
```

## Dependencies

- **Python 3.8+**
- **Python packages** (install with `pip install -r requirements.txt` or via uv/poetry)
  - numpy
  - pandas
  - networkx
  - scipy

## Quick Start

### 1. Run a Complete Simulation

Run both spatial (graph generation) and temporal (transaction execution) phases:

```bash
python python/run_simulation.py paramFiles/100_accounts/conf.json
```

This will:
1. Generate transaction graph from degree distribution
2. Execute temporal simulation over the specified time steps
3. Output transaction logs to the configured output directory

### 2. Configuration

Edit `paramFiles/<scenario>/conf.json` to customize:

```json
{
  "general": {
    "random_seed": 0,
    "simulation_name": "100_accounts",
    "total_steps": 367,
    "base_date": "2023-01-01"
  },
  "input": {
    "directory": "paramFiles/100_accounts",
    "accounts": "accounts.csv",
    "alert_patterns": "alertPatterns.csv",
    "normal_models": "normalModels.csv",
    "degree": "degree.csv"
  },
  "output": {
    "directory": "outputs",
    "transactions": "transactions.csv",
    "transaction_log": "tx_log.parquet",
    "accounts": "accounts.csv"
  }
}
```

### 3. Output Files

After running the simulation, find output in the configured directory:
- `tx_log.parquet` - Complete transaction log with all transaction details (efficient binary format)
- `accounts.csv` - Final account states
- `alert_transactions.csv` - Transactions flagged as suspicious

## Project Structure

```
AMLSim/
├── python/                          # Python implementation
│   ├── run_simulation.py           # Main entry point
│   ├── spatial_simulation/         # Graph generation
│   │   ├── generate_scalefree.py
│   │   ├── transaction_graph_generator.py
│   │   └── insert_patterns.py
│   └── temporal_simulation/        # Transaction execution
│       ├── simulator.py
│       ├── account.py
│       ├── alert_patterns.py
│       └── normal_models.py
├── paramFiles/                      # Simulation scenarios
│   ├── 100_accounts/
│   ├── 1K_accts/
│   ├── 10K_accts/
│   └── ...
└── tests/                          # Test suite
```

## Simulation Phases

### Spatial Simulation (Graph Generation)
Generates the transaction network structure:
1. **Degree Distribution** - Creates account connectivity patterns
2. **Transaction Graph** - Builds directed graph of potential transactions
3. **Pattern Insertion** - Adds suspicious activity patterns (optional)

**Key Outputs:**
- `alert_members.csv` - Alert pattern participants (accountID, isMain, isSAR, bankID, sourceType, phase)
- `normal_models.csv` - Normal model participants (modelID, type, accountID, isMain, isSAR)
- `transactions.csv` - Spatial graph edges (source, target, mean/std/min/max amounts)

### Temporal Simulation (Transaction Execution)
Executes transactions over time:
1. **Account Behaviors** - Random income/spending based on balance history
2. **Normal Patterns** - Regular transaction patterns between accounts
3. **Alert Patterns** - Suspicious money laundering typologies

**Dynamic Timing System:**
Each pattern (both normal and alert) has its timing dynamically determined:
- **Duration**: Sampled from lognormal distribution using `mean_duration_normal/alert` and `std_duration_normal/alert`
- **Start Time**: Uniformly sampled from valid range [0, total_steps - duration - 1]
- **Burstiness**: Level sampled from {1,2,3,4} with probabilities controlled by `burstiness_bias_normal/alert`
- **Transaction Steps**: Placed within pattern duration using beta distributions

**Key Outputs:**
- `tx_log.parquet` - Complete transaction history with timestamps, amounts, and metadata

## Transaction Types

- **TRANSFER** - Regular account-to-account transfers
- **CASH** - Cash deposits/withdrawals (used in suspicious patterns)
- **INITALBALANCE** - Initial account funding

## Alert Pattern Types

AMLSim supports various money laundering typologies:
- **fan_out** - One account sends to multiple recipients
- **fan_in** - Multiple accounts send to one recipient
- **cycle** - Circular transaction pattern
- **scatter_gather** - Fan-out followed by fan-in
- **gather_scatter** - Fan-in followed by fan-out
- **bipartite** - Transactions between two groups
- **stack** - Sequential chain of transfers

## Transaction Timing Configuration

### Duration Parameters

Control how long transaction patterns remain active (time window in which transactions occur):

```yaml
default:
  mean_duration_normal: 6      # Mean duration in steps for normal patterns
  std_duration_normal: 3       # Standard deviation in steps for normal patterns
  mean_duration_alert: 9       # Mean duration in steps for alert patterns
  std_duration_alert: 5        # Standard deviation in steps for alert patterns
```

**How it works:**
- Duration is sampled from a lognormal distribution: `duration ~ lognormal(μ, σ)`
- Parameters are specified in **linear space** (actual steps), converted internally to log-space
- Lower mean → shorter pattern time windows
- Higher std → more duration variability
- Durations are clipped to [2, total_steps-1] to ensure validity

**Important:** The configured duration is the **time window** in which transactions can occur. The actual observed span (first to last transaction) will typically be shorter because transactions are randomly placed within this window using beta distributions.

### Burstiness Parameters

Control how transactions cluster within pattern duration:

```yaml
default:
  burstiness_bias_normal: 0.0   # Neutral (equal probability for all levels)
  burstiness_bias_alert: 0.0    # Neutral (equal probability for all levels)

optimisation_bounds:
  burstiness_bias_alert: [-2.0, 2.0]
```

**Bias → Level Probabilities:**
- `bias = -2.0`: Strongly favor uniform (level 1)
- `bias = 0.0`: Equal probability for all levels
- `bias = 2.0`: Strongly favor clustered (level 4)

**Burstiness Levels:**
1. **Level 1 - Uniform** (Beta(1,1)): Evenly spaced transactions
2. **Level 2 - Regular** (Beta(2,2)): Slightly variable spacing
3. **Level 3 - Early Cluster** (Beta(0.5,3)): Transactions concentrate early
4. **Level 4 - Bimodal** (Beta(0.3,0.3)): Tight clusters with large gaps

**Implementation:**
```python
# Probability of level i given bias
prob[i] = exp(bias * i) / sum(exp(bias * j) for j in [0,1,2,3])

# Sample transaction steps using beta distribution
alpha, beta = BETA_PARAMS[level]
fractions = np.random.beta(alpha, beta, num_transactions)
steps = start + (fractions * duration).astype(int)
```

### Transaction Ordering Guarantees

Patterns with causal dependencies maintain correct ordering:
- **Forward (A→B→C)**: First transaction step < second transaction step
- **Mutual (loan→repay)**: Loan step < repayment step
- **Cycle**: All steps sorted in cycle order
- **Stack**: Layer N receives before sending to layer N+1
- **Scatter-Gather**: Scatter phase completes before gather starts
- **Gather-Scatter**: Gather phase completes before scatter starts

Beta distribution sorting (`np.sort(fractions)`) ensures temporal ordering is maintained.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

## Notes

- Use `random_seed` in config for reproducible simulations
- Alert patterns with `source_type: "CASH"` inject illicit funds as cash_balance
- Transaction validation ensures sufficient account balances
- SAR (Suspicious Activity Report) accounts have different spending behavior
