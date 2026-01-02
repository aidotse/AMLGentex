# AMLSim Test Suite

Comprehensive test suite for the AMLSim (Anti-Money Laundering Simulator) Python implementation.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components (150+ tests)
│   ├── spatial/            # Spatial simulation (graph generation) tests
│   │   ├── test_degree_distribution.py  # Scale-free degree distribution (10 tests)
│   │   ├── test_nominator.py            # Node nomination logic (39 tests)
│   │   └── test_utils.py                # Amount generators and utilities
│   └── temporal/           # Temporal simulation (time-step) tests
│       ├── test_account.py              # Account class behavior (14 tests)
│       ├── test_alert_pattern_execution.py  # Alert pattern ordering (9 tests)
│       ├── test_alert_pattern_ordering.py   # Comprehensive ordering tests (17 tests)
│       ├── test_normal_model_execution.py   # Normal model ordering (23 tests)
│       ├── test_pattern_generators.py   # Pattern generator functions (20 tests)
│       ├── test_salary_distribution.py  # Swedish salary distribution (16 tests)
│       ├── test_transaction_schedules.py  # Beta distribution timing (22 tests)
│       └── test_utils.py                # Utilities (sigmoid, distributions) (13 tests)
│
├── integration/            # Integration tests for component interactions (36 tests)
│   ├── spatial/            # Spatial pipeline tests
│   │   ├── test_alert_patterns.py       # Alert pattern generation (2 tests)
│   │   ├── test_normal_models.py        # Normal model generation (2 tests)
│   │   └── test_overlapping_patterns.py # Multi-pattern participation (1 test)
│   └── temporal/           # Temporal pipeline tests
│       └── test_temporal_pipeline.py    # Full temporal pipeline (6 tests)
│
├── e2e/                    # End-to-end tests for complete workflows (4 tests)
│   └── test_full_simulation.py          # Complete simulation runs
│
├── fixtures/               # Shared test data and fixtures
├── conftest.py             # Shared pytest fixtures
└── README.md               # This file
```

**Total: 190+ passing tests**

## Test Coverage

### Spatial Simulation Tests (51 tests)
- **Degree Distribution** (10 tests): Power-law degree generation, balance, parameters
- **Nominator** (39 tests): Candidate selection, count management, relationship checking
- **Alert Patterns** (2 tests): Pattern counts, structure validation

### Temporal Simulation Tests (140+ tests)
- **Account Behavior** (14 tests): Balance updates, transaction processing
- **Alert Pattern Execution** (9 tests): Stack layering, bipartite flow ordering
- **Alert Pattern Ordering** (17 tests): Cycle, forward, stack, scatter-gather, gather-scatter, bipartite, fan-in/out, random patterns
- **Normal Model Execution** (23 tests): Forward chains, mutual loans, periodical payments
- **Pattern Generators** (20 tests): Fan-in, fan-out, cycle, scatter-gather patterns
- **Salary Distribution** (16 tests): Swedish income sampling, log-normal distribution
- **Transaction Schedules** (22 tests): Beta distribution-based timing, burstiness levels, reproducibility
- **Utilities** (13 tests): Sigmoid, truncated normal distributions
- **Temporal Pipeline** (6 tests): Full simulation execution

### Integration & E2E Tests (40 tests)
- **Spatial Integration** (5 tests): Graph generation with multiple pattern types
- **Temporal Integration** (6 tests): Account loading, transaction execution
- **End-to-End** (4 tests): Complete simulation workflows

## Running Tests

### Run all tests
```bash
uv run pytest
```

### Run specific test categories
```bash
# Unit tests only
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# E2E tests only
uv run pytest -m e2e

# Spatial simulation tests
uv run pytest -m spatial

# Temporal simulation tests
uv run pytest -m temporal
```

### Run specific test files
```bash
# Test account functionality
uv run pytest tests/unit/temporal/test_account.py

# Test salary distribution
uv run pytest tests/unit/temporal/test_salary_distribution.py

# Test nominator (node nomination logic)
uv run pytest tests/unit/spatial/test_nominator.py

# Test pattern generators
uv run pytest tests/unit/temporal/test_pattern_generators.py

# Test normal model execution
uv run pytest tests/unit/temporal/test_normal_model_execution.py

# Test alert pattern execution
uv run pytest tests/unit/temporal/test_alert_pattern_execution.py

# Test full simulation
uv run pytest tests/e2e/test_full_simulation.py
```

### Run with verbose output
```bash
uv run pytest -v
```

### Run tests and skip slow ones
```bash
uv run pytest -m "not slow"
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for component interactions
- `@pytest.mark.e2e` - End-to-end tests for complete workflows
- `@pytest.mark.spatial` - Tests for spatial simulation (graph generation)
- `@pytest.mark.temporal` - Tests for temporal simulation (time-step execution)
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.requires_data` - Tests that require parameter files

## Writing New Tests

### Unit Tests

Unit tests should test individual functions or methods in isolation:

```python
import pytest

@pytest.mark.unit
@pytest.mark.temporal
def test_account_balance(sample_account):
    """Test account balance updates correctly"""
    initial = sample_account.balance
    sample_account.receive_income(1000.0, "TRANSFER")
    assert sample_account.balance == initial + 1000.0
```

### Integration Tests

Integration tests should test how components work together:

```python
import pytest

@pytest.mark.integration
@pytest.mark.temporal
def test_simulation_pipeline(project_root, test_config_path):
    """Test complete temporal simulation pipeline"""
    simulator = AMLSimulator(str(test_config_path))
    simulator.load_accounts()
    simulator.load_transactions()
    assert len(simulator.accounts) > 0
```

### End-to-End Tests

E2E tests should test complete workflows from start to finish:

```python
import pytest

@pytest.mark.e2e
@pytest.mark.slow
def test_full_simulation(project_root, test_config_path):
    """Test complete simulation run"""
    simulator = AMLSimulator(str(test_config_path))
    # ... run complete simulation
    assert len(simulator.transactions) > 0
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `project_root` - Path to project root directory
- `test_config_path` - Path to test configuration file
- `test_config` - Loaded test configuration
- `sample_account` - Sample non-SAR account
- `sample_sar_account` - Sample SAR account
- `configured_account` - Account with behavior parameters set
- `temp_output_dir` - Temporary directory for test outputs

## Coverage

To run tests with coverage reporting (requires pytest-cov):

```bash
uv run pytest --cov=python --cov-report=html --cov-report=term-missing
```

View coverage report:
```bash
open htmlcov/index.html
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines. Fast tests run on every commit, slow tests run nightly.

```bash
# Quick test suite (exclude slow tests)
uv run pytest -m "not slow"

# Full test suite (include all tests)
uv run pytest
```

## Test Data

Test data is located in `tests/data_creation/parameters/` with two configurations:

**small_test/** - Fast tests with minimal data (5 accounts)
- `data.yaml` - Test configuration
- `accounts.csv` - Sample account data
- `alertPatterns.csv` - Sample alert patterns
- `normalModels.csv` - Sample normal transaction models
- `degree.csv` - Degree distribution parameters
- `transactionType.csv` - Transaction type definitions

**large_test/** - Comprehensive tests with larger dataset (100K accounts)
- `data.yaml` - Test configuration
- Same CSV files as small_test but with larger scale

## Debugging Tests

### Run a single test with output
```bash
uv run pytest tests/unit/temporal/test_account.py::TestAccount::test_account_initialization -v -s
```

### Drop into debugger on failure
```bash
uv run pytest --pdb
```

### See all print statements
```bash
uv run pytest -s
```

## Key Test Files

### Timing System Tests (61 tests)

**Transaction Schedules** (`test_transaction_schedules.py` - 22 tests)
- **Burstiness Level Sampling**: Tests bias → probability conversion (5 tests)
- **Beta Distribution Scheduler**: Tests step generation with all 4 levels (8 tests)
- **Normal Models with Burstiness**: Tests ForwardModel, MutualModel, PeriodicalModel (5 tests)
- **Reproducibility**: Tests deterministic behavior with random seeds (3 tests)
- Validates that transactions maintain temporal ordering

**Alert Pattern Ordering** (`test_alert_pattern_ordering.py` - 17 tests)
- **Cycle Pattern**: Tests circular transaction ordering
- **Forward Pattern**: Tests sequential A→B→C chain ordering
- **Stack Pattern**: Tests layer-based execution (layer N receives before sending)
- **Scatter-Gather/Gather-Scatter**: Tests two-phase ordering constraints
- **Bipartite Pattern**: Tests unidirectional flow between groups
- **Fan-in/Fan-out**: Tests sorted transaction execution
- **Random Pattern**: Tests sorted random transactions
- **Boundary Tests**: Tests all patterns respect start/end time bounds

**Normal Model Execution** (`test_normal_model_execution.py` - 23 tests)
- Tests transaction ordering for ForwardModel, MutualModel, PeriodicalModel
- Validates execution constraints (loan before repayment, sequential forwarding)
- Tests with all 4 burstiness levels

### Other Important Test Files

**Salary Distribution** (`test_salary_distribution.py` - 16 tests)
- Tests Swedish salary sampling using dummy data (no external dependencies)
- Validates log-normal distribution properties, reproducibility, and caching
- Covers age group selection and monthly salary conversion

**Nominator** (`test_nominator.py` - 39 tests)
- Comprehensive testing of node nomination logic for spatial simulation
- Tests candidate selection for fan_in, fan_out, forward, single, mutual, periodical patterns
- Validates count management, relationship checking, and post-update behavior

**Pattern Generators** (`test_pattern_generators.py` - 20 tests)
- Tests individual pattern generator functions in isolation
- Covers fan_out, fan_in, cycle, bipartite, stack, scatter_gather, gather_scatter, random

**Alert Pattern Execution** (`test_alert_pattern_execution.py` - 9 tests)
- Tests stack pattern layered execution (receive before send)
- Tests bipartite pattern unidirectional flow constraint

**Degree Distribution** (`test_degree_distribution.py` - 10 tests)
- Tests scale-free power-law degree generation
- Validates degree balance and parameter effects

## Best Practices

1. **One assertion per test** - Each test should verify one specific behavior
2. **Use descriptive names** - Test names should explain what is being tested
3. **Arrange-Act-Assert** - Structure tests with setup, action, and verification
4. **Use fixtures** - Share common setup using pytest fixtures
5. **Mark appropriately** - Use markers to categorize tests
6. **Keep tests fast** - Unit tests should run in milliseconds
7. **Test edge cases** - Include boundary conditions and error cases
8. **Document intent** - Use docstrings to explain what is being tested
9. **Avoid external dependencies** - Use dummy data and fixtures instead of external files
10. **Test one thing** - Each test should focus on a single behavior or property
