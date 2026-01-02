"""
Shared pytest fixtures for AMLSim tests
"""
import os
import sys
import yaml
import pytest
import tempfile
from pathlib import Path

# Imports from src
from src.data_creation.python.temporal_simulation.account import Account
from src.data_creation.python.temporal_simulation.simulator import AMLSimulator
from src.data_creation.python.spatial_simulation.transaction_graph_generator import TransactionGenerator
from src.utils.config import load_data_config


@pytest.fixture
def project_root():
    """Return the project root directory (not tests directory)"""
    # Go up 3 levels: conftest.py -> data_creation -> tests -> project_root
    return Path(__file__).parent.parent.parent


@pytest.fixture
def test_config_path(project_root):
    """Return path to test configuration file"""
    return project_root / 'tests' / 'data_creation' / 'parameters' / 'small_test' / 'data.yaml'


@pytest.fixture
def test_config(test_config_path, project_root):
    """Load test configuration"""
    conf = load_data_config(str(test_config_path))

    # Override paths to match generate_test_data fixture structure
    test_config_dir = str(project_root / "tests" / "data_creation" / "parameters" / "small_test")
    test_spatial_dir = str(project_root / "tests" / "data_creation" / "parameters" / "small_test" / "spatial")
    test_temporal_dir = str(project_root / "tests" / "data_creation" / "parameters" / "small_test" / "temporal")

    conf['input']['directory'] = test_config_dir
    conf['temporal']['directory'] = test_spatial_dir
    conf['output']['directory'] = test_temporal_dir

    return conf


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_account():
    """Create a sample account for testing"""
    return Account(
        account_id=1,
        customer_id=100,
        initial_balance=10000.0,
        is_sar=False,
        bank_id='BANK001',
        random_state=42
    )


@pytest.fixture
def sample_sar_account():
    """Create a sample SAR account for testing"""
    return Account(
        account_id=2,
        customer_id=200,
        initial_balance=50000.0,
        is_sar=True,
        bank_id='BANK001',
        random_state=42
    )


@pytest.fixture
def configured_account(sample_account):
    """Create an account with behavior parameters set"""
    sample_account.set_parameters(
        prob_income=0.1,
        mean_income=1000,
        std_income=100,
        prob_income_sar=0.2,
        mean_income_sar=2000,
        std_income_sar=200,
        mean_outcome=500,
        std_outcome=100,
        mean_outcome_sar=1000,
        std_outcome_sar=200,
        prob_spend_cash=0.1,
        mean_phone_change_frequency=1460,
        std_phone_change_frequency=365,
        mean_phone_change_frequency_sar=1304,
        std_phone_change_frequency_sar=370,
        mean_bank_change_frequency=1460,
        std_bank_change_frequency=365,
        mean_bank_change_frequency_sar=1305,
        std_bank_change_frequency_sar=412
    )
    return sample_account


@pytest.fixture(scope="session", autouse=True)
def generate_test_data(request):
    """Generate spatial graph data needed for temporal/e2e tests (runs once per session)"""
    # Skip data generation for unit tests
    # Only run if integration or e2e tests are being executed
    marker_expr = request.config.getoption('-m')
    if marker_expr and 'unit' in marker_expr:
        # Skip if explicitly running unit tests
        return

    # Check if any integration/e2e tests are in the session
    has_integration_or_e2e = any(
        'integration' in str(item.fspath) or 'e2e' in str(item.fspath)
        for item in request.session.items
    )

    if has_integration_or_e2e:
        original_dir = os.getcwd()
        try:
            # Change to project root (tests is 2 levels up from conftest.py)
            project_root = Path(__file__).parent.parent.parent
            os.chdir(project_root)

            # Generate spatial data for small_test config
            config_path = project_root / "tests" / "data_creation" / "parameters" / "small_test" / "data.yaml"

            conf = load_data_config(str(config_path))

            # Override paths for test structure
            # Input: config files
            test_config_dir = str(project_root / "tests" / "data_creation" / "parameters" / "small_test")
            # Output: generated spatial data (separate directory to avoid overwriting config)
            test_spatial_dir = str(project_root / "tests" / "data_creation" / "parameters" / "small_test" / "spatial")
            # Output: generated temporal data
            test_temporal_dir = str(project_root / "tests" / "data_creation" / "parameters" / "small_test" / "temporal")

            # Create output directories if they don't exist
            Path(test_spatial_dir).mkdir(exist_ok=True)
            Path(test_temporal_dir).mkdir(exist_ok=True)

            conf['input']['directory'] = test_config_dir
            conf['temporal']['directory'] = test_spatial_dir
            conf['output']['directory'] = test_temporal_dir

            # Generate transaction graph
            txg = TransactionGenerator(conf, "test")
            txg.set_num_accounts()
            txg.generate_normal_transactions()
            txg.load_account_list()
            txg.load_normal_models()
            txg.build_normal_models()
            txg.set_main_acct_candidates()
            txg.load_alert_patterns()
            txg.mark_active_edges()

            # Write output files that temporal simulator expects
            txg.write_account_list()
            txg.write_transaction_list()
            txg.write_normal_models()
            txg.write_alert_account_list()

        finally:
            os.chdir(original_dir)


@pytest.fixture
def small_simulator_with_data(project_root, test_config, generate_test_data):
    """Create a small simulator instance with generated spatial data for testing"""
    # Change to project root for proper file paths
    original_dir = os.getcwd()
    os.chdir(project_root)

    try:
        # Data already generated by generate_test_data fixture
        simulator = AMLSimulator(test_config)
        yield simulator
    finally:
        os.chdir(original_dir)


@pytest.fixture
def small_simulator(project_root, test_config):
    """Create a small simulator instance for testing (without data generation)"""
    # Change to project root for proper file paths
    original_dir = os.getcwd()
    os.chdir(project_root)

    try:
        simulator = AMLSimulator(test_config)
        yield simulator
    finally:
        os.chdir(original_dir)


# ============================================================================
# Spatial Simulation (Graph Generation) Fixtures
# ============================================================================

class TransactionGraphFixture:
    """Helper class for creating transaction graph fixtures"""
    def __init__(self, config_str, clean=False):
        self.config_str = config_str

        base_dir = Path(__file__).parent
        config_path = base_dir / config_str

        conf = load_data_config(str(config_path))

        # Override paths for test structure (since it doesn't match the expected structure)
        test_config_dir = str(config_path.parent)
        conf['input']['directory'] = test_config_dir
        conf['temporal']['directory'] = test_config_dir
        conf['output']['directory'] = test_config_dir

        self.txg = TransactionGenerator(conf, "test")
        self.txg.set_num_accounts()
        self.txg.generate_normal_transactions()
        self.txg.load_account_list()
        self.txg.load_normal_models()
        self.txg.build_normal_models()

        if not clean:
            self.txg.set_main_acct_candidates()
            self.txg.load_alert_patterns()
            self.txg.mark_active_edges()


@pytest.fixture
def small_clean_graph():
    """Small transaction graph with normal models only (no alerts)"""
    config_str = "parameters/small_test/data.yaml"
    return TransactionGraphFixture(config_str, clean=True).txg


@pytest.fixture
def large_clean_graph():
    """Large transaction graph with normal models only (no alerts)"""
    config_str = "parameters/large_test/config/data.yaml"
    return TransactionGraphFixture(config_str, clean=True).txg


@pytest.fixture
def small_graph():
    """Small transaction graph with normal models and alert patterns"""
    config_str = "parameters/small_test/data.yaml"
    return TransactionGraphFixture(config_str, clean=False).txg


@pytest.fixture
def large_graph():
    """Large transaction graph with normal models and alert patterns"""
    config_str = "parameters/large_test/config/data.yaml"
    return TransactionGraphFixture(config_str, clean=False).txg
