"""
Shared pytest fixtures for preprocessing tests
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_nodes_df():
    """Create a sample nodes DataFrame for testing"""
    return pd.DataFrame({
        'account': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'is_sar': [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        'bank': ['BANK001'] * 10,
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10)
    })


@pytest.fixture
def sample_edges_df():
    """Create a sample edges DataFrame for testing"""
    return pd.DataFrame({
        'src': [1, 2, 3, 4, 5, 6, 7, 8],
        'dst': [2, 3, 4, 5, 6, 7, 8, 9],
        'amount': np.random.rand(8) * 1000
    })


@pytest.fixture
def sample_alert_members_df():
    """Create a sample alert_members DataFrame for testing"""
    return pd.DataFrame({
        'accountID': [2, 4, 7],
        'reason': ['fan_in', 'fan_out', 'cycle']
    })


@pytest.fixture
def sample_transactions_df():
    """Create a sample transaction DataFrame for testing"""
    np.random.seed(42)
    n_transactions = 100

    # Create realistic transaction data with numeric account IDs
    account_ids = np.random.randint(1000, 1010, n_transactions)
    dest_ids = np.random.randint(1000, 1010, n_transactions)

    # Ensure some transactions go to different banks
    bank_orig = []
    bank_dest = []
    for i in range(n_transactions):
        if i < 70:
            bank_orig.append('BANK001')
            bank_dest.append(np.random.choice(['BANK001', 'sink']))
        elif i < 90:
            bank_orig.append('BANK001')
            bank_dest.append('BANK002')
        else:
            bank_orig.append('BANK002')
            bank_dest.append(np.random.choice(['BANK001', 'BANK002', 'sink']))

    return pd.DataFrame({
        'step': np.random.randint(1, 31, n_transactions),
        'nameOrig': account_ids,
        'nameDest': dest_ids,
        'amount': np.random.uniform(10, 1000, n_transactions),
        'bankOrig': bank_orig,
        'bankDest': bank_dest,
        'type': ['TRANSFER'] * n_transactions,
        'oldbalanceOrig': np.random.uniform(1000, 10000, n_transactions),
        'newbalanceOrig': np.random.uniform(1000, 10000, n_transactions),
        'oldbalanceDest': np.random.uniform(1000, 10000, n_transactions),
        'newbalanceDest': np.random.uniform(1000, 10000, n_transactions),
        'daysInBankOrig': np.random.randint(0, 365, n_transactions),
        'daysInBankDest': np.random.randint(0, 365, n_transactions),
        'phoneChangesOrig': np.random.randint(0, 5, n_transactions),
        'phoneChangesDest': np.random.randint(0, 5, n_transactions),
        'isSAR': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05]),
        'patternID': np.random.randint(-1, 100, n_transactions),  # -1 for generic, 0+ for patterns
        'modelType': np.random.randint(0, 16, n_transactions)  # 0=generic, 1-8=SAR, 10-15=normal
    })


@pytest.fixture
def basic_preprocessor_config():
    """Create a basic configuration for DataPreprocessor"""
    return {
        'num_windows': 2,
        'window_len': 10,
        'train_start_step': 1,
        'train_end_step': 15,
        'val_start_step': 16,
        'val_end_step': 22,
        'test_start_step': 23,
        'test_end_step': 30,
        'include_edges': False
    }


@pytest.fixture
def preprocessor_config_with_edges():
    """Create a configuration with edge features enabled"""
    return {
        'num_windows': 2,
        'window_len': 10,
        'train_start_step': 1,
        'train_end_step': 15,
        'val_start_step': 16,
        'val_end_step': 22,
        'test_start_step': 23,
        'test_end_step': 30,
        'include_edges': True
    }


@pytest.fixture
def temp_parquet_file(sample_transactions_df):
    """Create a temporary parquet file with sample transactions"""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        sample_transactions_df.to_parquet(tmp.name)
        yield tmp.name
        Path(tmp.name).unlink()
