"""
Shared pytest fixtures for tune tests
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path


@pytest.fixture
def sample_config_yaml():
    """Create a sample configuration YAML"""
    return {
        "default": {
            "mean_amount": 1000,
            "std_amount": 200,
            "num_accounts": 1000
        },
        "optimisation_bounds": {
            "mean_amount": [500, 2000],
            "std_amount": [100, 500]
        }
    }


@pytest.fixture
def temp_yaml_config_file(sample_config_yaml):
    """Create a temporary YAML configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_yaml, f)
        path = f.name

    yield path

    # Cleanup
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def optimizer_config():
    """Sample optimizer configuration"""
    return {
        'preprocess': {
            'preprocessed_data_dir': '/tmp/preprocessed'
        },
        'DecisionTreeClassifier': {
            'default': {
                'client_type': 'TabularClient',
                'criterion': 'gini',
                'random_state': 42
            },
            'search_space': {
                'max_depth': (1, 20)
            },
            'isolated': {
                'clients': {
                    'BANK001': {'data_path': '/tmp/data/BANK001'}
                }
            }
        }
    }
