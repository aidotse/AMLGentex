"""
Unit tests for ml/clients/torch_geometric_client.py
"""
import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.ml.clients.torch_geometric_client import TorchGeometricClient
from src.ml.models.gnn_models import GCN


@pytest.fixture
def sample_nodes_data(tmp_path):
    """Create sample nodes parquet file"""
    np.random.seed(42)
    n_nodes = 100
    n_pos = 20

    df = pd.DataFrame({
        'account': [f'node_{i}' for i in range(n_nodes)],
        'bank': ['bank_A'] * n_nodes,
        'feature1': np.random.randn(n_nodes),
        'feature2': np.random.randn(n_nodes),
        'feature3': np.random.randn(n_nodes),
        'is_sar': [1] * n_pos + [0] * (n_nodes - n_pos)
    })
    path = tmp_path / "nodes.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_edges_data(tmp_path):
    """Create sample edges parquet file"""
    np.random.seed(42)
    n_edges = 200

    # Create random edges between nodes
    src = [f'node_{i}' for i in np.random.randint(0, 100, n_edges)]
    dst = [f'node_{i}' for i in np.random.randint(0, 100, n_edges)]

    df = pd.DataFrame({
        'src': src,
        'dst': dst
    })
    path = tmp_path / "edges.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_val_nodes_data(tmp_path):
    """Create sample validation nodes parquet file"""
    np.random.seed(43)
    n_nodes = 30
    n_pos = 6

    df = pd.DataFrame({
        'account': [f'val_node_{i}' for i in range(n_nodes)],
        'bank': ['bank_A'] * n_nodes,
        'feature1': np.random.randn(n_nodes),
        'feature2': np.random.randn(n_nodes),
        'feature3': np.random.randn(n_nodes),
        'is_sar': [1] * n_pos + [0] * (n_nodes - n_pos)
    })
    path = tmp_path / "val_nodes.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_val_edges_data(tmp_path):
    """Create sample validation edges parquet file"""
    np.random.seed(43)
    n_edges = 50

    src = [f'val_node_{i}' for i in np.random.randint(0, 30, n_edges)]
    dst = [f'val_node_{i}' for i in np.random.randint(0, 30, n_edges)]

    df = pd.DataFrame({
        'src': src,
        'dst': dst
    })
    path = tmp_path / "val_edges.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_test_nodes_data(tmp_path):
    """Create sample test nodes parquet file"""
    np.random.seed(44)
    n_nodes = 40
    n_pos = 8

    df = pd.DataFrame({
        'account': [f'test_node_{i}' for i in range(n_nodes)],
        'bank': ['bank_A'] * n_nodes,
        'feature1': np.random.randn(n_nodes),
        'feature2': np.random.randn(n_nodes),
        'feature3': np.random.randn(n_nodes),
        'is_sar': [1] * n_pos + [0] * (n_nodes - n_pos)
    })
    path = tmp_path / "test_nodes.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_test_edges_data(tmp_path):
    """Create sample test edges parquet file"""
    np.random.seed(44)
    n_edges = 80

    src = [f'test_node_{i}' for i in np.random.randint(0, 40, n_edges)]
    dst = [f'test_node_{i}' for i in np.random.randint(0, 40, n_edges)]

    df = pd.DataFrame({
        'src': src,
        'dst': dst
    })
    path = tmp_path / "test_edges.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.mark.unit
class TestTorchGeometricClientInit:
    """Tests for TorchGeometricClient initialization"""

    def test_init_with_separate_files(self, sample_nodes_data, sample_edges_data,
                                       sample_val_nodes_data, sample_val_edges_data,
                                       sample_test_nodes_data, sample_test_edges_data):
        """Test initialization with separate node/edge files"""
        client = TorchGeometricClient(
            id='test_client',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        assert client.id == 'test_client'
        assert client.seed == 42
        assert client.device == 'cpu'
        assert client.model is not None

    def test_init_creates_gnn_model(self, sample_nodes_data, sample_edges_data,
                                     sample_val_nodes_data, sample_val_edges_data,
                                     sample_test_nodes_data, sample_test_edges_data):
        """Test that GNN model is created correctly"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        # Model should have correct input dimension (3 features)
        assert client.model.input_layer.in_channels == 3

    def test_init_with_random_split(self, sample_nodes_data, sample_edges_data):
        """Test initialization with random node split"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_size=0.2,
            testset_size=0.1,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        # Should have train/val/test masks
        assert hasattr(client.trainset, 'train_mask')
        assert hasattr(client.trainset, 'val_mask')
        assert hasattr(client.trainset, 'test_mask')

    def test_init_with_trainset_size(self, sample_nodes_data, sample_edges_data):
        """Test initialization with explicit trainset_size"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            trainset_size=0.6,
            valset_size=0.2,
            testset_size=0.2,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        # Should have custom masks set
        assert hasattr(client.trainset, 'train_mask')


@pytest.mark.unit
class TestTorchGeometricClientTrain:
    """Tests for TorchGeometricClient training"""

    def test_train_updates_model(self, sample_nodes_data, sample_edges_data,
                                  sample_val_nodes_data, sample_val_edges_data,
                                  sample_test_nodes_data, sample_test_edges_data):
        """Test that train() updates model parameters"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        params_before = {k: v.clone() for k, v in client.model.named_parameters()}
        client.train()
        params_after = {k: v.clone() for k, v in client.model.named_parameters()}

        # At least some parameters should have changed
        changed = any(
            not torch.allclose(params_before[k], params_after[k])
            for k in params_before
        )
        assert changed


@pytest.mark.unit
class TestTorchGeometricClientEvaluate:
    """Tests for TorchGeometricClient evaluation"""

    def test_evaluate_returns_loss_and_predictions(self, sample_nodes_data, sample_edges_data,
                                                    sample_val_nodes_data, sample_val_edges_data,
                                                    sample_test_nodes_data, sample_test_edges_data):
        """Test that evaluate returns loss, predictions, and labels"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        loss, y_pred, y_true = client.evaluate(dataset='trainset')

        assert isinstance(loss, float)
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(y_true, np.ndarray)

    def test_evaluate_valset(self, sample_nodes_data, sample_edges_data,
                             sample_val_nodes_data, sample_val_edges_data,
                             sample_test_nodes_data, sample_test_edges_data):
        """Test evaluation on validation set"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        loss, y_pred, y_true = client.evaluate(dataset='valset')

        assert len(y_pred) == 30  # Validation set size


@pytest.mark.unit
class TestTorchGeometricClientRun:
    """Tests for TorchGeometricClient run"""

    def test_run_returns_results(self, sample_nodes_data, sample_edges_data,
                                  sample_val_nodes_data, sample_val_edges_data,
                                  sample_test_nodes_data, sample_test_edges_data):
        """Test that run() returns results dict"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        results = client.run(n_rounds=5, eval_every=2, n_warmup_rounds=2)

        assert 'trainset' in results
        assert 'valset' in results
        assert 'testset' in results


@pytest.mark.unit
class TestTorchGeometricClientParameters:
    """Tests for TorchGeometricClient parameter methods"""

    def test_get_parameters(self, sample_nodes_data, sample_edges_data,
                            sample_val_nodes_data, sample_val_edges_data,
                            sample_test_nodes_data, sample_test_edges_data):
        """Test get_parameters returns model parameters (excluding layer norms)"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        params = client.get_parameters()

        assert isinstance(params, dict)
        # Should exclude layer_norms
        for name in params:
            assert 'layer_norms' not in name

    def test_compute_gradients(self, sample_nodes_data, sample_edges_data,
                               sample_val_nodes_data, sample_val_edges_data,
                               sample_test_nodes_data, sample_test_edges_data):
        """Test compute_gradients returns gradient dict"""
        client = TorchGeometricClient(
            id='test',
            seed=42,
            device='cpu',
            trainset_nodes=sample_nodes_data,
            trainset_edges=sample_edges_data,
            valset_nodes=sample_val_nodes_data,
            valset_edges=sample_val_edges_data,
            testset_nodes=sample_test_nodes_data,
            testset_edges=sample_test_edges_data,
            Model=GCN,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_conv_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        gradients = client.compute_gradients()

        assert isinstance(gradients, dict)
        assert len(gradients) > 0
