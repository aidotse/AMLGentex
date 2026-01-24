"""
Unit tests for ml/clients/torch_client.py
"""
import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.ml.clients.torch_client import TorchClient
from src.ml.models.torch_models import MLP


@pytest.fixture
def sample_train_data(tmp_path):
    """Create sample training parquet file with balanced classes"""
    np.random.seed(42)
    n_samples = 200
    n_pos = 40  # 20% positive
    n_neg = n_samples - n_pos

    df = pd.DataFrame({
        'account': [f'acc_{i}' for i in range(n_samples)],
        'bank': ['bank_A'] * n_samples,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'is_sar': [1] * n_pos + [0] * n_neg
    })
    df = df.sample(frac=1, random_state=42)  # Shuffle
    path = tmp_path / "train.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_val_data(tmp_path):
    """Create sample validation parquet file"""
    np.random.seed(43)
    n_samples = 40
    n_pos = 8
    n_neg = n_samples - n_pos

    df = pd.DataFrame({
        'account': [f'acc_{i}' for i in range(n_samples)],
        'bank': ['bank_A'] * n_samples,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'is_sar': [1] * n_pos + [0] * n_neg
    })
    path = tmp_path / "val.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_test_data(tmp_path):
    """Create sample test parquet file"""
    np.random.seed(44)
    n_samples = 60
    n_pos = 12
    n_neg = n_samples - n_pos

    df = pd.DataFrame({
        'account': [f'acc_{i}' for i in range(n_samples)],
        'bank': ['bank_A'] * n_samples,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'is_sar': [1] * n_pos + [0] * n_neg
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.mark.unit
class TestTorchClientInit:
    """Tests for TorchClient initialization"""

    def test_init_with_separate_files(self, sample_train_data, sample_val_data, sample_test_data):
        """Test initialization with separate train/val/test files"""
        client = TorchClient(
            id='test_client',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        assert client.id == 'test_client'
        assert client.seed == 42
        assert client.device == 'cpu'
        assert client.model is not None

    def test_init_creates_model(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that model is created correctly"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        # Model should have correct input dimension (3 features)
        assert client.model.input_layer.in_features == 3

    def test_init_creates_optimizer(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that optimizer is created correctly"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        assert isinstance(client.optimizer, torch.optim.Adam)

    def test_init_creates_weighted_sampler(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that weighted sampler is created for class imbalance"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        # Trainloader should exist
        assert client.trainloader is not None


@pytest.mark.unit
class TestTorchClientTrain:
    """Tests for TorchClient training"""

    def test_train_updates_model(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that train() updates model parameters"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
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
class TestTorchClientEvaluate:
    """Tests for TorchClient evaluation"""

    def test_evaluate_returns_loss_and_predictions(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that evaluate returns loss, predictions, and labels"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        loss, y_pred, y_true = client.evaluate(dataset='trainset')

        assert isinstance(loss, float)
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(y_true, np.ndarray)
        assert len(y_pred) == len(y_true)

    def test_evaluate_valset(self, sample_train_data, sample_val_data, sample_test_data):
        """Test evaluation on validation set"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        loss, y_pred, y_true = client.evaluate(dataset='valset')

        assert len(y_pred) == 40  # Validation set size


@pytest.mark.unit
class TestTorchClientRun:
    """Tests for TorchClient run"""

    def test_run_returns_results(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that run() returns results dict"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        results = client.run(n_rounds=5, eval_every=2, n_warmup_rounds=2)

        assert 'trainset' in results
        assert 'valset' in results
        assert 'testset' in results


@pytest.mark.unit
class TestTorchClientParameters:
    """Tests for TorchClient parameter methods"""

    def test_get_parameters(self, sample_train_data, sample_val_data, sample_test_data):
        """Test get_parameters returns model parameters"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        params = client.get_parameters()

        assert isinstance(params, dict)
        assert len(params) > 0

    def test_set_parameters(self, sample_train_data, sample_val_data, sample_test_data):
        """Test set_parameters sets model parameters"""
        client1 = TorchClient(
            id='client1',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )
        client2 = TorchClient(
            id='client2',
            seed=123,  # Different seed = different init
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        params1 = client1.get_parameters()
        client2.set_parameters(params1)
        params2 = client2.get_parameters()

        for name in params1:
            assert torch.allclose(params1[name], params2[name])

    def test_compute_gradients(self, sample_train_data, sample_val_data, sample_test_data):
        """Test compute_gradients returns gradient dict"""
        client = TorchClient(
            id='test',
            seed=42,
            device='cpu',
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            batch_size=32,
            Model=MLP,
            optimizer='Adam',
            criterion='BCEWithLogitsLoss',
            n_hidden_layers=1,
            hidden_dim=16,
            output_dim=1,
            lr=0.01
        )

        gradients = client.compute_gradients()

        assert isinstance(gradients, dict)
        assert len(gradients) > 0
