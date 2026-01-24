"""
Unit tests for ml/clients/sklearn_client.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.ml.clients.sklearn_client import SklearnClient


@pytest.fixture
def sample_train_data(tmp_path):
    """Create sample training parquet file"""
    np.random.seed(42)
    df = pd.DataFrame({
        'account': [f'acc_{i}' for i in range(100)],
        'bank': ['bank_A'] * 100,
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'is_sar': np.random.randint(0, 2, 100)
    })
    path = tmp_path / "train.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_val_data(tmp_path):
    """Create sample validation parquet file"""
    np.random.seed(43)
    df = pd.DataFrame({
        'account': [f'acc_{i}' for i in range(20)],
        'bank': ['bank_A'] * 20,
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
        'feature3': np.random.randn(20),
        'is_sar': np.random.randint(0, 2, 20)
    })
    path = tmp_path / "val.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def sample_test_data(tmp_path):
    """Create sample test parquet file"""
    np.random.seed(44)
    df = pd.DataFrame({
        'account': [f'acc_{i}' for i in range(30)],
        'bank': ['bank_A'] * 30,
        'feature1': np.random.randn(30),
        'feature2': np.random.randn(30),
        'feature3': np.random.randn(30),
        'is_sar': np.random.randint(0, 2, 30)
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.mark.unit
class TestSklearnClientInit:
    """Tests for SklearnClient initialization"""

    def test_init_with_separate_files(self, sample_train_data, sample_val_data, sample_test_data):
        """Test initialization with separate train/val/test files"""
        client = SklearnClient(
            id='test_client',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        assert client.id == 'test_client'
        assert client.seed == 42
        assert client.X_train.shape[0] == 100
        assert client.X_val.shape[0] == 20
        assert client.X_test.shape[0] == 30

    def test_init_with_split_sizes(self, sample_train_data):
        """Test initialization with split sizes"""
        client = SklearnClient(
            id='test_client',
            seed=42,
            trainset=sample_train_data,
            valset_size=0.2,
            testset_size=0.1,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        # Original has 100 samples
        # 20 for val, 10 for test, ~70 for train
        assert client.X_val.shape[0] == 20
        assert client.X_test.shape[0] == 10

    def test_init_drops_mask_columns(self, tmp_path):
        """Test that mask columns are dropped during init"""
        # Need enough samples to split
        np.random.seed(42)
        df = pd.DataFrame({
            'account': [f'acc_{i}' for i in range(50)],
            'bank': ['bank_A'] * 50,
            'feature1': np.random.randn(50),
            'is_sar': np.random.randint(0, 2, 50),
            'train_mask': [True] * 50,
            'val_mask': [False] * 50,
            'test_mask': [False] * 50
        })
        path = tmp_path / "train_with_masks.parquet"
        df.to_parquet(path)

        client = SklearnClient(
            id='test',
            seed=42,
            trainset=str(path),
            valset_size=0.2,
            testset_size=0.1,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        # Should have 1 feature (feature1), not masks
        assert client.X_train.shape[1] == 1

    def test_init_normalizes_features(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that features are normalized"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        # Normalized features should be in [0, 1] range (approximately)
        assert client.X_train.min() >= -0.1  # Some tolerance
        assert client.X_train.max() <= 1.1


@pytest.mark.unit
class TestSklearnClientTrain:
    """Tests for SklearnClient training"""

    def test_train_fits_model(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that train() fits the model"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        client.train()

        # Model should be fitted
        assert hasattr(client.model, 'estimators_')


@pytest.mark.unit
class TestSklearnClientEvaluate:
    """Tests for SklearnClient evaluation"""

    def test_evaluate_trainset(self, sample_train_data, sample_val_data, sample_test_data):
        """Test evaluation on trainset"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )
        client.train()

        loss, y_pred, y_true = client.evaluate(dataset='trainset')

        assert loss == 0.0  # SklearnClient always returns 0.0 for loss
        assert len(y_pred) == len(client.y_train)
        assert len(y_true) == len(client.y_train)

    def test_evaluate_valset(self, sample_train_data, sample_val_data, sample_test_data):
        """Test evaluation on valset"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )
        client.train()

        loss, y_pred, y_true = client.evaluate(dataset='valset')

        assert len(y_pred) == len(client.y_val)

    def test_evaluate_testset(self, sample_train_data, sample_val_data, sample_test_data):
        """Test evaluation on testset"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )
        client.train()

        loss, y_pred, y_true = client.evaluate(dataset='testset')

        assert len(y_pred) == len(client.y_test)


@pytest.mark.unit
class TestSklearnClientRun:
    """Tests for SklearnClient run"""

    def test_run_returns_results(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that run() returns results dict"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        results = client.run()

        assert 'trainset' in results
        assert 'valset' in results
        assert 'testset' in results

    def test_run_logs_metrics(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that run() logs expected metrics"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        results = client.run()

        expected_metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall']
        for metric in expected_metrics:
            assert metric in results['trainset']
            assert metric in results['valset']
            assert metric in results['testset']

    def test_run_with_utility_metric(self, sample_train_data, sample_val_data, sample_test_data):
        """Test run() with utility metric saving"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        results = client.run(
            save_utility_metric=True,
            constraint_type='K',
            constraint_value=10,
            utility_metric='precision'
        )

        assert 'utility_metric' in results

    def test_run_with_feature_importances(self, sample_train_data, sample_val_data, sample_test_data):
        """Test run() with feature importances error saving"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        results = client.run(save_feature_importances_error=True)

        assert 'feature_importances_error' in results


@pytest.mark.unit
class TestSklearnClientLog:
    """Tests for SklearnClient logging"""

    def test_log_creates_dataset_entry(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that log() creates dataset entry in results"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        y_pred = np.array([0.1, 0.9, 0.3])
        y_true = np.array([0, 1, 0])

        client.log(dataset='custom', y_pred=y_pred, y_true=y_true, round=0, loss=0.5)

        assert 'custom' in client.results

    def test_log_appends_metrics(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that log() appends metrics"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        y_pred = np.array([0.1, 0.9, 0.3])
        y_true = np.array([0, 1, 0])

        client.log(dataset='test', y_pred=y_pred, y_true=y_true, round=0)
        client.log(dataset='test', y_pred=y_pred, y_true=y_true, round=1)

        assert len(client.results['test']['round']) == 2

    def test_log_computes_accuracy(self, sample_train_data, sample_val_data, sample_test_data):
        """Test that log() computes accuracy correctly"""
        client = SklearnClient(
            id='test',
            seed=42,
            trainset=sample_train_data,
            valset=sample_val_data,
            testset=sample_test_data,
            Model=RandomForestClassifier,
            n_estimators=10
        )

        # Perfect predictions (threshold 0.5)
        y_pred = np.array([0.1, 0.9, 0.1, 0.9])
        y_true = np.array([0, 1, 0, 1])

        client.log(dataset='test', y_pred=y_pred, y_true=y_true, metrics=['accuracy'])

        assert client.results['test']['accuracy'][0] == 1.0
