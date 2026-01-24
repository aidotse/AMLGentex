"""
Unit tests for ml/training/centralized.py
"""
import pytest
from unittest.mock import Mock

from src.ml.training.centralized import centralized


@pytest.mark.unit
class TestCentralized:
    """Tests for centralized training function"""

    def test_creates_single_client(self):
        """Test that centralized creates a single client"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {'accuracy': 0.9}
        MockModel = Mock()

        centralized(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            trainset='/path/to/train'
        )

        MockClient.assert_called_once()

    def test_client_id_is_cen(self):
        """Test that client is created with id='cen'"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {}
        MockModel = Mock()

        centralized(
            seed=42,
            Client=MockClient,
            Model=MockModel
        )

        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['id'] == 'cen'

    def test_passes_seed_to_client(self):
        """Test that seed is passed to client"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {}
        MockModel = Mock()

        centralized(
            seed=123,
            Client=MockClient,
            Model=MockModel
        )

        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['seed'] == 123

    def test_passes_model_to_client(self):
        """Test that Model is passed to client"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {}
        MockModel = Mock()

        centralized(
            seed=42,
            Client=MockClient,
            Model=MockModel
        )

        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['Model'] == MockModel

    def test_passes_kwargs_to_client(self):
        """Test that additional kwargs are passed to client"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {}
        MockModel = Mock()

        centralized(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            epochs=100,
            learning_rate=0.01
        )

        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['epochs'] == 100
        assert call_kwargs['learning_rate'] == 0.01

    def test_calls_client_run(self):
        """Test that client.run() is called"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {}
        MockModel = Mock()

        centralized(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            epochs=10
        )

        MockClient.return_value.run.assert_called_once()

    def test_returns_dict_with_client_id(self):
        """Test that results are returned as dict keyed by client id"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {'accuracy': 0.95}
        MockModel = Mock()

        result = centralized(
            seed=42,
            Client=MockClient,
            Model=MockModel
        )

        assert 'cen' in result
        assert result['cen'] == {'accuracy': 0.95}

    def test_passes_kwargs_to_run(self):
        """Test that kwargs are passed to client.run()"""
        MockClient = Mock()
        MockClient.return_value.id = 'cen'
        MockClient.return_value.run.return_value = {}
        MockModel = Mock()

        centralized(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            save_utility_metric=True,
            constraint_value=100
        )

        run_kwargs = MockClient.return_value.run.call_args.kwargs
        assert run_kwargs['save_utility_metric'] is True
        assert run_kwargs['constraint_value'] == 100
