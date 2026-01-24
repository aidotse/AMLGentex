"""
Unit tests for ml/training/hyperparameter_tuning.py
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import optuna

from src.ml.training.hyperparameter_tuning import HyperparamTuner


@pytest.mark.unit
class TestHyperparamTuner:
    """Tests for HyperparamTuner"""

    @pytest.fixture
    def mock_objective_fn(self):
        """Create mock objective function"""
        def obj_fn(seed, Server, Client, Model, n_workers, **kwargs):
            return {
                'client1': {
                    'valset': {'average_precision': [0.85]}
                }
            }
        return obj_fn

    @pytest.fixture
    def basic_tuner(self, mock_objective_fn):
        """Create basic tuner for testing"""
        return HyperparamTuner(
            study_name='test_study',
            obj_fn=mock_objective_fn,
            params={'epochs': 10},
            search_space={
                'lr': {'type': 'float', 'low': 0.001, 'high': 0.1},
                'hidden_dim': {'type': 'int', 'low': 16, 'high': 64}
            },
            Client=Mock(),
            Model=Mock(),
            Server=Mock(),
            seed=42
        )

    def test_init(self, basic_tuner):
        """Test initialization"""
        assert basic_tuner.study_name == 'test_study'
        assert basic_tuner.seed == 42
        assert 'lr' in basic_tuner.search_space
        assert 'hidden_dim' in basic_tuner.search_space

    def test_objective_suggests_float_params(self, basic_tuner):
        """Test that objective suggests float parameters"""
        trial = Mock()
        trial.suggest_float.return_value = 0.01
        trial.suggest_int.return_value = 32
        trial.suggest_categorical.return_value = 'value'

        basic_tuner.objective(trial)

        trial.suggest_float.assert_called_once()
        call_args = trial.suggest_float.call_args
        assert call_args[0][0] == 'lr'

    def test_objective_suggests_int_params(self, basic_tuner):
        """Test that objective suggests int parameters"""
        trial = Mock()
        trial.suggest_float.return_value = 0.01
        trial.suggest_int.return_value = 32

        basic_tuner.objective(trial)

        trial.suggest_int.assert_called_once()
        call_args = trial.suggest_int.call_args
        assert call_args[0][0] == 'hidden_dim'

    def test_objective_suggests_categorical_params(self, mock_objective_fn):
        """Test that objective suggests categorical parameters"""
        tuner = HyperparamTuner(
            study_name='test',
            obj_fn=mock_objective_fn,
            params={},
            search_space={
                'activation': {'type': 'categorical', 'values': ['relu', 'tanh']}
            },
            Client=Mock(),
            Model=Mock()
        )

        trial = Mock()
        trial.suggest_categorical.return_value = 'relu'

        tuner.objective(trial)

        trial.suggest_categorical.assert_called_once_with('activation', ['relu', 'tanh'])

    def test_objective_handles_log_scale(self, mock_objective_fn):
        """Test that objective handles log scale for floats"""
        tuner = HyperparamTuner(
            study_name='test',
            obj_fn=mock_objective_fn,
            params={},
            search_space={
                'lr': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True}
            },
            Client=Mock(),
            Model=Mock()
        )

        trial = Mock()
        trial.suggest_float.return_value = 0.01

        tuner.objective(trial)

        trial.suggest_float.assert_called_once()
        call_kwargs = trial.suggest_float.call_args[1]
        assert call_kwargs['log'] is True

    def test_objective_returns_metric_tuple(self, basic_tuner):
        """Test that objective returns tuple of metrics"""
        trial = Mock()
        trial.suggest_float.return_value = 0.01
        trial.suggest_int.return_value = 32

        result = basic_tuner.objective(trial)

        assert isinstance(result, tuple)
        assert len(result) == 1  # default metrics = ['average_precision']
        assert result[0] == 0.85

    def test_objective_averages_across_clients(self, mock_objective_fn):
        """Test that objective averages metrics across clients"""
        def multi_client_obj_fn(seed, Server, Client, Model, n_workers, **kwargs):
            return {
                'client1': {'valset': {'average_precision': [0.8]}},
                'client2': {'valset': {'average_precision': [0.9]}}
            }

        tuner = HyperparamTuner(
            study_name='test',
            obj_fn=multi_client_obj_fn,
            params={},
            search_space={},
            Client=Mock(),
            Model=Mock()
        )

        trial = Mock()
        result = tuner.objective(trial)

        # Should average: (0.8 + 0.9) / 2 = 0.85
        assert result[0] == pytest.approx(0.85)

    def test_objective_handles_utility_metric(self, mock_objective_fn):
        """Test that objective extracts utility metric when requested"""
        def obj_fn_with_utility(seed, Server, Client, Model, n_workers, **kwargs):
            return {
                'client1': {
                    'valset': {'average_precision': [0.85]},
                    'utility_metric': 0.75
                }
            }

        tuner = HyperparamTuner(
            study_name='test',
            obj_fn=obj_fn_with_utility,
            params={'save_utility_metric': True},
            search_space={},
            Client=Mock(),
            Model=Mock()
        )

        trial = Mock()
        tuner.objective(trial)

        assert tuner.utility_metric == 0.75

    def test_objective_handles_feature_importances_error(self, mock_objective_fn):
        """Test that objective extracts feature importances error"""
        def obj_fn_with_fi(seed, Server, Client, Model, n_workers, **kwargs):
            return {
                'client1': {
                    'valset': {'average_precision': [0.85]},
                    'feature_importances_error': 0.123
                }
            }

        tuner = HyperparamTuner(
            study_name='test',
            obj_fn=obj_fn_with_fi,
            params={'save_feature_importances_error': True},
            search_space={},
            Client=Mock(),
            Model=Mock()
        )

        trial = Mock()
        tuner.objective(trial)

        assert tuner.feature_importances_error == 0.123

    @patch('src.ml.training.hyperparameter_tuning.optuna.create_study')
    @patch('src.ml.training.hyperparameter_tuning.set_random_seed')
    def test_optimize_creates_study(self, mock_seed, mock_create_study, basic_tuner):
        """Test that optimize creates Optuna study"""
        mock_study = MagicMock()
        mock_study.best_trials = []
        mock_create_study.return_value = mock_study

        basic_tuner.optimize(n_trials=5)

        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()

    @patch('src.ml.training.hyperparameter_tuning.optuna.create_study')
    @patch('src.ml.training.hyperparameter_tuning.set_random_seed')
    def test_optimize_returns_best_trials(self, mock_seed, mock_create_study, basic_tuner):
        """Test that optimize returns best trials"""
        mock_trial = MagicMock()
        mock_study = MagicMock()
        mock_study.best_trials = [mock_trial]
        mock_create_study.return_value = mock_study

        result = basic_tuner.optimize(n_trials=5)

        assert result == [mock_trial]

    @patch('src.ml.training.hyperparameter_tuning.optuna.create_study')
    @patch('src.ml.training.hyperparameter_tuning.set_random_seed')
    def test_optimize_uses_tpe_sampler(self, mock_seed, mock_create_study, basic_tuner):
        """Test that optimize uses TPE sampler with correct seed"""
        mock_study = MagicMock()
        mock_study.best_trials = []
        mock_create_study.return_value = mock_study

        basic_tuner.optimize(n_trials=5)

        call_kwargs = mock_create_study.call_args[1]
        assert isinstance(call_kwargs['sampler'], optuna.samplers.TPESampler)

    def test_multiple_metrics(self, mock_objective_fn):
        """Test tuner with multiple metrics"""
        def multi_metric_obj_fn(seed, Server, Client, Model, n_workers, **kwargs):
            return {
                'client1': {
                    'valset': {
                        'average_precision': [0.85],
                        'f1': [0.75]
                    }
                }
            }

        tuner = HyperparamTuner(
            study_name='test',
            obj_fn=multi_metric_obj_fn,
            params={},
            search_space={},
            Client=Mock(),
            Model=Mock(),
            metrics=['average_precision', 'f1']
        )

        trial = Mock()
        result = tuner.objective(trial)

        assert len(result) == 2
        assert result[0] == 0.85
        assert result[1] == 0.75
