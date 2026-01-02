"""
Unit tests for DataTuner class
"""
import pytest
from unittest.mock import Mock, patch

from src.data_tuning.tuner import DataTuner


@pytest.mark.unit
class TestDataTunerInit:
    """Tests for DataTuner initialization"""

    def test_init_all_parameters(self):
        """Test initialization with all parameters"""
        mock_generator = Mock()
        mock_preprocessor = Mock()

        tuner = DataTuner(
            data_conf_file='/path/to/config.yaml',
            config={'key': 'value'},
            generator=mock_generator,
            preprocessor=mock_preprocessor,
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            bo_dir='/tmp/bo',
            seed=42,
            num_trials_model=100
        )

        assert tuner.data_conf_file == '/path/to/config.yaml'
        assert tuner.config == {'key': 'value'}
        assert tuner.generator == mock_generator
        assert tuner.preprocessor == mock_preprocessor
        assert tuner.target == 0.01
        assert tuner.constraint_type == 'fpr'
        assert tuner.constraint_value == 0.01
        assert tuner.utility_metric == 'recall'
        assert tuner.model == 'DecisionTreeClassifier'
        assert tuner.bo_dir == '/tmp/bo'
        assert tuner.seed == 42
        assert tuner.num_trials_model == 100

    def test_init_stores_all_attributes(self):
        """Test that all initialization parameters are stored as attributes"""
        params = {
            'data_conf_file': '/config.yaml',
            'config': {},
            'generator': Mock(),
            'preprocessor': Mock(),
            'target': 0.05,
            'constraint_type': 'K',
            'constraint_value': 100,
            'utility_metric': 'precision',
            'model': 'RandomForestClassifier',
            'bo_dir': '/bo',
            'seed': 123,
            'num_trials_model': 50
        }

        tuner = DataTuner(**params)

        for key, value in params.items():
            assert getattr(tuner, key) == value


@pytest.mark.unit
class TestDataTunerCall:
    """Tests for DataTuner __call__ method"""

    def test_call_creates_optimizer(self):
        """Test that calling tuner creates an Optimizer instance"""
        mock_generator = Mock()
        mock_preprocessor = Mock()

        tuner = DataTuner(
            data_conf_file='/config.yaml',
            config={},
            generator=mock_generator,
            preprocessor=mock_preprocessor,
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            bo_dir='/tmp',
            seed=42,
            num_trials_model=10
        )

        with patch('src.data_tuning.tuner.Optimizer') as mock_optimizer_class:
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize = Mock(return_value=[])
            mock_optimizer_class.return_value = mock_optimizer_instance

            tuner(n_trials=5)

            # Verify Optimizer was instantiated with correct parameters
            mock_optimizer_class.assert_called_once_with(
                data_conf_file='/config.yaml',
                config={},
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                model='DecisionTreeClassifier',
                bo_dir='/tmp',
                seed=42,
                num_trials_model=10,
                verbose=False
            )

    def test_call_runs_optimization(self):
        """Test that calling tuner runs optimization with correct n_trials"""
        tuner = DataTuner(
            data_conf_file='/config.yaml',
            config={},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            bo_dir='/tmp',
            seed=42,
            num_trials_model=10
        )

        with patch('src.data_tuning.tuner.Optimizer') as mock_optimizer_class:
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize = Mock(return_value=[])
            mock_optimizer_class.return_value = mock_optimizer_instance

            tuner(n_trials=20)

            # Verify optimize was called with correct n_trials
            mock_optimizer_instance.optimize.assert_called_once_with(n_trials=20)

    def test_call_prints_best_trials(self, capsys):
        """Test that calling tuner prints best trial information"""
        tuner = DataTuner(
            data_conf_file='/config.yaml',
            config={},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            bo_dir='/tmp',
            seed=42,
            num_trials_model=10,
            verbose=True  # Enable verbose to check printed output
        )

        # Mock trial
        mock_trial = Mock()
        mock_trial.number = 42
        mock_trial.values = [0.01, 0.05]
        mock_trial.params = {'param1': 100, 'param2': 0.5}

        with patch('src.data_tuning.tuner.Optimizer') as mock_optimizer_class:
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize = Mock(return_value=[mock_trial])
            mock_optimizer_class.return_value = mock_optimizer_instance

            tuner(n_trials=1)

        # Capture printed output
        captured = capsys.readouterr()

        # Verify trial information was printed
        assert 'trial: 42' in captured.out
        assert 'values: [0.01, 0.05]' in captured.out
        assert 'param1: 100' in captured.out
        assert 'param2: 0.5' in captured.out

    def test_call_handles_multiple_trials(self, capsys):
        """Test that calling tuner handles multiple best trials"""
        tuner = DataTuner(
            data_conf_file='/config.yaml',
            config={},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            bo_dir='/tmp',
            seed=42,
            num_trials_model=10,
            verbose=True  # Enable verbose to check printed output
        )

        # Mock multiple trials
        mock_trial1 = Mock()
        mock_trial1.number = 1
        mock_trial1.values = [0.01, 0.05]
        mock_trial1.params = {'param1': 100}

        mock_trial2 = Mock()
        mock_trial2.number = 2
        mock_trial2.values = [0.02, 0.04]
        mock_trial2.params = {'param1': 200}

        with patch('src.data_tuning.tuner.Optimizer') as mock_optimizer_class:
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize = Mock(return_value=[mock_trial1, mock_trial2])
            mock_optimizer_class.return_value = mock_optimizer_instance

            tuner(n_trials=10)

        captured = capsys.readouterr()

        # Verify both trials were printed
        assert 'trial: 1' in captured.out
        assert 'trial: 2' in captured.out

    def test_call_returns_best_trials(self):
        """Test that __call__ returns best trials"""
        tuner = DataTuner(
            data_conf_file='/config.yaml',
            config={},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            bo_dir='/tmp',
            seed=42,
            num_trials_model=10
        )

        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.values = [0.01, 0.05]
        mock_trial.params = {}

        with patch('src.data_tuning.tuner.Optimizer') as mock_optimizer_class:
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize = Mock(return_value=[mock_trial])
            mock_optimizer_class.return_value = mock_optimizer_instance

            result = tuner(n_trials=5)

        assert result == [mock_trial]

    def test_call_with_empty_trials(self, capsys):
        """Test that calling tuner handles empty trial list"""
        tuner = DataTuner(
            data_conf_file='/config.yaml',
            config={},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            bo_dir='/tmp',
            seed=42,
            num_trials_model=10
        )

        with patch('src.data_tuning.tuner.Optimizer') as mock_optimizer_class:
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize = Mock(return_value=[])
            mock_optimizer_class.return_value = mock_optimizer_instance

            tuner(n_trials=1)

        # Should not crash with empty trials
        captured = capsys.readouterr()
        assert captured.out == ''  # No output for empty trials
