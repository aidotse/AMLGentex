"""
Unit tests for Optimizer class
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

from src.data_tuning.optimizer import Optimizer


@pytest.mark.unit
class TestOptimizerInit:
    """Tests for Optimizer initialization"""

    def test_init_basic(self):
        """Test basic initialization"""
        optimizer = Optimizer(
            data_conf_file='/path/to/config.yaml',
            config={'preprocess': {'preprocessed_data_dir': '/tmp'}},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            model='DecisionTreeClassifier',
            seed=42
        )

        assert optimizer.data_conf_file == '/path/to/config.yaml'
        assert optimizer.target == 0.01
        assert optimizer.constraint_type == 'fpr'
        assert optimizer.constraint_value == 0.01
        assert optimizer.utility_metric == 'recall'
        assert optimizer.model == 'DecisionTreeClassifier'
        assert optimizer.seed == 42

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        optimizer = Optimizer(
            data_conf_file='/path/to/config.yaml',
            config={'preprocess': {'preprocessed_data_dir': '/tmp'}},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.01,
            constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
            seed=42
        )

        assert optimizer.model == 'DecisionTreeClassifier'  # Default
        assert optimizer.bank is None  # Default
        assert optimizer.bo_dir == 'tmp'  # Default
        assert optimizer.num_trials_model == 1  # Default

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters"""
        optimizer = Optimizer(
            data_conf_file='/path/to/config.yaml',
            config={'preprocess': {'preprocessed_data_dir': '/tmp'}},
            generator=Mock(),
            preprocessor=Mock(),
            target=0.05,
            constraint_type='K', constraint_value=100, utility_metric='precision',
            model='RandomForestClassifier',
            bank='BANK001',
            bo_dir='/custom/bo',
            seed=123,
            num_trials_model=10
        )

        assert optimizer.target == 0.05
        assert optimizer.constraint_type == 'K'
        assert optimizer.constraint_value == 100
        assert optimizer.utility_metric == 'precision'
        assert optimizer.model == 'RandomForestClassifier'
        assert optimizer.bank == 'BANK001'
        assert optimizer.bo_dir == '/custom/bo'
        assert optimizer.seed == 123
        assert optimizer.num_trials_model == 10


@pytest.mark.unit
class TestOptimizerObjective:
    """Tests for Optimizer objective function"""

    def test_objective_reads_config(self):
        """Test that objective function reads the data config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'mean_amount': 1000, 'std_amount': 200},
                'optimisation_bounds': {
                    'mean_amount': [500, 2000],
                    'std_amount': [100, 500]
                },
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1, 2], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'trainset_edges': pd.DataFrame({'src': [1], 'dst': [2]}),
                'valset_nodes': pd.DataFrame({'account': [3, 4], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'valset_edges': pd.DataFrame({'src': [3], 'dst': [4]}),
                'testset_nodes': pd.DataFrame({'account': [5, 6], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'testset_edges': pd.DataFrame({'src': [5], 'dst': [6]})
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient', 'random_state': 42},
                    'search_space': {'max_depth': (1, 20)},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            # Mock optuna trial
            mock_trial = Mock()
            mock_trial.suggest_int = Mock(return_value=1000)
            mock_trial.suggest_float = Mock(return_value=200.0)

            # Mock HyperparamTuner and clients to avoid actual training
            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.02
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.1
                mock_tuner_class.return_value = mock_tuner_instance

                # Mock client and model classes
                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                result = optimizer.objective(mock_trial)

            # Check that result is a tuple of two values
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], (int, float))
            assert isinstance(result[1], (int, float))

    def test_objective_updates_config(self):
        """Test that objective updates the config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'mean_amount': 1000},
                'optimisation_bounds': {'mean_amount': [500, 2000]},
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame({'src': [1], 'dst': [1]}),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame({'src': [1], 'dst': [1]}),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame({'src': [1], 'dst': [1]})
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()
            mock_trial.suggest_int = Mock(return_value=1500)

            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.01
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.05
                mock_tuner_class.return_value = mock_tuner_instance

                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                optimizer.objective(mock_trial)

            # Verify config was updated
            with open(config_path, 'r') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['default']['mean_amount'] == 1500

    def test_objective_calls_generator(self):
        """Test that objective calls the generator"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {},
                'optimisation_bounds': {},
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame(columns=['src', 'dst']),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame(columns=['src', 'dst']),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame(columns=['src', 'dst'])
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()

            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.01
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.05
                mock_tuner_class.return_value = mock_tuner_instance

                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                optimizer.objective(mock_trial)

            # Verify generator was called
            mock_generator.assert_called_once()

    def test_objective_calls_preprocessor(self):
        """Test that objective calls the preprocessor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {},
                'optimisation_bounds': {},
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            tx_log_path = '/tmp/tx_log.parquet'
            mock_generator = Mock(return_value=tx_log_path)
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame(columns=['src', 'dst']),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame(columns=['src', 'dst']),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame(columns=['src', 'dst'])
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()

            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.01
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.05
                mock_tuner_class.return_value = mock_tuner_instance

                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                optimizer.objective(mock_trial)

            # Verify preprocessor was called with tx_log_path
            mock_preprocessor.assert_called_once_with(tx_log_path)


@pytest.mark.unit
class TestOptimizerOptimize:
    """Tests for Optimizer optimize method"""

    def test_optimize_creates_study(self):
        """Test that optimize creates an Optuna study"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': tmpdir}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            with patch('src.data_tuning.optimizer.optuna') as mock_optuna:
                mock_study = Mock()
                mock_study.best_trials = []
                mock_optuna.create_study.return_value = mock_study
                mock_optuna.samplers.TPESampler = Mock()
                mock_optuna.pruners.HyperbandPruner = Mock()
                mock_optuna.visualization.matplotlib.plot_pareto_front = Mock()

                optimizer.optimize(n_trials=5)

                # Verify study was created
                mock_optuna.create_study.assert_called_once()
                call_kwargs = mock_optuna.create_study.call_args[1]
                assert 'study_name' in call_kwargs
                assert 'directions' in call_kwargs
                assert call_kwargs['directions'] == ['minimize', 'minimize']

    def test_optimize_runs_trials(self):
        """Test that optimize runs the specified number of trials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': tmpdir}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            with patch('src.data_tuning.optimizer.optuna') as mock_optuna:
                mock_study = Mock()
                mock_study.best_trials = []
                mock_optuna.create_study.return_value = mock_study
                mock_optuna.samplers.TPESampler = Mock()
                mock_optuna.pruners.HyperbandPruner = Mock()
                mock_optuna.visualization.matplotlib.plot_pareto_front = Mock()

                optimizer.optimize(n_trials=10)

                # Verify optimize was called with correct n_trials
                mock_study.optimize.assert_called_once()
                call_args = mock_study.optimize.call_args
                assert call_args[1]['n_trials'] == 10

    def test_optimize_saves_results(self):
        """Test that optimize saves results to files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            bo_dir = Path(tmpdir) / 'bo'
            os.makedirs(bo_dir)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': tmpdir}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42,
                bo_dir=str(bo_dir)
            )

            # Mock trial
            mock_trial = Mock()
            mock_trial.number = 1
            mock_trial.values = [0.01, 0.05]
            mock_trial.params = {'param1': 100, 'param2': 0.5}

            with patch('src.data_tuning.optimizer.optuna') as mock_optuna, \
                 patch('src.data_tuning.optimizer.plt') as mock_plt:
                mock_study = Mock()
                mock_study.best_trials = [mock_trial]
                mock_optuna.create_study.return_value = mock_study
                mock_optuna.samplers.TPESampler = Mock()
                mock_optuna.pruners.HyperbandPruner = Mock()
                mock_optuna.visualization.matplotlib.plot_pareto_front = Mock()

                result = optimizer.optimize(n_trials=1)

                # Verify results were saved
                assert Path(bo_dir, 'best_trials.txt').exists()

                # Verify plot was saved
                mock_plt.savefig.assert_called_once()

    def test_optimize_returns_best_trials(self):
        """Test that optimize returns best trials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': tmpdir}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()
            mock_trial.number = 1
            mock_trial.values = [0.01, 0.05]
            mock_trial.params = {}

            with patch('src.data_tuning.optimizer.optuna') as mock_optuna, \
                 patch('src.data_tuning.optimizer.plt'):
                mock_study = Mock()
                mock_study.best_trials = [mock_trial]
                mock_optuna.create_study.return_value = mock_study
                mock_optuna.samplers.TPESampler = Mock()
                mock_optuna.pruners.HyperbandPruner = Mock()
                mock_optuna.visualization.matplotlib.plot_pareto_front = Mock()

                result = optimizer.optimize(n_trials=1)

                assert result == [mock_trial]
                assert len(result) == 1


@pytest.mark.unit
class TestOptimizerBoundTypes:
    """Tests for handling different bound types"""

    def test_objective_handles_int_bounds(self):
        """Test that objective correctly handles integer bounds"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'num_accounts': 1000},
                'optimisation_bounds': {'num_accounts': [500, 2000]},  # Integer bounds
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame(columns=['src', 'dst']),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame(columns=['src', 'dst']),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame(columns=['src', 'dst'])
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()
            mock_trial.suggest_int = Mock(return_value=1500)
            mock_trial.suggest_float = Mock()

            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.01
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.05
                mock_tuner_class.return_value = mock_tuner_instance

                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                optimizer.objective(mock_trial)

            # Verify suggest_int was called
            mock_trial.suggest_int.assert_called_once_with('num_accounts', 500, 2000)

    def test_objective_handles_float_bounds(self):
        """Test that objective correctly handles float bounds"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'learning_rate': 0.1},
                'optimisation_bounds': {'learning_rate': [0.001, 0.5]},  # Float bounds
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame(columns=['src', 'dst']),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame(columns=['src', 'dst']),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame(columns=['src', 'dst'])
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()
            mock_trial.suggest_int = Mock()
            mock_trial.suggest_float = Mock(return_value=0.05)

            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.01
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.05
                mock_tuner_class.return_value = mock_tuner_instance

                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                optimizer.objective(mock_trial)

            # Verify suggest_float was called
            mock_trial.suggest_float.assert_called_once_with('learning_rate', 0.001, 0.5)


@pytest.mark.unit
class TestOptimizerReproducibility:
    """Tests for reproducibility and determinism in data tuning"""

    def test_objective_preserves_seed_in_config(self):
        """Test that objective function preserves the optimizer's seed in config (for data generation)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'mean_amount': 1000, 'std_amount': 200},
                'optimisation_bounds': {
                    'mean_amount': [500, 2000],
                    'std_amount': [100, 500]
                },
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1, 2], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'trainset_edges': pd.DataFrame({'src': [1], 'dst': [2]}),
                'valset_nodes': pd.DataFrame({'account': [3, 4], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'valset_edges': pd.DataFrame({'src': [3], 'dst': [4]}),
                'testset_nodes': pd.DataFrame({'account': [5, 6], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'testset_edges': pd.DataFrame({'src': [5], 'dst': [6]})
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            # Track that config file gets updated with the seed
            config_seeds_seen = []

            original_objective = optimizer.objective

            def tracking_objective(trial):
                # Read config after it's been updated by objective
                with open(config_path, 'r') as f:
                    current_config = yaml.safe_load(f)
                config_seeds_seen.append(current_config.get('general', {}).get('random_seed'))
                # Return dummy values
                return 0.0, 0.0

            with patch.object(optimizer, 'objective', tracking_objective), \
                 patch('src.data_tuning.optimizer.plt'):

                # Need to call the real objective first to see the seed handling
                mock_trial = Mock()
                mock_trial.suggest_int = Mock(return_value=1000)
                mock_trial.suggest_float = Mock(return_value=200.0)

                with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                     patch('src.data_tuning.optimizer.clients') as mock_clients, \
                     patch('src.data_tuning.optimizer.models') as mock_models:
                    mock_tuner_instance = Mock()
                    mock_tuner_instance.optimize = Mock(return_value=[])
                    mock_tuner_instance.fpr = 0.01
                    mock_tuner_instance.utility_metric = 0.8  # Default mock value
                    mock_tuner_instance.feature_importances_error = 0.05
                    mock_tuner_class.return_value = mock_tuner_instance

                    mock_clients.TabularClient = Mock()
                    mock_models.DecisionTreeClassifier = Mock()

                    # Call objective directly to test seed preservation
                    original_objective(mock_trial)

            # Verify seed was set to optimizer's seed
            with open(config_path, 'r') as f:
                final_config = yaml.safe_load(f)

            assert final_config['general']['random_seed'] == 42, \
                "Config should be updated with optimizer's seed for consistent data generation"

    def test_different_seeds_produce_different_parameters(self):
        """Test that different seeds produce different parameter suggestions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'mean_amount': 1000, 'std_amount': 200},
                'optimisation_bounds': {
                    'mean_amount': [500, 2000],
                    'std_amount': [100, 500]
                },
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1, 2], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'trainset_edges': pd.DataFrame({'src': [1], 'dst': [2]}),
                'valset_nodes': pd.DataFrame({'account': [3, 4], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'valset_edges': pd.DataFrame({'src': [3], 'dst': [4]}),
                'testset_nodes': pd.DataFrame({'account': [5, 6], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'testset_edges': pd.DataFrame({'src': [5], 'dst': [6]})
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            bo_dir1 = Path(tmpdir) / 'seed42'
            bo_dir2 = Path(tmpdir) / 'seed999'
            os.makedirs(bo_dir1)
            os.makedirs(bo_dir2)

            optimizer1 = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42,
                bo_dir=str(bo_dir1)
            )

            optimizer2 = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=999,
                bo_dir=str(bo_dir2)
            )

            # Track suggested parameters
            params_seed42 = []
            params_seed999 = []

            def make_objective_tracker(params_list):
                def tracked_objective(trial):
                    suggested = {}
                    for k, v in data_config['optimisation_bounds'].items():
                        if isinstance(v[0], int):
                            suggested[k] = trial.suggest_int(k, v[0], v[1])
                        else:
                            suggested[k] = trial.suggest_float(k, v[0], v[1])
                    params_list.append(suggested)
                    return 0.0, 0.0
                return tracked_objective

            with patch.object(optimizer1, 'objective', make_objective_tracker(params_seed42)), \
                 patch.object(optimizer2, 'objective', make_objective_tracker(params_seed999)), \
                 patch('src.data_tuning.optimizer.plt'):

                optimizer1.optimize(n_trials=5)
                optimizer2.optimize(n_trials=5)

            # With different seeds, at least some parameters should differ
            differences_found = False
            for i in range(5):
                for key in params_seed42[i].keys():
                    if params_seed42[i][key] != params_seed999[i][key]:
                        differences_found = True
                        break
                if differences_found:
                    break

            assert differences_found, "Different seeds should produce different parameter sequences"

    def test_config_seed_preserved_across_trials(self):
        """Test that the random seed in config is preserved (not overwritten by trial params)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'mean_amount': 1000},
                'optimisation_bounds': {'mean_amount': [500, 2000]},
                'general': {'random_seed': 999}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame(columns=['src', 'dst']),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame(columns=['src', 'dst']),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame(columns=['src', 'dst'])
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()
            mock_trial.suggest_int = Mock(return_value=1500)

            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.01
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.05
                mock_tuner_class.return_value = mock_tuner_instance

                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                optimizer.objective(mock_trial)

            # Verify that config file was updated with fixed seed
            with open(config_path, 'r') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['general']['random_seed'] == 42, \
                "Optimizer should override config seed with its own seed for fair comparison"


@pytest.mark.unit
class TestOptimizerParameterUpdates:
    """Tests for verifying parameters actually change during optimization"""

    def test_parameters_change_across_trials(self):
        """Test that optimization actually explores different parameter values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'mean_amount': 1000, 'std_amount': 200},
                'optimisation_bounds': {
                    'mean_amount': [500, 2000],
                    'std_amount': [100, 500]
                },
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1, 2], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'trainset_edges': pd.DataFrame({'src': [1], 'dst': [2]}),
                'valset_nodes': pd.DataFrame({'account': [3, 4], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'valset_edges': pd.DataFrame({'src': [3], 'dst': [4]}),
                'testset_nodes': pd.DataFrame({'account': [5, 6], 'bank': ['BANK001', 'BANK001'], 'is_sar': [0, 1]}),
                'testset_edges': pd.DataFrame({'src': [5], 'dst': [6]})
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            bo_dir = Path(tmpdir) / 'bo'
            os.makedirs(bo_dir)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42,
                bo_dir=str(bo_dir)
            )

            # Track parameters from each trial
            trial_params = []

            def track_objective(trial):
                suggested = {}
                for k, v in data_config['optimisation_bounds'].items():
                    if isinstance(v[0], int):
                        suggested[k] = trial.suggest_int(k, v[0], v[1])
                    else:
                        suggested[k] = trial.suggest_float(k, v[0], v[1])
                trial_params.append(suggested)
                # Return varying objectives to simulate exploration
                return abs(suggested['mean_amount'] - 1200) / 1000, abs(suggested['std_amount'] - 300) / 100

            with patch.object(optimizer, 'objective', track_objective), \
                 patch('src.data_tuning.optimizer.plt'):

                optimizer.optimize(n_trials=10)

            # Verify we explored different parameter values
            assert len(trial_params) == 10

            # Check that not all trials have the same parameters
            unique_mean_amounts = set(p['mean_amount'] for p in trial_params)
            unique_std_amounts = set(p['std_amount'] for p in trial_params)

            assert len(unique_mean_amounts) > 1, "mean_amount should vary across trials"
            assert len(unique_std_amounts) > 1, "std_amount should vary across trials"

            # Check that parameters are within bounds
            for params in trial_params:
                assert 500 <= params['mean_amount'] <= 2000
                assert 100 <= params['std_amount'] <= 500

    def test_config_file_updated_with_trial_parameters(self):
        """Test that the config file is actually updated with trial parameters during objective"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            initial_config = {
                'default': {'learning_rate': 0.1, 'num_epochs': 10},
                'optimisation_bounds': {
                    'learning_rate': [0.01, 0.5],
                    'num_epochs': [5, 20]
                },
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame(columns=['src', 'dst']),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame(columns=['src', 'dst']),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame(columns=['src', 'dst'])
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            mock_trial = Mock()
            mock_trial.suggest_float = Mock(return_value=0.25)
            mock_trial.suggest_int = Mock(return_value=15)

            with patch('src.data_tuning.optimizer.HyperparamTuner') as mock_tuner_class, \
                 patch('src.data_tuning.optimizer.clients') as mock_clients, \
                 patch('src.data_tuning.optimizer.models') as mock_models:
                mock_tuner_instance = Mock()
                mock_tuner_instance.optimize = Mock(return_value=[])
                mock_tuner_instance.fpr = 0.01
                mock_tuner_instance.utility_metric = 0.8  # Default mock value
                mock_tuner_instance.feature_importances_error = 0.05
                mock_tuner_class.return_value = mock_tuner_instance

                mock_clients.TabularClient = Mock()
                mock_models.DecisionTreeClassifier = Mock()

                optimizer.objective(mock_trial)

            # Read updated config
            with open(config_path, 'r') as f:
                updated_config = yaml.safe_load(f)

            # Verify parameters were updated
            assert updated_config['default']['learning_rate'] == 0.25
            assert updated_config['default']['num_epochs'] == 15


@pytest.mark.unit
class TestOptimizerUpdateConfigWithTrial:
    """Tests for update_config_with_trial method"""

    def test_update_config_with_specific_trial(self):
        """Test that config can be updated with a specific trial's parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {'param1': 100, 'param2': 0.5},
                'optimisation_bounds': {
                    'param1': [50, 200],
                    'param2': [0.1, 1.0]
                },
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            mock_generator = Mock(return_value='/tmp/tx_log.parquet')
            mock_preprocessor = Mock(return_value={
                'trainset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'trainset_edges': pd.DataFrame(columns=['src', 'dst']),
                'valset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'valset_edges': pd.DataFrame(columns=['src', 'dst']),
                'testset_nodes': pd.DataFrame({'account': [1], 'bank': ['BANK001'], 'is_sar': [0]}),
                'testset_edges': pd.DataFrame(columns=['src', 'dst'])
            })

            config = {
                'preprocess': {'preprocessed_data_dir': tmpdir},
                'DecisionTreeClassifier': {
                    'default': {'client_type': 'TabularClient'},
                    'search_space': {},
                    'isolated': {'clients': {'BANK001': {}}}
                }
            }

            bo_dir = Path(tmpdir) / 'bo'
            os.makedirs(bo_dir)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config=config,
                generator=mock_generator,
                preprocessor=mock_preprocessor,
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42,
                bo_dir=str(bo_dir)
            )

            # Mock optuna study with trials
            with patch('src.data_tuning.optimizer.optuna.load_study') as mock_load_study:
                mock_trial = Mock()
                mock_trial.number = 5
                mock_trial.values = [0.01, 0.05]
                mock_trial.params = {'param1': 150, 'param2': 0.75}

                mock_study = Mock()
                mock_study.trials = [mock_trial]
                mock_load_study.return_value = mock_study

                optimizer.update_config_with_trial(trial_number=5)

            # Verify config was updated
            with open(config_path, 'r') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['default']['param1'] == 150
            assert updated_config['default']['param2'] == 0.75

    def test_update_config_raises_error_for_nonexistent_trial(self):
        """Test that updating with nonexistent trial raises error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {
                'default': {},
                'optimisation_bounds': {},
                'general': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': tmpdir}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42,
                bo_dir=str(tmpdir)
            )

            with patch('src.data_tuning.optimizer.optuna.load_study') as mock_load_study:
                mock_study = Mock()
                mock_study.trials = []
                mock_load_study.return_value = mock_study

                with pytest.raises(ValueError, match="Trial 99 not found"):
                    optimizer.update_config_with_trial(trial_number=99)


@pytest.mark.unit
class TestOptimizerCleanup:
    """Tests for cleanup of intermediate files after optimization"""

    def test_cleanup_removes_intermediate_files(self):
        """Test that _cleanup_intermediate_files removes all intermediate parquet files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessed_dir = Path(tmpdir) / 'preprocessed'
            os.makedirs(preprocessed_dir)

            # Create dummy intermediate files that would be generated during optimization
            intermediate_files = [
                'trainset_nodes.parquet',
                'trainset_edges.parquet',
                'valset_nodes.parquet',
                'valset_edges.parquet',
                'testset_nodes.parquet',
                'testset_edges.parquet'
            ]

            for filename in intermediate_files:
                filepath = preprocessed_dir / filename
                # Create dummy file
                pd.DataFrame({'dummy': [1, 2, 3]}).to_parquet(filepath)
                assert filepath.exists(), f"{filename} should exist before cleanup"

            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': str(preprocessed_dir)}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            # Call cleanup
            optimizer._cleanup_intermediate_files()

            # Verify all intermediate files were removed
            for filename in intermediate_files:
                filepath = preprocessed_dir / filename
                assert not filepath.exists(), f"{filename} should be removed after cleanup"

    def test_cleanup_handles_missing_files(self):
        """Test that cleanup doesn't fail if some files are already missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessed_dir = Path(tmpdir) / 'preprocessed'
            os.makedirs(preprocessed_dir)

            # Create only some of the intermediate files
            partial_files = ['trainset_nodes.parquet', 'testset_edges.parquet']
            for filename in partial_files:
                filepath = preprocessed_dir / filename
                pd.DataFrame({'dummy': [1, 2, 3]}).to_parquet(filepath)

            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': str(preprocessed_dir)}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            # Cleanup should not fail even if some files don't exist
            optimizer._cleanup_intermediate_files()

            # Verify existing files were removed
            for filename in partial_files:
                filepath = preprocessed_dir / filename
                assert not filepath.exists(), f"{filename} should be removed after cleanup"

    def test_cleanup_called_during_optimize(self):
        """Test that cleanup is automatically called during optimize()"""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessed_dir = Path(tmpdir) / 'preprocessed'
            os.makedirs(preprocessed_dir)

            # Create intermediate files that would exist after optimization trials
            intermediate_files = [
                'trainset_nodes.parquet',
                'trainset_edges.parquet',
                'valset_nodes.parquet',
                'valset_edges.parquet',
                'testset_nodes.parquet',
                'testset_edges.parquet'
            ]

            for filename in intermediate_files:
                filepath = preprocessed_dir / filename
                pd.DataFrame({'dummy': [1, 2, 3]}).to_parquet(filepath)

            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            bo_dir = Path(tmpdir) / 'bo'
            os.makedirs(bo_dir)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': str(preprocessed_dir)}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42,
                bo_dir=str(bo_dir)
            )

            mock_trial = Mock()
            mock_trial.number = 1
            mock_trial.values = [0.01, 0.05]
            mock_trial.params = {}

            with patch('src.data_tuning.optimizer.optuna') as mock_optuna, \
                 patch('src.data_tuning.optimizer.plt'):
                mock_study = Mock()
                mock_study.best_trials = [mock_trial]
                mock_optuna.create_study.return_value = mock_study
                mock_optuna.samplers.TPESampler = Mock()
                mock_optuna.pruners.HyperbandPruner = Mock()
                mock_optuna.visualization.matplotlib.plot_pareto_front = Mock()

                optimizer.optimize(n_trials=1)

            # Verify all intermediate files were cleaned up after optimization
            for filename in intermediate_files:
                filepath = preprocessed_dir / filename
                assert not filepath.exists(), f"{filename} should be cleaned up after optimize()"

    def test_cleanup_preserves_other_files(self):
        """Test that cleanup only removes intermediate files, not other parquet files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessed_dir = Path(tmpdir) / 'preprocessed'
            os.makedirs(preprocessed_dir)

            # Create intermediate files
            intermediate_files = [
                'trainset_nodes.parquet',
                'trainset_edges.parquet',
                'valset_nodes.parquet',
                'valset_edges.parquet',
                'testset_nodes.parquet',
                'testset_edges.parquet'
            ]

            for filename in intermediate_files:
                filepath = preprocessed_dir / filename
                pd.DataFrame({'dummy': [1, 2, 3]}).to_parquet(filepath)

            # Create other files that should NOT be removed
            other_files = ['metadata.parquet', 'custom_dataset.parquet', 'results.parquet']
            for filename in other_files:
                filepath = preprocessed_dir / filename
                pd.DataFrame({'dummy': [1, 2, 3]}).to_parquet(filepath)
                assert filepath.exists()

            config_path = Path(tmpdir) / 'config.yaml'
            data_config = {'default': {}, 'optimisation_bounds': {}, 'general': {}}
            with open(config_path, 'w') as f:
                yaml.dump(data_config, f)

            optimizer = Optimizer(
                data_conf_file=str(config_path),
                config={'preprocess': {'preprocessed_data_dir': str(preprocessed_dir)}},
                generator=Mock(),
                preprocessor=Mock(),
                target=0.01,
                constraint_type='fpr', constraint_value=0.01, utility_metric='recall',
                seed=42
            )

            # Call cleanup
            optimizer._cleanup_intermediate_files()

            # Verify intermediate files were removed
            for filename in intermediate_files:
                filepath = preprocessed_dir / filename
                assert not filepath.exists(), f"{filename} should be removed"

            # Verify other files were NOT removed
            for filename in other_files:
                filepath = preprocessed_dir / filename
                assert filepath.exists(), f"{filename} should be preserved"
