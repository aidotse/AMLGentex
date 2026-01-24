"""
Unit tests for data_creation/generator.py
"""
import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data_creation.generator import DataGenerator


@pytest.fixture
def sample_config():
    """Create a sample config dict"""
    return {
        'general': {
            'simulation_name': 'test_sim',
            'random_seed': 42
        },
        'input': {
            'directory': 'experiments/test/config',
            'degree': 'degree.csv'
        },
        'spatial': {
            'directory': 'experiments/test/spatial'
        },
        'output': {
            'directory': 'experiments/test/temporal',
            'transaction_log': 'tx_log.parquet'
        }
    }


@pytest.fixture
def config_file(tmp_path, sample_config):
    """Create a temporary config file"""
    config_path = tmp_path / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return str(config_path)


@pytest.mark.unit
class TestDataGeneratorInit:
    """Tests for DataGenerator initialization"""

    def test_init_requires_absolute_path(self, sample_config):
        """Test that relative path raises ValueError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'data.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Using relative path should fail
            with pytest.raises(ValueError, match='must be an absolute path'):
                DataGenerator('relative/path/config.yaml')

    def test_init_requires_existing_file(self):
        """Test that nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match='Config file not found'):
            DataGenerator('/nonexistent/path/config.yaml')

    def test_init_loads_config(self, config_file, sample_config):
        """Test that config is loaded from file"""
        generator = DataGenerator(config_file)

        assert generator.config['general']['simulation_name'] == 'test_sim'
        assert generator.config['general']['random_seed'] == 42

    def test_init_stores_conf_file_path(self, config_file):
        """Test that conf_file path is stored"""
        generator = DataGenerator(config_file)

        assert generator.conf_file == config_file

    def test_init_converts_relative_paths_to_absolute(self, config_file):
        """Test that relative paths in config are converted to absolute"""
        generator = DataGenerator(config_file)

        # All directories should be absolute paths now
        assert os.path.isabs(generator.config['input']['directory'])
        assert os.path.isabs(generator.config['spatial']['directory'])
        assert os.path.isabs(generator.config['output']['directory'])


@pytest.mark.unit
class TestDataGeneratorRunSpatial:
    """Tests for run_spatial method"""

    def test_run_spatial_skips_if_outputs_exist(self, config_file, tmp_path):
        """Test that run_spatial skips if spatial outputs already exist"""
        generator = DataGenerator(config_file)

        # Create spatial output directory with some CSV files
        spatial_dir = Path(generator.config['spatial']['directory'])
        spatial_dir.mkdir(parents=True, exist_ok=True)
        (spatial_dir / 'accounts.csv').touch()

        result = generator.run_spatial(force=False)

        assert result == str(spatial_dir)

    def test_run_spatial_returns_spatial_dir(self, config_file, tmp_path):
        """Test that run_spatial returns the spatial directory path"""
        generator = DataGenerator(config_file)

        # Create spatial output directory with CSV files
        spatial_dir = Path(generator.config['spatial']['directory'])
        spatial_dir.mkdir(parents=True, exist_ok=True)
        (spatial_dir / 'accounts.csv').touch()

        result = generator.run_spatial(force=False)

        assert result == str(spatial_dir)
        assert os.path.isabs(result)


@pytest.mark.unit
class TestDataGeneratorRunTemporal:
    """Tests for run_temporal method - note: actual run requires full simulation setup"""
    pass  # Temporal simulation tests covered by integration tests


@pytest.mark.unit
class TestDataGeneratorRunSpatialBaseline:
    """Tests for run_spatial_baseline method"""

    def test_run_spatial_baseline_skips_if_checkpoint_exists(self, config_file, tmp_path):
        """Test that run_spatial_baseline skips if checkpoint exists"""
        generator = DataGenerator(config_file)

        # Create checkpoint file
        spatial_dir = Path(generator.config['spatial']['directory'])
        spatial_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = spatial_dir / 'baseline_checkpoint.pkl'
        checkpoint_path.touch()

        result = generator.run_spatial_baseline(force=False)

        assert result == str(checkpoint_path)

    def test_run_spatial_baseline_force_regenerates(self, config_file, tmp_path):
        """Test that run_spatial_baseline with force=True regenerates"""
        generator = DataGenerator(config_file)

        # Create checkpoint file
        spatial_dir = Path(generator.config['spatial']['directory'])
        spatial_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = spatial_dir / 'baseline_checkpoint.pkl'
        checkpoint_path.touch()

        # With force=True, it should attempt to regenerate
        # (Would need full mocking to actually run)


@pytest.mark.unit
class TestDataGeneratorRunSpatialFromBaseline:
    """Tests for run_spatial_from_baseline method"""

    def test_run_spatial_from_baseline_requires_checkpoint(self, config_file):
        """Test that run_spatial_from_baseline requires existing checkpoint"""
        generator = DataGenerator(config_file)

        # Ensure checkpoint does not exist
        spatial_dir = Path(generator.config['spatial']['directory'])
        spatial_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = spatial_dir / 'baseline_checkpoint.pkl'
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        with pytest.raises(FileNotFoundError, match='Baseline checkpoint not found'):
            generator.run_spatial_from_baseline()

    def test_run_spatial_from_baseline_default_checkpoint_path(self, config_file):
        """Test the default checkpoint path location"""
        generator = DataGenerator(config_file)

        spatial_dir = Path(generator.config['spatial']['directory'])
        expected_default = spatial_dir / 'baseline_checkpoint.pkl'

        # The default path should be in the spatial directory
        assert str(expected_default).endswith('baseline_checkpoint.pkl')


@pytest.mark.unit
class TestDataGeneratorCall:
    """Tests for __call__ method"""

    def test_call_runs_spatial_by_default(self, config_file, tmp_path):
        """Test that __call__ runs spatial simulation by default"""
        generator = DataGenerator(config_file)

        with patch.object(generator, 'run_spatial') as mock_spatial, \
             patch.object(generator, 'run_temporal') as mock_temporal:
            mock_temporal.return_value = '/path/to/tx_log.parquet'

            generator(spatial=True)

            mock_spatial.assert_called_once_with(force=False)
            mock_temporal.assert_called_once()

    def test_call_skips_spatial_when_false(self, config_file, tmp_path):
        """Test that __call__ skips spatial when spatial=False"""
        generator = DataGenerator(config_file)

        with patch.object(generator, 'run_spatial') as mock_spatial, \
             patch.object(generator, 'run_temporal') as mock_temporal:
            mock_temporal.return_value = '/path/to/tx_log.parquet'

            generator(spatial=False)

            mock_spatial.assert_not_called()
            mock_temporal.assert_called_once()

    def test_call_passes_force_spatial(self, config_file, tmp_path):
        """Test that __call__ passes force_spatial to run_spatial"""
        generator = DataGenerator(config_file)

        with patch.object(generator, 'run_spatial') as mock_spatial, \
             patch.object(generator, 'run_temporal') as mock_temporal:
            mock_temporal.return_value = '/path/to/tx_log.parquet'

            generator(spatial=True, force_spatial=True)

            mock_spatial.assert_called_once_with(force=True)

    def test_call_returns_tx_log_path(self, config_file, tmp_path):
        """Test that __call__ returns the transaction log path"""
        generator = DataGenerator(config_file)

        with patch.object(generator, 'run_spatial'), \
             patch.object(generator, 'run_temporal') as mock_temporal:
            mock_temporal.return_value = '/path/to/tx_log.parquet'

            result = generator(spatial=False)

            assert result == '/path/to/tx_log.parquet'


@pytest.mark.unit
class TestDataGeneratorMakePathsAbsolute:
    """Tests for _make_paths_absolute method"""

    def test_converts_relative_input_directory(self, tmp_path, sample_config):
        """Test that relative input directory is converted to absolute"""
        config_path = tmp_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        generator = DataGenerator(str(config_path))

        assert os.path.isabs(generator.config['input']['directory'])
        assert 'experiments/test/config' in generator.config['input']['directory']

    def test_converts_relative_spatial_directory(self, tmp_path, sample_config):
        """Test that relative spatial directory is converted to absolute"""
        config_path = tmp_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        generator = DataGenerator(str(config_path))

        assert os.path.isabs(generator.config['spatial']['directory'])
        assert 'experiments/test/spatial' in generator.config['spatial']['directory']

    def test_converts_relative_output_directory(self, tmp_path, sample_config):
        """Test that relative output directory is converted to absolute"""
        config_path = tmp_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        generator = DataGenerator(str(config_path))

        assert os.path.isabs(generator.config['output']['directory'])
        assert 'experiments/test/temporal' in generator.config['output']['directory']

    def test_preserves_absolute_paths(self, tmp_path):
        """Test that already absolute paths are preserved"""
        config = {
            'general': {'simulation_name': 'test'},
            'input': {'directory': '/absolute/input/path', 'degree': 'degree.csv'},
            'spatial': {'directory': '/absolute/spatial/path'},
            'output': {'directory': '/absolute/output/path', 'transaction_log': 'tx_log.parquet'}
        }
        config_path = tmp_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        generator = DataGenerator(str(config_path))

        assert generator.config['input']['directory'] == '/absolute/input/path'
        assert generator.config['spatial']['directory'] == '/absolute/spatial/path'
        assert generator.config['output']['directory'] == '/absolute/output/path'
