"""
Unit tests for configuration utilities
"""
import pytest
import tempfile
from pathlib import Path

from src.utils.config import discover_clients, build_data_paths


@pytest.mark.unit
class TestDiscoverClients:
    """Tests for discover_clients function"""

    def test_discovers_client_directories(self, tmp_path):
        """Test that client directories are discovered"""
        # Create preprocessed/clients structure
        clients_dir = tmp_path / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "bank_A").mkdir()
        (clients_dir / "bank_B").mkdir()
        (clients_dir / "bank_C").mkdir()

        clients = discover_clients(tmp_path)

        assert clients == ["bank_A", "bank_B", "bank_C"]

    def test_ignores_hidden_directories(self, tmp_path):
        """Test that hidden directories are ignored"""
        clients_dir = tmp_path / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "bank_A").mkdir()
        (clients_dir / ".hidden").mkdir()

        clients = discover_clients(tmp_path)

        assert clients == ["bank_A"]
        assert ".hidden" not in clients

    def test_returns_empty_if_no_clients_dir(self, tmp_path):
        """Test returns empty list if clients directory doesn't exist"""
        clients = discover_clients(tmp_path)
        assert clients == []

    def test_returns_sorted_clients(self, tmp_path):
        """Test that clients are returned in sorted order"""
        clients_dir = tmp_path / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "zebra_bank").mkdir()
        (clients_dir / "alpha_bank").mkdir()
        (clients_dir / "beta_bank").mkdir()

        clients = discover_clients(tmp_path)

        assert clients == ["alpha_bank", "beta_bank", "zebra_bank"]


@pytest.mark.unit
class TestBuildDataPaths:
    """Tests for build_data_paths function"""

    def test_centralized_paths_without_edges(self, tmp_path):
        """Test centralized paths for non-GNN clients"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="centralized"
        )

        assert "trainset_nodes" in paths
        assert "valset_nodes" in paths
        assert "testset_nodes" in paths
        assert "trainset_edges" not in paths
        assert "centralized" in paths["trainset_nodes"]

    def test_centralized_paths_with_edges(self, tmp_path):
        """Test centralized paths for GNN clients"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchGeometricClient",
            setting="centralized"
        )

        assert "trainset_nodes" in paths
        assert "valset_nodes" in paths
        assert "testset_nodes" in paths
        assert "trainset_edges" in paths
        assert "valset_edges" in paths
        assert "testset_edges" in paths

    def test_centralized_paths_sklearn(self, tmp_path):
        """Test centralized paths for sklearn clients use different keys"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="SklearnClient",
            setting="centralized"
        )

        # SklearnClient uses trainset/valset/testset (not *_nodes)
        assert "trainset" in paths
        assert "valset" in paths
        assert "testset" in paths
        assert "trainset_nodes" not in paths

    def test_federated_paths(self, tmp_path):
        """Test federated paths with explicit clients"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="federated",
            clients=["bank_A", "bank_B"]
        )

        assert "clients" in paths
        assert "bank_A" in paths["clients"]
        assert "bank_B" in paths["clients"]
        assert "trainset_nodes" in paths["clients"]["bank_A"]
        assert "valset_nodes" in paths["clients"]["bank_A"]
        assert "testset_nodes" in paths["clients"]["bank_A"]

    def test_federated_paths_with_edges(self, tmp_path):
        """Test federated paths for GNN clients include edges"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchGeometricClient",
            setting="federated",
            clients=["bank_A"]
        )

        assert "trainset_edges" in paths["clients"]["bank_A"]
        assert "valset_edges" in paths["clients"]["bank_A"]
        assert "testset_edges" in paths["clients"]["bank_A"]

    def test_isolated_paths(self, tmp_path):
        """Test isolated paths (same structure as federated)"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="isolated",
            clients=["bank_A"]
        )

        assert "clients" in paths
        assert "bank_A" in paths["clients"]
        assert "trainset_nodes" in paths["clients"]["bank_A"]
        assert "valset_nodes" in paths["clients"]["bank_A"]
        assert "testset_nodes" in paths["clients"]["bank_A"]

    def test_auto_discovers_clients(self, tmp_path):
        """Test that clients are auto-discovered if not provided"""
        # Create client directories
        clients_dir = tmp_path / "preprocessed" / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "discovered_bank").mkdir()

        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="federated",
            clients=None  # Auto-discover
        )

        assert "discovered_bank" in paths["clients"]


@pytest.mark.unit
class TestLoadTrainingConfig:
    """Tests for load_training_config function"""

    def test_loads_yaml_config(self, tmp_path):
        """Test that YAML config is loaded"""
        from src.utils.config import load_training_config

        # Create experiment structure
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        preprocessed_dir = tmp_path / "preprocessed" / "centralized"
        preprocessed_dir.mkdir(parents=True)

        config = {
            'TestModel': {
                'default': {'client_type': 'TorchClient', 'lr': 0.01, 'hidden_dim': 32},
                'centralized': {'batch_size': 64}
            }
        }
        config_path = config_dir / "models.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = load_training_config(
            str(config_path),
            model_type="TestModel",
            setting="centralized"
        )

        assert result['lr'] == 0.01
        assert result['hidden_dim'] == 32
        assert result['batch_size'] == 64

    def test_raises_error_for_unknown_model(self, tmp_path):
        """Test that unknown model type raises ValueError"""
        from src.utils.config import load_training_config

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)

        config = {'KnownModel': {'default': {'client_type': 'TorchClient'}}}
        config_path = config_dir / "models.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Model 'UnknownModel' not found"):
            load_training_config(
                str(config_path),
                model_type="UnknownModel",
                setting="centralized"
            )

    def test_merges_default_and_setting_config(self, tmp_path):
        """Test that default config is merged with setting-specific config"""
        from src.utils.config import load_training_config

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)

        config = {
            'TestModel': {
                'default': {'client_type': 'TorchClient', 'lr': 0.01, 'hidden_dim': 32, 'dropout': 0.1},
                'centralized': {'hidden_dim': 64}  # Override hidden_dim
            }
        }
        config_path = config_dir / "models.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = load_training_config(
            str(config_path),
            model_type="TestModel",
            setting="centralized"
        )

        assert result['lr'] == 0.01  # From default
        assert result['hidden_dim'] == 64  # Overridden
        assert result['dropout'] == 0.1  # From default

    def test_handles_isolated_client_overrides(self, tmp_path):
        """Test that isolated setting handles per-client overrides"""
        from src.utils.config import load_training_config

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        clients_dir = tmp_path / "preprocessed" / "clients" / "bank_A"
        clients_dir.mkdir(parents=True)

        config = {
            'TestModel': {
                'default': {'client_type': 'TorchClient', 'lr': 0.01},
                'isolated': {
                    'epochs': 100,
                    'clients': {
                        'bank_A': {'lr': 0.001}
                    }
                }
            }
        }
        config_path = config_dir / "models.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = load_training_config(
            str(config_path),
            model_type="TestModel",
            setting="isolated"
        )

        assert '_client_overrides' in result
        assert 'bank_A' in result['_client_overrides']

    def test_uses_experiment_root_override(self, tmp_path):
        """Test that experiment root can be overridden in config"""
        from src.utils.config import load_training_config

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        custom_root = tmp_path / "custom_root"
        preprocessed_dir = custom_root / "preprocessed" / "centralized"
        preprocessed_dir.mkdir(parents=True)

        config = {
            'experiment': {'root': str(custom_root)},
            'TestModel': {'default': {'client_type': 'TorchClient'}}
        }
        config_path = config_dir / "models.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = load_training_config(
            str(config_path),
            model_type="TestModel",
            setting="centralized"
        )

        # Data paths should be based on custom root
        assert str(custom_root) in result['trainset_nodes']

    def test_raises_error_for_missing_client_type(self, tmp_path):
        """Test that missing client_type raises ValueError"""
        from src.utils.config import load_training_config

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)

        config = {
            'TestModel': {
                'default': {'lr': 0.01}  # No client_type
            }
        }
        config_path = config_dir / "models.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="client_type not found"):
            load_training_config(
                str(config_path),
                model_type="TestModel",
                setting="centralized"
            )


@pytest.mark.unit
class TestGetClientConfig:
    """Tests for get_client_config function"""

    def test_returns_base_config_without_overrides(self):
        """Test returns base config when no overrides present"""
        from src.utils.config import get_client_config

        base_config = {'lr': 0.01, 'epochs': 100}
        result = get_client_config(base_config, 'bank_A')

        assert result == {'lr': 0.01, 'epochs': 100}

    def test_applies_client_specific_overrides(self):
        """Test applies client-specific overrides"""
        from src.utils.config import get_client_config

        base_config = {
            'lr': 0.01,
            'epochs': 100,
            '_client_overrides': {
                'bank_A': {'lr': 0.001, 'batch_size': 32}
            }
        }
        result = get_client_config(base_config, 'bank_A')

        assert result['lr'] == 0.001  # Overridden
        assert result['epochs'] == 100  # From base
        assert result['batch_size'] == 32  # Added from override
        assert '_client_overrides' not in result
