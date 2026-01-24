"""
Unit tests for ml/models/gnn_models.py
"""
import pytest
import torch
import torch_geometric

from src.ml.models.gnn_models import GCN, GAT, GraphSAGE


@pytest.fixture
def sample_graph_data():
    """Create sample PyG Data object for testing"""
    x = torch.randn(10, 16)  # 10 nodes, 16 features
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    ], dtype=torch.long)
    return torch_geometric.data.Data(x=x, edge_index=edge_index)


@pytest.mark.unit
class TestGCN:
    """Tests for GCN model"""

    def test_init(self):
        """Test initialization"""
        model = GCN(input_dim=16, n_conv_layers=2, hidden_dim=32, output_dim=2)

        assert model.input_layer.in_channels == 16
        assert model.input_layer.out_channels == 32
        assert len(model.conv_layers) == 2
        assert len(model.layer_norms) == 2
        assert model.output_layer.out_channels == 2

    def test_forward(self, sample_graph_data):
        """Test forward pass"""
        model = GCN(input_dim=16, n_conv_layers=2, hidden_dim=32, output_dim=2)
        model.eval()  # Disable dropout

        output = model(sample_graph_data)

        assert output.shape == (10, 2)  # 10 nodes, 2 classes

    def test_forward_binary(self, sample_graph_data):
        """Test forward pass with binary output"""
        model = GCN(input_dim=16, n_conv_layers=1, hidden_dim=16, output_dim=1)
        model.eval()

        output = model(sample_graph_data)

        assert output.shape == (10,)  # squeezed

    def test_get_parameters_excludes_layer_norms(self):
        """Test that get_parameters excludes layer norm parameters"""
        model = GCN(input_dim=16, n_conv_layers=2, hidden_dim=32, output_dim=2)

        params = model.get_parameters()

        # Should not include layer_norms
        for name in params.keys():
            assert 'layer_norms' not in name

    def test_set_parameters_excludes_layer_norms(self):
        """Test that set_parameters excludes layer norm parameters"""
        model1 = GCN(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)
        model2 = GCN(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        # Get params from model1 (excludes layer_norms)
        params = model1.get_parameters()

        # Set on model2
        model2.set_parameters(params)

        # Conv weights should match
        for name, param in model2.named_parameters():
            if 'layer_norms' not in name:
                assert torch.allclose(param.data, params[name])

    def test_get_gradients_excludes_layer_norms(self, sample_graph_data):
        """Test that get_gradients excludes layer norm gradients"""
        model = GCN(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        # Forward and backward pass
        output = model(sample_graph_data)
        loss = output.sum()
        loss.backward()

        grads = model.get_gradients()

        for name in grads.keys():
            assert 'layer_norms' not in name

    def test_set_gradients(self, sample_graph_data):
        """Test set_gradients"""
        model = GCN(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        # Forward and backward to create gradients
        output = model(sample_graph_data)
        loss = output.sum()
        loss.backward()

        grads = model.get_gradients()

        # Create new model and set gradients
        model2 = GCN(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)
        output2 = model2(sample_graph_data)
        loss2 = output2.sum()
        loss2.backward()

        model2.set_gradients(grads)

        # Check gradients were set for non-layer-norm params
        for name, param in model2.named_parameters():
            if name in grads:
                assert torch.allclose(param.grad, grads[name])


@pytest.mark.unit
class TestGAT:
    """Tests for GAT model"""

    def test_init(self):
        """Test initialization"""
        model = GAT(input_dim=16, n_conv_layers=2, hidden_dim=32, output_dim=2)

        assert model.input_layer.in_channels == 16
        assert len(model.conv_layers) == 2
        assert len(model.layer_norms) == 2

    def test_forward(self, sample_graph_data):
        """Test forward pass"""
        model = GAT(input_dim=16, n_conv_layers=2, hidden_dim=32, output_dim=2)
        model.eval()

        output = model(sample_graph_data)

        assert output.shape == (10, 2)

    def test_get_parameters_excludes_layer_norms(self):
        """Test that get_parameters excludes layer norm parameters"""
        model = GAT(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        params = model.get_parameters()

        for name in params.keys():
            assert 'layer_norms' not in name

    def test_gradient_flow(self, sample_graph_data):
        """Test that gradients flow through the model"""
        model = GAT(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        output = model(sample_graph_data)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.unit
class TestGraphSAGE:
    """Tests for GraphSAGE model"""

    def test_init(self):
        """Test initialization"""
        model = GraphSAGE(input_dim=16, n_conv_layers=2, hidden_dim=32, output_dim=2)

        assert model.input_layer.in_channels == 16
        assert len(model.conv_layers) == 2
        assert len(model.layer_norms) == 2

    def test_forward(self, sample_graph_data):
        """Test forward pass"""
        model = GraphSAGE(input_dim=16, n_conv_layers=2, hidden_dim=32, output_dim=2)
        model.eval()

        output = model(sample_graph_data)

        assert output.shape == (10, 2)

    def test_get_parameters_excludes_layer_norms(self):
        """Test that get_parameters excludes layer norm parameters"""
        model = GraphSAGE(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        params = model.get_parameters()

        for name in params.keys():
            assert 'layer_norms' not in name

    def test_set_parameters_strict_mode(self):
        """Test set_parameters with strict=True raises on missing param"""
        model = GraphSAGE(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        incomplete_params = {'nonexistent_param': torch.randn(10)}

        # Should raise KeyError in strict mode for missing params
        with pytest.raises(KeyError):
            model.set_parameters(incomplete_params, strict=True)

    def test_set_gradients_strict_mode(self, sample_graph_data):
        """Test set_gradients with strict=True raises on missing gradient"""
        model = GraphSAGE(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2)

        # Need to do forward/backward first to have gradients
        output = model(sample_graph_data)
        loss = output.sum()
        loss.backward()

        incomplete_grads = {'nonexistent_param': torch.randn(10)}

        with pytest.raises(KeyError):
            model.set_gradients(incomplete_grads, strict=True)

    def test_dropout_applied_in_training(self, sample_graph_data):
        """Test that dropout is applied during training"""
        model = GraphSAGE(input_dim=16, n_conv_layers=1, hidden_dim=32, output_dim=2, dropout=0.5)
        model.train()

        # Run multiple times - outputs should differ due to dropout
        output1 = model(sample_graph_data)
        output2 = model(sample_graph_data)

        # With 50% dropout, outputs are unlikely to be identical
        assert not torch.allclose(output1, output2)
