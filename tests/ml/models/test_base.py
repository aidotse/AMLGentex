"""
Unit tests for ml/models/base.py
"""
import pytest
import torch
import tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import numpy as np

from src.ml.models.base import TorchBaseModel, SklearnBaseModel


class SimpleTorchModel(TorchBaseModel):
    """Simple model for testing TorchBaseModel"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
class TestTorchBaseModel:
    """Tests for TorchBaseModel"""

    def test_get_parameters(self):
        """Test get_parameters returns all parameters"""
        model = SimpleTorchModel(10, 2)

        params = model.get_parameters()

        assert 'linear.weight' in params
        assert 'linear.bias' in params
        assert params['linear.weight'].shape == (2, 10)
        assert params['linear.bias'].shape == (2,)

    def test_set_parameters(self):
        """Test set_parameters sets all parameters"""
        model1 = SimpleTorchModel(10, 2)
        model2 = SimpleTorchModel(10, 2)

        # Get params from model1
        params = model1.get_parameters()

        # Set on model2
        model2.set_parameters(params)

        # Check they match
        for name, param in model2.named_parameters():
            assert torch.allclose(param.data, params[name])

    def test_set_parameters_strict(self):
        """Test set_parameters with strict=True"""
        model = SimpleTorchModel(10, 2)

        incomplete_params = {'linear.weight': torch.randn(2, 10)}

        # Should raise KeyError for missing bias
        with pytest.raises(KeyError, match="linear.bias"):
            model.set_parameters(incomplete_params, strict=True)

    def test_set_parameters_non_strict(self):
        """Test set_parameters with strict=False allows partial updates"""
        model = SimpleTorchModel(10, 2)

        # Only set weight, not bias
        new_weight = torch.randn(2, 10)
        partial_params = {'linear.weight': new_weight}

        # Should not raise
        model.set_parameters(partial_params, strict=False)

        assert torch.allclose(model.linear.weight.data, new_weight)

    def test_get_gradients(self):
        """Test get_gradients returns gradients after backward"""
        model = SimpleTorchModel(10, 2)
        x = torch.randn(5, 10)

        output = model(x)
        loss = output.sum()
        loss.backward()

        grads = model.get_gradients()

        assert 'linear.weight' in grads
        assert 'linear.bias' in grads
        assert grads['linear.weight'].shape == (2, 10)

    def test_get_gradients_returns_empty_before_backward(self):
        """Test get_gradients returns empty dict before backward"""
        model = SimpleTorchModel(10, 2)

        grads = model.get_gradients()

        assert grads == {}

    def test_set_gradients(self):
        """Test set_gradients sets gradients correctly"""
        model = SimpleTorchModel(10, 2)
        x = torch.randn(5, 10)

        # Create gradients
        output = model(x)
        loss = output.sum()
        loss.backward()

        grads = model.get_gradients()

        # Create new model and set gradients
        model2 = SimpleTorchModel(10, 2)
        output2 = model2(x)
        loss2 = output2.sum()
        loss2.backward()

        model2.set_gradients(grads)

        for name, param in model2.named_parameters():
            assert torch.allclose(param.grad, grads[name])

    def test_set_gradients_strict(self):
        """Test set_gradients with strict=True"""
        model = SimpleTorchModel(10, 2)
        x = torch.randn(5, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()

        incomplete_grads = {'linear.weight': torch.randn(2, 10)}

        with pytest.raises(KeyError, match="linear.bias"):
            model.set_gradients(incomplete_grads, strict=True)

    def test_forward_not_implemented(self):
        """Test that TorchBaseModel.forward raises NotImplementedError"""
        # Create instance without implementing forward
        model = TorchBaseModel()

        with pytest.raises(NotImplementedError):
            model.forward(torch.randn(5, 10))

    def test_parameters_are_detached(self):
        """Test that get_parameters returns detached tensors"""
        model = SimpleTorchModel(10, 2)

        params = model.get_parameters()

        for name, param in params.items():
            assert not param.requires_grad


@pytest.mark.unit
class TestSklearnBaseModel:
    """Tests for SklearnBaseModel"""

    def test_init(self):
        """Test initialization"""
        sklearn_model = LogisticRegression()
        model = SklearnBaseModel(sklearn_model)

        assert model.model is sklearn_model

    def test_fit(self):
        """Test fit method"""
        sklearn_model = LogisticRegression()
        model = SklearnBaseModel(sklearn_model)

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)

        # Model should be fitted
        assert hasattr(model.model, 'coef_')

    def test_predict(self):
        """Test predict method"""
        sklearn_model = LogisticRegression()
        model = SklearnBaseModel(sklearn_model)

        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 10)
        predictions = model.predict(X_test)

        assert predictions.shape == (20,)

    def test_get_parameters(self):
        """Test get_parameters returns sklearn params"""
        sklearn_model = LogisticRegression(C=0.5, max_iter=200)
        model = SklearnBaseModel(sklearn_model)

        params = model.get_parameters()

        assert params['C'] == 0.5
        assert params['max_iter'] == 200

    def test_set_parameters(self):
        """Test set_parameters sets sklearn params"""
        sklearn_model = LogisticRegression()
        model = SklearnBaseModel(sklearn_model)

        model.set_parameters({'C': 0.1, 'max_iter': 500})

        assert model.model.C == 0.1
        assert model.model.max_iter == 500

    def test_save_and_load_model(self, tmp_path):
        """Test save_model and load_model"""
        sklearn_model = LogisticRegression()
        model = SklearnBaseModel(sklearn_model)

        # Fit the model
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)

        # Save
        model_path = tmp_path / "model.joblib"
        model.save_model(str(model_path))

        assert model_path.exists()

        # Load into new model
        new_sklearn_model = LogisticRegression()
        new_model = SklearnBaseModel(new_sklearn_model)
        new_model.load_model(str(model_path))

        # Predictions should match
        X_test = np.random.randn(10, 10)
        original_preds = model.predict(X_test)
        loaded_preds = new_model.predict(X_test)

        np.testing.assert_array_equal(original_preds, loaded_preds)
