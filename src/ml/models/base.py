import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any
import joblib


class AbstractModel(ABC):
    """
    Abstract base class for all models.
    Ensures both PyTorch and Sklearn models implement necessary methods.
    """

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns the model's parameters.

        Returns:
            Dict[str, Any]: A dictionary containing parameter names and values.
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Sets the model's parameters.

        Args:
            parameters (Dict[str, Any]): A dictionary containing parameter names and values.
        """
        pass


class TorchBaseModel(nn.Module, AbstractModel):
    """
    Base class for PyTorch models, extending AbstractModel.
    Provides methods for getting/setting parameters and gradients.
    """

    def __init__(self):
        super(TorchBaseModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model (to be implemented by subclasses).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Retrieves the gradients of the model's parameters.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to gradients.
        """
        return {
            name: param.grad.clone().detach() 
            for name, param in self.named_parameters() if param.grad is not None
        }
    
    def set_gradients(self, gradients: Dict[str, torch.Tensor], strict: bool = False):
        """
        Sets the gradients of the model's parameters.

        Args:
            gradients (Dict[str, torch.Tensor]): A dictionary of parameter gradients.
            strict (bool): If True, raises an error if a parameter is missing.
        """
        for name, param in self.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone().detach()
            elif strict:
                raise KeyError(f"Gradient for parameter '{name}' not found in provided dictionary.")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Retrieves the model's parameters.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to values.
        """
        return {
            name: param.data.clone().detach() 
            for name, param in self.named_parameters()
        }

    def set_parameters(self, parameters: Dict[str, torch.Tensor], strict: bool = False):
        """
        Sets the model's parameters.

        Args:
            parameters (Dict[str, torch.Tensor]): A dictionary mapping parameter names to values.
            strict (bool): If True, raises an error if a parameter is missing.
        """
        for name, param in self.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone().detach()
            elif strict:
                raise KeyError(f"Parameter '{name}' not found in provided dictionary.")


class SklearnBaseModel(AbstractModel):
    """
    Base class for Scikit-Learn models, extending AbstractModel.
    Provides methods for saving/loading and retrieving model parameters.
    """

    def __init__(self, model: Any):
        """
        Initializes the Scikit-Learn model.

        Args:
            model (Any): A Scikit-Learn model instance.
        """
        self.model = model

    def fit(self, X, y):
        """Trains the Scikit-Learn model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predicts using the trained Scikit-Learn model."""
        return self.model.predict(X)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieves the model's parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the model parameters.
        """
        return self.model.get_params()

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Sets the model's parameters.

        Args:
            parameters (Dict[str, Any]): A dictionary containing parameter names and values.
        """
        self.model.set_params(**parameters)

    def save_model(self, path: str):
        """
        Saves the Scikit-Learn model.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """
        Loads a Scikit-Learn model.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
