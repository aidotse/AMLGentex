import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple

class AbstractClient(ABC):
    """
    Abstract base class for all clients.
    Ensures standard methods for clients (e.g., train, evaluate, run).
    """

    @abstractmethod
    def train(self):
        """Train the model on local dataset."""
        pass

    @abstractmethod
    def evaluate(self, dataset: str) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on given dataset."""
        pass

    @abstractmethod
    def run(self, **kwargs):
        """Run training and evaluation loop."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict:
        """Retrieve model parameters."""
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict):
        """Set model parameters."""
        pass
    
    @abstractmethod
    def get_gradients(self) -> Dict:
        """Retrieve model gradients."""
        pass

    @abstractmethod
    def set_gradients(self, gradients: Dict):
        """Set model gradients."""
        pass
    
    @abstractmethod
    def compute_gardients(self) -> Dict:
        """Train and retrive gradients"""
        pass
