from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AbstractServer(ABC):
    """Abstract base class for federated learning servers."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the server."""
        pass

    @abstractmethod
    def aggregate(self, *args, **kwargs):
        """Aggregate updates from clients."""
        pass

    @abstractmethod
    def distribute(self, *args, **kwargs):
        """Distribute model/gradients to clients."""
        pass
