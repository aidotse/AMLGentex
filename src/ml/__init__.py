"""
Machine Learning module for AMLGentex.

Contains clients, servers, models, and training orchestration for
centralized, federated, and isolated learning scenarios.
"""

from src.ml import clients, servers, models, training

__all__ = ['clients', 'servers', 'models', 'training']
