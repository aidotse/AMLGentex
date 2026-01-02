"""
Custom metrics for AML detection evaluation.

Provides specialized metrics including:
- average_precision_score: Average precision with configurable recall span
- confusion_matrix: Custom confusion matrix calculation
- precision, recall, balanced_accuracy: Metrics computed from confusion matrix
- constrained_utility_metric: Utility metrics under constraints (Precision@K, Recall@FPR, etc.)
"""

from src.ml.metrics.metrics import (
    average_precision_score,
    confusion_matrix,
    precision,
    recall,
    balanced_accuracy,
    constrained_utility_metric,
)

__all__ = [
    'average_precision_score',
    'confusion_matrix',
    'precision',
    'recall',
    'balanced_accuracy',
    'constrained_utility_metric',
]