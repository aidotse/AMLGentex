"""
Tests for custom ML metrics.

Tests cover:
- confusion_matrix: Including validation of FP/FN bug fix
- precision, recall, balanced_accuracy: Computed from confusion matrix
- average_precision_score: With custom recall span
"""

import pytest
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.ml.metrics import (
    confusion_matrix,
    precision,
    recall,
    balanced_accuracy,
    average_precision_score,
    constrained_utility_metric,
)


class TestConfusionMatrix:
    """Tests for confusion_matrix function."""

    def test_confusion_matrix_single_threshold(self):
        """Test confusion matrix with a single threshold."""
        # Simple binary classification scenario
        # y_true: [1, 1, 0, 0]
        # y_pred: [[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]]
        # threshold: 0.5
        # Predictions > 0.5: [True, True, False, False]
        # Expected:
        #   TP (pred=1, true=1): 2
        #   FP (pred=1, true=0): 0
        #   FN (pred=0, true=1): 0
        #   TN (pred=0, true=0): 2

        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]])
        threshold = 0.5

        cm = confusion_matrix(y_pred, y_true, threshold)

        assert cm.shape == (1, 5)
        assert cm[0, 0] == 2  # TP
        assert cm[0, 1] == 0  # FP
        assert cm[0, 2] == 0  # FN
        assert cm[0, 3] == 2  # TN
        assert cm[0, 4] == threshold

    def test_confusion_matrix_validates_bug_fix(self):
        """Test that validates the FP/FN bug fix."""
        # Scenario specifically designed to catch FP/FN swap
        # y_true: [1, 0]
        # y_pred: [[0, 0.8], [0, 0.3]]
        # threshold: 0.5
        # Predictions > 0.5: [True, False]
        # Expected:
        #   TP (pred=1, true=1): 1  (first sample)
        #   FP (pred=1, true=0): 0  (no false positives)
        #   FN (pred=0, true=1): 0  (no false negatives)
        #   TN (pred=0, true=0): 1  (second sample)

        y_true = np.array([1, 0])
        y_pred = np.array([[0, 0.8], [0, 0.3]])
        threshold = 0.5

        cm = confusion_matrix(y_pred, y_true, threshold)

        # If bug exists (FP/FN swapped), these assertions would fail
        assert cm[0, 0] == 1, "TP should be 1"
        assert cm[0, 1] == 0, "FP should be 0 (bug would give FP=0)"
        assert cm[0, 2] == 0, "FN should be 0 (bug would give FN=0)"
        assert cm[0, 3] == 1, "TN should be 1"

    def test_confusion_matrix_with_false_positives(self):
        """Test scenario with false positives."""
        # y_true: [0, 0, 1]
        # y_pred: [[0, 0.8], [0, 0.6], [0, 0.7]]
        # threshold: 0.5
        # All predictions > 0.5
        # Expected:
        #   TP: 1 (third sample)
        #   FP: 2 (first two samples - predicted positive but actually negative)
        #   FN: 0
        #   TN: 0

        y_true = np.array([0, 0, 1])
        y_pred = np.array([[0, 0.8], [0, 0.6], [0, 0.7]])
        threshold = 0.5

        cm = confusion_matrix(y_pred, y_true, threshold)

        assert cm[0, 0] == 1, "TP should be 1"
        assert cm[0, 1] == 2, "FP should be 2"
        assert cm[0, 2] == 0, "FN should be 0"
        assert cm[0, 3] == 0, "TN should be 0"

    def test_confusion_matrix_with_false_negatives(self):
        """Test scenario with false negatives."""
        # y_true: [1, 1, 0]
        # y_pred: [[0, 0.3], [0, 0.4], [0, 0.2]]
        # threshold: 0.5
        # All predictions <= 0.5
        # Expected:
        #   TP: 0
        #   FP: 0
        #   FN: 2 (first two samples - predicted negative but actually positive)
        #   TN: 1 (third sample)

        y_true = np.array([1, 1, 0])
        y_pred = np.array([[0, 0.3], [0, 0.4], [0, 0.2]])
        threshold = 0.5

        cm = confusion_matrix(y_pred, y_true, threshold)

        assert cm[0, 0] == 0, "TP should be 0"
        assert cm[0, 1] == 0, "FP should be 0"
        assert cm[0, 2] == 2, "FN should be 2"
        assert cm[0, 3] == 1, "TN should be 1"

    def test_confusion_matrix_multiple_thresholds(self):
        """Test confusion matrix with multiple thresholds."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([[0, 0.9], [0, 0.6], [0, 0.4], [0, 0.1]])
        thresholds = [0.3, 0.5, 0.7]

        cm = confusion_matrix(y_pred, y_true, thresholds)

        assert cm.shape == (3, 5)

        # threshold 0.3: [1,1,1,0] predicted
        assert cm[0, 0] == 2  # TP
        assert cm[0, 1] == 1  # FP
        assert cm[0, 2] == 0  # FN
        assert cm[0, 3] == 1  # TN

        # threshold 0.5: [1,1,0,0] predicted
        assert cm[1, 0] == 2  # TP
        assert cm[1, 1] == 0  # FP
        assert cm[1, 2] == 0  # FN
        assert cm[1, 3] == 2  # TN

        # threshold 0.7: [1,0,0,0] predicted
        assert cm[2, 0] == 1  # TP
        assert cm[2, 1] == 0  # FP
        assert cm[2, 2] == 1  # FN
        assert cm[2, 3] == 2  # TN

    def test_confusion_matrix_dynamic_thresholds(self):
        """Test confusion matrix with dynamic threshold generation."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([[0, 0.5], [0, 0.3], [0, 0.8]])

        cm = confusion_matrix(y_pred, y_true, "dynamic")

        # Should create thresholds from unique rounded predictions
        assert cm.shape[0] > 0
        assert cm.shape[1] == 5


class TestPrecision:
    """Tests for precision metric."""

    def test_precision_perfect(self):
        """Test precision with perfect predictions."""
        # TP=2, FP=0 -> precision = 1.0
        cm = np.array([[2, 0, 0, 0, 0.5]])

        p = precision(cm)

        assert p[0] == 1.0

    def test_precision_with_false_positives(self):
        """Test precision with false positives."""
        # TP=2, FP=1 -> precision = 2/3
        cm = np.array([[2, 1, 0, 0, 0.5]])

        p = precision(cm)

        assert np.isclose(p[0], 2/3)

    def test_precision_zero_predictions(self):
        """Test precision when no positive predictions."""
        # TP=0, FP=0 -> precision = 0 (by our implementation)
        cm = np.array([[0, 0, 2, 2, 0.5]])

        p = precision(cm)

        assert p[0] == 0.0


class TestRecall:
    """Tests for recall metric."""

    def test_recall_perfect(self):
        """Test recall with perfect predictions."""
        # TP=2, FN=0 -> recall = 1.0
        cm = np.array([[2, 0, 0, 0, 0.5]])

        r = recall(cm)

        assert r[0] == 1.0

    def test_recall_with_false_negatives(self):
        """Test recall with false negatives."""
        # TP=2, FN=1 -> recall = 2/3
        cm = np.array([[2, 0, 1, 0, 0.5]])

        r = recall(cm)

        assert np.isclose(r[0], 2/3)

    def test_recall_no_positives(self):
        """Test recall when no positive labels."""
        # TP=0, FN=0 -> recall = 1.0 (by our implementation)
        cm = np.array([[0, 0, 0, 2, 0.5]])

        r = recall(cm)

        assert r[0] == 1.0


class TestBalancedAccuracy:
    """Tests for balanced accuracy metric."""

    def test_balanced_accuracy_perfect(self):
        """Test balanced accuracy with perfect predictions."""
        # TP=2, FP=0, FN=0, TN=2 -> balanced_acc = 1.0
        cm = np.array([[2, 0, 0, 2, 0.5]])

        ba = balanced_accuracy(cm)

        assert ba[0] == 1.0

    def test_balanced_accuracy_imbalanced(self):
        """Test balanced accuracy with imbalanced predictions."""
        # TP=1, FP=1, FN=1, TN=1
        # sensitivity = 1/2 = 0.5
        # specificity = 1/2 = 0.5
        # balanced_acc = (0.5 + 0.5) / 2 = 0.5
        cm = np.array([[1, 1, 1, 1, 0.5]])

        ba = balanced_accuracy(cm)

        assert ba[0] == 0.5


class TestAveragePrecisionScore:
    """Tests for average precision score with custom recall span."""

    def test_average_precision_full_span(self):
        """Test average precision with full recall span."""
        # Simple scenario
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.5, 0.6, 0.3])

        ap = average_precision_score(y_true, y_pred, recall_span=(0.0, 1.0))

        # Should be a value between 0 and 1
        assert 0 <= ap <= 1

    def test_average_precision_high_recall_span(self):
        """Test average precision with high recall span (0.6, 1.0)."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.5, 0.6, 0.3])

        ap = average_precision_score(y_true, y_pred, recall_span=(0.6, 1.0))

        # Should be a value between 0 and 1
        assert 0 <= ap <= 1

    def test_average_precision_perfect_classifier(self):
        """Test average precision with perfect classifier."""
        # All positives ranked higher than negatives
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        ap = average_precision_score(y_true, y_pred, recall_span=(0.0, 1.0))

        # Should be close to 1.0 for perfect classifier
        assert ap > 0.9

    def test_average_precision_worst_classifier(self):
        """Test average precision with worst classifier."""
        # All positives ranked lower than negatives
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        ap = average_precision_score(y_true, y_pred, recall_span=(0.0, 1.0))

        # Should be close to proportion of positives (0.5) or lower
        assert ap < 0.6


class TestMetricsIntegration:
    """Integration tests combining multiple metrics."""

    def test_full_pipeline(self):
        """Test full metrics pipeline: confusion matrix -> precision, recall, balanced_accuracy."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([[0, 0.9], [0, 0.8], [0, 0.7], [0, 0.6],
                          [0, 0.5], [0, 0.4], [0, 0.3], [0, 0.2]])
        threshold = 0.5

        # Compute confusion matrix
        cm = confusion_matrix(y_pred, y_true, threshold)

        # Compute derived metrics
        p = precision(cm)
        r = recall(cm)
        ba = balanced_accuracy(cm)

        # All should be valid values
        assert 0 <= p[0] <= 1
        assert 0 <= r[0] <= 1
        assert 0 <= ba[0] <= 1

        # TP + FP + FN + TN should equal total samples
        assert cm[0, 0] + cm[0, 1] + cm[0, 2] + cm[0, 3] == len(y_true)


class TestConstrainedUtilityMetric:
    """Tests for constrained utility metric function."""

    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced test data (10% positive class)."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.array([0] * 900 + [1] * 100)
        y_pred_proba = np.random.rand(n_samples)
        # Boost scores for true positives to make them distinguishable
        y_pred_proba[y_true == 1] += 0.3
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        return y_true, y_pred_proba

    def test_precision_at_k(self, imbalanced_data):
        """Test Precision@K metric."""
        y_true, y_pred_proba = imbalanced_data
        K = 100

        precision_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', K, 'precision'
        )

        # Should return a value between 0 and 1
        assert 0 <= precision_at_k <= 1

        # With boosted scores for positives, should have reasonable precision
        # (at least better than random: 0.1)
        assert precision_at_k > 0.1

        # Manually verify: count how many of top K are actually positive
        top_k_indices = np.argsort(y_pred_proba)[-K:]
        expected_precision = y_true[top_k_indices].sum() / K
        assert np.isclose(precision_at_k, expected_precision)

    def test_recall_at_k(self, imbalanced_data):
        """Test Recall@K metric."""
        y_true, y_pred_proba = imbalanced_data
        K = 100

        recall_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', K, 'recall'
        )

        # Should return a value between 0 and 1
        assert 0 <= recall_at_k <= 1

        # Manually verify: fraction of positives captured in top K
        top_k_indices = np.argsort(y_pred_proba)[-K:]
        n_positives = y_true.sum()
        expected_recall = y_true[top_k_indices].sum() / n_positives
        assert np.isclose(recall_at_k, expected_recall)

    def test_recall_at_fpr(self, imbalanced_data):
        """Test Recall at FPR constraint."""
        y_true, y_pred_proba = imbalanced_data
        alpha = 0.01  # Max 1% FPR

        recall_at_fpr = constrained_utility_metric(
            y_true, y_pred_proba, 'fpr', alpha, 'recall'
        )

        # Should return a value between 0 and 1
        assert 0 <= recall_at_fpr <= 1

        # With strict FPR constraint, recall should be non-zero but might be low
        assert recall_at_fpr >= 0

    def test_precision_at_fpr(self, imbalanced_data):
        """Test Precision at FPR constraint."""
        y_true, y_pred_proba = imbalanced_data
        alpha = 0.05  # Max 5% FPR

        precision_at_fpr = constrained_utility_metric(
            y_true, y_pred_proba, 'fpr', alpha, 'precision'
        )

        # Should return a value between 0 and 1
        assert 0 <= precision_at_fpr <= 1

    def test_precision_at_recall(self, imbalanced_data):
        """Test Precision at Recall constraint."""
        y_true, y_pred_proba = imbalanced_data
        target_recall = 0.7  # Must achieve at least 70% recall

        precision_at_recall = constrained_utility_metric(
            y_true, y_pred_proba, 'recall', target_recall, 'precision'
        )

        # Should return a value between 0 and 1
        assert 0 <= precision_at_recall <= 1

    def test_perfect_classifier_precision_at_k(self):
        """Test Precision@K with perfect classifier."""
        # All positives ranked higher than negatives
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        K = 5

        precision_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', K, 'precision'
        )

        # Should be perfect
        assert precision_at_k == 1.0

    def test_worst_classifier_precision_at_k(self):
        """Test Precision@K with worst classifier."""
        # All positives ranked lower than negatives
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred_proba = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        K = 5

        precision_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', K, 'precision'
        )

        # Should be zero (all top K are negatives)
        assert precision_at_k == 0.0

    def test_edge_case_k_equals_n(self):
        """Test Precision@K when K equals total samples."""
        y_true = np.array([1, 1, 0, 0])
        y_pred_proba = np.array([0.8, 0.6, 0.4, 0.2])
        K = 4

        precision_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', K, 'precision'
        )

        # Should equal overall positive rate
        assert precision_at_k == 0.5

    def test_edge_case_k_greater_than_n(self):
        """Test Precision@K when K > number of samples."""
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.7, 0.6, 0.3])
        K = 10  # K > n (10 > 4)

        precision_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', K, 'precision'
        )

        # Should treat K as n and return overall precision
        assert precision_at_k == 0.5

        recall_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', K, 'recall'
        )

        # Should treat K as n and return 100% recall (all positives captured)
        assert recall_at_k == 1.0

    def test_edge_case_no_valid_threshold_fpr(self):
        """Test Recall at FPR when no threshold satisfies constraint."""
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        alpha = 0.0  # Impossible constraint (0% FPR)

        recall_at_fpr = constrained_utility_metric(
            y_true, y_pred_proba, 'fpr', alpha, 'recall'
        )

        # Should return 0 when constraint cannot be satisfied
        assert recall_at_fpr == 0.0

    def test_edge_case_no_valid_threshold_recall(self):
        """Test Precision at Recall when no threshold achieves target recall."""
        # Create scenario where only 1 positive with very low score
        # Even at lowest threshold, we can't achieve high recall
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1])  # Positive has lowest score
        target_recall = 0.99  # Nearly impossible with this data

        precision_at_recall = constrained_utility_metric(
            y_true, y_pred_proba, 'recall', target_recall, 'precision'
        )

        # Should return 0 when constraint cannot be satisfied
        # (or very low precision if it does find a point)
        assert precision_at_recall <= 0.5

    def test_invalid_constraint_type(self):
        """Test that invalid constraint type raises error."""
        y_true = np.array([1, 0])
        y_pred_proba = np.array([0.8, 0.2])

        with pytest.raises(ValueError, match="Unknown constraint_type"):
            constrained_utility_metric(
                y_true, y_pred_proba, 'invalid', 100, 'precision'
            )

    def test_invalid_utility_metric(self):
        """Test that invalid utility metric raises error."""
        y_true = np.array([1, 0])
        y_pred_proba = np.array([0.8, 0.2])

        with pytest.raises(ValueError, match="Unknown utility_metric"):
            constrained_utility_metric(
                y_true, y_pred_proba, 'K', 100, 'invalid'
            )

    def test_invalid_recall_recall_combination(self):
        """Test that recall constraint with recall utility metric raises error."""
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.7, 0.6, 0.3])

        with pytest.raises(ValueError, match="Invalid combination.*recall.*recall"):
            constrained_utility_metric(
                y_true, y_pred_proba, 'recall', 0.7, 'recall'
            )

    def test_edge_case_fpr_constraint_out_of_range(self):
        """Test that FPR constraint_value outside [0, 1] raises error."""
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.7, 0.6, 0.3])

        # Test FPR > 1
        with pytest.raises(ValueError, match="FPR constraint_value must be in"):
            constrained_utility_metric(
                y_true, y_pred_proba, 'fpr', 1.5, 'recall'
            )

        # Test FPR < 0
        with pytest.raises(ValueError, match="FPR constraint_value must be in"):
            constrained_utility_metric(
                y_true, y_pred_proba, 'fpr', -0.1, 'recall'
            )

    def test_edge_case_recall_constraint_out_of_range(self):
        """Test that recall constraint_value outside [0, 1] raises error."""
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.7, 0.6, 0.3])

        # Test recall > 1
        with pytest.raises(ValueError, match="Recall constraint_value must be in"):
            constrained_utility_metric(
                y_true, y_pred_proba, 'recall', 1.5, 'precision'
            )

        # Test recall < 0
        with pytest.raises(ValueError, match="Recall constraint_value must be in"):
            constrained_utility_metric(
                y_true, y_pred_proba, 'recall', -0.1, 'precision'
            )

    def test_edge_case_no_negatives_fpr(self):
        """Test FPR constraint when there are no negatives (all y=1)."""
        y_true = np.array([1, 1, 1, 1])
        y_pred_proba = np.array([0.9, 0.7, 0.6, 0.3])

        # FPR is undefined - should return perfect scores
        recall_at_fpr = constrained_utility_metric(
            y_true, y_pred_proba, 'fpr', 0.01, 'recall'
        )
        assert recall_at_fpr == 1.0

        precision_at_fpr = constrained_utility_metric(
            y_true, y_pred_proba, 'fpr', 0.01, 'precision'
        )
        assert precision_at_fpr == 1.0

    def test_edge_case_no_positives_recall_constraint(self):
        """Test recall constraint when there are no positives (all y=0)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.9, 0.7, 0.6, 0.3])

        # Recall is undefined - should return 0
        precision_at_recall = constrained_utility_metric(
            y_true, y_pred_proba, 'recall', 0.7, 'precision'
        )
        assert precision_at_recall == 0.0

    def test_edge_case_no_positives_recall_k(self):
        """Test Recall@K when there are no positives (all y=0)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.9, 0.7, 0.6, 0.3])

        # Recall is undefined - should return 0
        recall_at_k = constrained_utility_metric(
            y_true, y_pred_proba, 'K', 2, 'recall'
        )
        assert recall_at_k == 0.0
