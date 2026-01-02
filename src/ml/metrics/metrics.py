import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, roc_curve
from typing import Tuple, Union, List


def precision(confusion_matrix: np.ndarray) -> np.ndarray:
    tp = confusion_matrix[:, 0]
    fp = confusion_matrix[:, 1]
    return np.divide(tp, tp + fp, where=(tp + fp) > 0, out=np.zeros_like(tp, dtype=float))


def recall(confusion_matrix: np.ndarray) -> np.ndarray:
    tp = confusion_matrix[:, 0]
    fn = confusion_matrix[:, 2]
    return np.divide(tp, tp + fn, where=(tp + fn) > 0, out=np.ones_like(tp, dtype=float))


def average_precision_score(y_true: np.ndarray, y_pred: np.ndarray, recall_span: Tuple[int, int]=(0.0, 1.0)) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
    precisions = precisions[::-1]
    recalls = recalls[::-1]
    idxs = np.arange(np.where(recalls <= recall_span[0])[0][-1], np.where(recalls >= recall_span[1])[0][-1] + 1) # need "+ 1" due to np.arange 
    recall_diffs = np.diff(recalls[idxs])
    precision_areas = precisions[idxs[1:]] * recall_diffs
    return np.sum(precision_areas) / np.sum(recall_diffs)


def balanced_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
    tp = confusion_matrix[:, 0]
    fp = confusion_matrix[:, 1]
    fn = confusion_matrix[:, 2]
    tn = confusion_matrix[:, 3]
    return 0.5*(tp/(tp+fn) + tn/(tn+fp))


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, thresholds: Union[str, int, float, List[float]]) -> np.ndarray:
    if isinstance(thresholds, str) and thresholds == "dynamic":
        thresholds = np.unique(y_pred[:,1].round(decimals=4))
    if isinstance(thresholds, int):
        thresholds = np.linspace(0.0, 1.0, thresholds)
    if isinstance(thresholds, float):
        thresholds = np.array([thresholds])
    if isinstance(thresholds, list):
        thresholds = np.array(thresholds)
    y_true = y_true.astype(bool)
    cm = np.zeros((len(thresholds), 5), dtype=float)
    for i, threshold in enumerate(thresholds):
        cm[i, 0] = np.sum((y_pred[:,1] > threshold) & y_true)   # TP
        cm[i, 1] = np.sum((y_pred[:,1] > threshold) & ~y_true)  # FP
        cm[i, 2] = np.sum((y_pred[:,1] <= threshold) & y_true)  # FN
        cm[i, 3] = np.sum((y_pred[:,1] <= threshold) & ~y_true) # TN
        cm[i, 4] = threshold
    return cm


def constrained_utility_metric(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 constraint_type: str, constraint_value: float,
                                 utility_metric: str) -> float:
    """
    Compute utility metric under a constraint.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        constraint_type: Type of constraint - 'K' (alert budget), 'fpr' (max FPR), or 'recall' (min recall)
        constraint_value: Value of the constraint (e.g., K=100, alpha=0.01, min_recall=0.7)
        utility_metric: Metric to optimize - 'precision' or 'recall'

    Returns:
        Utility metric value under the given constraint

    Examples:
        # Precision@K: What is precision in top 100 predictions?
        constrained_utility_metric(y_true, y_pred, 'K', 100, 'precision')

        # Recall at FPR≤0.01: What recall can we achieve with FPR at most 1%?
        constrained_utility_metric(y_true, y_pred, 'fpr', 0.01, 'recall')

        # Precision at Recall≥0.7: What precision can we achieve while maintaining 70% recall?
        constrained_utility_metric(y_true, y_pred, 'recall', 0.7, 'precision')
    """
    n_positives = y_true.sum()
    n_negatives = len(y_true) - n_positives

    if constraint_type == 'K':
        # Metric at top K predictions (alert budget constraint)
        K = min(int(constraint_value), len(y_true))
        if K == 0:
            return 0.0

        # Use argpartition for faster top-K selection
        top_k_indices = np.argpartition(y_pred_proba, -K)[-K:]

        if utility_metric == 'precision':
            # Precision@K: fraction of top K that are positive
            return y_true[top_k_indices].sum() / K
        elif utility_metric == 'recall':
            # Recall@K: fraction of positives in top K
            if n_positives == 0:
                return 0.0  # Recall undefined when no positives
            return y_true[top_k_indices].sum() / n_positives
        else:
            raise ValueError(f"Unknown utility_metric: {utility_metric}. Use 'precision' or 'recall'")

    elif constraint_type == 'fpr':
        # Metric at FPR <= constraint_value (regulatory/risk constraint)
        # Validate constraint_value is in [0, 1]
        if not 0 <= constraint_value <= 1:
            raise ValueError(f"FPR constraint_value must be in [0, 1], got {constraint_value}")

        # FPR is undefined when there are no negatives
        if n_negatives == 0:
            # All samples are positive - no false positives possible
            if utility_metric == 'recall':
                return 1.0 if n_positives > 0 else 0.0
            elif utility_metric == 'precision':
                return 1.0 if n_positives > 0 else 0.0
            else:
                raise ValueError(f"Unknown utility_metric: {utility_metric}. Use 'precision' or 'recall'")

        fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_pred_proba)
        valid_indices = np.where(fpr_vals <= constraint_value)[0]

        if len(valid_indices) == 0:
            return 0.0

        # Pick threshold where FPR is closest to constraint (use full "FPR budget")
        # This gives the best recall and is the natural operating point
        best_idx = valid_indices[np.argmax(fpr_vals[valid_indices])]

        tpr = tpr_vals[best_idx]
        fpr = fpr_vals[best_idx]

        if utility_metric == 'recall':
            return tpr
        elif utility_metric == 'precision':
            # Compute precision at this operating point
            tp = tpr * n_positives
            fp = fpr * n_negatives
            if tp + fp > 0:
                return tp / (tp + fp)
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown utility_metric: {utility_metric}. Use 'precision' or 'recall'")

    elif constraint_type == 'recall':
        # Metric at Recall >= constraint_value (coverage constraint)
        # Validate constraint_value is in [0, 1]
        if not 0 <= constraint_value <= 1:
            raise ValueError(f"Recall constraint_value must be in [0, 1], got {constraint_value}")

        if utility_metric == 'precision':
            # Precision at Recall >= target
            # Recall is undefined when there are no positives
            if n_positives == 0:
                return 0.0

            # Use highest threshold that achieves target recall (common operational choice)
            prec_vals, rec_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)

            # Thresholds align with prec_vals[:-1] / rec_vals[:-1]
            # Last point is threshold below min score (everyone positive)
            valid_indices = np.where(rec_vals[:-1] >= constraint_value)[0]

            if len(valid_indices) == 0:
                return 0.0

            # Use highest threshold (most selective) that meets recall requirement
            # This is the first valid index (precision_recall_curve returns decreasing thresholds)
            best_idx = valid_indices[0]
            return prec_vals[best_idx]
        elif utility_metric == 'recall':
            # Disallow: recall at recall >= target is trivial (returns constant)
            raise ValueError(
                f"Invalid combination: constraint_type='recall' with utility_metric='recall'. "
                f"This returns a constant ({constraint_value}) which makes optimization impossible. "
                f"Use utility_metric='precision' instead."
            )
        else:
            raise ValueError(f"Unknown utility_metric: {utility_metric}. Use 'precision' or 'recall'")

    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}. Use 'K', 'fpr', or 'recall'")
