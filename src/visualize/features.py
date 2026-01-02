import numpy as np
import matplotlib.pyplot as plt

try:
    import umap
except ImportError:
    raise ImportError(
        "umap-learn is required for feature visualization. "
        "Install with: pip install umap-learn"
    )

def plot_umap(X: np.ndarray, y: np.ndarray) -> plt.Figure:
    """
    Create UMAP projection visualization of features.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - 0 for normal, 1 for SAR

    Returns:
        Matplotlib figure with UMAP scatter plot
    """
    points = umap.UMAP().fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = np.where(y == 0, 'C0', 'C1')
    ax.scatter(x=points[:, 0], y=points[:, 1], c=colors, alpha=0.6, s=10)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP Projection of Transaction Features')
    ax.legend(['Normal', 'SAR'], loc='best')
    return fig


# Note: This module is now used by scripts/visualize_features.py
# For usage, run: python scripts/visualize_features.py --experiment <experiment_name>
