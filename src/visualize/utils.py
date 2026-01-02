"""
Visualization utilities with auto-discovery of results files.
"""
from pathlib import Path
from typing import List, Dict, Optional
import pickle


def discover_results(
    experiment_root: str,
    settings: Optional[List[str]] = None,
    models: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Auto-discover results files from experiment directory structure.

    Convention:
    - Results at: {experiment_root}/results/{setting}/{model}/results.pkl

    Args:
        experiment_root: Path to experiment root directory
        settings: List of settings to include (centralized, federated, isolated).
                 If None, discovers all available settings.
        models: List of models to include. If None, discovers all available models.

    Returns:
        Dictionary mapping "{setting}/{model}" to results file path
    """
    experiment_root = Path(experiment_root)
    results_dir = experiment_root / "results"

    if not results_dir.exists():
        return {}

    results_files = {}

    # Auto-discover available settings if not specified
    if settings is None:
        settings = [
            d.name for d in results_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

    for setting in settings:
        setting_dir = results_dir / setting
        if not setting_dir.exists():
            continue

        # Auto-discover available models if not specified
        available_models = [
            d.name for d in setting_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

        # Filter by requested models if specified
        if models is not None:
            available_models = [m for m in available_models if m in models]

        for model in available_models:
            results_file = setting_dir / model / "results.pkl"
            if results_file.exists():
                key = f"{setting}/{model}"
                results_files[key] = results_file

    return results_files


def load_results(results_file: Path) -> Dict:
    """
    Load results from pickle file.

    Args:
        results_file: Path to results.pkl file

    Returns:
        Results dictionary
    """
    with open(results_file, 'rb') as f:
        return pickle.load(f)


def get_experiment_root_from_results(results_path: str) -> Path:
    """
    Extract experiment root from results file path.

    Args:
        results_path: Path to results file (e.g., experiments/10k_accounts/results/centralized/GCN/results.pkl)

    Returns:
        Experiment root path
    """
    results_path = Path(results_path)

    # Navigate up from results file: results.pkl -> model -> setting -> results -> experiment
    return results_path.parent.parent.parent.parent
