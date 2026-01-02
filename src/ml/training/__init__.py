# Note: These modules have __main__ blocks and are meant to be run as scripts.
# To avoid import warnings when using 'python -m', run them directly:
#   python src/ml/training/centralized.py --config ...
# Or import the functions when needed programmatically:
#   from src.ml.training.centralized import centralized

__all__ = []

# Only import when explicitly requested to avoid triggering __main__ blocks
def __getattr__(name):
    if name == 'centralized':
        from src.ml.training.centralized import centralized
        return centralized
    elif name == 'federated':
        from src.ml.training.federated import federated
        return federated
    elif name == 'isolated':
        from src.ml.training.isolated import isolated
        return isolated
    elif name == 'HyperparamTuner':
        from src.ml.training.hyperparameter_tuning import HyperparamTuner
        return HyperparamTuner
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
