from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.ml.models.base import SklearnBaseModel


# Export sklearn models with their base class
# These are used directly without custom wrappers
__all__ = ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']
