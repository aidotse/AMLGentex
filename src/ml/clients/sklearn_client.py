import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms
from src.ml.models import losses
from src.ml.metrics import average_precision_score, constrained_utility_metric
from src.utils import dataloaders, decrease_lr, filter_args, graphdataset, set_random_seed, tensordatasets
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve, confusion_matrix
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler


class SklearnClient():
    def __init__(self, id: str, seed: int, trainset: str, Model: Any, trainset_size: float = None, valset_size: float = None, testset_size: float  = None, valset: str = None, testset: str = None, **kwargs):
        self.id = id
        self.seed = seed
        self.results = {}
        
        set_random_seed(self.seed)

        train_df = pd.read_parquet(trainset).drop(columns=['account', 'bank'])
        # Drop mask columns if present (for transductive learning)
        mask_cols = [col for col in train_df.columns if col.endswith('_mask')]
        if mask_cols:
            train_df = train_df.drop(columns=mask_cols)

        n = len(train_df)
        if valset is not None:
            val_df = pd.read_parquet(valset).drop(columns=['account', 'bank'])
            if mask_cols:
                val_df = val_df.drop(columns=mask_cols)
        else:
            val_df = train_df.sample(n = int(n * valset_size), random_state=seed)
            train_df = train_df.drop(val_df.index)
        if testset is not None:
            test_df = pd.read_parquet(testset).drop(columns=['account', 'bank'])
            if mask_cols:
                test_df = test_df.drop(columns=mask_cols)
        else:
            test_df = train_df.sample(n = int(n * testset_size), random_state=seed)
            train_df = train_df.drop(test_df.index)
        if trainset_size is not None:
            train_df = train_df.sample(n = int(n * trainset_size), random_state=seed)
            
        self.X_train = train_df.drop(columns=['is_sar']).to_numpy()
        self.y_train = train_df['is_sar'].to_numpy()
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = val_df.drop(columns=['is_sar']).to_numpy()
        self.X_val = scaler.transform(self.X_val)
        self.y_val = val_df['is_sar'].to_numpy()
        self.X_test = test_df.drop(columns=['is_sar']).to_numpy()
        self.X_test = scaler.transform(self.X_test)
        self.y_test = test_df['is_sar'].to_numpy()
        
        self.model = Model(**filter_args(Model, kwargs))
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        return
    
    def evaluate(self, dataset='trainset'):
        dataset_mapping = {
            'trainset': (self.X_train, self.y_train),
            'valset': (self.X_val, self.y_val),
            'testset': (self.X_test, self.y_test)
        }
        x, y = dataset_mapping.get(dataset, (self.X_train, self.y_train))
        y_pred = self.model.predict_proba(x)
        return 0.0, y_pred[:,1], y
    
    def run(self, **kwargs):
        self.train()
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        self.log(dataset='trainset', round=0, loss=loss, y_pred=y_pred, y_true=y_true, metrics=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'precision_recall_curve', 'roc_curve'])
        loss, y_pred, y_true = self.evaluate(dataset='valset')
        self.log(dataset='valset', round=0, loss=loss, y_pred=y_pred, y_true=y_true, metrics=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'precision_recall_curve', 'roc_curve'])

        # Save constrained utility metric on testset for data tuning
        if kwargs.get('save_utility_metric', False) is True:
            constraint_type = kwargs.get('constraint_type', 'K')
            constraint_value = kwargs.get('constraint_value', 100)
            utility_metric = kwargs.get('utility_metric', 'precision')

            y_pred_proba = self.model.predict_proba(self.X_test)[:,1]
            utility_value = constrained_utility_metric(
                self.y_test, y_pred_proba,
                constraint_type, constraint_value, utility_metric
            )
            self.results['utility_metric'] = utility_value

        loss, y_pred, y_true = self.evaluate(dataset='testset')
        self.log(dataset='testset', round=0, loss=loss, y_pred=y_pred, y_true=y_true, metrics=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'precision_recall_curve', 'roc_curve'])

        if kwargs.get('save_feature_importances_error', False) is True:
            importances = self.model.feature_importances_
            error = abs(importances.mean() - importances).sum()
            self.results['feature_importances_error'] = error
        
        return self.results
    
    def log(self, dataset: str, y_pred: np.ndarray, y_true: np.ndarray, round: int = None, loss: float = None, metrics: List[str] = None):
        """
        Log training results for a given round.
        
        Args:
            dataset (str): Dataset name.
            y_pred (np.ndarray): Model predictions.
            y_true (np.ndarray): Ground truth labels.
            round (int): Training round.
            loss (float): Loss value.
            metrics (list): List of metrics to calculate.
        """
        if metrics is None:
            metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall']

        if dataset not in self.results:
            self.results[dataset] = {metric: [] for metric in metrics}
            self.results[dataset]['round'] = []
            self.results[dataset]['loss'] = []

        if round is not None:
            self.results[dataset]['round'].append(round)
        if loss is not None:
            self.results[dataset]['loss'].append(loss)

        for metric in metrics:
            if metric == 'accuracy':
                self.results[dataset]['accuracy'].append(accuracy_score(y_true, (y_pred > 0.5)))
            elif metric == 'average_precision':
                self.results[dataset]['average_precision'].append(average_precision_score(y_true, y_pred, recall_span=(0.6, 1.0)))
            elif metric == 'balanced_accuracy':
                self.results[dataset]['balanced_accuracy'].append(balanced_accuracy_score(y_true, (y_pred > 0.5)))
            elif metric == 'f1':
                self.results[dataset]['f1'].append(f1_score(y_true, (y_pred > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision':
                self.results[dataset]['precision'].append(precision_score(y_true, (y_pred > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'recall':
                self.results[dataset]['recall'].append(recall_score(y_true, (y_pred > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision_recall_curve':
                self.results[dataset]['precision_recall_curve'] = precision_recall_curve(y_true, y_pred)
            elif metric == 'roc_curve':
                self.results[dataset]['roc_curve'] = roc_curve(y_true, y_pred)