import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms
from src.ml.models import losses
from src.ml.metrics import average_precision_score
from src.utils import dataloaders, decrease_lr, filter_args, graphdataset, set_random_seed, tensordatasets
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve, confusion_matrix
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler


class TorchGeometricClient():
    """
    PyTorchGeometric-specific client for training and evaluation. 
    Can run in isolation and federation.
    """
    def __init__(self, id: str, seed: int, device: str, trainset_nodes: str, trainset_edges: str, Model: Any, optimizer: str, criterion: str, trainset_size: float = None, valset_nodes: str = None, valset_edges: str = None, valset_size: float = 0.0, testset_nodes: str = None, testset_edges: str = None, testset_size: float = 0.0, **kwargs):
        self.id = id
        self.seed = seed
        self.device = device
        self.results = {}
        
        set_random_seed(self.seed)
        
        # Read and prepare node dataframes
        train_nodes_df = pd.read_parquet(trainset_nodes).drop(columns=['bank']).rename(columns={'account': 'node'})
        # Drop mask columns if present (for transductive learning)
        mask_cols = [col for col in train_nodes_df.columns if col.endswith('_mask')]
        if mask_cols:
            train_nodes_df = train_nodes_df.drop(columns=mask_cols)

        train_edges_df = pd.read_parquet(trainset_edges)

        val_nodes_df = None
        if valset_nodes is not None:
            val_nodes_df = pd.read_parquet(valset_nodes).drop(columns=['bank']).rename(columns={'account': 'node'})
            if mask_cols:
                val_nodes_df = val_nodes_df.drop(columns=mask_cols)
        val_edges_df = pd.read_parquet(valset_edges) if valset_edges is not None else None

        test_nodes_df = None
        if testset_nodes is not None:
            test_nodes_df = pd.read_parquet(testset_nodes).drop(columns=['bank']).rename(columns={'account': 'node'})
            if mask_cols:
                test_nodes_df = test_nodes_df.drop(columns=mask_cols)
        test_edges_df = pd.read_parquet(testset_edges) if testset_edges is not None else None
        
        self.trainset, self.valset, self.testset = graphdataset(train_nodes_df, train_edges_df, val_nodes_df, val_edges_df, test_nodes_df, test_edges_df, device=device)
        if trainset_size is not None:
            num_nodes = len(self.trainset.y)
            neg_indices = torch.where(self.trainset.y == 0)[0]
            pos_indices = torch.where(self.trainset.y == 1)[0]
            neg_indices = neg_indices[torch.randperm(len(neg_indices))]
            pos_indices = pos_indices[torch.randperm(len(pos_indices))]
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            neg_train_end = int(trainset_size * num_neg)
            pos_train_end = int(trainset_size * num_pos)
            neg_val_end = neg_train_end + int(valset_size * num_neg)
            pos_val_end = pos_train_end + int(valset_size * num_pos)
            neg_test_end = neg_val_end + int(testset_size * num_neg)
            pos_test_end = pos_val_end + int(testset_size * num_pos)
            neg_train_indices = neg_indices[:neg_train_end]
            pos_train_indices = pos_indices[:pos_train_end]
            neg_val_indices = neg_indices[neg_train_end:neg_val_end]
            pos_val_indices = pos_indices[pos_train_end:pos_val_end]
            neg_test_indices = neg_indices[neg_val_end:neg_test_end]
            pos_test_indices = pos_indices[pos_val_end:pos_test_end]
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[neg_train_indices] = True
            train_mask[pos_train_indices] = True
            val_mask[neg_val_indices] = True
            val_mask[pos_val_indices] = True
            test_mask[neg_test_indices] = True
            test_mask[pos_test_indices] = True
            self.trainset.train_mask = train_mask
            self.trainset.val_mask = val_mask
            self.trainset.test_mask = test_mask
        else:
            self.trainset = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=valset_size, num_test=testset_size)(self.trainset)

        # Auto-detect input_dim from actual data (after dropping mask columns, etc.)
        # This must happen BEFORE creating the model
        actual_input_dim = self.trainset.x.shape[1]
        if 'input_dim' in kwargs and kwargs['input_dim'] != actual_input_dim:
            print(f"Warning: Configured input_dim={kwargs['input_dim']} doesn't match actual data dimension={actual_input_dim}. Using actual dimension.")
        kwargs['input_dim'] = actual_input_dim

        self.model = Model(**filter_args(Model, kwargs)).to(self.device)
        Optimizer = getattr(torch.optim, optimizer)
        self.optimizer = Optimizer(self.model.parameters(), **filter_args(Optimizer, kwargs))
        
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = torch.bincount(self.trainset.y).tolist()
            self.criterion = criterions.ClassBalancedLoss(n_samples_per_classes=n_samples_per_classes, gamma=kwargs.get('gamma', 0.9999))
        elif criterion == 'DAMLoss':
            self.criterion = criterions.DAMLoss(class_counts=torch.bincount(self.trainset.y))
        elif criterion == 'BCEWithLogitsLoss':
            Criterion = getattr(torch.nn, criterion)
            class_counts = torch.bincount(self.trainset.y)
            weight = class_counts.max() / class_counts
            self.criterion = Criterion(pos_weight = weight[1], **filter_args(Criterion, kwargs))
        else:
            class_counts = torch.bincount(self.trainset.y)
            self.criterion = getattr(torch.nn, criterion)(weight = class_counts.max() / class_counts)
        
    
    def train(self):
        """Train the model on local dataset."""
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(self.trainset)
        loss = self.criterion(y_pred[self.trainset.train_mask], self.trainset.y[self.trainset.train_mask].to(torch.float32))
        loss.backward()
        self.optimizer.step()
    
    def evaluate(self, dataset: str = 'trainset') -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a given dataset.
        
        Args:
            dataset (str): Dataset name (trainset, valset, testset).
        
        Returns:
            Tuple[float, np.ndarray, np.ndarray]: Loss, predicted logits, ground truth labels.
        """
        if dataset == 'trainset':
            dataset = self.trainset
            mask = dataset.train_mask
        elif dataset == 'valset':
            if self.valset is not None:
                dataset = self.valset
                mask = torch.tensor([True] * len(dataset.y))
            else:
                dataset = self.trainset
                mask = dataset.val_mask
        elif dataset == 'testset':
            if self.testset is not None:
                dataset = self.testset
                mask = torch.tensor([True] * len(dataset.y))
            else:
                dataset = self.trainset
                mask = dataset.test_mask
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(dataset)
            loss = self.criterion(y_pred[mask], dataset.y[mask].to(torch.float32)).item()
        return loss, torch.sigmoid(y_pred[mask]).cpu().numpy(), dataset.y[mask].cpu().numpy()
    
    def run(self, n_rounds: int = 100, eval_every: int = 5, lr_patience: int = 10, es_patience: int = 20, n_warmup_rounds: int = 30, **kwargs) -> Dict:
        """
        Run training and evaluation loop.
        
        Args: 
            n_rounds (int): Number of rounds (aka epochs).
            eval_every (int): Number of rounds between evalualtions.
            lr_patience (int): Learing rate patience.
            es_patience (int): Early stopping patience.
        
        Returns:
            Dict: Results from traning, validation and testing.
        """
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        self.log(dataset='trainset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
        previous_train_loss = loss
        
        loss, y_pred, y_true = self.evaluate(dataset='valset')
        self.log(dataset='valset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
        previous_es_value = average_precision_score(y_true, y_pred, recall_span=(0.6, 1.0))
        
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            
            set_random_seed(self.seed+round)
            
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            self.log(dataset='trainset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
            
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            
            if round % eval_every == 0:
                loss, y_pred, y_true = self.evaluate(dataset='testset')
                self.log(dataset='testset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                self.log(dataset='valset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
                es_value = average_precision_score(y_true, y_pred, recall_span=(0.6, 1.0))
                if es_value <= previous_es_value and round > n_warmup_rounds:
                    es_patience -= eval_every
                # else:
                #     es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_es_value = es_value
        
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        self.log(dataset='trainset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
        loss, y_pred, y_true = self.evaluate(dataset='valset')
        self.log(dataset='valset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
        loss, y_pred, y_true = self.evaluate(dataset='testset')
        self.log(dataset='testset', y_pred=y_pred, y_true=y_true, round=round, loss=loss, metrics=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'precision_recall_curve', 'roc_curve'])
        
        return self.results
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Retrieve model parameters."""
        return self.model.get_parameters()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters."""
        self.model.set_parameters(parameters)

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Retrieve model gradients."""
        return self.model.get_gradients()

    def set_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Set model gradients."""
        self.model.set_gradients(gradients)

    def compute_gradients(self) -> Dict[str, torch.Tensor]:
        """Train and retrive gradients"""
        self.model.train()
        self.model.zero_grad()
        y_pred = self.model(self.trainset)
        loss = self.criterion(y_pred[self.trainset.train_mask], self.trainset.y[self.trainset.train_mask].to(torch.float32))
        loss.backward()
        gradients = self.get_gradients()
        return gradients
    
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

