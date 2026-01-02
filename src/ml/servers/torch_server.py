import copy
import multiprocessing as mp
import numpy as np
import torch
from src.ml.metrics import average_precision_score
from src.utils import decrease_lr, filter_args, set_random_seed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from tqdm import tqdm
from typing import Any, Dict, List, Tuple


class TorchServer():
    def __init__(self, seed: int, device: str, Model: Any, optimizer: str, clients: List, n_workers: int = None, **kwargs):
        """Server for federated learning. Runs with TorchClients.

        Args:
            seed (int): Seed.
            device (str): Device. 
            Model (Any): Global model class.
            optimizer (str): Server optimizer
            clients (List): Clients. 
            n_workers (int, optional): Number of workers. Defaults to None.
        """
        self.seed = seed
        self.device = device
        self.n_workers = len(clients) if n_workers is None else n_workers
        self.results = {}
        
        self.global_model = Model(**filter_args(Model, kwargs)).to(self.device)
        Optimizer = getattr(torch.optim, optimizer)
        self.optimizer = Optimizer(self.global_model.parameters(), **filter_args(Optimizer, kwargs))
        
        self.clients = clients

    def compute_gradients(self, clients: List, seed: int) -> Tuple[List, List]:
        """Trains clients for one epoch and computes effective gradients.

        Args:
            clients (List): Clients to train. 
            seed (int): Seed.

        Returns:
            Tuple[List, List]: Lists of client ids and effective gradients. 
        """
        ids = []
        gradients = []
        for client in clients:
            set_random_seed(seed)
            ids.append(client.id)
            gradients.append(client.compute_gradients())
        return ids, gradients
    
    def train_clients(self, clients: List, seed: int):
        """Train clients.

        Args:
            clients (List): Clients to train.
            seed (int): Seed.
        """
        for client in clients:
            set_random_seed(seed)
            client.train()
    
    def evaluate_clients(self, clients: List, dataset: str) -> Tuple[List, List, List, List]:
        """Evaluates clients.

        Args:
            clients (List): Clients to evaluate.
            dataset (str): Dataset at clients to evaluate.

        Returns:
            Tuple[List, List, List, List]: Lists of client ids, losses, predicted logits, and ground truth labels.
        """
        ids = []
        losses = []
        y_preds = []
        y_trues = []
        for client in clients:
            loss, y_pred, y_true = client.evaluate(dataset=dataset)
            ids.append(client.id)
            losses.append(loss)
            y_preds.append(y_pred)
            y_trues.append(y_true)
        return ids, losses, y_preds, y_trues

    def aggregate_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Aggregate gradients by averaging."""
        avg_gradients = {}
        with torch.no_grad():
            for name in gradients[0]:
                avg_gradients[name] = torch.mean(torch.stack([grads[name] for grads in gradients]), dim=0)
        return avg_gradients
    
    def run(self, n_rounds: int = 100, eval_every: int = 5, lr_patience: int = 10, es_patience: int = 20, n_warmup_rounds: int = 50, **kwargs) -> Dict:
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

        with mp.Pool(self.n_workers) as p:
            
            client_splits = np.array_split(self.clients, self.n_workers)
            
            # sync state_dicts over clients
            global_parameters = self.global_model.get_parameters()
            for client in self.clients:
                client.set_parameters(copy.deepcopy(global_parameters))
            
            # evaluate initial model
            results = p.starmap(self.evaluate_clients, [(client_split, 'trainset') for client_split in client_splits])
            previous_train_loss = 0.0
            for result in results:
                for id, loss, y_pred, y_true in zip(result[0], result[1], result[2], result[3]):
                    self.log(id=id, dataset='trainset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
                    previous_train_loss += loss / len(self.clients)
            results = p.starmap(self.evaluate_clients, [(client_split, 'valset') for client_split in client_splits])
            previous_es_value = 0.0
            for result in results:
                for id, loss, y_pred, y_true in zip(result[0], result[1], result[2], result[3]):
                    self.log(id=id, dataset='valset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
                    previous_es_value += average_precision_score(y_true, y_pred, recall_span=(0.6, 1.0)) / len(self.clients)
            
            for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
                
                results = p.starmap(self.compute_gradients, [(client_split, self.seed+round) for client_split in client_splits])
                gradients = [grads for result in results for grads in result[1]]
                avg_gradients = self.aggregate_gradients(gradients)
                
                self.global_model.train()
                self.optimizer.zero_grad()
                self.global_model.set_gradients(avg_gradients)
                self.optimizer.step()
                
                global_state_dict = self.global_model.get_parameters()
                for client in self.clients:
                    client.set_parameters(global_state_dict)
                
                results = p.starmap(self.evaluate_clients, [(client_split, 'trainset') for client_split in client_splits])
                avg_loss = 0.0
                for result in results:
                    for id, loss, y_pred, y_true in zip(result[0], result[1], result[2], result[3]):
                        self.log(id=id, dataset='trainset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
                        avg_loss += loss / len(self.clients)
                
                if avg_loss >= previous_train_loss:
                    lr_patience -= 1
                else:
                    lr_patience = lr_patience_reset
                if lr_patience <= 0:
                    tqdm.write(f'Decreasing learning rate, round: {round}')
                    decrease_lr(self.optimizer, factor=0.5) # TODO: is this the correct way?
                    for client in self.clients:
                        decrease_lr(client.optimizer, factor=0.5)
                    lr_patience = lr_patience_reset
                previous_train_loss = avg_loss
                
                if round % eval_every == 0:
                    results = p.starmap(self.evaluate_clients, [(client_split, 'valset') for client_split in client_splits])
                    es_value = 0.0
                    for result in results:
                        for id, loss, y_pred, y_true in zip(result[0], result[1], result[2], result[3]):
                            self.log(id=id, dataset='valset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
                            es_value += average_precision_score(y_true, y_pred, recall_span=(0.6, 1.0)) / len(self.clients)
                    if es_value <= previous_es_value and round > n_warmup_rounds:
                        es_patience -= eval_every
                    # else:
                    #     es_patience = es_patience_reset
                    if es_patience <= 0:
                        tqdm.write(f'Early stopping, round: {round}')
                        break
                    previous_es_value = es_value
            
            results = p.starmap(self.evaluate_clients, [(client_split, 'trainset') for client_split in client_splits])
            for result in results:
                for id, loss, y_pred, y_true in zip(result[0], result[1], result[2], result[3]):
                    self.log(id=id, dataset='trainset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
            results = p.starmap(self.evaluate_clients, [(client_split, 'valset') for client_split in client_splits])
            for result in results:
                for id, loss, y_pred, y_true in zip(result[0], result[1], result[2], result[3]):
                    self.log(id=id, dataset='valset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
            results = p.starmap(self.evaluate_clients, [(client_split, 'testset') for client_split in client_splits])
            for result in results:
                for id, loss, y_pred, y_true in zip(result[0], result[1], result[2], result[3]):
                    self.log(id=id, dataset='testset', y_pred=y_pred, y_true=y_true, round=round, loss=loss, metrics=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'precision_recall_curve', 'roc_curve'])
        
        return self.results

    def log(self, id: str, dataset: str, y_pred: np.ndarray, y_true: np.ndarray, round: int = None, loss: float = None, metrics: List[str] = None):
        """
        Log training results for a given round.
        
        Args:
            id (str): Client ID.
            dataset (str): Dataset name.
            y_pred (np.ndarray): Model predictions.
            y_true (np.ndarray): Ground truth labels.
            round (int): Training round.
            loss (float): Loss value.
            metrics (list): List of metrics to calculate.
        """
        if metrics is None:
            metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall']

        if id not in self.results:
            self.results[id] = {} 
        
        if dataset not in self.results[id]:
            self.results[id][dataset] = {metric: [] for metric in metrics}
            self.results[id][dataset]['round'] = []
            self.results[id][dataset]['loss'] = []

        if round is not None:
            self.results[id][dataset]['round'].append(round)
        if loss is not None:
            self.results[id][dataset]['loss'].append(loss)

        for metric in metrics:
            if metric == 'accuracy':
                self.results[id][dataset]['accuracy'].append(accuracy_score(y_true, (y_pred > 0.5)))
            elif metric == 'average_precision':
                self.results[id][dataset]['average_precision'].append(average_precision_score(y_true, y_pred, recall_span=(0.6, 1.0)))
            elif metric == 'balanced_accuracy':
                self.results[id][dataset]['balanced_accuracy'].append(balanced_accuracy_score(y_true, (y_pred > 0.5)))
            elif metric == 'f1':
                self.results[id][dataset]['f1'].append(f1_score(y_true, (y_pred > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision':
                self.results[id][dataset]['precision'].append(precision_score(y_true, (y_pred > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'recall':
                self.results[id][dataset]['recall'].append(recall_score(y_true, (y_pred > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision_recall_curve':
                self.results[id][dataset]['precision_recall_curve'] = precision_recall_curve(y_true, y_pred)
            elif metric == 'roc_curve':
                self.results[id][dataset]['roc_curve'] = roc_curve(y_true, y_pred)
