import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Union

def accuracy(cm: np.ndarray, threshold: float) -> float:
    idx = np.argwhere(cm[:,4] > threshold)[0]
    tp = cm[idx, 0]
    fp = cm[idx, 1]
    fn = cm[idx, 2]
    tn = cm[idx, 3]
    return (tp+tn)/(tp+fp+tn+fn)

def balanced_accuracy(cm: np.ndarray, threshold: float) -> float:
    idx = np.argwhere(cm[:,4] > threshold)[0]
    tp = cm[idx, 0]
    fp = cm[idx, 1]
    fn = cm[idx, 2]
    tn = cm[idx, 3]
    return 0.5*(tp/(tp+fn) + tn/(tn+fp))

def recall(cm: np.ndarray, threshold: float) -> float:
    idx = np.argwhere(cm[:,4] > threshold)[0]
    tp = cm[idx, 0]
    fp = cm[idx, 1]
    fn = cm[idx, 2]
    if tp+fp == 0.0:
        return 0.0
    return tp/(tp+fn)

def precision(cm: np.ndarray, threshold: float) -> float:
    idx = np.argwhere(cm[:,4] > threshold)[0]
    tp = cm[idx, 0]
    fp = cm[idx, 1]
    if tp+fp == 0.0:
        return 1.0
    return tp/(tp+fp)

def f1(cm: np.ndarray, threshold: float) -> float:
    idx = np.argwhere(cm[:,4] > threshold)[0]
    tp = cm[idx, 0]
    fp = cm[idx, 1]
    fn = cm[idx, 2]
    return 2*tp/(2*tp+fp+fn)
    
def roc_curve(cm: np.ndarray) -> np.ndarray:
    tp = cm[:, 0]
    fp = cm[:, 1]
    fn = cm[:, 2]
    tn = cm[:, 3]
    tpr = np.divide(tp, tp + fn, where=(tp + fn) > 0, out=np.zeros_like(tp, dtype=float))
    fpr = np.divide(fp, fp + tn, where=(fp + tn) > 0, out=np.zeros_like(fp, dtype=float))
    return tpr, fpr

def precision_recall_curve(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tp = cm[:, 0]
    fp = cm[:, 1]
    fn = cm[:, 2]
    precision = np.divide(tp, tp + fp, where=(tp + fp) > 0, out=np.zeros_like(tp, dtype=float))
    recall = np.divide(tp, tp + fn, where=(tp + fn) > 0, out=np.ones_like(tp, dtype=float))
    sorted_indices = np.argsort(recall)
    precision = precision[sorted_indices]
    recall = recall[sorted_indices]
    return precision, recall

def average_precision(cm: np.ndarray, recall_span: Tuple[int, int]=(0.0, 1.0)) -> float:
    tp = cm[:, 0]
    fp = cm[:, 1]
    fn = cm[:, 2]
    precision_values = np.divide(tp, tp + fp, where=(tp + fp) > 0, out=np.zeros_like(tp, dtype=float))
    recall_values = np.divide(tp, tp + fn, where=(tp + fn) > 0, out=np.ones_like(tp, dtype=float))
    sorted_indices = np.argsort(recall_values)
    precision_values = precision_values[sorted_indices]
    recall_values = recall_values[sorted_indices]
    recall_diffs = np.diff(recall_values)
    precision_areas = precision_values[:-1] * recall_diffs
    idxs = (recall_values >= recall_span[0]) & (recall_values <= recall_span[1])
    area = np.sum(precision_areas[idxs[1:]])
    span = np.sum(recall_diffs[idxs[1:]])
    return area / span if span > 0 else 0

def plot_metrics(data: dict, dir: str, metrics: List[str], clients: List[str], datasets: List[str], reduction: str = 'none', formats: List[str] = ['png', 'csv']):
    os.makedirs(dir, exist_ok=True)
    if clients is None:
        clients = [client for client in data]
    if datasets is None:
        datasets = [dataset for dataset in data[clients[0]]]
    if reduction == "mean":
        reduced_data = {}
        for dataset in datasets:
            rounds = [0]
            for client in clients:
                if max(rounds) < max(data[client][dataset]["round"]):
                    rounds = data[client][dataset]["round"]
            reduced_data[dataset] = {"round": np.array(rounds)}
            for metric in metrics:
                if metric == "roc_curve" or metric == "precision_recall_curve":
                    continue
                sums = np.zeros(len(rounds))
                for client in clients:
                    values = np.array(data[client][dataset][metric])
                    if len(values) < len(rounds):
                        diff = len(rounds) - len(values)
                        extension = np.array([values[-1]] * diff)
                        values = np.append(values, extension)
                    sums = sums + values
                means = sums / len(clients)
                reduced_data[dataset][metric] = means
        if "png" in formats:
            os.makedirs(os.path.join(dir, 'png'), exist_ok=True)
            for metric in metrics:
                fig = plt.figure()
                for dataset in datasets:
                    if metric == "precision_recall_curve":
                        for client in clients:
                            y, x, _ = data[client][dataset][metric]
                            plt.step(x[::-1], y[::-1], where='pre', label=f'{client}: {dataset}')
                            plt.xlabel('recall')
                            plt.ylabel('precision')
                            plt.title('precision-recall curve')
                    elif metric == "roc_curve":
                        for client in clients:
                            x, y, _ = data[client][dataset][metric]
                            plt.step(x[::-1], y[::-1], where='pre', label=f'{client}: {dataset}')
                            plt.plot([0, 1], [0, 1], linestyle='--', color='k')
                            plt.xlabel('fpr')
                            plt.ylabel('tpr')
                            plt.title('roc curve')
                    elif dataset == "testset":
                        plt.plot(reduced_data[dataset]["round"][-1], reduced_data[dataset][metric][-1], "-o", label=dataset)
                        plt.xlabel("round")
                        plt.ylabel(metric)
                    else:
                        plt.plot(reduced_data[dataset]["round"], reduced_data[dataset][metric], "-o", label=dataset)
                        plt.xlabel("round")
                        plt.ylabel(metric)
                if metric not in ['precision_recall_curve', 'roc_curve']:
                    plt.legend()
                if metric in ['average_precision', 'f1', 'precision', 'precision_recall_curve']:
                    plt.yscale('log')
                # elif metric == 'loss':
                    # plt.ylim(0.0, 8.0)
                plt.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "png", f"{metric}.png"))
                plt.close()
        if "csv" in formats:
            os.makedirs(os.path.join(dir, 'csv'), exist_ok=True)
            for metric in metrics:
                for dataset in datasets:
                    if metric == 'precision_recall_curve':
                        for client in clients:
                            np.savetxt(os.path.join(dir, 'csv', f'{client}_{metric}_{dataset}.csv'), np.column_stack(data[client][dataset][metric][:2]), delimiter=',', header='precision,recall', comments="", fmt="%.6f")
                    elif metric == 'roc_curve':
                        for client in clients:
                            np.savetxt(os.path.join(dir, 'csv', f'{client}_{metric}_{dataset}.csv'), np.column_stack(data[client][dataset][metric][:2]), delimiter=',', header='fpr,tpr', comments="", fmt="%.6f")
                    else:
                        np.savetxt(os.path.join(dir, 'csv', f'{metric}_{dataset}.csv'), np.column_stack((reduced_data[dataset]['round'], reduced_data[dataset][metric])), delimiter=',', header=f'round,{metric}', comments="", fmt=("%d", "%.6f"))
    else:
        if "png" in formats:
            os.makedirs(os.path.join(dir, 'png'), exist_ok=True)
            for metric in metrics:
                fig = plt.figure()
                for dataset in datasets:
                    for client in clients:
                        if metric == "precision_recall_curve":
                            y, x, _ = data[client][dataset][metric]
                            plt.step(x[::-1], y[::-1], where='pre', label=f'{client}: {dataset}')
                            plt.xlabel('recall')
                            plt.ylabel('precision')
                            plt.title('precision-recall curve')
                        elif metric == "roc_curve":
                            x, y, _ = data[client][dataset][metric]
                            plt.step(x[::-1], y[::-1], where='pre', label=f'{client}: {dataset}')
                            plt.plot([0, 1], [0, 1], linestyle='--', color='k')
                            plt.xlabel('fpr')
                            plt.ylabel('tpr')
                            plt.title('roc curve')
                        else:
                            x = data[client][dataset]["round"]
                            y = data[client][dataset][metric]
                            plt.plot(x, y, '-o', label=f'{client}: {dataset}')
                            plt.xlabel("round")
                            plt.ylabel(metric)
                # plt.legend()
                if metric in ['average_precision', 'f1', 'precision']:
                    plt.yscale('log')
                # elif metric == 'loss':
                    # plt.ylim(0.0, 6.0)
                plt.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "png", f"{metric}.png"))
                plt.close()
        if 'csv' in formats:
            os.makedirs(os.path.join(dir, 'csv'), exist_ok=True)
            for metric in metrics:
                for dataset in datasets:
                    for client in clients:
                        if metric == 'precision_recall_curve':
                            np.savetxt(os.path.join(dir, 'csv', f'{client}_{metric}_{dataset}.csv'), np.column_stack(data[client][dataset][metric][:2]), delimiter=',', header='precision,recall', comments="", fmt="%.6f")
                        elif metric == 'roc_curve':
                            np.savetxt(os.path.join(dir, 'csv', f'{client}_{metric}_{dataset}.csv'), np.column_stack(data[client][dataset][metric][:2]), delimiter=',', header='fpr,tpr', comments="", fmt="%.6f")
                        else:
                            np.savetxt(os.path.join(dir, 'csv', f'{client}_{metric}_{dataset}.csv'), np.column_stack((data[client][dataset]['round'], data[client][dataset][metric])), delimiter=',', header=f'round,{metric}', comments="", fmt=("%d", "%.6f"))