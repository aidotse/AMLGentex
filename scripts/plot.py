import argparse
import pickle
from flib.vizualize import plot_metrics

def main():
    
    EXPERIMENT = '12_banks_homo_easy'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', nargs='+', help='Clients to include. If omited all clients will be included.', default=None)
    parser.add_argument('--datasets', nargs='+', help='Datasets to include. If omited all datasets will be included.', default=['trainset', 'valset', 'testset'])
    parser.add_argument('--metrics', nargs='+', help='Metrics to plot. Can be "loss", "accuracy", "balanced_accuracy", "recall", "precision", "average_precision", "f1", "roc_curve", "precision_recall_curve".', default=['loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'average_precision', 'f1', 'roc_curve', 'precision_recall_curve'])
    parser.add_argument('--reduction', help='Type of reduction if several clients. If "none", all clinets will be plotted individully.', default='mean')
    parser.add_argument('--results_files', type=str, help='Path to results file.', default=[
        # f'experiments/{EXPERIMENT}/results/centralized/DecisionTreeClassifier/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/DecisionTreeClassifier/results.pkl',
        # f'experiments/{EXPERIMENT}/results/centralized/RandomForestClassifier/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/RandomForestClassifier/results.pkl',
        # f'experiments/{EXPERIMENT}/results/centralized/GradientBoostingClassifier/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/GradientBoostingClassifier/results.pkl',
        # f'experiments/{EXPERIMENT}/results/centralized/LogisticRegressor/results.pkl',
        # f'experiments/{EXPERIMENT}/results/federated/LogisticRegressor/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/LogisticRegressor/results.pkl',
        # f'experiments/{EXPERIMENT}/results/centralized/MLP/results.pkl',
        # f'experiments/{EXPERIMENT}/results/federated/MLP/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/MLP/results.pkl',
        f'experiments/{EXPERIMENT}/results_incomplet_labels/centralized/GCN/results.pkl',
        # f'experiments/{EXPERIMENT}/results/federated/GCN/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/GCN/results.pkl',
        # f'experiments/{EXPERIMENT}/results/centralized/GAT/results.pkl',
        # f'experiments/{EXPERIMENT}/results/federated/GAT/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/GAT/results.pkl',
        # f'experiments/{EXPERIMENT}/results/centralized/GraphSAGE/results.pkl',
        # f'experiments/{EXPERIMENT}/results/federated/GraphSAGE/results.pkl',
        # f'experiments/{EXPERIMENT}/results/isolated/GraphSAGE/results.pkl',
    ])
    args = parser.parse_args()
    
    for results_file in args.results_files:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        output_dir = results_file.replace('results.pkl', '')
        plot_metrics(data, output_dir, args.metrics, args.clients, args.datasets, args.reduction)

if __name__ == '__main__':
    main()