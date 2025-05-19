import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def plot_n_sar_txs_over_tims():
    EXPERIMENT = '12_banks_homo_mid_dist_change'
    df = pd.read_csv(f'experiments/{EXPERIMENT}/data/raw/tx_log.csv')
    
    # neg_map = {0: 'single', 1: 'fan out', 2: 'fan in', 9: 'forward', 10: 'mutual', 11: 'periodical'}
    # pos_map = {1: 'fan out', 2: 'fan in', 3: 'cycle', 4: 'bipartite', 5: 'stack', 6: 'random', 7: 'scatter gather', 8: 'gather scatter'}
    
    n_steps = df['step'].max() + 1
    x = np.arange(n_steps)
    
    df_sar = df[df['isSAR'] == 1]
    
    df_sar_fan_in = df_sar[df_sar['modelType'] == 1]
    n_fan_in_per_step = df_sar_fan_in[['step', 'isSAR']].groupby('step').count()
    n_fan_in_per_step = n_fan_in_per_step.reindex(x, fill_value=0)
    n_fan_in_per_step = n_fan_in_per_step['isSAR'].to_numpy()
    
    df_sar_fan_out = df_sar[df_sar['modelType'] == 2]
    n_fan_out_per_step = df_sar_fan_out[['step', 'isSAR']].groupby('step').count()
    n_fan_out_per_step = n_fan_out_per_step.reindex(x, fill_value=0)
    n_fan_out_per_step = n_fan_out_per_step['isSAR'].to_numpy()
    
    df_sar_bipartite = df_sar[df_sar['modelType'] == 4]
    n_bipartite_per_step = df_sar_bipartite[['step', 'isSAR']].groupby('step').count()
    n_bipartite_per_step = n_bipartite_per_step.reindex(x, fill_value=0)
    n_bipartite_per_step = n_bipartite_per_step['isSAR'].to_numpy()
    
    df_sar_stack = df_sar[df_sar['modelType'] == 5]
    n_stack_per_step = df_sar_stack[['step', 'isSAR']].groupby('step').count()
    n_stack_per_step = n_stack_per_step.reindex(x, fill_value=0)
    n_stack_per_step = n_stack_per_step['isSAR'].to_numpy()
    
    df_sar_scatter_gather = df_sar[df_sar['modelType'] == 7]
    n_scatter_gather_per_step = df_sar_scatter_gather[['step', 'isSAR']].groupby('step').count()
    n_scatter_gather_per_step = n_scatter_gather_per_step.reindex(x, fill_value=0)
    n_scatter_gather_per_step = n_scatter_gather_per_step['isSAR'].to_numpy()
    
    df_sar_gather_scatter = df_sar[df_sar['modelType'] == 8]
    n_gather_scatter_per_step = df_sar_gather_scatter[['step', 'isSAR']].groupby('step').count()
    n_gather_scatter_per_step = n_gather_scatter_per_step.reindex(x, fill_value=0)
    n_gather_scatter_per_step = n_gather_scatter_per_step['isSAR'].to_numpy()
    
    plt.plot(x, n_fan_in_per_step, label='fan in')
    plt.fill_between(x, 0, n_fan_in_per_step, alpha=0.2)
    
    plt.plot(x, n_fan_in_per_step+n_fan_out_per_step, label='fan out')
    plt.fill_between(x, n_fan_in_per_step, n_fan_in_per_step+n_fan_out_per_step, alpha=0.2)
    
    plt.plot(x, n_fan_in_per_step+n_fan_out_per_step+n_bipartite_per_step, label='bipartite')
    plt.fill_between(x, n_fan_in_per_step+n_fan_out_per_step, n_fan_in_per_step+n_fan_out_per_step+n_bipartite_per_step, alpha=0.2)
    
    plt.plot(x, n_stack_per_step, label='stack')
    plt.fill_between(x, 0, n_stack_per_step, alpha=0.2)
    
    plt.plot(x, n_stack_per_step+n_scatter_gather_per_step, label='scatter gather')
    plt.fill_between(x, n_stack_per_step, n_stack_per_step+n_scatter_gather_per_step, alpha=0.2)
    
    plt.plot(x, n_stack_per_step+n_scatter_gather_per_step+n_gather_scatter_per_step, label='gather scatter')
    plt.fill_between(x, n_stack_per_step+n_scatter_gather_per_step, n_stack_per_step+n_scatter_gather_per_step+n_gather_scatter_per_step, alpha=0.2)
    
    plt.xlabel('step')
    plt.ylabel('number of sar txs')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.ylim((0, 50))
    plt.yticks(np.arange(0, 51, 10))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('difficult_dist_change_n_sar_txs_over_time.png')
    
    df = pd.DataFrame({
        'step': x, 
        'fan_in': n_fan_in_per_step,
        'fan_out': n_fan_out_per_step,
        'bipartite': n_bipartite_per_step,
        'stack': n_stack_per_step,
        'scatter_gather': n_scatter_gather_per_step,
        'gather_scatter': n_gather_scatter_per_step
    })
    df.to_csv('difficult_dist_change_n_sar_txs_over_time.csv', index=False)

def plot_n_active_sar_patterns_over_time():
    EXPERIMENT = '12_banks_homo_mid_no_dist_change'
    df = pd.read_csv(f'experiments/{EXPERIMENT}/data/raw/tx_log.csv')
    
    # neg_map = {0: 'single', 1: 'fan out', 2: 'fan in', 9: 'forward', 10: 'mutual', 11: 'periodical'}
    # pos_map = {1: 'fan out', 2: 'fan in', 3: 'cycle', 4: 'bipartite', 5: 'stack', 6: 'random', 7: 'scatter gather', 8: 'gather scatter'}
    
    fan_in_activity = np.zeros(113)
    fan_out_activity = np.zeros(113)
    bipartite_activity = np.zeros(113)
    stack_activity = np.zeros(113)
    scatter_gather_activity = np.zeros(113)
    gather_scatter_activity = np.zeros(113)
    
    df_sar = df[df['isSAR'] == 1]
    
    df_sar_fan_in = df_sar[df_sar['modelType'] == 1]
    gb = df_sar_fan_in.groupby('alertID')
    start_steps = gb['step'].min()
    end_steps = gb['step'].max()
    df_start_end = pd.merge(start_steps, end_steps, on='alertID')
    for row in df_start_end.itertuples():
        start = row.step_x
        end = row.step_y
        for i in range(start, end+1):
            fan_in_activity[i] += 1
    print('fan_in')
    print(fan_in_activity)
    print()
    
    df_sar_fan_out = df_sar[df_sar['modelType'] == 2]
    gb = df_sar_fan_out.groupby('alertID')
    start_steps = gb['step'].min()
    end_steps = gb['step'].max()
    df_start_end = pd.merge(start_steps, end_steps, on='alertID')
    for row in df_start_end.itertuples():
        start = row.step_x
        end = row.step_y
        for i in range(start, end+1):
            fan_out_activity[i] += 1
    print('fan_out')
    print(fan_out_activity)
    print()
    
    df_sar_bipartite = df_sar[df_sar['modelType'] == 4]
    gb = df_sar_bipartite.groupby('alertID')
    start_steps = gb['step'].min()
    end_steps = gb['step'].max()
    df_start_end = pd.merge(start_steps, end_steps, on='alertID')
    for row in df_start_end.itertuples():
        start = row.step_x
        end = row.step_y
        for i in range(start, end+1):
            bipartite_activity[i] += 1
    print('bipartite')
    print(bipartite_activity)
    print()
    
    # df_sar_stack = df_sar[df_sar['modelType'] == 5]
    # gb = df_sar_stack.groupby('alertID')
    # start_steps = gb['step'].min()
    # end_steps = gb['step'].max()
    # df_start_end = pd.merge(start_steps, end_steps, on='alertID')
    # for row in df_start_end.itertuples():
    #     start = row.step_x
    #     end = row.step_y
    #     for i in range(start, end+1):
    #         stack_activity[i] += 1
    # print('stack')
    # print(stack_activity)
    # print()
    
    # df_sar_scatter_gather = df_sar[df_sar['modelType'] == 7]
    # gb = df_sar_scatter_gather.groupby('alertID')
    # start_steps = gb['step'].min()
    # end_steps = gb['step'].max()
    # df_start_end = pd.merge(start_steps, end_steps, on='alertID')
    # for row in df_start_end.itertuples():
    #     start = row.step_x
    #     end = row.step_y
    #     for i in range(start, end+1):
    #         scatter_gather_activity[i] += 1
    # print('scatter_gather')
    # print(scatter_gather_activity)
    # print()
    
    # df_sar_gather_scatter = df_sar[df_sar['modelType'] == 8]
    # gb = df_sar_gather_scatter.groupby('alertID')
    # start_steps = gb['step'].min()
    # end_steps = gb['step'].max()
    # df_start_end = pd.merge(start_steps, end_steps, on='alertID')
    # for row in df_start_end.itertuples():
    #     start = row.step_x
    #     end = row.step_y
    #     for i in range(start, end+1):
    #         gather_scatter_activity[i] += 1
    # print('gather_scatter')
    # print(gather_scatter_activity)
    # print()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(113)
    
    plt.plot(x, fan_in_activity, label='fan in')
    plt.fill_between(x, 0, fan_in_activity, alpha=0.2)
    
    plt.plot(x, fan_in_activity+fan_out_activity, label='fan out')
    plt.fill_between(x, fan_in_activity, fan_in_activity+fan_out_activity, alpha=0.2)
    
    plt.plot(x, fan_in_activity+fan_out_activity+bipartite_activity, label='bipartite')
    plt.fill_between(x, fan_in_activity+fan_out_activity, fan_in_activity+fan_out_activity+bipartite_activity, alpha=0.2)
    
    # plt.plot(x, stack_activity, label='stack')
    # plt.fill_between(x, 0, stack_activity, alpha=0.2)
    # 
    # plt.plot(x, stack_activity+scatter_gather_activity, label='scatter gather')
    # plt.fill_between(x, stack_activity, stack_activity+scatter_gather_activity, alpha=0.2)
    # 
    # plt.plot(x, stack_activity+scatter_gather_activity+gather_scatter_activity, label='gather scatter')
    # plt.fill_between(x, stack_activity+scatter_gather_activity, stack_activity+scatter_gather_activity+gather_scatter_activity, alpha=0.2)
    
    plt.xlabel('step')
    plt.ylabel('number of sar patterns')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
    plt.ylim((0, 200))
    plt.yticks(np.arange(0, 201, 10))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('sar_pattern_activity_over_time.png')
    
    df = pd.DataFrame({
        'step': x,
        'fan_in': fan_in_activity,
        'fan_out': fan_out_activity,
        'bipartite': bipartite_activity,
        # 'stack': stack_activity,
        # 'scatter_gather': scatter_gather_activity,
        # 'gather_scatter': gather_scatter_activity
    })
    df.to_csv('sar_pattern_activity_over_time.csv', index=False)

def plot_overlap_results():
    EXPERIMENT = '12_banks_homo_mid_dist_change'
    model_types = ['LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'] # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    seeds = ['seed_42'] # , 'seed_43', 'seed_44', 'seed_45', 'seed_46', 'seed_47', 'seed_48', 'seed_49', 'seed_50', 'seed_51']
    datasets = ['trainset', 'valset', 'testset']
    metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']
    
    overlap_results = {
        model_type: {
            'trainset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
            'valset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
            'testset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []}
        } for model_type in model_types
    }
    for model_type in model_types:
        overlaps = ['0.982_overlap']
        for overlap in overlaps:
            avg_results = {
                'trainset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
                'valset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
                'testset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []}
            }
            for seed in seeds:
                file = f'experiments/{EXPERIMENT}/results/centralized/{model_type}/{overlap}/{seed}/results.pkl'
                with open(file, 'rb') as f:
                    results = pickle.load(f)
                for dataset in ['trainset', 'valset', 'testset']:
                    for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                        avg_results[dataset][metric].append(results['cen'][dataset][metric])
            for dataset in ['trainset', 'valset', 'testset']:
                for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                    avg_results[dataset][metric] = np.mean(avg_results[dataset][metric], axis=0)
            for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'loss']:
                fig, ax = plt.subplots()
                ax.plot(avg_results['trainset']['round'], avg_results['trainset'][metric], label='trainset')
                ax.plot(avg_results['valset']['round'], avg_results['valset'][metric], label='valset')
                ax.plot(avg_results['testset']['round'], avg_results['testset'][metric], label='testset')
                ax.set_xlabel('round')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid()
                os.makedirs(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{overlap}', exist_ok=True)
                fig.savefig(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{overlap}/{metric}.png')
                plt.close(fig)
                d = {'round': avg_results['trainset']['round'], metric: avg_results['trainset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{overlap}/{metric}_trainset.csv')
                d = {'round': avg_results['valset']['round'], metric: avg_results['valset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{overlap}/{metric}_valset.csv')
                d = {'round': avg_results['testset']['round'], metric: avg_results['testset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{overlap}/{metric}_testset.csv')
            for dataset in ['trainset', 'valset', 'testset']:
                for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                    overlap_results[model_type][dataset][metric].append(avg_results[dataset][metric][-1])
    for model_type in model_types:
        for dataset in datasets: 
            for metric in metrics:
                overlap_results[model_type][dataset][metric] = np.array(overlap_results[model_type][dataset][metric])
    x = np.array([0.982]) # , 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0])
    for dataset in datasets:
        for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'loss']:
            fig, ax = plt.subplots()
            d = {'overlap': x}
            for model_type in model_types:
                ax.plot(x, overlap_results[model_type][dataset][metric], label=model_type)
                d[model_type] = overlap_results[model_type][dataset][metric]
            ax.set_xlim(1.0, 0.0)
            ax.set_xlabel('overlap')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.legend()
            ax.grid()
            os.makedirs(f'experiments/{EXPERIMENT}/results/centralized/plots', exist_ok=True)
            fig.savefig(f'experiments/{EXPERIMENT}/results/centralized/plots/{dataset}_{metric}_overlap.png')
            plt.close(fig)
            df = pd.DataFrame(d)
            df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{dataset}_{metric}_overlap.csv')

def plot_fed_results():
    EXPERIMENT = '12_banks'
    model_types = ['LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE']
    seeds = ['seed_42'] # , 'seed_43', 'seed_44', 'seed_45', 'seed_46', 'seed_47', 'seed_48', 'seed_49', 'seed_50', 'seed_51']
    datasets = ['trainset', 'valset', 'testset']
    metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']
    
    overlap_results = {
        model_type: {
            'trainset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
            'valset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
            'testset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []}
        } for model_type in model_types
    }
    for model_type in model_types:
        n_clients = ['1_clients', '2_clients', '3_clients', '4_clients', '5_clients', '6_clients', '7_clients', '8_clients', '9_clients', '10_clients', '11_clients', '12_clients']
        for n in n_clients:
            avg_results = {
                'trainset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
                'valset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
                'testset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []}
            }
            for seed in seeds:
                file = f'experiments/{EXPERIMENT}/results/federated/{model_type}/{n}/{seed}/results.pkl'
                with open(file, 'rb') as f:
                    results = pickle.load(f)
                for dataset in ['trainset', 'valset', 'testset']:
                    for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                        clients = list(results.keys())
                        for client in clients:
                            avg_results[dataset][metric].append(results[client][dataset][metric])
            for dataset in ['trainset', 'valset', 'testset']:
                for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                    avg_results[dataset][metric] = np.mean(avg_results[dataset][metric], axis=0)
            for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'loss']:
                fig, ax = plt.subplots()
                ax.plot(avg_results['trainset']['round'], avg_results['trainset'][metric], label='trainset')
                ax.plot(avg_results['valset']['round'], avg_results['valset'][metric], label='valset')
                ax.plot(avg_results['testset']['round'], avg_results['testset'][metric], label='testset')
                ax.set_xlabel('round')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid()
                os.makedirs(f'experiments/{EXPERIMENT}/results/federated/plots/{model_type}/{n}', exist_ok=True)
                fig.savefig(f'experiments/{EXPERIMENT}/results/federated/plots/{model_type}/{n}/{metric}.png')
                plt.close(fig)
                d = {'round': avg_results['trainset']['round'], metric: avg_results['trainset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/federated/plots/{model_type}/{n}/{metric}_trainset.csv')
                d = {'round': avg_results['valset']['round'], metric: avg_results['valset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/federated/plots/{model_type}/{n}/{metric}_valset.csv')
                d = {'round': avg_results['testset']['round'], metric: avg_results['testset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/federated/plots/{model_type}/{n}/{metric}_testset.csv')
            for dataset in ['trainset', 'valset', 'testset']:
                for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                    overlap_results[model_type][dataset][metric].append(avg_results[dataset][metric][-1])
    for model_type in model_types:
        for dataset in datasets: 
            for metric in metrics:
                overlap_results[model_type][dataset][metric] = np.array(overlap_results[model_type][dataset][metric])
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    for dataset in datasets:
        for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'loss']:
            fig, ax = plt.subplots()
            d = {'n_clients': x}
            for model_type in model_types:
                ax.plot(x, overlap_results[model_type][dataset][metric], label=model_type)
                d[model_type] = overlap_results[model_type][dataset][metric]
            ax.set_xlabel('n clients')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.legend()
            ax.grid()
            os.makedirs(f'experiments/{EXPERIMENT}/results/federated/plots', exist_ok=True)
            fig.savefig(f'experiments/{EXPERIMENT}/results/federated/plots/{dataset}_{metric}_clients.png')
            plt.close(fig)
            df = pd.DataFrame(d)
            df.to_csv(f'experiments/{EXPERIMENT}/results/federated/plots/{dataset}_{metric}_clients.csv')

def plot_missing_label_results():
    EXPERIMENT = '12_banks'
    model_types = ['LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE']
    seeds = ['seed_42'] # , 'seed_43', 'seed_44', 'seed_45', 'seed_46', 'seed_47', 'seed_48', 'seed_49', 'seed_50', 'seed_51']
    datasets = ['trainset', 'valset', 'testset']
    metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']
    
    missing_labels_results = {
        model_type: {
            'trainset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
            'valset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
            'testset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []}
        } for model_type in model_types
    }
    for model_type in model_types:
        fractions = ['trainset_size_0.6', 'trainset_size_0.18', 'trainset_size_0.054', 'trainset_size_0.0162', 'trainset_size_0.00486', 'trainset_size_0.001458']
        for frac in fractions:
            avg_results = {
                'trainset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
                'valset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []},
                'testset': {'accuracy': [], 'average_precision': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'round': [], 'loss': []}
            }
            for seed in seeds:
                file = f'experiments/{EXPERIMENT}/results/centralized/{model_type}/{frac}/{seed}/results.pkl'
                with open(file, 'rb') as f:
                    results = pickle.load(f)
                for dataset in ['trainset', 'valset', 'testset']:
                    for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                        clients = list(results.keys())
                        for client in clients:
                            avg_results[dataset][metric].append(results[client][dataset][metric])
            for dataset in ['trainset', 'valset', 'testset']:
                for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                    avg_results[dataset][metric] = np.mean(avg_results[dataset][metric], axis=0)
            for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'loss']:
                fig, ax = plt.subplots()
                ax.plot(avg_results['trainset']['round'], avg_results['trainset'][metric], label='trainset')
                ax.plot(avg_results['valset']['round'], avg_results['valset'][metric], label='valset')
                ax.plot(avg_results['testset']['round'], avg_results['testset'][metric], label='testset')
                ax.set_xlabel('round')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid()
                os.makedirs(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{frac}', exist_ok=True)
                fig.savefig(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{frac}/{metric}.png')
                plt.close(fig)
                d = {'round': avg_results['trainset']['round'], metric: avg_results['trainset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{frac}/{metric}_trainset.csv')
                d = {'round': avg_results['valset']['round'], metric: avg_results['valset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{frac}/{metric}_valset.csv')
                d = {'round': avg_results['testset']['round'], metric: avg_results['testset'][metric]}
                df = pd.DataFrame(d)
                df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{model_type}/{frac}/{metric}_testset.csv')
            for dataset in ['trainset', 'valset', 'testset']:
                for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'round', 'loss']:
                    missing_labels_results[model_type][dataset][metric].append(avg_results[dataset][metric][-1])
    for model_type in model_types:
        for dataset in datasets: 
            for metric in metrics:
                missing_labels_results[model_type][dataset][metric] = np.array(missing_labels_results[model_type][dataset][metric])
    x = np.array([0.6, 0.18, 0.054, 0.0162, 0.00486, 0.001458])
    for dataset in datasets:
        for metric in ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'loss']:
            fig, ax = plt.subplots()
            d = {'n_clients': x}
            for model_type in model_types:
                ax.plot(x, missing_labels_results[model_type][dataset][metric], label=model_type)
                d[model_type] = missing_labels_results[model_type][dataset][metric]
            ax.set_xlim(0.7, 0.0)
            ax.set_xlabel('trainset fraction')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.legend()
            ax.grid()
            os.makedirs(f'experiments/{EXPERIMENT}/results/centralized/plots', exist_ok=True)
            fig.savefig(f'experiments/{EXPERIMENT}/results/centralized/plots/{dataset}_{metric}_missing_labels.png')
            plt.close(fig)
            df = pd.DataFrame(d)
            df.to_csv(f'experiments/{EXPERIMENT}/results/centralized/plots/{dataset}_{metric}_missing_labels.csv')

if __name__ == '__main__':
    plot_n_sar_txs_over_tims()
