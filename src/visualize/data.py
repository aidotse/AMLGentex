import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def edge_label_hist(df:pd.DataFrame, banks:list, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    fig, ax = plt.subplots()
    width = 0.30
    x = np.array([-1*width/2, width/2])
    csv = {'bank': [], 'neg_count': [], 'pos_count': [], 'neg_ratio': [], 'pos_ratio': []}
    for i, bank in enumerate(banks):
        df_bank = df if bank == 'all' else df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        n_pos = df_bank[df_bank['isSAR'] == 1].shape[0]
        n_neg = df_bank[df_bank['isSAR'] == 0].shape[0]
        rects = ax.bar(x+i, np.array([n_neg, n_pos]), width, color=['C0', 'C1'], label=['neg', 'pos'], zorder=3)
        ax.bar_label(rects, [f'{n_neg/(n_neg+n_pos):.4f}', f'{n_pos/(n_neg+n_pos):.4f}'], padding=1)
        csv['bank'].append(bank)
        csv['neg_count'].append(n_neg)
        csv['pos_count'].append(n_pos)
        csv['neg_ratio'].append(n_neg/(n_neg+n_pos))
        csv['pos_ratio'].append(n_pos/(n_neg+n_pos))
    ax.grid(axis='y', zorder=0)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(banks)), banks)
    ax.legend(rects, ['neg', 'pos'], ncols=2)
    plt.tight_layout(pad=2.0)
    plt.title('Edge label distribution')
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)
    


def node_label_hist(df:pd.DataFrame, banks:list, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    fig, ax = plt.subplots()
    width = 0.30
    x = np.array([-1*width/2, width/2])
    csv = {'bank': [], 'neg_count': [], 'pos_count': [], 'neg_ratio': [], 'pos_ratio': []}
    for i, bank in enumerate(banks):
        df_bank = df if bank == 'all' else df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        df_bank = pd.concat([df_bank[['nameOrig', 'bankOrig', 'isSAR']].rename(columns={'nameOrig': 'name', 'bankOrig': 'bank'}), df_bank[['nameDest', 'bankDest', 'isSAR']].rename(columns={'nameDest': 'name', 'bankDest': 'bank'})])
        if bank != 'all':
            df_bank = df_bank[df_bank['bank'] == bank]
        gb = df_bank.groupby('name')
        accts = gb['isSAR'].max()
        n_pos = accts[accts == 1].shape[0]
        n_neg = accts[accts == 0].shape[0]
        rects = ax.bar(x+i, np.array([n_neg, n_pos]), width, color=['C0', 'C1'], label=['neg', 'pos'], zorder=3)
        ax.bar_label(rects, [f'{n_neg/(n_neg+n_pos):.4f}', f'{n_pos/(n_neg+n_pos):.4f}'], padding=1)
        csv['bank'].append(bank)
        csv['neg_count'].append(n_neg)
        csv['pos_count'].append(n_pos)
        csv['neg_ratio'].append(n_neg/(n_neg+n_pos))
        csv['pos_ratio'].append(n_pos/(n_neg+n_pos))
    ax.grid(axis='y', zorder=0)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(banks)), banks)
    ax.legend(rects, ['neg', 'pos'], ncols=2)
    plt.tight_layout(pad=2.0)
    plt.title('Node label distribution')
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def balance_curves(df:pd.DataFrame, file:str, bank=None):
    x = np.arange(df['step'].min(), df['step'].max()+1)
    accts = pd.concat([df[['nameOrig', 'bankOrig', 'isSAR']].rename(columns={'nameOrig': 'name', 'bankOrig': 'bank'}), df[['nameDest', 'bankDest', 'isSAR']].rename(columns={'nameDest': 'name', 'bankDest': 'bank'})])
    if bank is not None:
        accts = accts[accts['bank'] == bank]
    gb = accts.groupby('name')
    accts = gb['isSAR'].max()
    if -1 in accts.index or -2 in accts.index:
        accts = accts.drop([-1, -2])
    neg_accts = accts[accts == 0].sample(3).index.tolist()
    pos_accts = accts[accts == 1].sample(3).index.tolist()
    fig, ax = plt.subplots()
    csv = {'step': x}
    for acct in neg_accts:
        in_txs = df[df['nameDest'] == acct][['step', 'newbalanceDest']].rename(columns={'newbalanceDest': 'balance'})
        out_txs = df[df['nameOrig'] == acct][['step', 'newbalanceOrig']].rename(columns={'newbalanceOrig': 'balance'})
        txs = pd.concat([in_txs, out_txs])
        txs = txs.sort_values('step')
        y = np.interp(x, txs['step'], txs['balance'])
        ax.step(x, y, where='pre', color='C0', label=acct)
        csv[f'{acct}'] = y
    for acct in pos_accts:
        if acct == 9633 or acct == 68717 or acct == 32407:
            print(acct)
        in_txs = df[df['nameDest'] == acct][['step', 'newbalanceDest']].rename(columns={'newbalanceDest': 'balance'})
        out_txs = df[df['nameOrig'] == acct][['step', 'newbalanceOrig']].rename(columns={'newbalanceOrig': 'balance'})
        txs = pd.concat([in_txs, out_txs])
        txs = txs.sort_values('step')
        y = np.interp(x, txs['step'], txs['balance'])
        ax.step(x, y, where='pre', color='C1', label=acct)
        csv[f'{acct}(sar)'] = y
    ax.legend()
    ax.grid()
    ax.set_xlabel('step')
    ax.set_ylabel('balance')
    plt.title('Balance curves')
    plt.tight_layout(pad=2.0)
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def pattern_hist(df:pd.DataFrame, file:str):
    # OBS! This doesnt show the dist of patterns, it shows the dist of edges belonging to a certain pattern
    # TODO: Change this to show the dist of patterns. Unclear how...
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    neg_pattern_count = df[df['isSAR'] == 0]['modelType'].value_counts()
    pos_pattern_count = df[df['isSAR'] == 1]['modelType'].value_counts()
    neg_map = {0: 'single', 1: 'fan out', 2: 'fan in', 9: 'forward', 10: 'mutual', 11: 'periodical'}
    pos_map = {1: 'fan out', 2: 'fan in', 3: 'cycle', 4: 'bipartite', 5: 'stack', 6: 'random', 7: 'scatter gather', 8: 'gather scatter'}
    fig, ax = plt.subplots()
    width = 0.30
    x = 0    
    neg_names = []
    csv = {'pattern': [], 'count': []}
    for pattern, count in neg_pattern_count.items():
        neg_names.append(neg_map[pattern])
        rect = ax.bar(x, count, width, color='C0', zorder=3)
        ax.bar_label(rect, [f'{count}'], padding=1)
        csv['pattern'].append(neg_map[pattern])
        csv['count'].append(count)
        x += 1
    pos_names = []
    for pattern, count in pos_pattern_count.items():
        pos_names.append(pos_map[pattern])
        rect = ax.bar(x, count, width, color='C1', zorder=3)
        ax.bar_label(rect, [f'{count}'], padding=1)
        x += 1
        csv['pattern'].append(f'{pos_map[pattern]} (sar)')
        csv['count'].append(count)
    ax.grid(axis='y', zorder=0)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(neg_names)+len(pos_names)), neg_names+pos_names)
    neg_proxy = plt.Rectangle((0, 0), 1, 1, fc="C0")  # Blue box for 'neg'
    pos_proxy = plt.Rectangle((0, 0), 1, 1, fc="C1")  # Orange box for 'pos'
    ax.legend([neg_proxy, pos_proxy], ['neg', 'pos'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=2.0)
    plt.title('Pattern distribution')
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def amount_hist(df:pd.DataFrame, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    #df = df[df['amount'] < 30000] # TODO: remove this, here now because 3_banks_homo_mid contains some very high sar txs, which it shouldn't
    df_neg = df[df['isSAR'] == 0]
    df_pos = df[df['isSAR'] == 1]
    fig, ax = plt.subplots()
    width = 100
    sns.histplot(df_neg, x='amount', binwidth=width, stat='proportion', kde=True, ax=ax, color='C0')
    sns.histplot(df_pos, x='amount', binwidth=width, stat='proportion', kde=True, ax=ax, color='C1')
    ax.grid(axis='y')
    #ax.set_yscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Amount distribution')
    plt.savefig(file)
    plt.close()
    bins = np.arange(0, df['amount'].max() + width, width)
    counts_neg, _ = np.histogram(df_neg['amount'], bins=bins)
    counts_pos, _ = np.histogram(df_pos['amount'], bins=bins)
    total_neg = counts_neg.sum()
    total_pos = counts_pos.sum()
    proportions_neg = counts_neg / total_neg if total_neg > 0 else np.zeros_like(counts_neg)
    proportions_pos = counts_pos / total_pos if total_pos > 0 else np.zeros_like(counts_pos)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_df = pd.DataFrame({f'amount(binwidth={width})': bin_centers, 'count_neg': counts_neg, 'count_pos': counts_pos, 'proportion_neg': proportions_neg, 'proportion_pos': proportions_pos})
    hist_df.to_csv(file.replace('.png', '.csv'), index=False)


def spending_hist(df:pd.DataFrame, file:str):
    df = df[df['bankDest'] == 'sink']
    df_neg = df[df['isSAR'] == 0]
    df_pos = df[df['isSAR'] == 1]
    fig, ax = plt.subplots()
    width = 3000
    sns.histplot(df_neg, x='amount', binwidth=width, stat='proportion', kde=False, ax=ax, color='C0')
    sns.histplot(df_pos, x='amount', binwidth=width, stat='proportion', kde=False, ax=ax, color='C1')
    ax.grid(axis='y')
    ax.set_yscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Spending distribution')
    plt.savefig(file)
    plt.close()
    bins = np.arange(0, df['amount'].max() + width, width)
    counts_neg, _ = np.histogram(df_neg['amount'], bins=bins)
    counts_pos, _ = np.histogram(df_pos['amount'], bins=bins)
    total_neg = counts_neg.sum()
    total_pos = counts_pos.sum()
    proportions_neg = counts_neg / total_neg if total_neg > 0 else np.zeros_like(counts_neg)
    proportions_pos = counts_pos / total_pos if total_pos > 0 else np.zeros_like(counts_pos)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_df = pd.DataFrame({f'amount(binwidth={width})': bin_centers, 'count_neg': counts_neg, 'count_pos': counts_pos, 'proportion_neg': proportions_neg, 'proportion_pos': proportions_pos})
    hist_df.to_csv(file.replace('.png', '.csv'), index=False)


def n_txs_hist(df:pd.DataFrame, bank:str, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_in = df[['amount', 'nameDest', 'bankDest', 'isSAR']].rename(columns={'nameDest': 'name', 'bankDest': 'bank'})
    df_out = df[['amount', 'nameOrig', 'bankOrig', 'isSAR']].rename(columns={'nameOrig': 'name', 'bankOrig': 'bank'})
    df = pd.concat([df_in, df_out])
    df = df[df['bank']==bank]
    d = {}
    gb = df.groupby('name')
    d['n_txs'] = gb['amount'].count()
    d['isSAR'] = gb['isSAR'].max()
    df = pd.concat(d, axis=1)
    df_neg = df[df['isSAR'] == 0]
    df_pos = df[df['isSAR'] == 1]
    fig, ax = plt.subplots()
    sns.histplot(df_neg, x='n_txs', binwidth=1, stat='proportion', kde=True, ax=ax, color='C0', zorder=3)
    sns.histplot(df_pos, x='n_txs', binwidth=1, stat='proportion', kde=True, ax=ax, color='C1', zorder=3)
    ax.grid(axis='y', zorder=0)
    ax.set_xscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Number of transactions distribution')
    plt.savefig(file)
    plt.close()
    bins = np.arange(0, df['n_txs'].max() + 1, 1)
    counts_neg, _ = np.histogram(df_neg['n_txs'], bins=bins)
    counts_pos, _ = np.histogram(df_pos['n_txs'], bins=bins)
    total_neg = counts_neg.sum()
    total_pos = counts_pos.sum()
    proportions_neg = counts_neg / total_neg if total_neg > 0 else np.zeros_like(counts_neg)
    proportions_pos = counts_pos / total_pos if total_pos > 0 else np.zeros_like(counts_pos)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_df = pd.DataFrame({'n_txs(binwidth=1)': bin_centers, 'count_neg': counts_neg, 'count_pos': counts_pos, 'proportion_neg': proportions_neg, 'proportion_pos': proportions_pos})
    hist_df.to_csv(file.replace('.png', '.csv'), index=False)


def n_spending_hist(df:pd.DataFrame, file:str):
    df = df[df['bankDest'] == 'sink']
    d = {}
    gb = df.groupby('nameOrig')
    d['n_spending_txs'] = gb['amount'].count()
    d['isSAR'] = gb['isSAR'].max()
    df = pd.concat(d, axis=1)
    df_neg = df[df['isSAR'] == 0]
    df_pos = df[df['isSAR'] == 1]
    fig, ax = plt.subplots()
    sns.histplot(df_neg, x='n_spending_txs', binwidth=1, stat='proportion', kde=False, ax=ax, color='C0', zorder=3)
    sns.histplot(df_pos, x='n_spending_txs', binwidth=1, stat='proportion', kde=False, ax=ax, color='C1', zorder=3)
    ax.grid(axis='y', zorder=0)
    #ax.set_xscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Number of spending transactions distribution')
    plt.savefig(file)
    bins = np.arange(0, df['n_spending_txs'].max() + 1, 1)
    counts_neg, _ = np.histogram(df_neg['n_spending_txs'], bins=bins)
    counts_pos, _ = np.histogram(df_pos['n_spending_txs'], bins=bins)
    total_neg = counts_neg.sum()
    total_pos = counts_pos.sum()
    proportions_neg = counts_neg / total_neg if total_neg > 0 else np.zeros_like(counts_neg)
    proportions_pos = counts_pos / total_pos if total_pos > 0 else np.zeros_like(counts_pos)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_df = pd.DataFrame({'n_spending_txs(binwidth=1)': bin_centers, 'count_neg': counts_neg, 'count_pos': counts_pos, 'proportion_neg': proportions_neg, 'proportion_pos': proportions_pos})
    hist_df.to_csv(file.replace('.png', '.csv'), index=False)


def powerlaw_degree_dist(df:pd.DataFrame, file:str):
    # OBS! This dist is not equivelent to the dist of the "blueprint". Here we look at all txs, we don't aggregate the txs between two accounts into one "edge".
    # TODO: Change this to show the dist of the "blueprint". 
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_in = df[['amount', 'nameDest', 'isSAR']].rename(columns={'nameDest': 'name'})
    df_out = df[['amount', 'nameOrig', 'isSAR']].rename(columns={'nameOrig': 'name'})
    df = pd.concat([df_in, df_out])
    d = {}
    gb = df.groupby('name')
    d['degree'] = gb['amount'].count()
    d['isSAR'] = gb['isSAR'].max()
    df = pd.concat(d, axis=1)
    df_neg = df[df['isSAR'] == 0]['degree'].value_counts().reset_index()
    df_pos = df[df['isSAR'] == 1]['degree'].value_counts().reset_index()
    
    def func(x, scale, gamma):
        return scale * np.power(x, -gamma)
    
    plt.figure(figsize=(10, 10))
    x = np.linspace(1, 1000, 1000)
    csv = {}
    
    counts = df_neg['count'].values
    degrees = df_neg['degree'].values
    probs = counts / counts.sum()
    log_degrees = np.log(degrees)
    log_probs = np.log(probs)
    coeffs = np.polyfit(log_degrees, log_probs, 1)
    gamma, scale = coeffs
    y = func(x, np.exp(scale), -gamma)
    plt.plot(x, y, label=f'pareto sampling fit\n  gamma={-gamma:.2f}\n  scale={np.exp(scale):.2f}', color='C0')
    plt.scatter(degrees, probs, label='original neg', color='C0')
    csv['degrees_pos'] = degrees
    csv['probs_pos'] = probs
    csv['fit_pos_x'] = x
    csv['fit_pos_y'] = y
    
    counts = df_pos['count'].values
    degrees = df_pos['degree'].values
    probs = counts / counts.sum()
    log_degrees = np.log(degrees)
    log_probs = np.log(probs)
    coeffs = np.polyfit(log_degrees, log_probs, 1)
    gamma, scale = coeffs
    y = func(x, np.exp(scale), -gamma)
    plt.plot(x, y, label=f'pareto sampling fit\n  gamma={-gamma:.2f}\n  scale={np.exp(scale):.2f}', color='C1')
    plt.scatter(degrees, probs, label='original pos', color='C1')
    csv['degrees_neg'] = degrees
    csv['probs_neg'] = probs
    csv['fit_neg_x'] = x
    csv['fit_neg_y'] = y
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('log(degree)')
    plt.ylabel('log(probability)')
    plt.legend()
    plt.grid()
    plt.title('Powerlaw degree distribution')
    plt.savefig(file)
    plt.close()
    
    max_len = max(len(v) for v in csv.values())
    for k, v in csv.items():
        diff = max_len - len(v)
        if diff > 0:
            padding = [None] * diff
            csv[k] = np.concatenate([v, padding])
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def graph(df:pd.DataFrame, file:str, alert_ids=None, n_alerts=None):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    if alert_ids is None:
        alert_ids = df['patternID'].unique()
    if n_alerts is not None:
        alert_ids = np.random.choice(alert_ids, n_alerts, replace=False)
    for alert_id in alert_ids:
        if alert_id == -1:
            continue
        accts = pd.concat([df[df['patternID'] == alert_id]['nameOrig'], df[df['patternID'] == alert_id]['nameDest']]).unique()
        l = []
        for acct in accts:
            l.append(df[df['nameOrig']==acct])
            l.append(df[df['nameDest']==acct])
        df_alert = pd.concat(l)
        graph = nx.from_pandas_edgelist(df=df_alert, source='nameOrig', target='nameDest', edge_attr='patternID', create_using=nx.DiGraph)
        plt.figure(figsize=(10, 10))
        pos = nx.forceatlas2_layout(graph, max_iter=300, scaling_ratio=0.1)
        colors = ['C0' if graph[u][v]['patternID'] == -1 else 'C1' for u, v in graph.edges()]
        nx.draw(graph, pos, with_labels=True, node_size=1000, node_color='C0', edge_color=colors, arrows=True, arrowsize=30, font_size=10, font_weight='bold', width=1, connectionstyle='arc3,rad=0.1')
        plt.title(f'Alert {alert_id}')
        plt.savefig(file.replace('.png', f'_{alert_id}.png'))
        plt.close()


def get_blueprint(df:pd.DataFrame):
    df_edges = df[['nameOrig', 'nameDest', 'isSAR']].rename(columns={'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'class'})
    df_edges = df_edges.drop_duplicates()
    df_out = df[['nameOrig', 'bankOrig', 'isSAR']].rename(columns={'nameOrig': 'id', 'bankOrig': 'bank', 'isSAR': 'class'})
    df_in = df[['nameDest', 'bankDest', 'isSAR']].rename(columns={'nameDest': 'id', 'bankDest': 'bank', 'isSAR': 'class'})
    df_nodes = pd.concat([df_out, df_in])
    df_nodes = df_nodes.groupby('id')['class'].max().reset_index()
    return df_nodes, df_edges


def homophily(df:pd.DataFrame, banks, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_nodes, df_edges = get_blueprint(df)
    n_pos_edges = df_edges[df_edges['class'] == 1].shape[0]
    n_neg_edges = df_edges[(df_edges['src'].isin(df_nodes[df_nodes['class'] == 0]['id'])) & (df_edges['dst'].isin(df_nodes[df_nodes['class'] == 0]['id']))].shape[0]
    n_edges = df_edges.shape[0]
    homophily_edge = (n_pos_edges + n_neg_edges) / n_edges
    n_nodes = df_nodes.shape[0]
    homophiliy_node = 0
    homophiliy_class_sum1 = [0.0, 0.0]
    homophiliy_class_sum2 = [0.0, 0.0]
    for v in range(n_nodes):
        node = df_nodes.iloc[v]['id']
        node_class = df_nodes.iloc[v]['class']
        df_node_edges = df_edges[(df_edges['src'] == node) | (df_edges['dst'] == node)]
        neighbours = set(df_node_edges['src'].tolist() + df_node_edges['dst'].tolist())
        neighbours.remove(node)
        n_neighbours = len(neighbours)
        df_neighbours = df_nodes[df_nodes['id'].isin(neighbours)]
        n_similar_neighbours = df_neighbours[df_neighbours['class'] == node_class].shape[0]
        homophiliy_node += n_similar_neighbours / n_neighbours / n_nodes
        homophiliy_class_sum1[int(node_class)] += n_similar_neighbours
        homophiliy_class_sum2[int(node_class)] += n_neighbours
    h_neg = homophiliy_class_sum1[0] / homophiliy_class_sum2[0] if homophiliy_class_sum2[0] > 0 else 1
    # homophily_class_neg = max(h_neg - len(df_nodes[df_nodes['class']==0]) / n_nodes, 0)
    h_pos = homophiliy_class_sum1[1] / homophiliy_class_sum2[1] if homophiliy_class_sum2[1] > 0 else 1
    # homophily_class_pos = max(h_pos - len(df_nodes[df_nodes['class']==1]) / n_nodes, 0)
    
    # homophily_class = 0.0
    # for i, (sum1, sum2) in enumerate(zip(homophiliy_class_sum1, homophiliy_class_sum2)):
    #     h = sum1 / sum2 if sum2 > 0 else 1
    #     homophily_class += 1/(len(homophiliy_class_sum1)-1) * max(h - len(df_nodes[df_nodes['class']==i]) / n_nodes, 0)
    
    print(h_neg,h_pos)
    
    with open(file, 'w') as f:
        f.write('homophily_edge,homophiliy_node,homophily_class_neg,homophily_class_pos\n')
        f.write(f'{homophily_edge},{homophiliy_node},{h_neg},{h_pos}\n')


def sar_over_n_banks_hist(df:pd.DataFrame, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_sar = df[df['isSAR'] == 1]
    gb = df_sar.groupby(by='patternID')
    n_unique_banks_orig = gb['bankOrig'].nunique()
    n_unique_banks_dest = gb['bankDest'].nunique()
    n_unique_banks_orig_dest = pd.concat([n_unique_banks_orig, n_unique_banks_dest], axis=1)
    n_unique_banks_counts = n_unique_banks_orig_dest.max(axis=1).value_counts()
    n_banks = n_unique_banks_counts.idxmax()
    n_unique_banks_counts = n_unique_banks_counts.reindex(range(1, n_banks+1), fill_value=0)
    n_patterns = n_unique_banks_counts.values.sum()
    fig, ax = plt.subplots()
    width = 0.30
    rects = ax.bar(np.arange(1, n_banks+1), n_unique_banks_counts.values / n_patterns, width, color='C0', zorder=3)
    ax.bar_label(rects, n_unique_banks_counts.values, padding=1)
    ax.set_xticks(np.arange(1, n_banks+1), np.arange(1, n_banks+1))
    ax.grid(axis='y', zorder=0)
    ax.set_xlabel('Number of banks')
    ax.set_ylabel('Ratio of SAR patterns')
    plt.tight_layout(pad=2.0)
    plt.title('SAR patterns spread over banks')
    plt.savefig(file)
    plt.close()
    csv = {'n_banks': np.arange(1, n_banks+1), 'sar_pattern_count': n_unique_banks_counts.values, 'sar_pattern_ratio': n_unique_banks_counts.values / n_patterns}
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def sar_pattern_txs_hist(df:pd.DataFrame, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_sar = df[df['isSAR'] == 1]
    gb = df_sar.groupby(by='patternID')
    sar_pattern_txs_count = gb['patternID'].count()
    sar_pattern_txs_vc = sar_pattern_txs_count.value_counts().reset_index().rename(columns={'patternID': 'sar_pattern_txs_size'})
    fig, ax = plt.subplots()
    width = 0.30
    rects = ax.bar(sar_pattern_txs_vc['sar_pattern_txs_size'].values, sar_pattern_txs_vc['count'].values, width, color='C0', zorder=3)
    ax.grid(axis='y', zorder=0)
    ax.set_xlabel('Size')
    ax.set_ylabel('Count')
    plt.tight_layout(pad=2.0)
    plt.title('SAR patterns txs size hist')
    plt.savefig(file)
    plt.close()    
    sar_pattern_txs_vc.to_csv(file.replace('.png', '.csv'), index=False)


def sar_pattern_account_hist(df:pd.DataFrame, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_sar = df[df['isSAR'] == 1]
    #gb = df_sar.groupby(by='patternID')
    #n_ = gb['accountOrig', 'accountDest'].apply(lambda x: pd.unique(x.values.ravel()).tolist())
    df_sar_acct = df_sar.set_index('patternID')[['nameOrig', 'nameDest']].stack().reset_index(level=1, drop=True).reset_index(name='account').drop_duplicates()
    df_sar_pattern_acct_count = df_sar_acct.groupby(by='patternID')['patternID'].count()
    df_sar_pattern_acct_vc = df_sar_pattern_acct_count.value_counts().reset_index().rename(columns={'patternID': 'sar_pattern_account_size'})
    fig, ax = plt.subplots()
    width = 0.30
    rects = ax.bar(df_sar_pattern_acct_vc['sar_pattern_account_size'].values, df_sar_pattern_acct_vc['count'].values, width, color='C0', zorder=3)
    ax.grid(axis='y', zorder=0)
    ax.set_xlabel('Size')
    ax.set_ylabel('Count')
    plt.tight_layout(pad=2.0)
    plt.title('SAR patterns account size hist')
    plt.savefig(file)
    plt.close()    
    df_sar_pattern_acct_vc.to_csv(file.replace('.png', '.csv'), index=False)


def n_sar_account_participation(df:pd.DataFrame, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_sar = df[df['isSAR'] == 1]
    df_in = df_sar[['nameDest', 'patternID']].rename(columns={'nameDest': 'name'})
    df_out = df_sar[['nameOrig', 'patternID']].rename(columns={'nameOrig': 'name'})
    df = pd.concat([df_in, df_out])
    n_pattern_participation_per_account = df.groupby(by='name')['patternID'].nunique()
    vc = n_pattern_participation_per_account.value_counts().reset_index().rename(columns={'patternID': 'n_patterns', 'count': 'n_accounts'})
    vc['fraction'] = vc['n_accounts'] / vc['n_accounts'].sum()
    fig, ax = plt.subplots()
    width = 0.30
    rects = ax.bar(vc['n_patterns'].values, vc['n_accounts'].values, width, color='C0', zorder=3)
    ax.bar_label(rects, [f'{value:.4f}' for value in vc['fraction'].values], padding=1)
    ax.grid(axis='y', zorder=0)
    ax.set_xlabel('Number of patterns')
    ax.set_ylabel('Number of accounts')
    plt.tight_layout(pad=2.0)
    plt.title('SAR accounts participation hist')
    plt.savefig(file)
    plt.close()
    vc.to_csv(file.replace('.png', '.csv'), index=False)


def plot(df:pd.DataFrame, plot_dir:str):
    banks = pd.concat([df['bankOrig'], df['bankDest']]).unique().tolist()
    banks = sorted(banks)
    if 'source' in banks:
        banks.remove('source')
    if 'sink' in banks:
        banks.remove('sink')
    sar_pattern_account_hist(df, os.path.join(plot_dir, 'sar_pattern_account_hist.png'))
    sar_pattern_txs_hist(df, os.path.join(plot_dir, 'sar_pattern_txs_hist.png'))
    n_sar_account_participation(df, os.path.join(plot_dir, 'sar_account_participation_hist.png'))
    sar_over_n_banks_hist(df, os.path.join(plot_dir, 'sar_over_n_banks_hist.png'))
    edge_label_hist(df, banks+['all'], os.path.join(plot_dir, 'edge_label_hist.png'))
    node_label_hist(df, banks+['all'], os.path.join(plot_dir, 'node_label_hist.png'))
    homophily(df, banks+['all'], os.path.join(plot_dir, os.path.join(plot_dir, 'homophily.csv')))
    for bank in banks:
        os.makedirs(os.path.join(plot_dir, bank), exist_ok=True)
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        balance_curves(df_bank, os.path.join(plot_dir, bank, 'balance_curves.png'), bank=bank)
        pattern_hist(df_bank, os.path.join(plot_dir, bank, 'pattern_hist.png'))
        amount_hist(df_bank, os.path.join(plot_dir, bank, 'amount_hist.png'))
        spending_hist(df_bank, os.path.join(plot_dir, bank, 'spending_hist.png'))
        n_txs_hist(df_bank, bank, os.path.join(plot_dir, bank, 'n_txs_hist.png'))
        n_spending_hist(df_bank, os.path.join(plot_dir, bank, 'n_spending_hist.png'))
        powerlaw_degree_dist(df_bank, os.path.join(plot_dir, bank, 'powerlaw_degree_dist.png'))
        graph(df_bank, os.path.join(plot_dir, bank, 'graph.png'), n_alerts=10)


# Note: This module is now used by scripts/visualize_data.py
# For usage, run: python scripts/visualize_data.py --experiment <experiment_name>