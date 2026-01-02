import pandas as pd
import os
from typing import Optional, Union, List


def flip_labels(nodes:pd.DataFrame, labels:list=[0, 1], fracs:list=[0.01, 0.1], seed:int=42):
    
    # copy nodes
    nodes = nodes.copy()
    
    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']
    
    for label, frac in zip(labels, fracs):
        accounts_to_flip = nodes[nodes['true_label'] == label]['account'].sample(frac=frac, random_state=seed)
        nodes.loc[nodes['account'].isin(accounts_to_flip), 'is_sar'] = 1 - label
    
    return nodes


def missing_labels(nodes:pd.DataFrame, labels:list=[0, 1], fracs:list=[0.01, 0.1], seed:int=42):
    
    # copy nodes
    nodes = nodes.copy()
    
    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']
    
    for label, frac in zip(labels, fracs):
        accounts_to_miss = nodes[nodes['true_label'] == label]['account'].sample(frac=frac, random_state=seed)
        nodes.loc[nodes['account'].isin(accounts_to_miss), 'is_sar'] = -1
    
    return nodes


def flip_neighbours(nodes:pd.DataFrame, edges:pd.DataFrame, frac:float=0.1, seed:int=42):
    
    # copy nodes
    nodes = nodes.copy()
    
    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']
    
    # find all normal accounts that are connected to SAR accounts
    sar_accounts = set(nodes[nodes['true_label'] == 1]['account'])
    edges_with_sar = edges[edges['src'].isin(sar_accounts) | edges['dst'].isin(sar_accounts)]
    normal_accounts_with_edges_to_sar_accounts = set(edges_with_sar['src']).union(set(edges_with_sar['dst'])) - sar_accounts
    normal_accounts_with_edges_to_sar_accounts = normal_accounts_with_edges_to_sar_accounts.intersection(set(nodes[nodes['true_label'] == 0]['account']))
    normal_accounts_with_edges_to_sar_accounts = pd.DataFrame(normal_accounts_with_edges_to_sar_accounts, columns=['account'])
    normal_accounts_with_edges_to_sar_accounts = normal_accounts_with_edges_to_sar_accounts.sort_values(by='account')
    
    # flip labels of normal accounts that are connected to SAR accounts
    accounts_to_flip = normal_accounts_with_edges_to_sar_accounts['account'].sample(frac=frac, random_state=seed)
    nodes.loc[nodes['account'].isin(accounts_to_flip), 'is_sar'] = 1 - nodes.loc[nodes['account'].isin(accounts_to_flip), 'true_label']
    
    return nodes


def topology_noise(nodes:pd.DataFrame, alert_members:pd.DataFrame, topologies:list=['fan_in', 'fan_out', 'cycle', 'bipartite', 'stack', 'gather_scatter', 'scatter_gather'], fracs:Optional[Union[float, List[float]]]=None, seed:int=42):

    # TODO: remove alert_members, future updates should ensure that nodes has the necessary information to find the topologies

    # copy nodes
    nodes = nodes.copy()

    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']

    if isinstance(fracs, list):
        assert len(fracs) == len(topologies), 'fracs should have the same length as topologies'
    elif isinstance(fracs, float):
        fracs = [fracs] * len(topologies)
    elif fracs is None:
        fracs = [0.5] * len(topologies)

    node_sar = nodes[nodes['is_sar'] == 1][['account','is_sar']]
    alert_members = alert_members[['reason','accountID']].rename(columns={'accountID':'account'})
    alert_members = alert_members[alert_members['account'].isin(node_sar['account'])]

    for topology, frac in zip(topologies, fracs):
        accounts_to_flip = alert_members[alert_members['reason'] == topology]['account'].sample(frac=frac, random_state=seed)
        nodes.loc[nodes['account'].isin(accounts_to_flip), 'is_sar'] = 0

    return nodes


def apply_train_noise(preprocessed_dir: str, noise_fn, noise_kwargs: dict = None):
    """
    Apply label noise only to training split labels.

    Handles both transductive (masks) and inductive (separate files) settings.
    Preserves validation and test labels for proper evaluation.

    Args:
        preprocessed_dir: Directory containing preprocessed data
        noise_fn: Noise function to apply (e.g., missing_labels, flip_labels)
        noise_kwargs: Keyword arguments for the noise function

    Example:
        >>> # Make 10% of SAR and 5% of normal training labels unknown
        >>> apply_train_noise(
        ...     'experiments/my_exp/preprocessed',
        ...     missing_labels,
        ...     {'labels': [1, 0], 'fracs': [0.1, 0.05], 'seed': 42}
        ... )
    """
    if noise_kwargs is None:
        noise_kwargs = {}

    centralized_dir = os.path.join(preprocessed_dir, 'centralized')
    train_nodes_file = os.path.join(centralized_dir, 'trainset_nodes.parquet')
    val_nodes_file = os.path.join(centralized_dir, 'valset_nodes.parquet')
    test_nodes_file = os.path.join(centralized_dir, 'testset_nodes.parquet')

    # Load training data
    df_train = pd.read_parquet(train_nodes_file)

    # Check if transductive (has masks) or inductive (no masks)
    is_transductive = 'train_mask' in df_train.columns

    if is_transductive:
        print("\nApplying noise to transductive training labels...")

        # Extract indices of nodes with training labels
        train_mask = df_train['train_mask'].copy()
        train_indices = df_train[train_mask].index

        # Create true_label column preserving original labels for all nodes
        if 'true_label' not in df_train.columns:
            df_train['true_label'] = df_train['is_sar'].copy()

        # Create temporary dataframe with only training nodes
        df_train_subset = df_train.loc[train_indices].copy()

        # Apply noise function
        df_train_subset_noisy = noise_fn(df_train_subset, **noise_kwargs)

        # Update is_sar column for training nodes only (by index)
        df_train.loc[train_indices, 'is_sar'] = df_train_subset_noisy['is_sar'].values

        # Save updated file (same file for all splits in transductive)
        df_train.to_parquet(train_nodes_file, index=False)
        df_train.to_parquet(val_nodes_file, index=False)
        df_train.to_parquet(test_nodes_file, index=False)

        # Report statistics
        n_unknown = (df_train[df_train['train_mask']]['is_sar'] == -1).sum()
        n_train = df_train['train_mask'].sum()
        print(f"  Training labels: {n_train}")
        print(f"  Unknown labels: {n_unknown} ({100*n_unknown/n_train:.1f}%)")

    else:
        print("\nApplying noise to inductive training labels...")

        # Apply noise to training set only
        df_train_noisy = noise_fn(df_train, **noise_kwargs)
        df_train_noisy.to_parquet(train_nodes_file, index=False)

        # Report statistics
        n_unknown = (df_train_noisy['is_sar'] == -1).sum()
        n_train = len(df_train_noisy)
        print(f"  Training nodes: {n_train}")
        print(f"  Unknown labels: {n_unknown} ({100*n_unknown/n_train:.1f}%)")

    print("  Validation and test labels preserved for evaluation")
    print(f"  Updated: {train_nodes_file}")
