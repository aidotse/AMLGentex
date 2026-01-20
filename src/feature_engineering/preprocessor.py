import pandas as pd
import numpy as np
from scipy import stats


def _calc_timing_agg(steps: pd.Series) -> pd.Series:
    """
    Calculate timing distribution features for a group of transactions.

    Returns Series with:
    - first_step: time of first transaction
    - last_step: time of last transaction
    - time_span: range between first and last
    - time_std: standard deviation of transaction times
    - time_skew: skewness (positive = clustered late, negative = clustered early)
    - burstiness: measure of temporal concentration (0 = evenly distributed, 1 = all at once)
    """
    vals = steps.values
    n = len(vals)

    if n == 0:
        return pd.Series({
            'first_step': np.nan, 'last_step': np.nan, 'time_span': np.nan,
            'time_std': np.nan, 'time_skew': np.nan, 'burstiness': np.nan,
        })

    first_step = vals.min()
    last_step = vals.max()
    time_span = last_step - first_step

    if n == 1:
        return pd.Series({
            'first_step': first_step, 'last_step': last_step, 'time_span': 0.0,
            'time_std': 0.0, 'time_skew': 0.0, 'burstiness': 1.0,
        })

    time_std = np.std(vals)
    time_skew = stats.skew(vals) if time_std > 0 else 0.0

    # Burstiness: B = (std - mean) / (std + mean), ranges from -1 to 1
    sorted_vals = np.sort(vals)
    gaps = np.diff(sorted_vals)
    if len(gaps) > 0 and gaps.mean() > 0:
        gap_std = gaps.std()
        gap_mean = gaps.mean()
        burstiness = (gap_std - gap_mean) / (gap_std + gap_mean) if (gap_std + gap_mean) > 0 else 0.0
    else:
        burstiness = 0.0

    return pd.Series({
        'first_step': first_step, 'last_step': last_step, 'time_span': time_span,
        'time_std': time_std, 'time_skew': time_skew, 'burstiness': burstiness,
    })


def _calc_gap_agg(steps: pd.Series) -> pd.Series:
    """
    Calculate inter-transaction gap features for a group of transactions.

    Returns Series with:
    - mean_gap: average time between consecutive transactions
    - median_gap: median time between consecutive transactions
    - std_gap: standard deviation of gaps
    - max_gap: longest gap between transactions
    - min_gap: shortest gap between transactions
    """
    vals = steps.values
    n = len(vals)

    if n < 2:
        return pd.Series({
            'mean_gap': np.nan, 'median_gap': np.nan, 'std_gap': np.nan,
            'max_gap': np.nan, 'min_gap': np.nan,
        })

    sorted_vals = np.sort(vals)
    gaps = np.diff(sorted_vals)

    return pd.Series({
        'mean_gap': gaps.mean(), 'median_gap': np.median(gaps),
        'std_gap': gaps.std() if len(gaps) > 1 else 0.0,
        'max_gap': gaps.max(), 'min_gap': gaps.min(),
    })


class DataPreprocessor:
    def __init__(self, config, verbose: bool = True):
        self.num_windows = config['num_windows']
        self.window_len = config['window_len']
        self.include_edge_features = config.get('include_edge_features', False)
        self.bank = None
        self.verbose = verbose

        # Learning mode: transductive or inductive
        self.learning_mode = config['learning_mode']
        self.is_transductive = self.learning_mode == 'transductive'

        if self.is_transductive:
            # Transductive: single time window for all splits
            self.time_start = config['time_start']
            self.time_end = config['time_end']
            # Use same window for train/val/test
            self.train_start_step = self.time_start
            self.train_end_step = self.time_end
            self.val_start_step = self.time_start
            self.val_end_step = self.time_end
            self.test_start_step = self.time_start
            self.test_end_step = self.time_end

            # Transductive-only settings
            self.transductive_train_fraction = config.get('transductive_train_fraction', 0.6)
            self.transductive_val_fraction = config.get('transductive_val_fraction', 0.2)
            self.transductive_test_fraction = config.get('transductive_test_fraction', 0.2)
            self.seed = config.get('seed', 42)
            self.split_by_pattern = config.get('split_by_pattern', False)
        else:
            # Inductive: separate time windows for each split
            self.train_start_step = config['train_start_step']
            self.train_end_step = config['train_end_step']
            self.val_start_step = config['val_start_step']
            self.val_end_step = config['val_end_step']
            self.test_start_step = config['test_start_step']
            self.test_end_step = config['test_end_step']
            # These are not used in inductive mode
            self.transductive_train_fraction = None
            self.transductive_val_fraction = None
            self.transductive_test_fraction = None
            self.seed = config.get('seed', 42)
            self.split_by_pattern = False
    
    
    def load_data(self, path:str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df = df[df['bankOrig'] != 'source'] # TODO: create features based on source transactions
        # Drop columns not needed for feature engineering
        # Note: patternID is kept if split_by_pattern is enabled (for splitting, not as a feature)
        # modelType is always dropped to avoid direct pattern leakage
        cols_to_drop = ['type', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'modelType']
        if not self.split_by_pattern:
            cols_to_drop.append('patternID')
        df.drop(columns=cols_to_drop, inplace=True)
        return df

    
    def cal_node_features(self, df:pd.DataFrame, start_step, end_step) -> pd.DataFrame:
        # Validate window coverage
        total_span = end_step - start_step + 1
        if self.num_windows == 1 and self.window_len < total_span:
            raise ValueError(f"Window configuration (num_windows={self.num_windows}, window_len={self.window_len}) "
                           f"do not allow coverage of the full range [{start_step}, {end_step}] ({total_span} steps)")

        # Calculate windows - allow both overlapping and non-overlapping strategies
        if self.num_windows > 1:
            total_span = end_step - start_step + 1
            total_coverage = self.num_windows * self.window_len

            if total_coverage >= total_span:
                # Full coverage with overlapping windows
                window_overlap = (total_coverage - total_span) // (self.num_windows - 1)
                windows = [(start_step + i*(self.window_len-window_overlap),
                           start_step + i*(self.window_len-window_overlap) + self.window_len-1)
                          for i in range(self.num_windows)]
                windows[-1] = (end_step - self.window_len + 1, end_step)
            else:
                # Partial coverage with non-overlapping windows (evenly spaced)
                if self.verbose:
                    print(f"Warning: Windows cover {total_coverage}/{total_span} steps. Using non-overlapping windows.")
                step_size = total_span // self.num_windows
                windows = [(start_step + i*step_size,
                           min(start_step + i*step_size + self.window_len - 1, end_step))
                          for i in range(self.num_windows)]
        else:
            windows = [(start_step, end_step)]

        # Filter transactions based on bank setting
        # When self.bank is set, bank sees ALL transactions involving its accounts
        # (incoming from any bank, outgoing to any bank, spending to sink)
        # Define column mappings for incoming and outgoing transactions
        in_cols = ['step', 'nameDest', 'bankDest', 'amount', 'nameOrig', 'daysInBankDest', 'phoneChangesDest', 'isSAR']
        in_rename = {'nameDest': 'account', 'bankDest': 'bank', 'nameOrig': 'counterpart',
                     'daysInBankDest': 'days_in_bank', 'phoneChangesDest': 'n_phone_changes', 'isSAR': 'is_sar'}
        out_cols = ['step', 'nameOrig', 'bankOrig', 'amount', 'nameDest', 'daysInBankOrig', 'phoneChangesOrig', 'isSAR']
        out_rename = {'nameOrig': 'account', 'bankOrig': 'bank', 'nameDest': 'counterpart',
                      'daysInBankOrig': 'days_in_bank', 'phoneChangesOrig': 'n_phone_changes', 'isSAR': 'is_sar'}

        # Include patternID if split_by_pattern is enabled
        if self.split_by_pattern and 'patternID' in df.columns:
            in_cols.append('patternID')
            in_rename['patternID'] = 'pattern_id'
            out_cols.append('patternID')
            out_rename['patternID'] = 'pattern_id'

        if self.bank is not None:
            # Accounts belonging to this bank
            accounts_orig = df[df['bankOrig'] == self.bank]['nameOrig'].unique()
            accounts_dest = df[df['bankDest'] == self.bank]['nameDest'].unique()
            accounts = pd.unique(np.concatenate([accounts_orig, accounts_dest]))

            # Spending: transactions from this bank's accounts to sink
            df_spending = df[(df['bankOrig'] == self.bank) & (df['bankDest'] == 'sink')].rename(columns={'nameOrig': 'account'})

            # Incoming: transactions TO this bank's accounts (from any bank)
            df_network_in = df[(df['bankDest'] == self.bank) & (df['bankDest'] != 'sink')]
            # Outgoing: transactions FROM this bank's accounts (to any bank, excluding sink)
            df_network_out = df[(df['bankOrig'] == self.bank) & (df['bankDest'] != 'sink')]

            df_in = df_network_in[in_cols].rename(columns=in_rename)
            df_out = df_network_out[out_cols].rename(columns=out_rename)
        else:
            # No bank filter: use all transactions
            accounts = pd.unique(df[['nameOrig', 'nameDest']].values.ravel('K'))
            df_spending = df[df['bankDest'] == 'sink'].rename(columns={'nameOrig': 'account'})
            df_network = df[df['bankDest'] != 'sink']

            df_in = df_network[in_cols].rename(columns=in_rename)
            df_out = df_network[out_cols].rename(columns=out_rename)

        df_nodes = pd.DataFrame()
        df_nodes = pd.concat([df_out[['account', 'bank']], df_in[['account', 'bank']]]).drop_duplicates().set_index('account')
        node_features = {}
        
        # calculate spending features
        for window in windows:
            gb_spending = df_spending[(df_spending['step']>=window[0])&(df_spending['step']<=window[1])].groupby(['account'])
            node_features[f'sums_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].sum()
            node_features[f'means_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].mean()
            node_features[f'medians_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].median()
            node_features[f'stds_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].std()
            node_features[f'maxs_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].max()
            node_features[f'mins_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].min()
            node_features[f'counts_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].count()
            # Incoming transaction features
            df_in_window = df_in[(df_in['step']>=window[0])&(df_in['step']<=window[1])]
            gb_in = df_in_window.groupby(['account'])
            node_features[f'sum_in_{window[0]}_{window[1]}'] = gb_in['amount'].apply(lambda x: x[x > 0].sum())
            node_features[f'mean_in_{window[0]}_{window[1]}'] = gb_in['amount'].mean()
            node_features[f'median_in_{window[0]}_{window[1]}'] = gb_in['amount'].median()
            node_features[f'std_in_{window[0]}_{window[1]}'] = gb_in['amount'].std()
            node_features[f'max_in_{window[0]}_{window[1]}'] = gb_in['amount'].max()
            node_features[f'min_in_{window[0]}_{window[1]}'] = gb_in['amount'].min()
            node_features[f'count_in_{window[0]}_{window[1]}'] = gb_in['amount'].count()
            node_features[f'count_unique_in_{window[0]}_{window[1]}'] = gb_in['counterpart'].nunique()
            # Timing features for incoming transactions
            if len(df_in_window) > 0:
                timing_in_raw = gb_in['step'].apply(_calc_timing_agg)
                if isinstance(timing_in_raw.index, pd.MultiIndex):
                    timing_in = timing_in_raw.unstack()
                    for feat in ['first_step', 'last_step', 'time_span', 'time_std', 'time_skew', 'burstiness']:
                        if feat in timing_in.columns:
                            node_features[f'{feat}_in_{window[0]}_{window[1]}'] = timing_in[feat]
                # Gap features for incoming transactions
                gap_in_raw = gb_in['step'].apply(_calc_gap_agg)
                if isinstance(gap_in_raw.index, pd.MultiIndex):
                    gap_in = gap_in_raw.unstack()
                    for feat in ['mean_gap', 'median_gap', 'std_gap', 'max_gap', 'min_gap']:
                        if feat in gap_in.columns:
                            node_features[f'{feat}_in_{window[0]}_{window[1]}'] = gap_in[feat]

            # Outgoing transaction features
            df_out_window = df_out[(df_out['step']>=window[0])&(df_out['step']<=window[1])]
            gb_out = df_out_window.groupby(['account'])
            node_features[f'sum_out_{window[0]}_{window[1]}'] = gb_out['amount'].apply(lambda x: x[x > 0].sum())
            node_features[f'mean_out_{window[0]}_{window[1]}'] = gb_out['amount'].mean()
            node_features[f'median_out_{window[0]}_{window[1]}'] = gb_out['amount'].median()
            node_features[f'std_out_{window[0]}_{window[1]}'] = gb_out['amount'].std()
            node_features[f'max_out_{window[0]}_{window[1]}'] = gb_out['amount'].max()
            node_features[f'min_out_{window[0]}_{window[1]}'] = gb_out['amount'].min()
            node_features[f'count_out_{window[0]}_{window[1]}'] = gb_out['amount'].count()
            node_features[f'count_unique_out_{window[0]}_{window[1]}'] = gb_out['counterpart'].nunique()
            # Timing features for outgoing transactions
            if len(df_out_window) > 0:
                timing_out_raw = gb_out['step'].apply(_calc_timing_agg)
                if isinstance(timing_out_raw.index, pd.MultiIndex):
                    timing_out = timing_out_raw.unstack()
                    for feat in ['first_step', 'last_step', 'time_span', 'time_std', 'time_skew', 'burstiness']:
                        if feat in timing_out.columns:
                            node_features[f'{feat}_out_{window[0]}_{window[1]}'] = timing_out[feat]
                # Gap features for outgoing transactions
                gap_out_raw = gb_out['step'].apply(_calc_gap_agg)
                if isinstance(gap_out_raw.index, pd.MultiIndex):
                    gap_out = gap_out_raw.unstack()
                    for feat in ['mean_gap', 'median_gap', 'std_gap', 'max_gap', 'min_gap']:
                        if feat in gap_out.columns:
                            node_features[f'{feat}_out_{window[0]}_{window[1]}'] = gap_out[feat]
        # Calculate inter-window timing features (patterns across windows)
        if len(windows) > 1:
            # Collect per-window counts for inter-window analysis
            in_counts_per_window = []
            out_counts_per_window = []
            for window in windows:
                in_count_col = f'count_in_{window[0]}_{window[1]}'
                out_count_col = f'count_out_{window[0]}_{window[1]}'
                if in_count_col in node_features:
                    in_counts_per_window.append(node_features[in_count_col])
                if out_count_col in node_features:
                    out_counts_per_window.append(node_features[out_count_col])

            # Number of active windows (windows with at least one transaction)
            if in_counts_per_window:
                in_counts_df = pd.concat(in_counts_per_window, axis=1).fillna(0)
                node_features['n_active_windows_in'] = (in_counts_df > 0).sum(axis=1)
                # Activity consistency: coefficient of variation across windows
                in_means = in_counts_df.mean(axis=1)
                in_stds = in_counts_df.std(axis=1)
                node_features['window_activity_cv_in'] = (in_stds / in_means).replace([np.inf, -np.inf], 0).fillna(0)
                # Volume trend: correlation with window index (positive = increasing activity)
                window_indices = np.arange(len(windows))
                def calc_trend(row):
                    if row.std() == 0:
                        return 0.0
                    return np.corrcoef(window_indices, row.values)[0, 1] if len(row) > 1 else 0.0
                node_features['volume_trend_in'] = in_counts_df.apply(calc_trend, axis=1).fillna(0)

            if out_counts_per_window:
                out_counts_df = pd.concat(out_counts_per_window, axis=1).fillna(0)
                node_features['n_active_windows_out'] = (out_counts_df > 0).sum(axis=1)
                out_means = out_counts_df.mean(axis=1)
                out_stds = out_counts_df.std(axis=1)
                node_features['window_activity_cv_out'] = (out_stds / out_means).replace([np.inf, -np.inf], 0).fillna(0)
                def calc_trend(row):
                    if row.std() == 0:
                        return 0.0
                    return np.corrcoef(window_indices, row.values)[0, 1] if len(row) > 1 else 0.0
                node_features['volume_trend_out'] = out_counts_df.apply(calc_trend, axis=1).fillna(0)

        # calculate non window related features
        combine_cols = ['account', 'days_in_bank', 'n_phone_changes', 'is_sar']
        if self.split_by_pattern and 'pattern_id' in df_in.columns:
            combine_cols.append('pattern_id')
        df_combined = pd.concat([df_in[combine_cols], df_out[combine_cols]])
        gb = df_combined.groupby('account')
        node_features['counts_days_in_bank'] = gb['days_in_bank'].max()
        node_features['counts_phone_changes'] = gb['n_phone_changes'].max()
        # find label
        node_features['is_sar'] = gb['is_sar'].max()
        # track pattern ID for pattern-based splitting (use max to get the SAR pattern if any)
        if self.split_by_pattern and 'pattern_id' in df_in.columns:
            node_features['pattern_id'] = gb['pattern_id'].max()
        # concat features
        node_features_df = pd.concat(node_features, axis=1)
        # merge with nodes
        df_nodes = df_nodes.join(node_features_df)
        # filter out nodes not belonging to the bank
        
        df_nodes = df_nodes.reset_index()
        df_nodes = df_nodes[df_nodes['account'].isin(accounts)]
        if self.bank is not None:
            df_nodes = df_nodes[df_nodes['bank'] == self.bank]
        else:
            # When processing all banks, deduplicate by account to avoid duplicate nodes
            # (features are computed per-account, not per-account-bank pair)
            df_nodes = df_nodes.drop_duplicates(subset='account', keep='first')
        # if any value is nan, there was no transaction in the window for that account and hence the feature should be 0
        df_nodes = df_nodes.fillna(0.0).infer_objects(copy=False)
        # check if there is any missing values
        assert df_nodes.isnull().sum().sum() == 0, 'There are missing values in the node features'
        # Validate feature values
        # 1. Amount-based features must be non-negative
        amount_cols = [col for col in df_nodes.columns
                      if col not in ['account', 'bank', 'pattern_id', 'is_sar']
                      and not any(p in col for p in ['burstiness_', 'time_skew_', 'volume_trend_'])]
        assert (df_nodes[amount_cols] < 0).sum().sum() == 0, 'There are negative values in amount-based features'

        # 2. Timing features that can be negative should be bounded
        # burstiness and volume_trend are bounded [-1, 1], time_skew is typically [-3, 3]
        bounded_cols = [col for col in df_nodes.columns if 'burstiness_' in col or 'volume_trend_' in col]
        if bounded_cols:
            assert (df_nodes[bounded_cols].abs() <= 1.0).all().all(), 'Burstiness/trend features out of [-1, 1] range'
        skew_cols = [col for col in df_nodes.columns if 'time_skew_' in col]
        if skew_cols:
            assert (df_nodes[skew_cols].abs() <= 10.0).all().all(), 'Time skew features have extreme values'
        return df_nodes
    
    
    def extract_edge_list(self, df: pd.DataFrame, start_step, end_step) -> pd.DataFrame:
        """
        Extract unique edge list (graph structure) from network transactions.

        Returns minimal edge dataframe with only src, dst columns.
        Graph structure is purely topological - labels and features are separate.
        """
        # Filter network transactions only (exclude source/sink)
        df_network = df[(df['bankOrig'] != 'source') & (df['bankDest'] != 'sink')]

        # Filter by bank if specified - include edges where at least one endpoint belongs to this bank
        if self.bank is not None:
            df_network = df_network[(df_network['bankOrig'] == self.bank) | (df_network['bankDest'] == self.bank)]

        # Extract unique edges (pure graph structure)
        edges = df_network[['nameOrig', 'nameDest']].drop_duplicates()

        # Rename columns to standard names
        edges = edges.rename(columns={
            'nameOrig': 'src',
            'nameDest': 'dst'
        })

        return edges

    def cal_edge_features(self, df: pd.DataFrame, start_step, end_step, directional: bool = False) -> pd.DataFrame:
        # Validate window coverage
        total_span = end_step - start_step + 1
        if self.num_windows == 1 and self.window_len < total_span:
            raise ValueError(f"Window configuration (num_windows={self.num_windows}, window_len={self.window_len}) "
                           f"do not allow coverage of the full range [{start_step}, {end_step}] ({total_span} steps)")

        # Calculate windows - allow both overlapping and non-overlapping strategies
        if self.num_windows > 1:
            total_span = end_step - start_step + 1
            total_coverage = self.num_windows * self.window_len

            if total_coverage >= total_span:
                # Full coverage with overlapping windows
                window_overlap = (total_coverage - total_span) // (self.num_windows - 1)
                windows = [(start_step + i * (self.window_len - window_overlap),
                           start_step + i * (self.window_len - window_overlap) + self.window_len - 1)
                          for i in range(self.num_windows)]
                windows[-1] = (end_step - self.window_len + 1, end_step)
            else:
                # Partial coverage with non-overlapping windows (evenly spaced)
                step_size = total_span // self.num_windows
                windows = [(start_step + i * step_size,
                           min(start_step + i * step_size + self.window_len - 1, end_step))
                          for i in range(self.num_windows)]
        else:
            windows = [(start_step, end_step)]

        # Filter by bank if set - include edges where at least one endpoint belongs to this bank
        if self.bank is not None:
            df = df[(df['bankOrig'] == self.bank) | (df['bankDest'] == self.bank)]

        # Rename columns
        df = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})

        # If directional=False then sort src and dst
        if not directional:
            df[['src', 'dst']] = np.sort(df[['src', 'dst']], axis=1)

        # Initialize final dataframe
        df_edges = pd.DataFrame()

        # Iterate over windows
        for window in windows:
            window_df = df[(df['step'] >= window[0]) & (df['step'] <= window[1])].groupby(['src', 'dst']).agg(
                sums=('amount', 'sum'),
                means=('amount', 'mean'),
                medians=('amount', 'median'),
                stds=('amount', 'std'),
                maxs=('amount', 'max'),
                mins=('amount', 'min'),
                counts=('amount', 'count')
            ).fillna(0.0).reset_index()

            # Rename the columns with window information
            window_df = window_df.rename(columns={col: f'{col}_{window[0]}_{window[1]}' for col in window_df.columns if col not in ['src', 'dst']})

            # Merge window data with the main df_edges DataFrame
            if df_edges.empty:
                df_edges = window_df
            else:
                df_edges = pd.merge(df_edges, window_df, on=['src', 'dst'], how='outer').fillna(0.0)

        # Aggregate 'is_sar' for the entire dataset (use max to capture any SAR)
        sar_df = df.groupby(['src', 'dst'])['is_sar'].max().reset_index()

        # Merge SAR information into df_edges
        df_edges = pd.merge(df_edges, sar_df, on=['src', 'dst'], how='outer').fillna(0.0)

        # Ensure no missing values
        assert df_edges.isnull().sum().sum() == 0, 'There are missing values in the edge features'

        # Ensure no negative values (except for src and dst columns)
        assert (df_edges.drop(columns=['src', 'dst']) < 0).sum().sum() == 0, 'There are negative values in the edge features'

        return df_edges


    def create_transductive_masks(self, df_nodes: pd.DataFrame):
        """
        Create train/val/test masks for transductive learning.

        If split_by_pattern is enabled, splits by pattern ID to ensure no pattern
        appears in multiple splits (prevents data leakage). Otherwise, randomly
        splits SAR and normal nodes.
        """
        if self.split_by_pattern and 'pattern_id' in df_nodes.columns:
            return self._create_masks_by_pattern(df_nodes)
        else:
            return self._create_masks_random(df_nodes)

    def _create_masks_random(self, df_nodes: pd.DataFrame):
        """Random split of SAR and normal nodes into non-overlapping sets."""
        n_nodes = len(df_nodes)
        train_mask = np.zeros(n_nodes, dtype=bool)
        val_mask = np.zeros(n_nodes, dtype=bool)
        test_mask = np.zeros(n_nodes, dtype=bool)

        # Split SAR nodes
        sar_indices = np.where(df_nodes['is_sar'] == 1)[0]
        np.random.seed(self.seed)
        sar_indices = np.random.permutation(sar_indices)

        n_sar = len(sar_indices)
        n_sar_train = int(self.transductive_train_fraction * n_sar)
        n_sar_val = int(self.transductive_val_fraction * n_sar)

        sar_train = sar_indices[:n_sar_train]
        sar_val = sar_indices[n_sar_train:n_sar_train + n_sar_val]
        sar_test = sar_indices[n_sar_train + n_sar_val:]

        # Split normal nodes
        normal_indices = np.where(df_nodes['is_sar'] == 0)[0]
        normal_indices = np.random.permutation(normal_indices)

        n_normal = len(normal_indices)
        n_normal_train = int(self.transductive_train_fraction * n_normal)
        n_normal_val = int(self.transductive_val_fraction * n_normal)

        normal_train = normal_indices[:n_normal_train]
        normal_val = normal_indices[n_normal_train:n_normal_train + n_normal_val]
        normal_test = normal_indices[n_normal_train + n_normal_val:]

        # Set masks
        train_mask[sar_train] = True
        train_mask[normal_train] = True
        val_mask[sar_val] = True
        val_mask[normal_val] = True
        test_mask[sar_test] = True
        test_mask[normal_test] = True

        df_nodes['train_mask'] = train_mask
        df_nodes['val_mask'] = val_mask
        df_nodes['test_mask'] = test_mask

        if self.verbose:
            print(f"\nTransductive label splitting (random):")
            print(f"  Total nodes: {n_nodes}, SAR nodes: {n_sar}, Normal nodes: {n_normal}")
            print(f"  Train: {len(sar_train)} SAR + {len(normal_train)} normal = {train_mask.sum()}")
            print(f"  Val:   {len(sar_val)} SAR + {len(normal_val)} normal = {val_mask.sum()}")
            print(f"  Test:  {len(sar_test)} SAR + {len(normal_test)} normal = {test_mask.sum()}")

        return df_nodes

    def _create_masks_by_pattern(self, df_nodes: pd.DataFrame):
        """
        Split by pattern ID to ensure no pattern appears in multiple splits.

        SAR patterns (pattern_id >= 0) are split by pattern, keeping all nodes
        of a pattern together. Normal nodes (pattern_id < 0) are split randomly.
        """
        n_nodes = len(df_nodes)
        train_mask = np.zeros(n_nodes, dtype=bool)
        val_mask = np.zeros(n_nodes, dtype=bool)
        test_mask = np.zeros(n_nodes, dtype=bool)

        np.random.seed(self.seed)

        # Get unique SAR patterns (pattern_id >= 0)
        sar_mask = df_nodes['pattern_id'] >= 0
        sar_patterns = df_nodes.loc[sar_mask, 'pattern_id'].unique()
        sar_patterns = np.random.permutation(sar_patterns)

        n_patterns = len(sar_patterns)
        n_patterns_train = int(self.transductive_train_fraction * n_patterns)
        n_patterns_val = int(self.transductive_val_fraction * n_patterns)

        train_patterns = set(sar_patterns[:n_patterns_train])
        val_patterns = set(sar_patterns[n_patterns_train:n_patterns_train + n_patterns_val])
        test_patterns = set(sar_patterns[n_patterns_train + n_patterns_val:])

        # Assign SAR nodes based on their pattern
        for idx, row in df_nodes.iterrows():
            node_idx = df_nodes.index.get_loc(idx)
            pattern = row['pattern_id']
            if pattern >= 0:  # SAR node
                if pattern in train_patterns:
                    train_mask[node_idx] = True
                elif pattern in val_patterns:
                    val_mask[node_idx] = True
                elif pattern in test_patterns:
                    test_mask[node_idx] = True

        # Split normal nodes (pattern_id < 0) randomly
        normal_indices = np.where(df_nodes['pattern_id'] < 0)[0]
        normal_indices = np.random.permutation(normal_indices)

        n_normal = len(normal_indices)
        n_normal_train = int(self.transductive_train_fraction * n_normal)
        n_normal_val = int(self.transductive_val_fraction * n_normal)

        normal_train = normal_indices[:n_normal_train]
        normal_val = normal_indices[n_normal_train:n_normal_train + n_normal_val]
        normal_test = normal_indices[n_normal_train + n_normal_val:]

        train_mask[normal_train] = True
        val_mask[normal_val] = True
        test_mask[normal_test] = True

        df_nodes['train_mask'] = train_mask
        df_nodes['val_mask'] = val_mask
        df_nodes['test_mask'] = test_mask

        # Remove pattern_id from output (it was only needed for splitting)
        df_nodes = df_nodes.drop(columns=['pattern_id'])

        if self.verbose:
            n_sar_train = train_mask.sum() - len(normal_train)
            n_sar_val = val_mask.sum() - len(normal_val)
            n_sar_test = test_mask.sum() - len(normal_test)
            print(f"\nTransductive label splitting (by pattern):")
            print(f"  Total nodes: {n_nodes}, SAR patterns: {n_patterns}, Normal nodes: {n_normal}")
            print(f"  Train: {len(train_patterns)} patterns ({n_sar_train} SAR nodes) + {len(normal_train)} normal = {train_mask.sum()}")
            print(f"  Val:   {len(val_patterns)} patterns ({n_sar_val} SAR nodes) + {len(normal_val)} normal = {val_mask.sum()}")
            print(f"  Test:  {len(test_patterns)} patterns ({n_sar_test} SAR nodes) + {len(normal_test)} normal = {test_mask.sum()}")

        return df_nodes


    def preprocess_transductive(self, df: pd.DataFrame):
        """
        Preprocess for transductive learning (same time window, split labels).

        In transductive learning, all splits share the same graph structure and features.
        The train/val/test distinction is made via boolean mask columns (train_mask,
        val_mask, test_mask) on the nodes. The returned dict contains three keys for
        API consistency with inductive preprocessing, but they reference the same
        DataFrame objects.
        """
        df_window = df[(df['step'] >= self.train_start_step) & (df['step'] <= self.train_end_step)]

        # Compute features once (same graph for all splits)
        df_nodes = self.cal_node_features(df_window, self.train_start_step, self.train_end_step)
        df_nodes = self.create_transductive_masks(df_nodes)

        df_edges = self.extract_edge_list(df_window, self.train_start_step, self.train_end_step)

        # Optionally compute edge features
        if self.include_edge_features:
            df_network = df_window[(df_window['bankOrig'] != 'source') & (df_window['bankDest'] != 'sink')]
            if self.bank is not None:
                df_network = df_network[(df_network['bankOrig'] == self.bank) | (df_network['bankDest'] == self.bank)]

            df_edge_features = self.cal_edge_features(df=df_network, start_step=self.train_start_step,
                                                     end_step=self.train_end_step, directional=True)
            df_edges = df_edges.merge(df_edge_features, on=['src', 'dst'], how='left')

        # Return same graph for all splits (same object references for API consistency)
        # Note: If saving to disk, only one copy needs to be written
        return {
            'trainset_nodes': df_nodes,
            'trainset_edges': df_edges,
            'valset_nodes': df_nodes,   # same as trainset_nodes
            'valset_edges': df_edges,   # same as trainset_edges
            'testset_nodes': df_nodes,  # same as trainset_nodes
            'testset_edges': df_edges   # same as trainset_edges
        }


    def preprocess_inductive(self, df: pd.DataFrame):
        """Preprocess for inductive learning (different time windows)."""
        df_train = df[(df['step'] >= self.train_start_step) & (df['step'] <= self.train_end_step)]
        df_val = df[(df['step'] >= self.val_start_step) & (df['step'] <= self.val_end_step)]
        df_test = df[(df['step'] >= self.test_start_step) & (df['step'] <= self.test_end_step)]

        # Compute features separately for each split
        df_nodes_train = self.cal_node_features(df_train, self.train_start_step, self.train_end_step)
        df_nodes_val = self.cal_node_features(df_val, self.val_start_step, self.val_end_step)
        df_nodes_test = self.cal_node_features(df_test, self.test_start_step, self.test_end_step)

        if self.verbose:
            print(f"\nInductive setting:")
            print(f"  Train: {len(df_nodes_train)} nodes ({int(df_nodes_train['is_sar'].sum())} SAR)")
            print(f"  Val:   {len(df_nodes_val)} nodes ({int(df_nodes_val['is_sar'].sum())} SAR)")
            print(f"  Test:  {len(df_nodes_test)} nodes ({int(df_nodes_test['is_sar'].sum())} SAR)")

        # Extract edge lists
        df_edges_train = self.extract_edge_list(df_train, self.train_start_step, self.train_end_step)
        df_edges_val = self.extract_edge_list(df_val, self.val_start_step, self.val_end_step)
        df_edges_test = self.extract_edge_list(df_test, self.test_start_step, self.test_end_step)

        # Optionally compute edge features
        if self.include_edge_features:
            df_train_network = df_train[(df_train['bankOrig'] != 'source') & (df_train['bankDest'] != 'sink')]
            df_val_network = df_val[(df_val['bankOrig'] != 'source') & (df_val['bankDest'] != 'sink')]
            df_test_network = df_test[(df_test['bankOrig'] != 'source') & (df_test['bankDest'] != 'sink')]

            if self.bank is not None:
                df_train_network = df_train_network[(df_train_network['bankOrig'] == self.bank) | (df_train_network['bankDest'] == self.bank)]
                df_val_network = df_val_network[(df_val_network['bankOrig'] == self.bank) | (df_val_network['bankDest'] == self.bank)]
                df_test_network = df_test_network[(df_test_network['bankOrig'] == self.bank) | (df_test_network['bankDest'] == self.bank)]

            df_edge_features_train = self.cal_edge_features(df=df_train_network, start_step=self.train_start_step,
                                                           end_step=self.train_end_step, directional=True)
            df_edge_features_val = self.cal_edge_features(df=df_val_network, start_step=self.val_start_step,
                                                         end_step=self.val_end_step, directional=True)
            df_edge_features_test = self.cal_edge_features(df=df_test_network, start_step=self.test_start_step,
                                                          end_step=self.test_end_step, directional=True)

            df_edges_train = df_edges_train.merge(df_edge_features_train, on=['src', 'dst'], how='left')
            df_edges_val = df_edges_val.merge(df_edge_features_val, on=['src', 'dst'], how='left')
            df_edges_test = df_edges_test.merge(df_edge_features_test, on=['src', 'dst'], how='left')

        return {
            'trainset_nodes': df_nodes_train,
            'trainset_edges': df_edges_train,
            'valset_nodes': df_nodes_val,
            'valset_edges': df_edges_val,
            'testset_nodes': df_nodes_test,
            'testset_edges': df_edges_test
        }


    def preprocess(self, df: pd.DataFrame):
        """Dispatcher for transductive vs inductive preprocessing."""
        if self.is_transductive:
            return self.preprocess_transductive(df)
        else:
            return self.preprocess_inductive(df)
    
    
    def __call__(self, raw_data_file):
        if self.verbose:
            print('\nPreprocessing data...', end='')
        raw_df = self.load_data(raw_data_file)
        preprocessed_df = self.preprocess(raw_df)
        if self.verbose:
            print(' done\n')
        return preprocessed_df
