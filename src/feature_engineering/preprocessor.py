import pandas as pd
import numpy as np
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Feature names for timing calculations
TIMING_FEATURES = ['first_step', 'last_step', 'time_span', 'time_std', 'time_skew', 'burstiness']
GAP_FEATURES = ['mean_gap', 'median_gap', 'std_gap', 'max_gap', 'min_gap']
ALL_TIMING_FEATURES = TIMING_FEATURES + GAP_FEATURES


def _calc_all_timing_features(steps: pd.Series) -> pd.Series:
    """
    Calculate all timing distribution and gap features in a single pass.

    Combines timing and gap calculations to avoid redundant sorting.

    Returns Series with 11 features:
    Timing features:
    - first_step: time of first transaction
    - last_step: time of last transaction
    - time_span: range between first and last
    - time_std: standard deviation of transaction times
    - time_skew: skewness (positive = clustered late, negative = clustered early)
    - burstiness: temporal concentration, ranges -1 to 1
      (-1 = perfectly regular, 0 = random/Poisson, +1 = maximally bursty)

    Gap features:
    - mean_gap: average time between consecutive transactions
    - median_gap: median time between consecutive transactions
    - std_gap: standard deviation of gaps
    - max_gap: longest gap between transactions
    - min_gap: shortest gap between transactions
    """
    timestamps = steps.values
    n = len(timestamps)

    # No transactions
    if n == 0:
        return pd.Series({feat: np.nan for feat in ALL_TIMING_FEATURES})

    # Sort once, use for all calculations
    sorted_timestamps = np.sort(timestamps)
    first_step = sorted_timestamps[0]
    last_step = sorted_timestamps[-1]
    time_span = last_step - first_step

    # Single transaction - no gaps possible
    if n == 1:
        return pd.Series({
            'first_step': first_step, 'last_step': last_step, 'time_span': 0.0,
            'time_std': 0.0, 'time_skew': 0.0, 'burstiness': 1.0,
            'mean_gap': np.nan, 'median_gap': np.nan, 'std_gap': np.nan,
            'max_gap': np.nan, 'min_gap': np.nan,
        })

    # Calculate timing features
    time_std = np.std(sorted_timestamps)
    time_skew = stats.skew(sorted_timestamps) if time_std > 0 else 0.0

    # Calculate gaps (already sorted)
    gaps = np.diff(sorted_timestamps)

    # Burstiness: B = (std - mean) / (std + mean), ranges from -1 to 1
    gap_mean = gaps.mean()
    gap_std = gaps.std() if len(gaps) > 1 else 0.0
    denominator = gap_std + gap_mean
    burstiness = (gap_std - gap_mean) / denominator if denominator > 0 else 0.0

    # Gap features
    mean_gap = gap_mean
    median_gap = np.median(gaps)
    std_gap = gap_std
    max_gap = gaps.max()
    min_gap = gaps.min()

    return pd.Series({
        'first_step': first_step, 'last_step': last_step, 'time_span': time_span,
        'time_std': time_std, 'time_skew': time_skew, 'burstiness': burstiness,
        'mean_gap': mean_gap, 'median_gap': median_gap, 'std_gap': std_gap,
        'max_gap': max_gap, 'min_gap': min_gap,
    })


def _add_timing_features(
    df_window: pd.DataFrame,
    groupby_obj,
    direction: str,
    window: tuple,
    node_features: dict
) -> None:
    """
    Add timing and gap features for a given direction (incoming/outgoing).

    Args:
        df_window: Filtered DataFrame for the current window
        groupby_obj: GroupBy object grouped by account
        direction: 'in' or 'out'
        window: Tuple of (start_step, end_step)
        node_features: Dict to add features to (modified in place)
    """
    if len(df_window) == 0:
        return

    timing_raw = groupby_obj['step'].apply(_calc_all_timing_features)

    # apply() with Series return creates MultiIndex; unstack to DataFrame
    if isinstance(timing_raw.index, pd.MultiIndex):
        timing_df = timing_raw.unstack()
        for feat in ALL_TIMING_FEATURES:
            if feat in timing_df.columns:
                node_features[f'{feat}_{direction}_{window[0]}_{window[1]}'] = timing_df[feat]


def _calc_volume_trend(row: pd.Series, window_indices: np.ndarray) -> float:
    """Calculate correlation between window index and transaction counts."""
    if row.std() == 0 or len(row) <= 1:
        return 0.0
    return np.corrcoef(window_indices, row.values)[0, 1]


class DataPreprocessor:
    def __init__(self, config):
        self.num_windows = config['num_windows']
        self.window_len = config['window_len']
        self.include_edge_features = config.get('include_edge_features', False)
        self.bank = None

        # Static account features (from spatial/accounts.csv)
        # These are demographics that don't change over time: age, salary, city
        # Path is derived automatically from experiment structure
        self.static_accounts_path = None  # Set when processing, derived from tx_log path

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
        # patternID and modelType are dropped to avoid direct pattern leakage
        # (split_by_pattern uses alert_models.csv instead)
        # Balance columns are kept for computing balance at window start
        cols_to_drop = ['type', 'modelType', 'patternID']
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        return df

    def load_static_features(self) -> pd.DataFrame:
        """
        Load static account features from spatial/accounts.csv.

        Returns DataFrame with columns: account, age, salary, city, init_balance
        These are demographics/initial values that don't change over time.
        init_balance is used as fallback for balance_at_start_* features.
        """
        if not self.static_accounts_path:
            return None

        from pathlib import Path
        path = Path(self.static_accounts_path)
        if not path.exists():
            logger.info(f"Warning: Static accounts file not found: {path}")
            return None

        df = pd.read_csv(path)

        # Normalize column names (spatial output uses uppercase)
        df.columns = df.columns.str.upper()

        # Select and rename columns we need
        static_cols = {
            'ACCOUNT_ID': 'account',
            'AGE': 'age',
            'SALARY': 'salary',
            'CITY': 'city',
            'INIT_BALANCE': 'init_balance'
        }

        available_cols = {k: v for k, v in static_cols.items() if k in df.columns}
        df_static = df[list(available_cols.keys())].rename(columns=available_cols)

        logger.info(f"Loaded static features for {len(df_static)} accounts: {list(df_static.columns)}")

        return df_static

    def load_alert_models(self) -> dict:
        """
        Load alert_models.csv to get account-to-patterns mapping.

        Returns dict mapping account_id -> set of pattern IDs.
        Used for split_by_pattern to properly handle accounts in multiple patterns.
        """
        if not hasattr(self, 'alert_models_path') or not self.alert_models_path:
            return {}

        from pathlib import Path
        path = Path(self.alert_models_path)
        if not path.exists():
            logger.warning(f"alert_models.csv not found: {path}. Pattern-based splitting may be incomplete.")
            return {}

        df = pd.read_csv(path)

        # Build account -> set of patterns mapping
        # alert_models.csv has columns: modelID, type, accountID, isMain, sourceType, phase
        account_to_patterns = {}
        for _, row in df.iterrows():
            account_id = row['accountID']
            pattern_id = row['modelID']
            if account_id not in account_to_patterns:
                account_to_patterns[account_id] = set()
            account_to_patterns[account_id].add(pattern_id)

        n_multi_pattern = sum(1 for patterns in account_to_patterns.values() if len(patterns) > 1)
        if n_multi_pattern > 0:
            logger.info(f"Loaded alert_models: {len(account_to_patterns)} SAR accounts, "
                       f"{n_multi_pattern} in multiple patterns")
        else:
            logger.info(f"Loaded alert_models: {len(account_to_patterns)} SAR accounts")

        return account_to_patterns

    def load_normal_models(self) -> dict:
        """
        Load normal_models.csv to get normal account-to-patterns mapping.

        Returns dict mapping account_id -> set of pattern IDs.
        Used for split_by_pattern to split normal accounts by pattern (not randomly).
        """
        if not hasattr(self, 'normal_models_path') or not self.normal_models_path:
            return {}

        from pathlib import Path
        path = Path(self.normal_models_path)
        if not path.exists():
            logger.info(f"normal_models.csv not found: {path}. Normal accounts will be split randomly.")
            return {}

        df = pd.read_csv(path)

        # Build account -> set of patterns mapping
        # normal_models.csv has columns: modelID, type, accountID, isMain
        account_to_patterns = {}
        for _, row in df.iterrows():
            account_id = row['accountID']
            pattern_id = row['modelID']
            if account_id not in account_to_patterns:
                account_to_patterns[account_id] = set()
            account_to_patterns[account_id].add(pattern_id)

        n_multi_pattern = sum(1 for patterns in account_to_patterns.values() if len(patterns) > 1)
        if n_multi_pattern > 0:
            logger.info(f"Loaded normal_models: {len(account_to_patterns)} normal accounts, "
                       f"{n_multi_pattern} in multiple patterns")
        else:
            logger.info(f"Loaded normal_models: {len(account_to_patterns)} normal accounts")

        return account_to_patterns

    def merge_static_features(self, df_nodes: pd.DataFrame, df_static: pd.DataFrame) -> pd.DataFrame:
        """
        Merge static account features into node features DataFrame.

        Args:
            df_nodes: Node features with 'account' column
            df_static: Static features with 'account', 'age', 'salary', 'city', 'init_balance' columns

        Returns:
            df_nodes with static features added (city one-hot encoded, init_balance used as fallback)
        """
        if df_static is None:
            return df_nodes

        # Merge on account
        df_merged = df_nodes.merge(df_static, on='account', how='left')

        # Log any missing accounts
        missing = df_merged['age'].isna().sum() if 'age' in df_merged.columns else 0
        if missing > 0:
            logger.info(f"Warning: {missing} accounts missing static features")

        # Use init_balance as fallback for balance_at_start_* columns
        if 'init_balance' in df_merged.columns:
            balance_cols = [col for col in df_merged.columns if col.startswith('balance_at_start_')]
            for col in balance_cols:
                # Handle both NaN and 'unknown' string values (can occur when no transactions before window start)
                # First convert to numeric (coercing 'unknown' to NaN), then fill with init_balance
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(df_merged['init_balance'])
            # Drop init_balance after using it as fallback (it's not a feature itself)
            df_merged = df_merged.drop(columns=['init_balance'])

        # One-hot encode city column if present
        if 'city' in df_merged.columns:
            df_merged['city'] = df_merged['city'].fillna('unknown')
            city_dummies = pd.get_dummies(df_merged['city'], prefix='city', dtype=float)
            df_merged = pd.concat([df_merged.drop(columns=['city']), city_dummies], axis=1)

        return df_merged

    def compute_balance_at_step(self, df: pd.DataFrame, target_step: int, accounts: list) -> dict:
        """
        Compute balance for each account at a specific step.

        Uses the most recent transaction before target_step to get the balance.
        If no prior transactions, returns NaN (will use init_balance from static features).

        Args:
            df: Transaction DataFrame with balance columns
            target_step: The step to compute balance at
            accounts: List of account IDs to compute balances for

        Returns:
            Dict mapping account -> balance at target_step
        """
        # Filter transactions before target_step
        df_prior = df[df['step'] < target_step]

        balances = {}

        # For outgoing transactions: use newbalanceOrig
        if 'newbalanceOrig' in df_prior.columns:
            df_out = df_prior[['nameOrig', 'step', 'newbalanceOrig']].copy()
            df_out = df_out.sort_values('step').groupby('nameOrig').last()
            for acc in df_out.index:
                if acc in accounts:
                    balances[acc] = df_out.loc[acc, 'newbalanceOrig']

        # For incoming transactions: use newbalanceDest (override if more recent)
        if 'newbalanceDest' in df_prior.columns:
            df_in = df_prior[['nameDest', 'step', 'newbalanceDest']].copy()
            df_in = df_in.sort_values('step').groupby('nameDest').last()
            for acc in df_in.index:
                if acc in accounts:
                    # Check if incoming tx is more recent than outgoing
                    if acc in balances:
                        # Get the step of the last outgoing tx
                        last_out_step = df_prior[df_prior['nameOrig'] == acc]['step'].max()
                        last_in_step = df_prior[df_prior['nameDest'] == acc]['step'].max()
                        if pd.notna(last_in_step) and (pd.isna(last_out_step) or last_in_step > last_out_step):
                            balances[acc] = df_in.loc[acc, 'newbalanceDest']
                    else:
                        balances[acc] = df_in.loc[acc, 'newbalanceDest']

        return balances

    def cal_node_features(self, df:pd.DataFrame, start_step, end_step) -> pd.DataFrame:
        # Calculate windows - allow both overlapping and non-overlapping strategies
        total_span = end_step - start_step + 1
        if self.num_windows > 1:
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
                logger.info(f"Warning: Windows cover {total_coverage}/{total_span} steps. Using non-overlapping windows.")
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
        
        # Get all accounts for balance calculation
        all_accounts = set(df_nodes.index)

        # calculate spending features
        for window in windows:
            # Balance at window start
            balance_at_start = self.compute_balance_at_step(df, window[0], all_accounts)
            node_features[f'balance_at_start_{window[0]}_{window[1]}'] = pd.Series(balance_at_start)

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
            # Timing and gap features for incoming transactions
            _add_timing_features(df_in_window, gb_in, 'in', window, node_features)

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
            # Timing and gap features for outgoing transactions
            _add_timing_features(df_out_window, gb_out, 'out', window, node_features)

            # Combined (in+out) timing features - captures cross-direction temporal patterns
            df_combined_window = pd.concat([df_in_window, df_out_window])
            if len(df_combined_window) > 0:
                gb_combined = df_combined_window.groupby(['account'])
                _add_timing_features(df_combined_window, gb_combined, 'combined', window, node_features)

        # Calculate inter-window timing features (patterns across windows)
        if len(windows) > 1:
            # Prepare window indices for trend calculation (used for both directions)
            window_indices = np.arange(len(windows))

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

            # Incoming inter-window features
            if in_counts_per_window:
                in_counts_df = pd.concat(in_counts_per_window, axis=1).fillna(0)
                node_features['n_active_windows_in'] = (in_counts_df > 0).sum(axis=1)
                # Activity consistency: coefficient of variation across windows
                in_means = in_counts_df.mean(axis=1).values
                in_stds = in_counts_df.std(axis=1).values
                node_features['window_activity_cv_in'] = pd.Series(
                    np.divide(in_stds, in_means, out=np.zeros_like(in_stds), where=(in_means != 0)),
                    index=in_counts_df.index
                )
                # Volume trend: correlation with window index (positive = increasing activity)
                node_features['volume_trend_in'] = in_counts_df.apply(
                    lambda row: _calc_volume_trend(row, window_indices), axis=1
                ).fillna(0)

            # Outgoing inter-window features
            if out_counts_per_window:
                out_counts_df = pd.concat(out_counts_per_window, axis=1).fillna(0)
                node_features['n_active_windows_out'] = (out_counts_df > 0).sum(axis=1)
                out_means = out_counts_df.mean(axis=1).values
                out_stds = out_counts_df.std(axis=1).values
                node_features['window_activity_cv_out'] = pd.Series(
                    np.divide(out_stds, out_means, out=np.zeros_like(out_stds), where=(out_means != 0)),
                    index=out_counts_df.index
                )
                node_features['volume_trend_out'] = out_counts_df.apply(
                    lambda row: _calc_volume_trend(row, window_indices), axis=1
                ).fillna(0)

            # Combined (in+out) inter-window features
            if in_counts_per_window and out_counts_per_window:
                combined_counts_df = in_counts_df.add(out_counts_df, fill_value=0)
                node_features['n_active_windows_combined'] = (combined_counts_df > 0).sum(axis=1)
                combined_means = combined_counts_df.mean(axis=1).values
                combined_stds = combined_counts_df.std(axis=1).values
                node_features['window_activity_cv_combined'] = pd.Series(
                    np.divide(combined_stds, combined_means, out=np.zeros_like(combined_stds), where=(combined_means != 0)),
                    index=combined_counts_df.index
                )
                node_features['volume_trend_combined'] = combined_counts_df.apply(
                    lambda row: _calc_volume_trend(row, window_indices), axis=1
                ).fillna(0)

        # calculate non window related features
        combine_cols = ['account', 'days_in_bank', 'n_phone_changes', 'is_sar']
        df_combined = pd.concat([df_in[combine_cols], df_out[combine_cols]])
        gb = df_combined.groupby('account')
        node_features['counts_days_in_bank'] = gb['days_in_bank'].max()
        node_features['counts_phone_changes'] = gb['n_phone_changes'].max()
        # find label
        node_features['is_sar'] = gb['is_sar'].max()
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
        # Fill numeric columns with 0.0 (for missing feature values)
        numeric_cols = df_nodes.select_dtypes(include=[np.number]).columns
        df_nodes[numeric_cols] = df_nodes[numeric_cols].fillna(0.0)
        # Fill any remaining object columns (e.g., 'bank') with 'unknown' as safety
        for col in df_nodes.select_dtypes(include=['object', 'category']).columns:
            df_nodes[col] = df_nodes[col].fillna('unknown')
        # check if there is any missing values
        assert df_nodes.isnull().sum().sum() == 0, 'There are missing values in the node features'
        # Validate feature values
        # 1. Amount-based features must be non-negative (only check numeric columns)
        amount_cols = [col for col in numeric_cols
                      if col not in ['account', 'is_sar']
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
        # Calculate windows - allow both overlapping and non-overlapping strategies
        total_span = end_step - start_step + 1
        if self.num_windows > 1:
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
                df_edges = pd.merge(df_edges, window_df, on=['src', 'dst'], how='outer')
                # Fill numeric columns with 0.0 for missing feature values
                edge_numeric_cols = df_edges.select_dtypes(include=[np.number]).columns
                df_edges[edge_numeric_cols] = df_edges[edge_numeric_cols].fillna(0.0)

        # Aggregate 'is_sar' for the entire dataset (use max to capture any SAR)
        sar_df = df.groupby(['src', 'dst'])['is_sar'].max().reset_index()

        # Merge SAR information into df_edges
        df_edges = pd.merge(df_edges, sar_df, on=['src', 'dst'], how='outer')
        # Fill numeric columns with 0.0 for missing feature values
        sar_numeric_cols = df_edges.select_dtypes(include=[np.number]).columns
        df_edges[sar_numeric_cols] = df_edges[sar_numeric_cols].fillna(0.0)

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
        if self.split_by_pattern:
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

        # Add masks using concat to avoid fragmentation warning
        mask_df = pd.DataFrame({
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }, index=df_nodes.index)
        df_nodes = pd.concat([df_nodes, mask_df], axis=1)

        logger.info(f"\nTransductive label splitting (random):")
        logger.info(f"  Total nodes: {n_nodes}, SAR nodes: {n_sar}, Normal nodes: {n_normal}")
        logger.info(f"  Train: {len(sar_train)} SAR + {len(normal_train)} normal = {train_mask.sum()}")
        logger.info(f"  Val:   {len(sar_val)} SAR + {len(normal_val)} normal = {val_mask.sum()}")
        logger.info(f"  Test:  {len(sar_test)} SAR + {len(normal_test)} normal = {test_mask.sum()}")

        return df_nodes

    def _create_masks_by_pattern(self, df_nodes: pd.DataFrame):
        """
        Split by pattern to prevent data leakage between train/val/test.

        Uses alert_models.csv to get ground truth account-to-pattern mapping.
        Patterns are split into train/val/test. For accounts in multiple patterns,
        one pattern is chosen uniformly at random to determine the account's split.
        Normal nodes use normal_models.csv if available, otherwise split randomly.
        """
        n_nodes = len(df_nodes)
        train_mask = np.zeros(n_nodes, dtype=bool)
        val_mask = np.zeros(n_nodes, dtype=bool)
        test_mask = np.zeros(n_nodes, dtype=bool)

        np.random.seed(self.seed)

        # Load account-to-patterns mappings
        sar_account_to_patterns = self.load_alert_models()
        normal_account_to_patterns = self.load_normal_models()

        if not sar_account_to_patterns:
            logger.warning("No alert_models data found. Falling back to random SAR splitting.")
            return self._create_masks_random(df_nodes)

        # Split SAR patterns
        sar_patterns = set()
        for patterns in sar_account_to_patterns.values():
            sar_patterns.update(patterns)
        sar_patterns = list(np.random.permutation(list(sar_patterns)))

        n_sar_patterns = len(sar_patterns)
        n_sar_patterns_train = int(self.transductive_train_fraction * n_sar_patterns)
        n_sar_patterns_val = int(self.transductive_val_fraction * n_sar_patterns)

        train_sar_patterns = set(sar_patterns[:n_sar_patterns_train])
        val_sar_patterns = set(sar_patterns[n_sar_patterns_train:n_sar_patterns_train + n_sar_patterns_val])
        test_sar_patterns = set(sar_patterns[n_sar_patterns_train + n_sar_patterns_val:])

        # Split normal patterns (if available)
        if normal_account_to_patterns:
            normal_patterns = set()
            for patterns in normal_account_to_patterns.values():
                normal_patterns.update(patterns)
            normal_patterns = list(np.random.permutation(list(normal_patterns)))

            n_normal_patterns = len(normal_patterns)
            n_normal_patterns_train = int(self.transductive_train_fraction * n_normal_patterns)
            n_normal_patterns_val = int(self.transductive_val_fraction * n_normal_patterns)

            train_normal_patterns = set(normal_patterns[:n_normal_patterns_train])
            val_normal_patterns = set(normal_patterns[n_normal_patterns_train:n_normal_patterns_train + n_normal_patterns_val])
            test_normal_patterns = set(normal_patterns[n_normal_patterns_train + n_normal_patterns_val:])
        else:
            n_normal_patterns = 0
            train_normal_patterns = val_normal_patterns = test_normal_patterns = set()

        # Identify SAR nodes
        sar_mask = df_nodes['is_sar'] == 1

        # Assign SAR nodes based on their pattern (random choice for accounts in multiple patterns)
        n_sar_assigned = {'train': 0, 'val': 0, 'test': 0}
        for idx, row in df_nodes.iterrows():
            node_idx = df_nodes.index.get_loc(idx)
            if row['is_sar'] == 1:
                account_id = row['account']
                patterns = sar_account_to_patterns.get(account_id, set())
                if patterns:
                    pattern = np.random.choice(list(patterns))
                    if pattern in train_sar_patterns:
                        train_mask[node_idx] = True
                        n_sar_assigned['train'] += 1
                    elif pattern in val_sar_patterns:
                        val_mask[node_idx] = True
                        n_sar_assigned['val'] += 1
                    elif pattern in test_sar_patterns:
                        test_mask[node_idx] = True
                        n_sar_assigned['test'] += 1

        # Assign normal nodes based on their pattern (or randomly if no normal_models.csv)
        n_normal_assigned = {'train': 0, 'val': 0, 'test': 0}
        normal_indices = np.where(~sar_mask)[0]

        if normal_account_to_patterns:
            # Split normal nodes by pattern (random choice for accounts in multiple patterns)
            for idx, row in df_nodes.iterrows():
                node_idx = df_nodes.index.get_loc(idx)
                if row['is_sar'] != 1:
                    account_id = row['account']
                    patterns = normal_account_to_patterns.get(account_id, set())
                    if patterns:
                        pattern = np.random.choice(list(patterns))
                        if pattern in train_normal_patterns:
                            train_mask[node_idx] = True
                            n_normal_assigned['train'] += 1
                        elif pattern in val_normal_patterns:
                            val_mask[node_idx] = True
                            n_normal_assigned['val'] += 1
                        elif pattern in test_normal_patterns:
                            test_mask[node_idx] = True
                            n_normal_assigned['test'] += 1
                    else:
                        # Account not in normal_models.csv - assign randomly based on fractions
                        r = np.random.random()
                        if r < self.transductive_train_fraction:
                            train_mask[node_idx] = True
                            n_normal_assigned['train'] += 1
                        elif r < self.transductive_train_fraction + self.transductive_val_fraction:
                            val_mask[node_idx] = True
                            n_normal_assigned['val'] += 1
                        else:
                            test_mask[node_idx] = True
                            n_normal_assigned['test'] += 1
        else:
            # No normal_models.csv - split randomly
            normal_indices = np.random.permutation(normal_indices)
            n_normal = len(normal_indices)
            n_normal_train = int(self.transductive_train_fraction * n_normal)
            n_normal_val = int(self.transductive_val_fraction * n_normal)

            train_mask[normal_indices[:n_normal_train]] = True
            val_mask[normal_indices[n_normal_train:n_normal_train + n_normal_val]] = True
            test_mask[normal_indices[n_normal_train + n_normal_val:]] = True

            n_normal_assigned = {
                'train': n_normal_train,
                'val': n_normal_val,
                'test': n_normal - n_normal_train - n_normal_val
            }

        # Add masks using concat to avoid fragmentation warning
        mask_df = pd.DataFrame({
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }, index=df_nodes.index)
        df_nodes = pd.concat([df_nodes, mask_df], axis=1)

        logger.info(f"\nTransductive label splitting (by pattern):")
        logger.info(f"  Total nodes: {n_nodes}, SAR patterns: {n_sar_patterns}, Normal patterns: {n_normal_patterns}")
        logger.info(f"  Train: {len(train_sar_patterns)} SAR patterns ({n_sar_assigned['train']} nodes) + "
                   f"{len(train_normal_patterns)} normal patterns ({n_normal_assigned['train']} nodes) = {train_mask.sum()}")
        logger.info(f"  Val:   {len(val_sar_patterns)} SAR patterns ({n_sar_assigned['val']} nodes) + "
                   f"{len(val_normal_patterns)} normal patterns ({n_normal_assigned['val']} nodes) = {val_mask.sum()}")
        logger.info(f"  Test:  {len(test_sar_patterns)} SAR patterns ({n_sar_assigned['test']} nodes) + "
                   f"{len(test_normal_patterns)} normal patterns ({n_normal_assigned['test']} nodes) = {test_mask.sum()}")

        # Verify no account appears in multiple splits
        all_train_patterns = train_sar_patterns | train_normal_patterns
        all_val_patterns = val_sar_patterns | val_normal_patterns
        all_test_patterns = test_sar_patterns | test_normal_patterns
        self._verify_pattern_split(df_nodes, train_mask, val_mask, test_mask,
                                   all_train_patterns, all_val_patterns, all_test_patterns)

        return df_nodes

    def _verify_pattern_split(self, df_nodes: pd.DataFrame, train_mask, val_mask, test_mask,
                               train_patterns: set, val_patterns: set, test_patterns: set):
        """
        Verify that pattern-based splitting has no account-level overlap.

        With random assignment for multi-pattern accounts, patterns may be "incomplete"
        across splits (some members in train, others in val). This is acceptable noise.
        The critical check is that no single account appears in multiple splits.
        """
        # Verify no account appears in multiple splits (the real leakage concern)
        multi_mask = (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)) > 1
        if multi_mask.any():
            overlap_accounts = df_nodes.loc[multi_mask, 'account'].tolist()
            raise AssertionError(f"Accounts in multiple splits: {overlap_accounts[:10]}...")

        logger.info("  ✓ Split verified: no account appears in multiple splits")


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

        # Merge static features (age, salary, city) if configured
        df_static = self.load_static_features()
        df_nodes = self.merge_static_features(df_nodes, df_static)

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

        # Merge static features (age, salary, city) if configured
        df_static = self.load_static_features()
        df_nodes_train = self.merge_static_features(df_nodes_train, df_static)
        df_nodes_val = self.merge_static_features(df_nodes_val, df_static)
        df_nodes_test = self.merge_static_features(df_nodes_test, df_static)

        logger.info(f"\nInductive setting:")
        logger.info(f"  Train: {len(df_nodes_train)} nodes ({int(df_nodes_train['is_sar'].sum())} SAR)")
        logger.info(f"  Val:   {len(df_nodes_val)} nodes ({int(df_nodes_val['is_sar'].sum())} SAR)")
        logger.info(f"  Test:  {len(df_nodes_test)} nodes ({int(df_nodes_test['is_sar'].sum())} SAR)")

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

    def plot_feature_analysis(self, df_nodes: pd.DataFrame, output_dir: str):
        """
        Generate plots comparing feature distributions between SAR and non-SAR accounts.

        Creates a multi-panel figure showing how each feature separates the classes.

        Args:
            df_nodes: Node features DataFrame with 'is_sar' column
            output_dir: Directory to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("matplotlib not available, skipping feature analysis plots")
            return

        import os

        if 'is_sar' not in df_nodes.columns:
            logger.info("No 'is_sar' column found, skipping feature analysis plots")
            return

        # Split into SAR and non-SAR
        sar_df = df_nodes[df_nodes['is_sar'] == 1]
        non_sar_df = df_nodes[df_nodes['is_sar'] == 0]

        if len(sar_df) == 0 or len(non_sar_df) == 0:
            logger.info(f"Insufficient data for comparison (SAR: {len(sar_df)}, Non-SAR: {len(non_sar_df)})")
            return

        # Get numeric feature columns (exclude identifiers and labels)
        # Note: city is now one-hot encoded (city_* columns) so it's included as numeric features
        exclude_cols = ['account', 'bank', 'is_sar', 'train_mask', 'val_mask', 'test_mask']
        numeric_cols = df_nodes.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        if len(feature_cols) == 0:
            logger.info("No numeric features to plot")
            return

        # Group features by category for organized plotting
        # Exclude city_* columns from main plot (handled separately in city analysis)
        non_city_features = [c for c in feature_cols if not c.startswith('city_')]

        feature_groups = {
            'Static': [c for c in non_city_features if c in ['age', 'salary', 'init_balance']],
            'Balance': [c for c in non_city_features if 'balance' in c and c != 'init_balance'],
            'Incoming Amount': [c for c in non_city_features if any(p in c for p in ['sum_in', 'mean_in', 'median_in', 'std_in', 'max_in', 'min_in'])],
            'Outgoing Amount': [c for c in non_city_features if any(p in c for p in ['sum_out', 'mean_out', 'median_out', 'std_out', 'max_out', 'min_out'])],
            'Spending': [c for c in non_city_features if 'spending' in c],
            'Counts': [c for c in non_city_features if 'count' in c],
            'Intra-Window Timing': [c for c in non_city_features if any(p in c for p in ['first_step', 'last_step', 'time_span', 'time_std', 'gap'])],
            'Inter-Window Timing': [c for c in non_city_features if any(p in c for p in ['burstiness', 'time_skew', 'volume_trend', 'activity_cv', 'n_active'])],
            'Global': [c for c in non_city_features if c in ['counts_days_in_bank', 'counts_phone_changes']],
        }

        # Collect features that weren't categorized (excluding city_* columns)
        categorized = set()
        for cols in feature_groups.values():
            categorized.update(cols)
        uncategorized = [c for c in non_city_features if c not in categorized]
        if uncategorized:
            feature_groups['Other'] = uncategorized

        # Remove empty groups
        feature_groups = {k: v for k, v in feature_groups.items() if v}

        logger.info(f"\nGenerating feature analysis plots: {len(sar_df)} SAR, {len(non_sar_df)} non-SAR accounts")
        logger.info(f"Feature groups: {', '.join(f'{k}({len(v)})' for k, v in feature_groups.items())}")

        os.makedirs(output_dir, exist_ok=True)
        summary_stats = []
        saved_plots = []

        # Create separate figure for each feature group
        for group_name, cols in feature_groups.items():
            # Calculate median ratios for each feature
            ratios = []
            labels = []
            for col in cols:
                sar_vals = sar_df[col].dropna()
                non_sar_vals = non_sar_df[col].dropna()

                if len(sar_vals) > 0 and len(non_sar_vals) > 0:
                    sar_median = np.median(sar_vals)
                    non_sar_median = np.median(non_sar_vals)
                    sar_mean = np.mean(sar_vals)
                    non_sar_mean = np.mean(non_sar_vals)

                    # Use absolute values for comparison
                    sar_abs = max(abs(sar_median), abs(sar_mean))
                    non_sar_abs = max(abs(non_sar_median), abs(non_sar_mean))

                    if non_sar_abs < 0.01 and sar_abs < 0.01:
                        # Both groups essentially zero - no meaningful difference
                        ratio = 1.0
                    elif abs(non_sar_median) > 1e-6 and np.sign(sar_median) == np.sign(non_sar_median):
                        # Same sign - normal ratio
                        ratio = sar_median / non_sar_median
                    elif abs(non_sar_median) > 1e-6:
                        # Different signs - use absolute ratio, show SAR is "different"
                        ratio = abs(sar_median / non_sar_median)
                    elif abs(non_sar_mean) > 1e-6 and np.sign(sar_mean) == np.sign(non_sar_mean):
                        # Fallback to mean if same sign
                        ratio = sar_mean / non_sar_mean
                    else:
                        # Can't compute meaningful ratio
                        ratio = 1.0

                    ratios.append(ratio)
                    # Shorten column name for display
                    short_name = col.replace('_', ' ').replace('  ', ' ')
                    if len(short_name) > 30:
                        short_name = short_name[:27] + '...'
                    labels.append(short_name)

                    summary_stats.append({
                        'group': group_name,
                        'feature': col,
                        'sar_median': sar_median,
                        'non_sar_median': non_sar_median,
                        'ratio': ratio
                    })

            if ratios:
                # Create figure sized to fit the features
                fig_height = max(3, len(ratios) * 0.4)
                fig, ax = plt.subplots(figsize=(10, fig_height))

                # Color bars based on ratio (green = SAR higher, red = SAR lower)
                colors = ['green' if r > 1.1 else 'red' if r < 0.9 else 'gray' for r in ratios]

                # Use linear scale centered at 1.0 (no difference)
                # Plot (ratio - 1) so that 1.0x shows as 0, 2.0x shows as 1, 0.5x shows as -0.5
                centered_ratios = [r - 1.0 for r in ratios]

                bars = ax.barh(range(len(ratios)), centered_ratios, color=colors, alpha=0.7)
                ax.set_yticks(range(len(ratios)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('SAR median / Non-SAR median (centered at 1.0x)')
                ax.set_title(f'{group_name} Features (n={len(cols)})')

                # Add ratio values on bars
                for i, (bar, ratio) in enumerate(zip(bars, ratios)):
                    width = bar.get_width()
                    x_pos = width + 0.05 if width >= 0 else width - 0.05
                    ha = 'left' if width >= 0 else 'right'
                    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                           f'{ratio:.2f}x', va='center', ha=ha, fontsize=8)

                plt.tight_layout()

                # Save plot with sanitized filename
                safe_name = group_name.lower().replace(' ', '_').replace('-', '_')
                plot_path = os.path.join(output_dir, f'features_{safe_name}.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                saved_plots.append(plot_path)

        logger.info(f"Saved {len(saved_plots)} feature analysis plots to {output_dir}")

        # Print top discriminative features
        summary_stats.sort(key=lambda x: abs(np.log2(max(x['ratio'], 0.01)) if x['ratio'] > 0 else 10), reverse=True)
        logger.info("\nTop 10 most discriminative features (by median ratio):")
        for stat in summary_stats[:10]:
            direction = "SAR higher" if stat['ratio'] > 1 else "SAR lower"
            logger.info(f"  {stat['feature']}: {stat['ratio']:.2f}x ({direction})")

        # Plot city distribution if available (check for one-hot encoded city columns)
        city_cols = [col for col in df_nodes.columns if col.startswith('city_')]
        if city_cols:
            self._plot_city_analysis(df_nodes, sar_df, non_sar_df, output_dir, city_cols)

    def _plot_city_analysis(self, df_nodes: pd.DataFrame, sar_df: pd.DataFrame,
                            non_sar_df: pd.DataFrame, output_dir: str, city_cols: list):
        """
        Plot city distribution analysis comparing SAR rates across cities.

        Args:
            df_nodes: Full node DataFrame
            sar_df: SAR accounts subset
            non_sar_df: Non-SAR accounts subset
            output_dir: Directory to save the plot
            city_cols: List of one-hot encoded city column names
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        import os

        # Reconstruct city from one-hot encoded columns
        # Each row has exactly one city column = 1.0
        city_data = df_nodes[city_cols]
        cities = city_data.idxmax(axis=1).str.replace('city_', '', n=1)
        sar_city_data = sar_df[city_cols]
        sar_cities = sar_city_data.idxmax(axis=1).str.replace('city_', '', n=1)

        # Count accounts per city
        city_counts = cities.value_counts()
        sar_city_counts = sar_cities.value_counts()

        # Calculate SAR rate per city
        city_stats = []
        for city in city_counts.index:
            total = city_counts.get(city, 0)
            sar_count = sar_city_counts.get(city, 0)
            if total > 0:
                sar_rate = sar_count / total
                city_stats.append({
                    'city': city,
                    'total': total,
                    'sar_count': sar_count,
                    'sar_rate': sar_rate
                })

        if not city_stats:
            return

        # Sort by SAR rate
        city_stats.sort(key=lambda x: x['sar_rate'], reverse=True)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(city_stats) * 0.3)))

        # Plot 1: SAR rate by city
        cities_sorted = [s['city'] for s in city_stats]
        sar_rates = [s['sar_rate'] for s in city_stats]
        totals = [s['total'] for s in city_stats]

        # Color by SAR rate (higher = more red)
        overall_sar_rate = len(sar_df) / len(df_nodes)
        colors = ['red' if r > overall_sar_rate * 1.5 else 'orange' if r > overall_sar_rate else 'green' for r in sar_rates]

        bars = ax1.barh(range(len(cities_sorted)), sar_rates, color=colors, alpha=0.7)
        ax1.axvline(overall_sar_rate, color='black', linestyle='--', linewidth=1, label=f'Overall SAR rate: {overall_sar_rate:.1%}')
        ax1.set_yticks(range(len(cities_sorted)))
        ax1.set_yticklabels([f"{c} (n={totals[i]})" for i, c in enumerate(cities_sorted)], fontsize=8)
        ax1.set_xlabel('SAR Rate')
        ax1.set_title('SAR Rate by City')
        ax1.legend(loc='lower right')

        # Add percentage labels
        for i, (bar, rate) in enumerate(zip(bars, sar_rates)):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1%}', va='center', fontsize=7)

        # Plot 2: City distribution comparison (SAR vs Non-SAR)
        # Normalize to show proportion
        sar_proportions = [sar_city_counts.get(c, 0) / len(sar_df) if len(sar_df) > 0 else 0 for c in cities_sorted]
        non_sar_proportions = [(city_counts.get(c, 0) - sar_city_counts.get(c, 0)) / len(non_sar_df) if len(non_sar_df) > 0 else 0 for c in cities_sorted]

        y_pos = np.arange(len(cities_sorted))
        width = 0.35

        ax2.barh(y_pos - width/2, non_sar_proportions, width, label='Non-SAR', alpha=0.7, color='C0')
        ax2.barh(y_pos + width/2, sar_proportions, width, label='SAR', alpha=0.7, color='C1')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(cities_sorted, fontsize=8)
        ax2.set_xlabel('Proportion of Accounts')
        ax2.set_title('City Distribution: SAR vs Non-SAR')
        ax2.legend()

        plt.tight_layout()

        # Save plot
        city_plot_path = os.path.join(output_dir, 'city_analysis.png')
        plt.savefig(city_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved city analysis plot to {city_plot_path}")

        # Print city summary
        logger.info(f"\nCity Analysis (overall SAR rate: {overall_sar_rate:.1%}):")
        logger.info("  Cities with elevated SAR rates:")
        for stat in city_stats[:5]:
            if stat['sar_rate'] > overall_sar_rate:
                logger.info(f"    {stat['city']}: {stat['sar_rate']:.1%} SAR rate ({stat['sar_count']}/{stat['total']} accounts)")

    def __call__(self, raw_data_file):
        logger.info('Preprocessing data...')

        # Derive static accounts path from tx_log path
        # tx_log: experiments/<exp>/temporal/tx_log.parquet
        # accounts: experiments/<exp>/spatial/accounts.csv
        from pathlib import Path
        tx_path = Path(raw_data_file)
        experiment_dir = tx_path.parent.parent  # Go up from temporal/ to experiment root
        self.static_accounts_path = experiment_dir / 'spatial' / 'accounts.csv'
        self.alert_models_path = experiment_dir / 'spatial' / 'alert_models.csv'
        self.normal_models_path = experiment_dir / 'spatial' / 'normal_models.csv'

        raw_df = self.load_data(raw_data_file)
        preprocessed_df = self.preprocess(raw_df)

        logger.info(' done\n')

        return preprocessed_df
