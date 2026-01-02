import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, config, verbose: bool = True):
        self.num_windows = config['num_windows']
        self.window_len = config['window_len']
        self.train_start_step = config['train_start_step']
        self.train_end_step = config['train_end_step']
        self.val_start_step = config['val_start_step']
        self.val_end_step = config['val_end_step']
        self.test_start_step = config['test_start_step']
        self.test_end_step = config['test_end_step']
        self.include_edges = config['include_edges']
        self.bank = None
        self.verbose = verbose

        # Detect learning setting from time windows
        # Transductive: All splits have same time window (complete overlap)
        # Inductive: Splits have different time windows (temporal separation)
        self.is_transductive = (
            self.train_start_step == self.val_start_step == self.test_start_step and
            self.train_end_step == self.val_end_step == self.test_end_step
        )

        # Label splitting configuration for transductive learning
        self.train_label_fraction = config.get('train_label_fraction', 0.6)
        self.val_label_fraction = config.get('val_label_fraction', 0.2)
        self.test_label_fraction = config.get('test_label_fraction', 0.2)
        self.seed = config.get('seed', 42)
    
    
    def load_data(self, path:str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df = df[df['bankOrig'] != 'source'] # TODO: create features based on source transactions
        # Drop columns not needed for feature engineering
        # Note: patternID and modelType are dropped to avoid direct pattern leakage
        df.drop(columns=['type', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'patternID', 'modelType'], inplace=True)
        return df

    
    def cal_node_features(self, df:pd.DataFrame, start_step, end_step) -> pd.DataFrame:
        df_bank = df[(df['bankOrig'] == self.bank) & (df['bankDest'] == self.bank)] if self.bank is not None else df
        accounts = pd.unique(df_bank[['nameOrig', 'nameDest']].values.ravel('K'))

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
        # filter out transactions to the sink
        df_spending = df[df['bankDest'] == 'sink'].rename(columns={'nameOrig': 'account'})
        # filter out and reform transactions within the network 
        df_network = df[df['bankDest'] != 'sink']
        df_in = df_network[['step', 'nameDest', 'bankDest', 'amount', 'nameOrig', 'daysInBankDest', 'phoneChangesDest', 'isSAR']].rename(columns={'nameDest': 'account', 'bankDest': 'bank', 'nameOrig': 'counterpart', 'daysInBankDest': 'days_in_bank', 'phoneChangesDest': 'n_phone_changes', 'isSAR': 'is_sar'})
        df_out = df_network[['step', 'nameOrig', 'bankOrig', 'amount', 'nameDest', 'daysInBankOrig', 'phoneChangesOrig', 'isSAR']].rename(columns={'nameOrig': 'account', 'bankOrig': 'bank', 'nameDest': 'counterpart', 'daysInBankOrig': 'days_in_bank', 'phoneChangesOrig': 'n_phone_changes', 'isSAR': 'is_sar'})

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
            gb_in = df_in[(df_in['step']>=window[0])&(df_in['step']<=window[1])].groupby(['account'])
            node_features[f'sum_in_{window[0]}_{window[1]}'] = gb_in['amount'].apply(lambda x: x[x > 0].sum())
            node_features[f'mean_in_{window[0]}_{window[1]}'] = gb_in['amount'].mean()
            node_features[f'median_in_{window[0]}_{window[1]}'] = gb_in['amount'].median()
            node_features[f'std_in_{window[0]}_{window[1]}'] = gb_in['amount'].std()
            node_features[f'max_in_{window[0]}_{window[1]}'] = gb_in['amount'].max()
            node_features[f'min_in_{window[0]}_{window[1]}'] = gb_in['amount'].min()
            node_features[f'count_in_{window[0]}_{window[1]}'] = gb_in['amount'].count()
            node_features[f'count_unique_in_{window[0]}_{window[1]}'] = gb_in['counterpart'].nunique()
            gb_out = df_out[(df_out['step']>=window[0])&(df_out['step']<=window[1])].groupby(['account'])
            node_features[f'sum_out_{window[0]}_{window[1]}'] = gb_out['amount'].apply(lambda x: x[x > 0].sum())
            node_features[f'mean_out_{window[0]}_{window[1]}'] = gb_out['amount'].mean()
            node_features[f'median_out_{window[0]}_{window[1]}'] = gb_out['amount'].median()
            node_features[f'std_out_{window[0]}_{window[1]}'] = gb_out['amount'].std()
            node_features[f'max_out_{window[0]}_{window[1]}'] = gb_out['amount'].max()
            node_features[f'min_out_{window[0]}_{window[1]}'] = gb_out['amount'].min()
            node_features[f'count_out_{window[0]}_{window[1]}'] = gb_out['amount'].count()
            node_features[f'count_unique_out_{window[0]}_{window[1]}'] = gb_out['counterpart'].nunique()
        # calculate non window related features
        df_combined = pd.concat([df_in[['account', 'days_in_bank', 'n_phone_changes', 'is_sar']], df_out[['account', 'days_in_bank', 'n_phone_changes', 'is_sar']]])
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
        df_nodes = df_nodes[df_nodes['bank'] == self.bank] if self.bank is not None else df_nodes
        # if any value is nan, there was no transaction in the window for that account and hence the feature should be 0
        df_nodes = df_nodes.fillna(0.0).infer_objects(copy=False)
        # check if there is any missing values
        assert df_nodes.isnull().sum().sum() == 0, 'There are missing values in the node features'
        # check if there are any negative values in all comuns except the bank column
        assert (df_nodes.drop(columns='bank') < 0).sum().sum() == 0, 'There are negative values in the node features'
        return df_nodes
    
    
    def extract_edge_list(self, df: pd.DataFrame, start_step, end_step) -> pd.DataFrame:
        """
        Extract unique edge list (graph structure) from network transactions.

        Returns minimal edge dataframe with only src, dst columns.
        Graph structure is purely topological - labels and features are separate.
        """
        # Filter network transactions only (exclude source/sink)
        df_network = df[(df['bankOrig'] != 'source') & (df['bankDest'] != 'sink')]

        # Filter by bank if specified
        if self.bank is not None:
            df_network = df_network[(df_network['bankOrig'] == self.bank) & (df_network['bankDest'] == self.bank)]

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
        Randomly splits SAR and normal nodes into non-overlapping sets.
        """
        n_nodes = len(df_nodes)
        train_mask = np.zeros(n_nodes, dtype=bool)
        val_mask = np.zeros(n_nodes, dtype=bool)
        test_mask = np.zeros(n_nodes, dtype=bool)

        # Split SAR nodes
        sar_indices = np.where(df_nodes['is_sar'] == 1)[0]
        np.random.seed(self.seed)
        sar_indices = np.random.permutation(sar_indices)

        n_sar = len(sar_indices)
        n_sar_train = int(self.train_label_fraction * n_sar)
        n_sar_val = int(self.val_label_fraction * n_sar)

        sar_train = sar_indices[:n_sar_train]
        sar_val = sar_indices[n_sar_train:n_sar_train + n_sar_val]
        sar_test = sar_indices[n_sar_train + n_sar_val:]

        # Split normal nodes
        normal_indices = np.where(df_nodes['is_sar'] == 0)[0]
        normal_indices = np.random.permutation(normal_indices)

        n_normal = len(normal_indices)
        n_normal_train = int(self.train_label_fraction * n_normal)
        n_normal_val = int(self.val_label_fraction * n_normal)

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
            print(f"\nTransductive label splitting:")
            print(f"  Total nodes: {n_nodes}, SAR nodes: {n_sar}, Normal nodes: {n_normal}")
            print(f"  Train: {len(sar_train)} SAR + {len(normal_train)} normal = {train_mask.sum()}")
            print(f"  Val:   {len(sar_val)} SAR + {len(normal_val)} normal = {val_mask.sum()}")
            print(f"  Test:  {len(sar_test)} SAR + {len(normal_test)} normal = {test_mask.sum()}")

        return df_nodes


    def preprocess_transductive(self, df: pd.DataFrame):
        """Preprocess for transductive learning (same time window, split labels)."""
        df_window = df[(df['step'] >= self.train_start_step) & (df['step'] <= self.train_end_step)]

        # Compute features once (same graph for all splits)
        df_nodes = self.cal_node_features(df_window, self.train_start_step, self.train_end_step)
        df_nodes = self.create_transductive_masks(df_nodes)

        df_edges = self.extract_edge_list(df_window, self.train_start_step, self.train_end_step)

        # Optionally compute edge features
        if self.include_edges:
            df_network = df_window[(df_window['bankOrig'] != 'source') & (df_window['bankDest'] != 'sink')]
            if self.bank is not None:
                df_network = df_network[(df_network['bankOrig'] == self.bank) & (df_network['bankDest'] == self.bank)]

            df_edge_features = self.cal_edge_features(df=df_network, start_step=self.train_start_step,
                                                     end_step=self.train_end_step, directional=True)
            df_edges = df_edges.merge(df_edge_features, on=['src', 'dst'], how='left')

        # Return same graph for all splits
        return {
            'trainset_nodes': df_nodes,
            'trainset_edges': df_edges,
            'valset_nodes': df_nodes,
            'valset_edges': df_edges,
            'testset_nodes': df_nodes,
            'testset_edges': df_edges
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
        if self.include_edges:
            df_train_network = df_train[(df_train['bankOrig'] != 'source') & (df_train['bankDest'] != 'sink')]
            df_val_network = df_val[(df_val['bankOrig'] != 'source') & (df_val['bankDest'] != 'sink')]
            df_test_network = df_test[(df_test['bankOrig'] != 'source') & (df_test['bankDest'] != 'sink')]

            if self.bank is not None:
                df_train_network = df_train_network[(df_train_network['bankOrig'] == self.bank) & (df_train_network['bankDest'] == self.bank)]
                df_val_network = df_val_network[(df_val_network['bankOrig'] == self.bank) & (df_val_network['bankDest'] == self.bank)]
                df_test_network = df_test_network[(df_test_network['bankOrig'] == self.bank) & (df_test_network['bankDest'] == self.bank)]

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
