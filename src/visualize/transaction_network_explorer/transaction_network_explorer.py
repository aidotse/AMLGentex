import pandas as pd
import numpy as np
import holoviews as hv
import panel as pn
import datashader as ds
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

class TransactionNetwork():
    
    def __init__(self, path:str) -> None:
        self.path = path  # Store path for later use
        self.df = self.load_data(path)
        self.df_nodes, self.df_edges = self.format_data(self.df)
        self.BANK_IDS = self.df_nodes['bank'].unique().tolist()
        self.N_ACCOUNTS = len(self.df_nodes)
        n_legit_illicit = self.df_edges['is_sar'].value_counts()
        self.N_LEGIT_TXS = n_legit_illicit[0]
        self.N_LAUND_TXS = n_legit_illicit[1] if 1 in n_legit_illicit else 0
        self.START_STEP = int(self.df_edges['step'].min())
        self.END_STEP = int(self.df_edges['step'].max()) + 1
        self.N_STEPS = self.END_STEP - self.START_STEP + 1
        self.LEGIT_MODEL_IDS = self.df_edges[self.df_edges['is_sar']==0]['model_type'].unique()
        self.LAUND_MODEL_IDS = self.df_edges[self.df_edges['is_sar']==1]['model_type'].unique()
        self.HOMOPHILY_EDGE, self.HOMOPHILY_NODE, self.HOMOPHILY_CLASS = self.calc_homophily()

        # Load spatial simulation data to get actual pattern type mappings
        self.legitimate_type_map, self.LEGIT_MODEL_NAMES = self._build_normal_model_map()
        self.laundering_type_map, self.LAUND_MODEL_NAMES = self._build_alert_model_map()

        # Load preprocessed features if available
        self.df_features = self._load_preprocessed_features()

        self.nodes = hv.Points(self.df_nodes, ['x', 'y'], ['name'])

        pass
    
    def load_data(self, path:str) -> pd.DataFrame:
        """Load transaction data from CSV or Parquet"""
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        df = df.loc[df['type']!='CASH']
        return df

    def _build_normal_model_map(self):
        """Build mapping from normal pattern names to model IDs based on actual data"""
        from src.utils.pattern_types import NORMAL_ID_TO_NAME

        # Use centralized pattern type mappings
        type_to_ids = {}
        available_types = set()

        # Get unique normal model types that appear in the data
        normal_model_types = self.df[self.df['isSAR'] == 0]['modelType'].unique()

        # Build mapping from pattern name to model type IDs
        for model_type_id in normal_model_types:
            if model_type_id in NORMAL_ID_TO_NAME:
                pattern_name = NORMAL_ID_TO_NAME[model_type_id]
                available_types.add(pattern_name)
                if pattern_name not in type_to_ids:
                    type_to_ids[pattern_name] = []
                type_to_ids[pattern_name].append(model_type_id)
            elif model_type_id == 0:
                # Include generic behaviors
                available_types.add("generic")
                if "generic" not in type_to_ids:
                    type_to_ids["generic"] = []
                type_to_ids["generic"].append(model_type_id)

        # Return mapping and sorted available types
        return type_to_ids, sorted(available_types)

    def _build_alert_model_map(self):
        """Build mapping from SAR pattern names to model IDs based on actual data"""
        from src.utils.pattern_types import SAR_ID_TO_NAME

        # Use centralized pattern type mappings
        type_to_ids = {}
        available_types = set()

        # Get unique SAR model types that appear in the data
        sar_model_types = self.df[self.df['isSAR'] == 1]['modelType'].unique()

        # Build mapping from pattern name to model type IDs
        for model_type_id in sar_model_types:
            if model_type_id in SAR_ID_TO_NAME:
                pattern_name = SAR_ID_TO_NAME[model_type_id]
                available_types.add(pattern_name)
                if pattern_name not in type_to_ids:
                    type_to_ids[pattern_name] = []
                type_to_ids[pattern_name].append(model_type_id)

        # Return mapping and sorted available types
        return type_to_ids, sorted(available_types)

    def _load_preprocessed_features(self):
        """Load preprocessed features if available"""
        import os
        # Try to load preprocessed features from centralized training set
        base_path = os.path.dirname(self.path)  # temporal/
        experiment_path = os.path.dirname(base_path)  # experiment root
        features_path = os.path.join(experiment_path, 'preprocessed', 'centralized', 'trainset_nodes.parquet')

        if os.path.exists(features_path):
            try:
                df_features = pd.read_parquet(features_path)
                # Keep only feature columns (exclude metadata)
                feature_cols = [col for col in df_features.columns
                               if col not in ['account', 'bank', 'is_sar', 'train_mask', 'val_mask', 'test_mask']]
                print(f"Loaded {len(feature_cols)} preprocessed features for {len(df_features)} accounts")
                return df_features
            except Exception as e:
                print(f"Warning: Could not load preprocessed features: {e}")
                return None
        else:
            print(f"Warning: Preprocessed features not found at {features_path}")
            return None

    def _add_plot_info(self, plot, title, description, extra_button=None):
        """
        Wrap a plot with a title and info tooltip

        Args:
            plot: HoloViews plot object
            title: Short title for the plot
            description: Detailed description shown in tooltip
            extra_button: Optional additional button to add to header (e.g., settings)

        Returns:
            Panel Column with title/info and plot
        """
        # Create info button with tooltip
        info_button = pn.widgets.TooltipIcon(value=description)

        # Create header with title, info icon, and optional extra button
        header_items = [
            pn.pane.Markdown(f"**{title}**", margin=(5, 5, 0, 5), width=200),
            info_button
        ]

        if extra_button is not None:
            header_items.append(extra_button)

        header = pn.Row(
            *header_items,
            margin=(0, 0, 0, 0)
        )

        # Apply consistent sizing to plots (280x180 for 2x4 grid)
        plot_pane = pn.pane.HoloViews(plot, width=280, height=180)

        # Wrap plot with header
        return pn.Column(header, plot_pane, margin=5, width=290)

    def format_data(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df['nameOrig']!=-2]
        df = df.loc[df['nameDest']!=-1]
        df.reset_index(inplace=True, drop=True)
        
        df1 = df[['nameOrig', 'bankOrig']].rename(columns={'nameOrig': 'name', 'bankOrig': 'bank'})
        df2 = df[['nameDest', 'bankDest']].rename(columns={'nameDest': 'name', 'bankDest': 'bank'})
        
        df_nodes = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        df_nodes = self.spread_nodes(df_nodes)
        
        df_edges = df[['nameOrig', 'nameDest', 'bankOrig', 'bankDest', 'step', 'amount', 'modelType', 'isSAR']].rename(columns={'nameOrig': 'source', 'nameDest': 'target', 'bankOrig': 'source_bank', 'bankDest': 'target_bank', 'modelType': 'model_type', 'isSAR': 'is_sar'})
        df_edges['x0'] = df_edges['source'].map(df_nodes.set_index('name')['x'])
        df_edges['y0'] = df_edges['source'].map(df_nodes.set_index('name')['y'])
        df_edges['x1'] = df_edges['target'].map(df_nodes.set_index('name')['x'])
        df_edges['y1'] = df_edges['target'].map(df_nodes.set_index('name')['y'])
        # Removed: df_edges.loc[df_edges['is_sar']==1, 'model_type'] += 20
        # No need to offset SAR model IDs since we have is_sar flag to distinguish them
        
        df1 = df_edges[df_edges['is_sar']==1][['source', 'target']]
        df1 = pd.concat([df1['source'], df1['target']]).drop_duplicates()
        df_nodes['is_sar'] = df_nodes['name'].isin(df1).astype(int)
        
        return df_nodes, df_edges
    
    def spread_nodes(self, df:pd.DataFrame) -> pd.DataFrame:
        # spred nodes randomly
        rs = np.random.random(size=df.shape[0])
        ts = np.random.random(size=df.shape[0])*2*np.pi
        df['x'] = rs * np.cos(ts)
        df['y'] = rs * np.sin(ts)
        
        # spread nodes by bank
        unique_banks = df['bank'].unique()
        r = 1
        n = 3
        ts = [0, 120, 240]
        for i, bank in enumerate(unique_banks):
            if i % n == 0:
                r *= 1.6
                ts = [t+60 for t in ts]
            t = ts[i % n]
            df.loc[df['bank']==bank, 'x'] += r*np.cos(t*np.pi/180)
            df.loc[df['bank']==bank, 'y'] += r*np.sin(t*np.pi/180)
        
        return df
    
    def calc_homophily(self) -> float:
        # TODO: make this dynamic with streams?
        edges_sar = self.df_edges[self.df_edges['is_sar']==1]
        edges_normal = self.df_edges[self.df_edges['is_sar']==0]
        edges_normal = edges_normal[~edges_normal['source'].isin(edges_sar['source'])]
        edges_normal = edges_normal[~edges_normal['target'].isin(edges_sar['target'])]
        homophily_edge = (len(edges_normal) + len(edges_sar)) / len(self.df_edges)
        homophily_nodes = 0.0
        h_class_sum1 = [0.0, 0.0]
        h_class_sum2 = [0.0, 0.0]
        df1 = self.df_edges[['source', 'target', 'step', 'is_sar']].rename(columns={'source': 'name', 'target': 'counterpart'})
        df2 = self.df_edges[['source', 'target', 'step', 'is_sar']].rename(columns={'target': 'name', 'source': 'counterpart'})
        df = pd.concat([df1, df2]).reset_index(drop=True)
        gb = df.groupby('name')
        n_nodes = len(gb)
        n_neighbours = gb['counterpart'].nunique().tolist()
        labels = gb['is_sar'].max().tolist()
        n_sar = gb['is_sar'].sum().tolist()
        for i in range(n_nodes):
            n_similar = n_sar[i] if labels[i] == 1 else n_neighbours[i] - n_sar[i]
            homophily_nodes += n_similar / n_neighbours[i] / n_nodes
            h_class_sum1[labels[i]] += n_similar
            h_class_sum2[labels[i]] += n_neighbours[i]
        homophily_class = 0.0
        for i, (h_sum1, h_sum2) in enumerate(zip(h_class_sum1, h_class_sum2)):
            h = h_sum1 / h_sum2 if h_sum2 > 0 else 1
            homophily_class += 1/(len(h_class_sum1)-1) * max(h - len(self.df_nodes[self.df_nodes['is_sar']==i]) / n_nodes, 0)
        return homophily_edge, homophily_nodes, homophily_class
    
    def get_all_nodes(self):
        nodes = hv.Points(self.df_nodes, ['x', 'y'], ['name'])
        return nodes

    def select_nodes(self, banks, laundering_models, legitimate_models, steps, x_range, y_range):
        # Filter nodes by selected banks first
        nodes_in_banks = self.df_nodes[self.df_nodes['bank'].isin(banks)]['name'].tolist()

        # filter edges
        df = self.df_edges
        # AND logic - both endpoints must be in selected banks
        df = df[df['source'].isin(nodes_in_banks) & df['target'].isin(nodes_in_banks)]
        # Get all model IDs for selected pattern names
        laundering_model_ids = []
        for m in laundering_models:
            if m in self.laundering_type_map:
                laundering_model_ids.extend(self.laundering_type_map[m])
        legitimate_model_ids = []
        for m in legitimate_models:
            if m in self.legitimate_type_map:
                legitimate_model_ids.extend(self.legitimate_type_map[m])

        # Filter by model types AND is_sar flag (since model IDs can overlap)
        df_filtered_parts = []
        if laundering_model_ids:
            df_sar = df[(df['model_type'].isin(laundering_model_ids)) & (df['is_sar'] == 1)]
            df_filtered_parts.append(df_sar)
        if legitimate_model_ids:
            df_normal = df[(df['model_type'].isin(legitimate_model_ids)) & (df['is_sar'] == 0)]
            df_filtered_parts.append(df_normal)

        df = pd.concat(df_filtered_parts, ignore_index=True) if df_filtered_parts else pd.DataFrame()
        df = df[(df['step'] >= steps[0]) & (df['step'] <= steps[1])]
        names = set(df['source'].unique().tolist() + df['target'].unique().tolist())
        df = self.df_nodes[self.df_nodes['name'].isin(names)]
        names_legit = df[df['is_sar']==0]['name'].tolist()
        names_illicit = df[df['is_sar']==1]['name'].tolist()
        nodes_legit = self.nodes.select(name=names_legit).opts(color='green', size=10, alpha=1.0)
        nodes_illicit = self.nodes.select(name=names_illicit).opts(color='red', size=10, alpha=1.0)
        return nodes_legit * nodes_illicit
     
    def get_edges(self, df:pd.DataFrame, x_range, y_range):
        """
        Returns an image of the edges in the transaction network within the specified x and y ranges.

        Parameters:
        df (pd.DataFrame): A dataframe containing the edges in the transaction network.
        x_range (): TODO 
        y_range (): TODO
        
        Returns:
        hv.Image: An image of the edges in the transaction network within the specified x and y ranges.
        """
        df_legit = df[df['is_sar']==0]
        df_illicit = df[df['is_sar']==1]
        if df_legit.empty:
            cmap=['#FFFFFF', '#FF0000']
        elif df_illicit.empty:
            cmap=['#FFFFFF', '#000000']
        else:
            cmap=['#FFFFFF', '#000000', '#FF0000']
        edges = ds.Canvas(
            plot_width=600, 
            plot_height=600, 
            x_range=x_range,
            y_range=y_range,
        ).line(source=df_legit, x=['x0', 'x1'], y=['y0', 'y1'], axis=1)
        edges_illicit = ds.Canvas(
            plot_width=600, 
            plot_height=600, 
            x_range=x_range,
            y_range=y_range,
        ).line(source=df_illicit, x=['x0', 'x1'], y=['y0', 'y1'], axis=1)
        edges.data = 0.5*edges.data + edges_illicit.data
        edges = hv.Image(edges).opts(cmap=cmap)
        return edges
    
    def select_edges(self, banks, laundering_models, legitimate_models, steps, x_range, y_range):
        # Filter nodes by selected banks first
        nodes_in_banks = self.df_nodes[self.df_nodes['bank'].isin(banks)]['name'].tolist()

        # filter edges
        df = self.df_edges
        # AND logic - both endpoints must be in selected banks
        df = df[df['source'].isin(nodes_in_banks) & df['target'].isin(nodes_in_banks)]
        # Get all model IDs for selected pattern names
        laundering_model_ids = []
        for m in laundering_models:
            if m in self.laundering_type_map:
                laundering_model_ids.extend(self.laundering_type_map[m])
        legitimate_model_ids = []
        for m in legitimate_models:
            if m in self.legitimate_type_map:
                legitimate_model_ids.extend(self.legitimate_type_map[m])

        # Filter by model types AND is_sar flag (since model IDs can overlap)
        df_filtered_parts = []
        if laundering_model_ids:
            df_sar = df[(df['model_type'].isin(laundering_model_ids)) & (df['is_sar'] == 1)]
            df_filtered_parts.append(df_sar)
        if legitimate_model_ids:
            df_normal = df[(df['model_type'].isin(legitimate_model_ids)) & (df['is_sar'] == 0)]
            df_filtered_parts.append(df_normal)

        df = pd.concat(df_filtered_parts, ignore_index=True) if df_filtered_parts else pd.DataFrame()
        df = df[(df['step'] >= steps[0]) & (df['step'] <= steps[1])]
        edges = self.get_edges(df, x_range, y_range)
        return edges

    def update_graph(self, banks, laundering_models, legitimate_models, steps, x_range, y_range): # not used
        # Filter nodes by selected banks first
        nodes_in_banks = self.df_nodes[self.df_nodes['bank'].isin(banks)]['name'].tolist()

        # filter edges
        df = self.df_edges
        # AND logic - both endpoints must be in selected banks
        df = df[df['source'].isin(nodes_in_banks) & df['target'].isin(nodes_in_banks)]
        # Get all model IDs for selected pattern names
        laundering_model_ids = []
        for m in laundering_models:
            if m in self.laundering_type_map:
                laundering_model_ids.extend(self.laundering_type_map[m])
        legitimate_model_ids = []
        for m in legitimate_models:
            if m in self.legitimate_type_map:
                legitimate_model_ids.extend(self.legitimate_type_map[m])

        # Filter by model types AND is_sar flag (since model IDs can overlap)
        df_filtered_parts = []
        if laundering_model_ids:
            df_sar = df[(df['model_type'].isin(laundering_model_ids)) & (df['is_sar'] == 1)]
            df_filtered_parts.append(df_sar)
        if legitimate_model_ids:
            df_normal = df[(df['model_type'].isin(legitimate_model_ids)) & (df['is_sar'] == 0)]
            df_filtered_parts.append(df_normal)

        df = pd.concat(df_filtered_parts, ignore_index=True) if df_filtered_parts else pd.DataFrame()
        df = df[(df['step'] >= steps[0]) & (df['step'] <= steps[1])]
        names = set(df['source'].unique().tolist() + df['target'].unique().tolist())
        nodes = self.nodes.select(name=names)
        edges = self.get_edges(df[df['is_sar']==0], x_range, y_range)
        return edges * nodes

    def get_balances(self, index):
        names = self.df_nodes.loc[index, 'name'].tolist()

        # For each account, create a proper time series of balance over steps
        curves = {}
        for i, name in enumerate(names):
            # Get all transactions where this account is involved
            df1 = self.df[self.df['nameOrig'] == name][['step', 'newbalanceOrig']].rename(columns={'newbalanceOrig': 'balance'})
            df2 = self.df[self.df['nameDest'] == name][['step', 'newbalanceDest']].rename(columns={'newbalanceDest': 'balance'})

            # Combine and sort by step
            account_df = pd.concat([df1, df2])
            if len(account_df) == 0:
                continue

            # Sort by step
            account_df = account_df.sort_values('step').reset_index(drop=True)

            # For each step, keep only the last balance value (after all transactions in that step)
            # This removes the "jumping" within a single step
            account_df = account_df.groupby('step', as_index=False).last()

            # Create a proper time series with step interpolation
            # This shows balance staying constant between transactions
            steps = account_df['step'].values
            balances = account_df['balance'].values

            # Add initial point at step 0 if first transaction is later
            if steps[0] > 0:
                # Try to get initial balance from oldbalance fields
                first_tx_orig = self.df[(self.df['nameOrig'] == name) & (self.df['step'] == steps[0])]
                first_tx_dest = self.df[(self.df['nameDest'] == name) & (self.df['step'] == steps[0])]

                if len(first_tx_orig) > 0:
                    initial_balance = first_tx_orig['oldbalanceOrig'].iloc[0]
                elif len(first_tx_dest) > 0:
                    initial_balance = first_tx_dest['oldbalanceDest'].iloc[0]
                else:
                    initial_balance = balances[0]  # Fallback

                steps = np.insert(steps, 0, 0)
                balances = np.insert(balances, 0, initial_balance)

            # Create curve with step interpolation
            curves[i] = hv.Curve((steps, balances), 'step', 'balance').opts(
                interpolation='steps-post'  # Step interpolation for proper time series
            )

        if curves:
            curves = hv.NdOverlay(curves).opts(
                hv.opts.Curve(xlim=(0, 365), ylim=(0, 100000), xlabel='step', ylabel='balance')
            )
        else:
            curves = hv.NdOverlay({0: hv.Curve(data=sorted(zip([0], [0])))}).opts(
                hv.opts.Curve(xlim=(0, 365), ylim=(0, 100000), xlabel='step', ylabel='balance')
            )
        return curves.opts(shared_axes=False, show_legend=False)
    
    def get_amount_hist(self, df:pd.DataFrame, bins:int=20):
        # Adjust bins if we have fewer unique values
        n_unique = df['amount'].nunique()
        actual_bins = min(bins, max(1, n_unique))
        vc = df['amount'].value_counts(bins=actual_bins)
        vc.sort_index(inplace=True)
        x_ls = vc.index.left
        x_rs = vc.index.right
        y_bs = [1]*len(vc)
        y_ts = vc.values
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append([x_l, y_b, x_r, y_t])
        # Dynamic axis scaling
        y_max = max(y_ts) * 2.0 if len(y_ts) > 0 else None
        x_min = min(x_ls) * 0.9 if len(x_ls) > 0 else 0
        x_max = max(x_rs) * 1.1 if len(x_rs) > 0 else None
        histogram = hv.Rectangles(rects).opts(
            shared_axes=False,
            xlabel='amount',
            ylabel='count',
            logy=True,
            xlim=(x_min, x_max),
            ylim=(1, y_max),
            width=250,
            height=200
        )
        return histogram
    
    def get_indegree_hist(self, df:pd.DataFrame, bins:int=20):
        # Filter out sink/source nodes (-1, -2)
        df_filtered = df[~df['target'].isin([-1, -2])]
        # Count unique sources for each target (true indegree)
        vc = df_filtered.groupby('target')['source'].nunique()

        # Adjust bins if we have fewer unique values
        n_unique = vc.nunique()
        actual_bins = min(bins, max(1, n_unique))
        vcc = vc.value_counts(bins=actual_bins)
        vcc.sort_index(inplace=True)
        x_ls = vcc.index.left
        x_rs = vcc.index.right
        y_bs = [1]*len(vcc)
        y_ts = vcc.values
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        # Dynamic axis scaling
        y_max = max(y_ts) * 2.0 if len(y_ts) > 0 else None
        x_min = min(x_ls) * 0.9 if len(x_ls) > 0 else 0
        x_max = max(x_rs) * 1.1 if len(x_rs) > 0 else None
        histogram = hv.Rectangles(rects).opts(
            shared_axes=False,
            xlabel='indegree',
            ylabel='count',
            logy=True,
            xlim=(x_min, x_max),
            ylim=(1, y_max),
            width=250,
            height=200
        )
        return histogram

    def get_outdegree_hist(self, df:pd.DataFrame, bins:int=20):
        # Filter out sink/source nodes (-1, -2)
        df_filtered = df[~df['source'].isin([-1, -2])]
        # Count unique targets for each source (true outdegree)
        vc = df_filtered.groupby('source')['target'].nunique()
        # Adjust bins if we have fewer unique values
        n_unique = vc.nunique()
        actual_bins = min(bins, max(1, n_unique))
        vcc = vc.value_counts(bins=actual_bins)
        vcc.sort_index(inplace=True)
        x_ls = vcc.index.left
        x_rs = vcc.index.right
        y_bs = [1]*len(vcc)
        y_ts = vcc.values
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        # Dynamic axis scaling
        y_max = max(y_ts) * 2.0 if len(y_ts) > 0 else None
        x_min = min(x_ls) * 0.9 if len(x_ls) > 0 else 0
        x_max = max(x_rs) * 1.1 if len(x_rs) > 0 else None
        histogram = hv.Rectangles(rects).opts(
            shared_axes=False,
            xlabel='outdegree',
            ylabel='count',
            logy=True,
            xlim=(x_min, x_max),
            ylim=(1, y_max),
            width=250,
            height=200
        )
        return histogram

    def get_n_payments_hist(self, df:pd.DataFrame):
        # Count transactions per individual step
        step_counts = df.groupby('step').size()

        if len(step_counts) == 0:
            return hv.Rectangles([]).opts(
                shared_axes=False,
                xlabel='step',
                ylabel='transactions per step',
                width=250,
                height=200
            )

        # Create rectangles for each step
        rects = []
        bar_width = 0.8  # Width of each bar
        for step, count in step_counts.items():
            x_l = step - bar_width / 2
            x_r = step + bar_width / 2
            rects.append((x_l, 0, x_r, count))

        # Dynamic axis scaling
        start = df['step'].min()
        end = df['step'].max()
        y_max = step_counts.max() * 1.1 if len(step_counts) > 0 else None
        x_min = start - 1
        x_max = end + 1

        histogram = hv.Rectangles(rects).opts(
            shared_axes=False,
            xlabel='step',
            ylabel='transactions per step',
            logy=False,
            xlim=(x_min, x_max),
            ylim=(0, y_max),
            width=250,
            height=200
        )
        return histogram

    def get_volume_hist(self, df:pd.DataFrame, interval:int=28):
        start = df['step'].min()
        end = df['step'].max()
        x_ls = []
        x_rs = []
        y_ts = []
        for i in range(start, end, interval):
            x_ls.append(i+interval*0.2)
            x_rs.append(i+interval*0.8)
            y_ts.append(sum(df[(df['step'] >= i) & (df['step'] <= i+interval)]['amount']))
        y_bs = [0]*len(y_ts)
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        # Dynamic axis scaling
        y_max = max(y_ts) * 1.1 if len(y_ts) > 0 and max(y_ts) > 0 else None
        x_min = start - interval * 0.1 if len(x_ls) > 0 else 0
        x_max = end + interval * 0.1 if len(x_rs) > 0 else None
        histogram = hv.Rectangles(rects).opts(shared_axes=False, xlabel=f'step', ylabel=f'volume per {interval} steps', logy=False, xlim=(x_min, x_max), ylim=(0, y_max))
        return histogram

    def get_n_users_hist(self, df:pd.DataFrame, interval:int=28):
        start = df['step'].min()
        end = df['step'].max()
        x_ls = []
        x_rs = []
        y_ts = []
        for i in range(start, end, interval):
            x_ls.append(i+interval*0.2)
            x_rs.append(i+interval*0.8)
            y_ts.append(len(set(
                df[(df['step'] >= i) & (df['step'] <= i+interval)]['source'].unique().tolist() +
                df[(df['step'] >= i) & (df['step'] <= i+interval)]['target'].unique().tolist()
            )))
        y_bs = [1E-1]*len(y_ts)
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        histogram = hv.Rectangles(rects).opts(shared_axes=False, xlabel=f'step', ylabel='number of active users per month', logy=False)
        return histogram

    def get_umap_plot(self, df:pd.DataFrame, umap_params=None):
        """
        Create UMAP 2D visualization of accounts colored by SAR status.
        Uses preprocessed feature vectors from the feature engineering pipeline.

        Args:
            df: Filtered edge DataFrame
            umap_params: Dictionary with UMAP hyperparameters (n_neighbors, min_dist, metric)

        Returns:
            Holoviews scatter plot or empty plot if insufficient data
        """
        # Default UMAP parameters
        if umap_params is None:
            umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'}
        if not UMAP_AVAILABLE:
            return hv.Scatter([]).opts(
                shared_axes=False,
                xlabel='UMAP not available',
                ylabel='',
                width=220,
                height=160,
                title='Install umap-learn'
            )

        if len(df) == 0:
            return hv.Scatter([]).opts(
                shared_axes=False,
                xlabel='UMAP 1',
                ylabel='UMAP 2',
                width=220,
                height=160,
                title='No data'
            )

        # Extract accounts from filtered edges
        accounts = set(df['source'].unique().tolist() + df['target'].unique().tolist())

        # Need at least 2 accounts for UMAP
        if len(accounts) < 2:
            return hv.Scatter([]).opts(
                shared_axes=False,
                xlabel='UMAP 1',
                ylabel='UMAP 2',
                width=220,
                height=160,
                title='Insufficient data'
            )

        # Use preprocessed features
        if self.df_features is None:
            return hv.Scatter([]).opts(
                shared_axes=False,
                xlabel='UMAP 1',
                ylabel='UMAP 2',
                width=220,
                height=160,
                title='No preprocessed features'
            )

        # Filter to accounts in current view
        df_view = self.df_features[self.df_features['account'].isin(accounts)].copy()

        if len(df_view) < 2:
            return hv.Scatter([]).opts(
                shared_axes=False,
                xlabel='UMAP 1',
                ylabel='UMAP 2',
                width=220,
                height=160,
                title='Insufficient data'
            )

        # Get feature columns (exclude metadata)
        feature_cols = [col for col in df_view.columns
                       if col not in ['account', 'bank', 'is_sar', 'train_mask', 'val_mask', 'test_mask']]

        X = df_view[feature_cols].values
        y = df_view['is_sar'].values
        account_ids = df_view['account'].values

        # Handle case with all zeros or constant features
        if X.std() == 0 or np.all(X == X[0]):
            return hv.Scatter([]).opts(
                shared_axes=False,
                xlabel='UMAP 1',
                ylabel='UMAP 2',
                width=220,
                height=160,
                title='No feature variance'
            )

        # Normalize features (important for UMAP with mixed scales)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Compute UMAP embedding with user-specified parameters
        try:
            n_neighbors = min(umap_params['n_neighbors'], len(X)-1)
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=n_neighbors,
                min_dist=umap_params['min_dist'],
                metric=umap_params['metric']
            )
            embedding = reducer.fit_transform(X_normalized)

            # Create scatter data with colors
            scatter_data = []
            for i, (x, y_coord) in enumerate(embedding):
                color = 'red' if y[i] == 1 else 'lightblue'
                scatter_data.append((x, y_coord, color, int(account_ids[i]), int(y[i])))

            # Create scatter plot
            scatter = hv.Scatter(
                scatter_data,
                kdims=['UMAP 1', 'UMAP 2'],
                vdims=['color', 'account', 'is_sar']
            )

            return scatter.opts(
                shared_axes=False,
                color='color',
                size=6,
                alpha=0.7,
                width=250,
                height=200,
                title='Account UMAP',
                fontsize={'title': 11},
                tools=['hover'],
                hover_tooltips=[('Account', '@account'), ('SAR', '@is_sar')]
            )

        except Exception as e:
            # Return empty plot with error message
            return hv.Scatter([]).opts(
                shared_axes=False,
                xlabel='UMAP 1',
                ylabel='UMAP 2',
                width=220,
                height=160,
                title=f'UMAP failed'
            )

    def get_pattern_timing_plot(self, df:pd.DataFrame):
        """
        Create box plots showing duration (in steps) for normal vs alert patterns.

        Duration: Time span from first to last transaction in each pattern instance
        Burstiness: Coefficient of variation of inter-transaction intervals
                   CV = std(intervals) / mean(intervals)
                   Higher values = more clustered/bursty transactions

        Note: Requires at least 3 transactions per pattern to calculate meaningful statistics.

        Args:
            df: Filtered edge DataFrame (from df_edges)

        Returns:
            Holoviews BoxWhisker plot comparing actual duration values
        """
        if len(df) == 0:
            return hv.BoxWhisker([]).opts(
                shared_axes=False,
                xlabel='',
                ylabel='',
                width=250,
                height=200,
                title='No data'
            )

        # Match filtered edges back to original data to get patternID
        # df has: source, target, step, model_type, is_sar
        # self.df has: nameOrig, nameDest, step, modelType, patternID, etc.

        # Create a lookup key to match transactions
        df_lookup = df.copy()
        df_lookup['lookup_key'] = (
            df_lookup['source'].astype(str) + '_' +
            df_lookup['target'].astype(str) + '_' +
            df_lookup['step'].astype(str)
        )

        # Create lookup in original data
        df_orig = self.df.copy()
        df_orig['lookup_key'] = (
            df_orig['nameOrig'].astype(str) + '_' +
            df_orig['nameDest'].astype(str) + '_' +
            df_orig['step'].astype(str)
        )

        # Merge to get patternID
        df_with_pattern = df_lookup.merge(
            df_orig[['lookup_key', 'patternID']],
            on='lookup_key',
            how='left'
        )

        # Calculate pattern-level statistics
        pattern_stats = []

        # For SAR patterns, group by patternID (each alert is a unique pattern instance)
        sar_patterns = df_with_pattern[df_with_pattern['is_sar'] == 1].groupby('patternID')
        for pattern_id, group in sar_patterns:
            if pd.notna(pattern_id) and len(group) >= 3:  # Need at least 3 txs for 2+ intervals
                    steps = sorted(group['step'].values)
                    duration = steps[-1] - steps[0]  # Steps between first and last

                    # Calculate burstiness (coefficient of variation of intervals)
                    intervals = np.diff(steps)
                    if len(intervals) > 1 and intervals.mean() > 0:
                        burstiness = intervals.std() / intervals.mean()
                    else:
                        burstiness = 0

                    pattern_stats.append({
                        'type': 'Alert',
                        'duration': duration,
                        'burstiness': burstiness
                    })

        # For normal patterns, group by patternID (unified structure with SAR patterns)
        # Each patternID represents a distinct pattern instance
        # Exclude generic transactions (patternID = -1)
        normal_data = df_with_pattern[(df_with_pattern['is_sar'] == 0) & (df_with_pattern['patternID'] >= 0)].copy()
        normal_patterns = normal_data.groupby('patternID')
        for pattern_id, group in normal_patterns:
            if len(group) >= 3:  # Need at least 3 txs for 2+ intervals
                # Remove duplicates from merge
                steps = sorted(group['step'].unique())
                if len(steps) < 3:
                    continue

                duration = steps[-1] - steps[0]  # Steps between first and last

                # Calculate burstiness
                intervals = np.diff(steps)
                if len(intervals) > 1 and intervals.mean() > 0:
                    burstiness = intervals.std() / intervals.mean()
                else:
                    burstiness = 0

                pattern_stats.append({
                    'type': 'Normal',
                    'duration': duration,
                    'burstiness': burstiness
                })

        if len(pattern_stats) == 0:
            return hv.BoxWhisker([]).opts(
                shared_axes=False,
                xlabel='',
                ylabel='',
                width=250,
                height=200,
                title='Insufficient data'
            )

        # Create DataFrame for plotting
        stats_df = pd.DataFrame(pattern_stats)

        # Show duration in actual steps (not normalized)
        plot_data = [(row['type'], row['duration']) for _, row in stats_df.iterrows()]

        # Create box whisker plot
        boxwhisker = hv.BoxWhisker(
            plot_data,
            kdims=['Type'],
            vdims=['Duration']
        ).opts(
            shared_axes=False,
            width=250,
            height=200,
            title='Pattern Duration (steps)',
            fontsize={'title': 11},
            ylabel='Duration (steps)',
            box_fill_color=hv.dim('Type').categorize({'Normal': 'lightblue', 'Alert': 'lightcoral'})
        )

        return boxwhisker

    def get_pattern_concentration_plot(self, df:pd.DataFrame, q=0.8):
        """
        Create box plots showing burstiness via Wasserstein distance from uniform distribution.

        Wasserstein Distance: Measures "Earth Mover's Distance" between actual transaction
        timing and a uniform reference distribution over the allocated pattern duration.

        Algorithm:
        - For pattern with sorted times t₁ ≤ ... ≤ tₙ in allocated [start, end]
        - Normalize: uᵢ = (tᵢ - start)/(end - start) ∈ [0,1]
        - Create uniform reference: [0, 1/(n-1), 2/(n-1), ..., 1]
        - Distance = wasserstein_distance(actual, uniform)

        Interpretation:
        - Distance = 0: perfectly uniform distribution
        - Higher distance = more deviation from uniform = more bursty
        - Example: [0, 0, 0.1, 0.1, 1, 1] vs uniform → high distance (bursty)
        - Example: [0, 0.2, 0.4, 0.6, 0.8, 1] vs uniform → low distance (uniform)

        Note: Requires at least 3 transactions per pattern to calculate meaningful statistics.

        Args:
            df: Filtered edge DataFrame
            q: Not used (kept for API compatibility)

        Returns:
            Holoviews BoxWhisker plot comparing Wasserstein distances across pattern types
        """
        if len(df) == 0:
            return hv.BoxWhisker([]).opts(
                shared_axes=False,
                xlabel='',
                ylabel='',
                width=250,
                height=200,
                title='No data'
            )

        # Match filtered edges to original data for patternID
        df_lookup = df.copy()
        df_lookup['lookup_key'] = (
            df_lookup['source'].astype(str) + '_' +
            df_lookup['target'].astype(str) + '_' +
            df_lookup['step'].astype(str)
        )

        df_orig = self.df.copy()
        df_orig['lookup_key'] = (
            df_orig['nameOrig'].astype(str) + '_' +
            df_orig['nameDest'].astype(str) + '_' +
            df_orig['step'].astype(str)
        )

        df_with_pattern = df_lookup.merge(
            df_orig[['lookup_key', 'patternID']],
            on='lookup_key',
            how='left'
        )

        pattern_stats = []
        excluded_counts = {'Alert': 0, 'Normal': 0}  # Track patterns with < 3 transactions

        # SAR patterns: group by patternID
        if 'patternID' in df_with_pattern.columns:
            sar_patterns = df_with_pattern[df_with_pattern['is_sar'] == 1].groupby('patternID')
            for pattern_id, group in sar_patterns:
                if pd.notna(pattern_id):
                    if len(group) < 3:
                        excluded_counts['Alert'] += 1
                        continue
                    steps = sorted(group['step'].values)
                    n = len(steps)

                    # Normalize times to [0,1] based on effective duration
                    t1, tn = steps[0], steps[-1]
                    if tn > t1:
                        actual_normalized = np.array([(t - t1) / (tn - t1) for t in steps])
                    else:
                        # All at same step
                        actual_normalized = np.zeros(n)

                    # Create uniform reference: distribute n transactions evenly across unique steps
                    unique_steps = np.unique(actual_normalized)
                    n_unique = len(unique_steps)

                    # Uniform: each unique step gets approximately n/n_unique transactions
                    uniform_reference = []
                    txs_per_step = n // n_unique
                    remainder = n % n_unique

                    for i, step_val in enumerate(unique_steps):
                        # Distribute remainder evenly
                        count = txs_per_step + (1 if i < remainder else 0)
                        uniform_reference.extend([step_val] * count)

                    uniform_reference = np.array(sorted(uniform_reference))

                    # Calculate Wasserstein distance
                    distance = wasserstein_distance(actual_normalized, uniform_reference)

                    pattern_stats.append({
                        'type': 'Alert',
                        'burstiness': distance
                    })

        # Normal patterns: group by patternID (unified structure with SAR patterns)
        # Each patternID represents a distinct pattern instance
        # Exclude generic transactions (patternID = -1)
        normal_data = df_with_pattern[(df_with_pattern['is_sar'] == 0) & (df_with_pattern['patternID'] >= 0)].copy()
        normal_patterns = normal_data.groupby('patternID')
        for pattern_id, group in normal_patterns:
            if len(group) < 3:
                excluded_counts['Normal'] += 1
                continue

            # Get all transaction steps (not unique, since we care about clustering)
            steps = sorted(group['step'].values)
            n = len(steps)

            # Normalize times to [0,1] based on effective duration
            t1, tn = steps[0], steps[-1]
            if tn > t1:
                actual_normalized = np.array([(t - t1) / (tn - t1) for t in steps])
            else:
                # All at same step
                actual_normalized = np.zeros(n)

            # Create uniform reference: distribute n transactions evenly across unique steps
            unique_steps = np.unique(actual_normalized)
            n_unique = len(unique_steps)

            # Uniform: each unique step gets approximately n/n_unique transactions
            uniform_reference = []
            txs_per_step = n // n_unique
            remainder = n % n_unique

            for i, step_val in enumerate(unique_steps):
                # Distribute remainder evenly
                count = txs_per_step + (1 if i < remainder else 0)
                uniform_reference.extend([step_val] * count)

            uniform_reference = np.array(sorted(uniform_reference))

            # Calculate Wasserstein distance
            distance = wasserstein_distance(actual_normalized, uniform_reference)

            pattern_stats.append({
                'type': 'Normal',
                'burstiness': distance
            })

        if len(pattern_stats) == 0:
            return hv.Bars([]).opts(
                shared_axes=False,
                xlabel='',
                ylabel='',
                width=250,
                height=200,
                title='Insufficient data'
            )

        # Define burstiness thresholds
        # Low: 0-0.15 (nearly uniform)
        # Medium: 0.15-0.35 (moderate clustering)
        # High: >0.35 (strong clustering/highly bursty)
        LOW_THRESHOLD = 0.15
        HIGH_THRESHOLD = 0.35

        stats_df = pd.DataFrame(pattern_stats)

        # Categorize each pattern
        def categorize_burstiness(distance):
            if distance < LOW_THRESHOLD:
                return 'Low'
            elif distance < HIGH_THRESHOLD:
                return 'Medium'
            else:
                return 'High'

        stats_df['category'] = stats_df['burstiness'].apply(categorize_burstiness)

        # Count patterns in each category for each type and normalize to proportions
        count_data = []
        for pattern_type in ['Alert', 'Normal']:
            type_df = stats_df[stats_df['type'] == pattern_type]
            total_patterns = len(type_df)

            if total_patterns > 0:
                for category in ['Low', 'Medium', 'High']:
                    count = len(type_df[type_df['category'] == category])
                    proportion = count / total_patterns  # Normalize to [0, 1]
                    count_data.append({
                        'Type': pattern_type,
                        'Category': category,
                        'Proportion': proportion
                    })

        count_df = pd.DataFrame(count_data)

        # Calculate total patterns analyzed (included) and excluded
        total_included = len(pattern_stats)
        total_excluded = excluded_counts['Alert'] + excluded_counts['Normal']

        # Create title with exclusion info if any patterns were excluded
        if total_excluded > 0:
            title = f'Burstiness ({total_included} patterns, {total_excluded} excluded <3 txs)'
        else:
            title = f'Burstiness Categories ({total_included} patterns)'

        # Create grouped bar chart
        bars = hv.Bars(
            count_df,
            kdims=['Category', 'Type'],
            vdims=['Proportion']
        ).opts(
            shared_axes=False,
            width=250,
            height=200,
            title=title,
            fontsize={'title': 9},
            xlabel='Burstiness',
            ylabel='Proportion',
            ylim=(0, 1),
            color=hv.dim('Type').categorize({'Normal': 'lightblue', 'Alert': 'lightcoral'}),
            legend_position='top_right',
            show_legend=True,
            show_grid=True,
            xrotation=0,  # Keep x-axis labels horizontal
            multi_level=False  # Don't show Type labels on x-axis
        )

        return bars

    def update_hists(self, steps, banks, legitimate_models, laundering_models, x_range, y_range, include_interbank=False):
        # Filter nodes by selected banks first
        nodes_in_banks = self.df_nodes[self.df_nodes['bank'].isin(banks)]['name'].tolist()

        # filter edges
        df = self.df_edges
        # Filter based on inter-bank toggle
        if include_interbank:
            # OR logic - show transactions if either endpoint is in selected banks
            df = df[df['source'].isin(nodes_in_banks) | df['target'].isin(nodes_in_banks)]
        else:
            # AND logic - both endpoints must be in selected banks
            df = df[df['source'].isin(nodes_in_banks) & df['target'].isin(nodes_in_banks)]
        # Get all model IDs for selected pattern names
        laundering_model_ids = []
        for m in laundering_models:
            if m in self.laundering_type_map:
                laundering_model_ids.extend(self.laundering_type_map[m])
        legitimate_model_ids = []
        for m in legitimate_models:
            if m in self.legitimate_type_map:
                legitimate_model_ids.extend(self.legitimate_type_map[m])

        # Filter by model types AND is_sar flag (since model IDs can overlap)
        df_filtered_parts = []
        if laundering_model_ids:
            df_sar = df[(df['model_type'].isin(laundering_model_ids)) & (df['is_sar'] == 1)]
            df_filtered_parts.append(df_sar)
        if legitimate_model_ids:
            df_normal = df[(df['model_type'].isin(legitimate_model_ids)) & (df['is_sar'] == 0)]
            df_filtered_parts.append(df_normal)

        df = pd.concat(df_filtered_parts, ignore_index=True) if df_filtered_parts else pd.DataFrame()
        df = df[(df['step'] >= steps[0]) & (df['step'] <= steps[1])]

        # Return empty layout if no data after filtering
        if len(df) == 0:
            empty_hist = hv.Rectangles([]).opts(
                shared_axes=False,
                xlabel='No data',
                ylabel='',
                width=250,
                height=200
            )
            return (empty_hist + empty_hist + empty_hist + empty_hist + empty_hist + empty_hist).cols(3)

        amount_hist = self.get_amount_hist(df)
        indegree_hist = self.get_indegree_hist(df)
        outdegree_hist = self.get_outdegree_hist(df)
        n_payments_hist = self.get_n_payments_hist(df)
        volume_hist = self.get_volume_hist(df)
        n_users_hist = self.get_n_users_hist(df)
        return (amount_hist + indegree_hist + outdegree_hist + volume_hist + n_payments_hist + n_users_hist).cols(3)

    def update_hists_from_filtered_data(self, df, x_range, y_range, selected_nodes=None, umap_params=None, umap_button=None):
        """Update histograms with balance plot and UMAP visualization using pre-filtered data.

        This method accepts already-filtered data to avoid redundant filtering operations.

        Args:
            df: Pre-filtered edge DataFrame
            x_range: X-axis range for visualization
            y_range: Y-axis range for visualization
            selected_nodes: List of selected node IDs for balance plot
            umap_params: Dict with UMAP hyperparameters
            umap_button: Optional button to add to UMAP plot header (for settings)
        """
        # Return empty layout if no data
        if len(df) == 0:
            empty_hist = hv.Rectangles([]).opts(
                shared_axes=False,
                xlabel='No data',
                ylabel='',
                width=250,
                height=200
            )
            return (empty_hist + empty_hist + empty_hist +
                    empty_hist + empty_hist + empty_hist +
                    empty_hist).cols(3)

        amount_hist = self.get_amount_hist(df)
        indegree_hist = self.get_indegree_hist(df)
        outdegree_hist = self.get_outdegree_hist(df)

        # Pattern timing plot (duration)
        timing_plot = self.get_pattern_timing_plot(df)

        # Pattern concentration plot (tightest window burstiness metric)
        concentration_plot = self.get_pattern_concentration_plot(df, q=0.8)

        # Balance plot for selected nodes
        if selected_nodes and len(selected_nodes) > 0:
            # Get indices for all selected nodes
            all_indices = []
            valid_nodes = []
            for node_id in selected_nodes:
                node_indices = self.df_nodes[self.df_nodes['name'] == node_id].index
                if len(node_indices) > 0:
                    all_indices.extend(node_indices.tolist())
                    valid_nodes.append(node_id)

            if len(all_indices) > 0:
                balance_plot = self.get_balances(all_indices)

                # Determine max balance for y-axis scaling
                max_balance = 0
                for node_id in valid_nodes:
                    df1 = self.df[self.df['nameOrig'] == node_id][['newbalanceOrig', 'oldbalanceOrig']]
                    df2 = self.df[self.df['nameDest'] == node_id][['newbalanceDest', 'oldbalanceDest']]

                    if len(df1) > 0:
                        max_balance = max(max_balance, df1['newbalanceOrig'].max(), df1['oldbalanceOrig'].max())
                    if len(df2) > 0:
                        max_balance = max(max_balance, df2['newbalanceDest'].max(), df2['oldbalanceDest'].max())

                if max_balance == 0:
                    max_balance = 100000

                # Get actual step range from filtered data
                min_step = df['step'].min() if len(df) > 0 else 0
                max_step = df['step'].max() if len(df) > 0 else self.END_STEP
                step_range = max_step - min_step
                x_min = max(0, min_step - step_range * 0.05)
                x_max = max_step + step_range * 0.05

                # Format title
                if len(valid_nodes) <= 5:
                    title = f'Balance: {", ".join(map(str, valid_nodes))}'
                else:
                    title = f'Balance: {len(valid_nodes)} accounts'

                balance_plot = balance_plot.opts(
                    hv.opts.Curve(
                        width=250,
                        height=200,
                        title=title,
                        fontsize={'title': 11},
                        xlim=(x_min, x_max),
                        ylim=(0, max_balance * 1.1)
                    )
                )
            else:
                balance_plot = hv.Curve([]).opts(
                    shared_axes=False,
                    xlabel='step',
                    ylabel='balance',
                    width=250,
                    height=200,
                    title='Click nodes to see balance',
                    fontsize={'title': 11}
                )
        else:
            balance_plot = hv.Curve([]).opts(
                shared_axes=False,
                xlabel='step',
                ylabel='balance',
                width=250,
                height=200,
                title='Click nodes to see balance',
                fontsize={'title': 11}
            )

        # Compute UMAP visualization
        umap_plot = self.get_umap_plot(df, umap_params)

        # Also get transaction count histogram
        n_payments_hist = self.get_n_payments_hist(df)

        # Wrap plots with titles and info tooltips
        amount_wrapped = self._add_plot_info(
            amount_hist,
            "Transaction Amount",
            "Distribution of transaction amounts (log scale). Shows how money flows vary in size."
        )

        n_payments_wrapped = self._add_plot_info(
            n_payments_hist,
            "Activity Over Time",
            "Number of transactions at each timestep. Peaks indicate periods of high activity."
        )

        outdegree_wrapped = self._add_plot_info(
            outdegree_hist,
            "Outdegree",
            "Number of unique recipients per account. Higher values indicate accounts sending to many others."
        )

        indegree_wrapped = self._add_plot_info(
            indegree_hist,
            "Indegree",
            "Number of unique senders per account. Higher values indicate accounts receiving from many others."
        )

        timing_wrapped = self._add_plot_info(
            timing_plot,
            "Pattern Duration",
            "Number of timesteps each pattern spans. SAR patterns may show tighter timing than normal activity."
        )

        concentration_wrapped = self._add_plot_info(
            concentration_plot,
            "Burstiness",
            "Counts patterns by burstiness level using Wasserstein distance from uniform distribution. "
            "Thresholds: Low (0-0.15) = nearly uniform, Medium (0.15-0.35) = moderate clustering, High (>0.35) = strong clustering/highly bursty."
        )

        umap_wrapped = self._add_plot_info(
            umap_plot,
            "UMAP Projection",
            "2D projection of account behavior features. Clusters may reveal distinct activity patterns. Red = SAR accounts, Blue = Normal.",
            extra_button=umap_button
        )

        balance_wrapped = self._add_plot_info(
            balance_plot,
            "Account Balance",
            "Balance over time for selected accounts. Click nodes in the graph to select. Shows financial trajectory."
        )

        # 2x4 grid layout:
        # Row 1: amount, transactions per step, outdegree, indegree
        # Row 2: timing, concentration, umap, balance
        return pn.GridBox(
            amount_wrapped, n_payments_wrapped, outdegree_wrapped, indegree_wrapped,
            timing_wrapped, concentration_wrapped, umap_wrapped, balance_wrapped,
            ncols=4,
            sizing_mode='stretch_both'
        )

    def update_hists_with_balance(self, steps, banks, legitimate_models, laundering_models, x_range, y_range, include_interbank=False, selected_nodes=None, umap_params=None):
        """Update histograms with balance plot and UMAP visualization"""
        # Filter nodes by selected banks first
        nodes_in_banks = self.df_nodes[self.df_nodes['bank'].isin(banks)]['name'].tolist()

        # filter edges
        df = self.df_edges
        # Filter based on inter-bank toggle
        if include_interbank:
            # OR logic - show transactions if either endpoint is in selected banks
            df = df[df['source'].isin(nodes_in_banks) | df['target'].isin(nodes_in_banks)]
        else:
            # AND logic - both endpoints must be in selected banks
            df = df[df['source'].isin(nodes_in_banks) & df['target'].isin(nodes_in_banks)]
        # Get all model IDs for selected pattern names
        laundering_model_ids = []
        for m in laundering_models:
            if m in self.laundering_type_map:
                laundering_model_ids.extend(self.laundering_type_map[m])
        legitimate_model_ids = []
        for m in legitimate_models:
            if m in self.legitimate_type_map:
                legitimate_model_ids.extend(self.legitimate_type_map[m])

        # Filter by model types AND is_sar flag (since model IDs can overlap)
        df_filtered_parts = []
        if laundering_model_ids:
            df_sar = df[(df['model_type'].isin(laundering_model_ids)) & (df['is_sar'] == 1)]
            df_filtered_parts.append(df_sar)
        if legitimate_model_ids:
            df_normal = df[(df['model_type'].isin(legitimate_model_ids)) & (df['is_sar'] == 0)]
            df_filtered_parts.append(df_normal)

        df = pd.concat(df_filtered_parts, ignore_index=True) if df_filtered_parts else pd.DataFrame()
        df = df[(df['step'] >= steps[0]) & (df['step'] <= steps[1])]

        # Delegate to the new method that works with pre-filtered data
        return self.update_hists_from_filtered_data(df, x_range, y_range, selected_nodes, umap_params)