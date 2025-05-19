import pandas as pd
import numpy as np
import holoviews as hv
import datashader as ds

class TransactionNetwork():
    
    def __init__(self, path:str) -> None:
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
        self.LEGIT_MODEL_NAMES = ['single', 'fan-out', 'fan-in', 'forward', 'mutal', 'periodical']
        self.LEGIT_MODEL_IDS = self.df_edges[self.df_edges['is_sar']==0]['model_type'].unique()
        self.LAUND_MODEL_NAMES = ['fan-out', 'fan-in', 'cycle', 'bipartite', 'stacked', 'random', 'scatter-gather', 'gather-scatter']
        self.LAUND_MODEL_IDS = self.df_edges[self.df_edges['is_sar']==1]['model_type'].unique()
        self.HOMOPHILY_EDGE, self.HOMOPHILY_NODE, self.HOMOPHILY_CLASS = self.calc_homophily()
        
        self.legitimate_type_map = {'single': 0, 'fan-out': 1, 'fan-in': 2,  'forward': 9, 'mutal': 10, 'periodical': 11}
        self.laundering_type_map = {'fan-out': 21, 'fan-in': 22, 'cycle': 23, 'bipartite': 24, 'stacked': 25, 'random': 26, 'scatter-gather': 27, 'gather-scatter': 28}
        
        self.nodes = hv.Points(self.df_nodes, ['x', 'y'], ['name'])
        
        pass
    
    def load_data(self, path:str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.loc[df['type']!='CASH']
        return df
    
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
        df_edges.loc[df_edges['is_sar']==1, 'model_type'] += 20
        
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
        # filter
        df = self.df_edges
        df = df[df['source_bank'].isin(banks) & df['target_bank'].isin(banks)]
        laundering_models = [self.laundering_type_map[m] for m in laundering_models]
        legitimate_models = [self.legitimate_type_map[m] for m in legitimate_models]
        df = df[df['model_type'].isin(laundering_models + legitimate_models)]
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
        # filter
        df = self.df_edges
        df = df[df['source_bank'].isin(banks) & df['target_bank'].isin(banks)]
        laundering_models = [self.laundering_type_map[m] for m in laundering_models]
        legitimate_models = [self.legitimate_type_map[m] for m in legitimate_models]
        df = df[df['model_type'].isin(laundering_models + legitimate_models)]
        df = df[(df['step'] >= steps[0]) & (df['step'] <= steps[1])]
        edges = self.get_edges(df, x_range, y_range)
        return edges

    def update_graph(self, banks, laundering_models, legitimate_models, steps, x_range, y_range): # not used
        df = self.df_edges
        df = df[df['source_bank'].isin(banks) & df['target_bank'].isin(banks)]
        laundering_models = [self.laundering_type_map[m] for m in laundering_models]
        legitimate_models = [self.legitimate_type_map[m] for m in legitimate_models]
        df = df[df['model_type'].isin(laundering_models + legitimate_models)]
        df = df[(df['step'] >= steps[0]) & (df['step'] <= steps[1])]
        names = set(df['source'].unique().tolist() + df['target'].unique().tolist())
        nodes = self.nodes.select(name=names)
        edges = self.get_edges(df[df['is_sar']==0], x_range, y_range)
        return edges * nodes

    def get_balances(self, index):
        names = self.df_nodes.loc[index, 'name'].tolist()
        # TODO: filter on steps aswell?
        
        '''
        df1 = self.df[self.df['nameOrig'].isin(names)][['nameOrig', 'step', 'amount']]
        df1 = df1.rename(columns={'nameOrig': 'name'})
        df2 = self.df[self.df['nameDest'].isin(names)][['nameDest', 'step', 'amount']]
        df2 = df2.rename(columns={'nameDest': 'name'})
        df1['amount'] = -df1['amount']
        df = pd.concat([df1, df2]).reset_index(drop=True)
        df = df.sort_values(by=['step'])
        gb = df.groupby('name')
        df['balance'] = gb['amount'].cumsum()
        '''
        
        df1 = self.df[self.df['nameOrig'].isin(names)][['nameOrig', 'step', 'newbalanceOrig']]
        df1 = df1.rename(columns={'nameOrig': 'name', 'newbalanceOrig': 'balance'})
        df2 = self.df[self.df['nameDest'].isin(names)][['nameDest', 'step', 'newbalanceDest']]
        df2 = df2.rename(columns={'nameDest': 'name', 'balance': 'newbalanceDest'})
        df = pd.concat([df1, df2]).reset_index(drop=True)
        #df = df.sort_values(by=['step']).reset_index(drop=True)
        
        curves = {}
        for i, name in enumerate(names):
            curves[i] = hv.Curve(df[df['name']==name][['step', 'balance']], 'step', 'balance')
        if curves:
            curves = hv.NdOverlay(curves).opts(
                hv.opts.Curve(xlim=(0, 365), ylim=(0, 100000), xlabel='step', ylabel='balance')#, interpolation='steps-post')
            )
        else:
            curves = hv.NdOverlay({0: hv.Curve(data=sorted(zip([0], [0])))}).opts(
                hv.opts.Curve(xlim=(0, 365), ylim=(0, 100000), xlabel='step', ylabel='balance')#, interpolation='steps-post')
            )
        return curves.opts(shared_axes=False, show_legend=False)
    
    def get_amount_hist(self, df:pd.DataFrame, bins:int=20):
        vc = df['amount'].value_counts(bins=bins)
        vc.sort_index(inplace=True)
        x_ls = vc.index.left
        x_rs = vc.index.right
        y_bs = [1E-1]*bins
        y_ts = vc.values
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append([x_l, y_b, x_r, y_t])
        histogram = hv.Rectangles(rects).opts(shared_axes=False, xlabel='amount', ylabel='count', logy=True, ylim=(0.8E-1, None))
        return histogram
    
    def get_indegree_hist(self, df:pd.DataFrame, bins:int=20):
        vc = df['target'].value_counts()
        vcc = vc.value_counts(bins=bins)
        vcc.sort_index(inplace=True)
        x_ls = vcc.index.left
        x_rs = vcc.index.right
        y_bs = [1E-1]*bins
        y_ts = vcc.values
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        histogram = hv.Rectangles(rects).opts(shared_axes=False, xlabel='indegree', ylabel='count', logy=True, ylim=(0.8E-1, None))
        return histogram

    def get_outdegree_hist(self, df:pd.DataFrame, bins:int=20):
        vc = df['source'].value_counts()
        vcc = vc.value_counts(bins=bins)
        vcc.sort_index(inplace=True)
        x_ls = vcc.index.left
        x_rs = vcc.index.right
        y_bs = [1E-1]*bins
        y_ts = vcc.values
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        histogram = hv.Rectangles(rects).opts(shared_axes=False, xlabel='outdegree', ylabel='count', logy=True, ylim=(0.8E-1, None))
        return histogram

    def get_n_payments_hist(self, df:pd.DataFrame, interval:int=28):
        start = df['step'].min()
        end = df['step'].max()
        x_ls = []
        x_rs = []
        y_ts = []
        for i in range(start, end, interval):
            x_ls.append(i+interval*0.2)
            x_rs.append(i+interval*0.8)
            y_ts.append(len(df[(df['step'] >= i) & (df['step'] <= i+interval)]))
        y_bs = [1E-1]*len(y_ts)
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        histogram = hv.Rectangles(rects).opts(shared_axes=False, xlabel=f'step', ylabel='number of transactions per month', logy=False)
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
        y_bs = [1E-1]*len(y_ts)
        rects = []
        for x_l, x_r, y_b, y_t in zip(x_ls, x_rs, y_bs, y_ts):
            rects.append((x_l, y_b, x_r, y_t))
        histogram = hv.Rectangles(rects).opts(shared_axes=False, xlabel=f'step', ylabel='volume per month', logy=False)
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
    
    def update_hists(self, steps, banks, legitimate_models, laundering_models, x_range, y_range):
        # filter
        df = self.df_edges
        df = df[df['source_bank'].isin(banks) & df['target_bank'].isin(banks)]
        laundering_models = [self.laundering_type_map[m] for m in laundering_models]
        legitimate_models = [self.legitimate_type_map[m] for m in legitimate_models]
        df = df[df['model_type'].isin(laundering_models + legitimate_models)]
        df = df[(df['step']  > steps[0]) & (df['step'] < steps[1])]
        amount_hist = self.get_amount_hist(df)
        indegree_hist = self.get_indegree_hist(df)
        outdegree_hist = self.get_outdegree_hist(df)
        n_payments_hist = self.get_n_payments_hist(df)
        volume_hist = self.get_volume_hist(df)
        n_users_hist = self.get_n_users_hist(df)
        return (amount_hist + indegree_hist + outdegree_hist + volume_hist + n_payments_hist + n_users_hist).cols(3)