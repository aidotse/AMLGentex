#!/usr/bin/env python3
"""
Transaction Network Explorer Dashboard

Standalone Panel app for exploring AMLGentex transaction networks.

Usage:
    python dashboard.py --experiment tutorial_demo
    # Or with uv:
    uv run python src/visualize/transaction_network_explorer/dashboard.py --experiment tutorial_demo

Then open http://localhost:5006 in your browser.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import panel as pn
import holoviews as hv
from src.visualize.transaction_network_explorer.transaction_network_explorer import TransactionNetwork

pn.extension()
hv.extension('bokeh')


def create_dashboard(tx_net: TransactionNetwork):
    """Create interactive dashboard for transaction network exploration"""

    # === REACTIVE STATE ===

    # Store selected nodes using a reactive list
    import param

    class SelectionState(param.Parameterized):
        selected_nodes = param.List(default=[])

    state = SelectionState()

    # === WIDGETS ===

    # Bank selection - dropdown with multiple selection
    bank_select = pn.widgets.MultiChoice(
        name='Banks',
        options=tx_net.BANK_IDS,
        value=tx_net.BANK_IDS,
        width=300
    )

    # Alert (laundering) model selection - dropdown with multiple selection
    # Use actual pattern names from the loaded data
    alert_model_names = tx_net.LAUND_MODEL_NAMES if tx_net.LAUND_MODEL_NAMES else []
    alert_select = pn.widgets.MultiChoice(
        name='SAR Patterns',
        options=alert_model_names,
        value=alert_model_names,
        width=300
    )

    # Normal (legitimate) model selection - dropdown with multiple selection
    # Use actual pattern names from the loaded data
    normal_model_names = tx_net.LEGIT_MODEL_NAMES if tx_net.LEGIT_MODEL_NAMES else []
    normal_select = pn.widgets.MultiChoice(
        name='Normal Patterns',
        options=normal_model_names,
        value=normal_model_names,
        width=300
    )

    # Time range slider
    time_range = pn.widgets.IntRangeSlider(
        name='Time Range (steps)',
        start=tx_net.START_STEP,
        end=tx_net.END_STEP,
        value=(tx_net.START_STEP, tx_net.END_STEP),
        step=1,
        width=300
    )

    # Inter-bank transaction toggle
    show_interbank = pn.widgets.Checkbox(
        name='Include inter-bank transactions',
        value=False
    )

    # UMAP hyperparameter controls (collapsible)
    umap_neighbors = pn.widgets.IntInput(
        name='n_neighbors',
        start=2,
        end=50,
        step=1,
        value=15,
        width=120
    )

    umap_min_dist = pn.widgets.FloatInput(
        name='min_dist',
        start=0.0,
        end=0.99,
        step=0.05,
        value=0.1,
        width=120
    )

    umap_metric = pn.widgets.Select(
        name='metric',
        options=['euclidean', 'cosine', 'manhattan'],
        value='euclidean',
        width=120
    )

    # Bank account statistics (reactive)
    @pn.depends(bank_select)
    def bank_account_stats(banks):
        """Display statistics about accounts in selected banks"""
        if not banks:
            return pn.pane.Markdown('*Select banks to see account statistics*', sizing_mode='fixed')

        # Get all accounts in selected banks
        nodes_in_banks = tx_net.df_nodes[tx_net.df_nodes['bank'].isin(banks)]['name'].tolist()
        total_accounts = len(nodes_in_banks)

        # Get accounts that have at least one intra-bank transaction (both source and target in selected banks)
        df_intra = tx_net.df_edges[
            (tx_net.df_edges['source'].isin(nodes_in_banks)) &
            (tx_net.df_edges['target'].isin(nodes_in_banks))
        ]
        accounts_with_intra = set(df_intra['source'].unique().tolist() + df_intra['target'].unique().tolist())
        n_intra = len(accounts_with_intra)

        # Get all accounts that have any transaction
        df_any = tx_net.df_edges[
            (tx_net.df_edges['source'].isin(nodes_in_banks)) |
            (tx_net.df_edges['target'].isin(nodes_in_banks))
        ]
        accounts_with_any = set(df_any['source'].unique().tolist() + df_any['target'].unique().tolist())

        # Accounts that ONLY have inter-bank transactions (have transactions but none are intra-bank)
        # These are accounts in selected banks that only transact with accounts in non-selected banks
        accounts_only_inter = (accounts_with_any & set(nodes_in_banks)) - accounts_with_intra
        n_only_inter = len(accounts_only_inter)

        # Accounts with no transactions at all
        n_no_tx = total_accounts - len(accounts_with_any & set(nodes_in_banks))

        return pn.pane.Markdown(
            f'**{total_accounts}** accounts in selected bank(s): '
            f'**{n_intra}** with intra-bank txs, '
            f'**{n_only_inter}** with only inter-bank txs' +
            (f', **{n_no_tx}** with no txs' if n_no_tx > 0 else ''),
            sizing_mode='fixed'
        )

    # === NETWORK VISUALIZATION ===

    # X/Y ranges for the network (needed by histogram function)
    x_range = (-10, 10)
    y_range = (-10, 10)

    # Cache for filtered data to avoid redundant computation
    class FilterCache:
        def __init__(self):
            self.last_params = None
            self.cached_df = None

        def get_filtered_data(self, banks, alert_models, normal_models, time_steps, include_interbank):
            """Get filtered edge DataFrame with caching"""
            current_params = (
                tuple(sorted(banks)),
                tuple(sorted(alert_models)),
                tuple(sorted(normal_models)),
                time_steps,
                include_interbank
            )

            if current_params == self.last_params and self.cached_df is not None:
                return self.cached_df

            # Filter nodes by selected banks first
            nodes_in_banks = tx_net.df_nodes[tx_net.df_nodes['bank'].isin(banks)]['name'].tolist()

            # Start with all edges - use view instead of copy for speed
            df = tx_net.df_edges

            # Filter edges based on inter-bank toggle
            if include_interbank:
                df = df[df['source'].isin(nodes_in_banks) | df['target'].isin(nodes_in_banks)]
            else:
                df = df[df['source'].isin(nodes_in_banks) & df['target'].isin(nodes_in_banks)]

            # Build list of model IDs
            alert_model_ids = []
            normal_model_ids = []

            for model_name in alert_models:
                if model_name in tx_net.laundering_type_map:
                    alert_model_ids.extend(tx_net.laundering_type_map[model_name])

            for model_name in normal_models:
                if model_name in tx_net.legitimate_type_map:
                    normal_model_ids.extend(tx_net.legitimate_type_map[model_name])

            # Filter by model types
            df_filtered_parts = []
            if alert_model_ids:
                df_sar = df[(df['model_type'].isin(alert_model_ids)) & (df['is_sar'] == 1)]
                df_filtered_parts.append(df_sar)
            if normal_model_ids:
                df_normal = df[(df['model_type'].isin(normal_model_ids)) & (df['is_sar'] == 0)]
                df_filtered_parts.append(df_normal)

            if not df_filtered_parts:
                self.cached_df = pd.DataFrame()
                self.last_params = current_params
                return self.cached_df

            df_filtered = pd.concat(df_filtered_parts, ignore_index=True)

            # Filter by time range
            df_filtered = df_filtered[(df_filtered['step'] >= time_steps[0]) & (df_filtered['step'] <= time_steps[1])]

            self.cached_df = df_filtered
            self.last_params = current_params
            return df_filtered

    filter_cache = FilterCache()

    @pn.depends(bank_select, alert_select, normal_select, time_range, show_interbank, state.param.selected_nodes)
    def update_network(banks, alert_models, normal_models, time_steps, include_interbank, selected_nodes_list):
        """Update network visualization based on filters"""
        # Show empty if nothing selected
        if not banks:
            return pn.pane.Markdown('### Select at least one bank to visualize')

        # Convert to lists if needed (handle empty tuples/None)
        alert_models = list(alert_models) if alert_models else []
        normal_models = list(normal_models) if normal_models else []

        if not alert_models and not normal_models:
            return pn.pane.Markdown('### Select at least one pattern (alert or normal) to visualize')

        # Use cached filtering
        df = filter_cache.get_filtered_data(banks, alert_models, normal_models, time_steps, include_interbank)

        if len(df) == 0:
            available_normal = sorted(tx_net.LEGIT_MODEL_NAMES)
            available_sar = sorted(tx_net.LAUND_MODEL_NAMES)

            return pn.pane.Markdown(
                f'### No transactions match the selected filters\n\n'
                f'**Available normal patterns in data:** {", ".join(available_normal) if available_normal else "none"}\n\n'
                f'**Available SAR patterns in data:** {", ".join(available_sar) if available_sar else "none"}\n\n'
                f'*Try adjusting the time range, bank selection, or pattern selection.*'
            )

        # Limit number of edges for performance (sample if too many)
        MAX_EDGES = 10000
        if len(df) > MAX_EDGES:
            df = df.sample(n=MAX_EDGES, random_state=42)
            show_warning = True
        else:
            show_warning = False

        # Get unique nodes from filtered edges
        node_names = set(df['source'].unique().tolist() + df['target'].unique().tolist())
        nodes_df = tx_net.df_nodes[tx_net.df_nodes['name'].isin(node_names)].copy()

        # Vectorized selection state and styling - MUCH faster than apply()
        selected_nodes_set = set(state.selected_nodes)
        nodes_df['selected'] = nodes_df['name'].isin(selected_nodes_set).astype(int)
        nodes_df['node_size'] = 15 + nodes_df['selected'] * 10  # 15 or 25

        # Vectorized color assignment
        nodes_df['node_color'] = '#00BFFF'  # Default: normal (blue)
        nodes_df.loc[nodes_df['is_sar'] == 1, 'node_color'] = '#FF0000'  # SAR (red)
        nodes_df.loc[nodes_df['selected'] == 1, 'node_color'] = '#FFA500'  # Selected (orange)

        nodes_df['border_width'] = 1.5 + nodes_df['selected'] * 1.5  # 1.5 or 3.0

        # Create node data with all styling applied
        nodes = hv.Nodes(
            (nodes_df['x'], nodes_df['y'], nodes_df['name'], nodes_df['is_sar'],
             nodes_df['selected'], nodes_df['node_size'], nodes_df['node_color'], nodes_df['border_width']),
            vdims=['name', 'is_sar', 'selected', 'node_size', 'node_color', 'border_width']
        )

        # VECTORIZED edge color assignment - much faster than iterrows()
        edge_colors = np.where(df['is_sar'] == 1, 'red', 'gray')
        edges_data = list(zip(df['source'], df['target'], df['is_sar'], edge_colors))

        # Create graph with edge color as additional dimension
        if edges_data:
            graph = hv.Graph((edges_data, nodes), vdims=['is_sar', 'color'])

            # Style the graph - reasonable sizes for good visibility
            title_text = f'Transaction Network ({len(df):,} transactions, {len(nodes_df):,} accounts)'
            if show_warning:
                title_text += f' [Sampled from {MAX_EDGES:,}+ total]'

            styled_graph = graph.opts(
                hv.opts.Graph(
                    width=750,
                    height=750,
                    node_size=12,  # Reasonable fixed size
                    node_color='node_color',  # Use our color column
                    node_alpha=1.0,  # Fully opaque nodes
                    edge_color='color',
                    edge_alpha=0.3,  # Semi-transparent edges
                    edge_line_width=0.8,  # Visible but not dominant
                    node_line_color='black',
                    node_line_width=1.5,  # Visible border
                    directed=True,
                    arrowhead_length=0.008,
                    tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'tap'],
                    title=title_text,
                    xaxis=None,
                    yaxis=None,
                    padding=0.1,
                    inspection_policy='nodes'
                )
            )

            # Render to Bokeh and add native Bokeh tap handling
            bokeh_plot = hv.render(styled_graph)

            # Find the graph renderer
            graph_renderer = None
            for r in bokeh_plot.renderers:
                if hasattr(r, 'node_renderer'):
                    graph_renderer = r
                    break

            if graph_renderer:
                # Capture nodes_df for this render
                nodes_copy = nodes_df.copy()

                # Add custom hover tool for nodes
                from bokeh.models import HoverTool

                # Add name and is_sar fields to the node data source
                graph_renderer.node_renderer.data_source.data['name'] = nodes_copy['name'].tolist()
                graph_renderer.node_renderer.data_source.data['is_sar'] = nodes_copy['is_sar'].tolist()

                hover = HoverTool(
                    tooltips=[('Node ID', '@name'), ('SAR Account', '@is_sar')],
                    renderers=[graph_renderer.node_renderer]
                )
                bokeh_plot.add_tools(hover)

                # Add Python callback for node selection
                def on_change(attr, old, new):
                    if new and len(new) > 0:
                        idx = new[0]
                        if idx < len(nodes_copy):
                            node_id = int(nodes_copy.iloc[idx]['name'])
                            # Toggle selection
                            curr = list(state.selected_nodes)
                            if node_id in curr:
                                curr.remove(node_id)
                            else:
                                curr.append(node_id)
                            state.selected_nodes = curr

                graph_renderer.node_renderer.data_source.selected.on_change('indices', on_change)

            return bokeh_plot
        else:
            # Handle case with no edges (shouldn't happen given earlier check, but defensive)
            return pn.pane.Markdown('### No edges to display')

    # === HISTOGRAMS ===

    @pn.depends(bank_select, alert_select, normal_select, time_range, show_interbank, state.param.selected_nodes,
                umap_neighbors, umap_min_dist, umap_metric)
    def update_histograms(banks, alert_models, normal_models, time_steps, include_interbank, selected_nodes_list,
                         n_neighbors, min_dist, metric):
        """Update histograms based on filters"""
        # Show empty if nothing selected
        if not banks:
            return pn.pane.Markdown('*Select banks to view distributions*')

        if not alert_models and not normal_models:
            return pn.pane.Markdown('*Select at least one pattern to view distributions*')

        # Convert to lists
        alert_models = list(alert_models) if alert_models else []
        normal_models = list(normal_models) if normal_models else []

        # Use cached filtered data instead of recomputing
        df = filter_cache.get_filtered_data(banks, alert_models, normal_models, time_steps, include_interbank)

        # Use the selected nodes passed as parameter
        selected_node_ids = selected_nodes_list if selected_nodes_list else []

        # UMAP parameters
        umap_params = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric
        }

        # Pass the already-filtered dataframe to avoid redundant filtering
        # Returns a Panel GridBox with wrapped plots (includes titles and info tooltips)
        hists = tx_net.update_hists_from_filtered_data(df, x_range, y_range, selected_node_ids, umap_params, umap_settings_button)

        return hists

    # === DATASET INFORMATION ===

    # Show which patterns are available in the loaded dataset
    # Pattern names are already validated to exist in the data
    available_normal = sorted(tx_net.LEGIT_MODEL_NAMES)
    available_sar = sorted(tx_net.LAUND_MODEL_NAMES)

    # Calculate per-bank account counts
    bank_account_counts = []
    for bank in tx_net.BANK_IDS:
        count = len(tx_net.df_nodes[tx_net.df_nodes['bank'] == bank])
        bank_account_counts.append(f"{bank}: {count}")

    # Homophily info as separate row with tooltip
    homophily_info = pn.Row(
        pn.pane.Markdown(
            f'**Homophily:** Edge={tx_net.HOMOPHILY_EDGE:.3f}, '
            f'Node={tx_net.HOMOPHILY_NODE:.3f}, '
            f'Class={tx_net.HOMOPHILY_CLASS:.3f}',
            margin=(0, 5, 0, 0)
        ),
        pn.widgets.TooltipIcon(
            value="Homophily measures the tendency of similar nodes to connect. "
                  "Edge: fraction of edges between same-type nodes (SAR-SAR or Normal-Normal). "
                  "Node: average node-level homophily. "
                  "Class: homophily broken down by class. "
                  "Values: 0 = random mixing, 1 = complete segregation."
        ),
        margin=(0, 0, 0, 0)
    )

    dataset_info = pn.Column(
        pn.pane.Markdown(
            f'**Dataset:** {tx_net.N_ACCOUNTS:,} accounts, '
            f'{tx_net.N_LEGIT_TXS:,} normal + {tx_net.N_LAUND_TXS:,} SAR txs\n\n'
            f'**Patterns:** {len(available_normal)} normal, {len(available_sar)} SAR types',
            sizing_mode='fixed'
        ),
        homophily_info,
        sizing_mode='fixed'
    )

    # === DASHBOARD LAYOUT ===

    # Show selected nodes (only when nodes are selected)
    @pn.depends(state.param.selected_nodes)
    def show_selected(selected):
        if selected:
            return pn.pane.Markdown(f'**Selected nodes:** {", ".join(map(str, selected))}')
        return pn.pane.Markdown('')  # Empty when no selection

    # UMAP controls in a simple collapsible card in the left panel
    umap_controls = pn.Card(
        umap_neighbors,
        umap_min_dist,
        umap_metric,
        title='UMAP Settings',
        collapsed=True,  # Start collapsed
        collapsible=True,
        width=290
    )

    # No button needed - settings are in the left panel
    umap_settings_button = None

    # Left column: Controls with collapsible UMAP settings at bottom
    controls_panel = pn.Column(
        pn.pane.Markdown('# AMLGentex Network Explorer'),
        dataset_info,
        pn.layout.Divider(),
        bank_select,
        show_interbank,
        bank_account_stats,
        alert_select,
        normal_select,
        time_range,
        show_selected,
        umap_controls,  # Collapsible card with UMAP settings
        width=300,
        scroll=True,  # Make it scrollable
        sizing_mode='stretch_height',  # Fill available height
        max_height=750  # Match the top row height
    )

    # Right: Network graph
    graph_panel = pn.Column(
        update_network,
        sizing_mode='stretch_both'
    )

    # Top row: Controls + Graph
    top_row = pn.Row(
        controls_panel,
        graph_panel,
        sizing_mode='stretch_width',
        height=750
    )

    # Bottom row: Histograms in 2x4 grid (full width)
    bottom_row = pn.Row(
        update_histograms,
        sizing_mode='stretch_width'
    )

    # Main layout: Top (controls + graph) | Bottom (histograms) + FloatPanel overlay
    dashboard_content = pn.Column(
        top_row,
        bottom_row,
        sizing_mode='stretch_both'
    )

    # Main dashboard layout
    dashboard = pn.Column(
        dashboard_content,
        sizing_mode='stretch_both'
    )

    return dashboard


def main():
    parser = argparse.ArgumentParser(description='Transaction Network Explorer Dashboard')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--port', type=int, default=5006, help='Server port (default: 5006)')
    parser.add_argument('--project-root', type=str, help='Project root directory (default: auto-detect)')
    args = parser.parse_args()

    # Find project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Auto-detect: go up from this file to find project root
        project_root = Path(__file__).parent.parent.parent.parent

    # Find transaction log (prefer parquet for better performance)
    experiment_root = project_root / 'experiments' / args.experiment
    parquet_path = experiment_root / 'temporal' / 'tx_log.parquet'
    csv_path = experiment_root / 'temporal' / 'tx_log.csv'

    # Use parquet if available, otherwise fall back to CSV
    if parquet_path.exists():
        tx_log_path = parquet_path
    elif csv_path.exists():
        tx_log_path = csv_path
    else:
        raise FileNotFoundError(
            f'Transaction log not found. Expected:\n'
            f'  - {parquet_path}\n'
            f'  - {csv_path}'
        )

    # Load transaction network
    print(f"\nLoading transaction network from: {tx_log_path}")
    tx_net = TransactionNetwork(str(tx_log_path))

    print(f"\nTransaction Network Statistics:")
    print(f"  Banks: {tx_net.BANK_IDS}")
    print(f"  Accounts: {tx_net.N_ACCOUNTS:,}")
    print(f"  Normal transactions: {tx_net.N_LEGIT_TXS:,}")
    print(f"  SAR transactions: {tx_net.N_LAUND_TXS:,}")
    print(f"  Time range: {tx_net.START_STEP} to {tx_net.END_STEP}")

    # Create and serve dashboard
    print(f"\nStarting dashboard server...")
    print(f"  Open http://localhost:{args.port} in your browser")
    print(f"  Press Ctrl+C to stop\n")

    dashboard = create_dashboard(tx_net)
    dashboard.show(port=args.port, open=True)


if __name__ == '__main__':
    main()
