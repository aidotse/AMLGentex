# Transaction Network Explorer

Interactive visualization tool for exploring large-scale AML transaction networks using Holoviews and Datashader.

## Overview

The Transaction Network Explorer provides an interactive, scalable way to visualize and analyze transaction networks from AML simulations. It uses Datashader for efficient rendering of large networks (thousands to millions of transactions) and Holoviews for interactive exploration.

## Features

### Network Visualization
- **Scalable rendering**: Uses Datashader to visualize networks with millions of edges
- **Interactive filtering**: Filter by banks, transaction models, time steps, and spatial regions
- **Node layout**: Automatic layout that clusters nodes by bank affiliation
- **SAR highlighting**: Visual distinction between normal and suspicious (SAR) transactions

### Network Analysis
- **Homophily metrics**: Measures how well SAR transactions cluster together
  - Edge homophily: Proportion of edges connecting same-label nodes
  - Node homophily: Average homophily per node
  - Class homophily: Separate metrics for normal vs SAR transactions
- **Temporal analysis**: Track network evolution over time steps
- **Statistical summaries**: Compute network properties like degree distributions

### Interactive Features
- **Dynamic histograms**: Real-time updates of:
  - Transaction amounts
  - Degree distributions (in-degree, out-degree)
  - Payment volumes over time
  - Number of active accounts over time
- **Balance tracking**: Visualize account balances over time
- **Spatial filtering**: Select regions of the network to analyze
- **Model filtering**: Focus on specific transaction patterns (fan-out, cycle, etc.)

## Installation

Install the Transaction Network Explorer dependencies as an optional extra:

```bash
# Using uv (recommended)
uv pip install -e ".[network-explorer]"

# Or using pip
pip install -e ".[network-explorer]"
```

This installs the required dependencies:
- `datashader` - Scalable rendering of large datasets
- `holoviews` - Interactive plots
- `panel` - Dashboard interface
- `jupyter_bokeh` - Jupyter integration

Core dependencies (pandas, numpy, matplotlib) are included in the main AMLGentex installation.

## Usage

### Standalone Dashboard (Recommended)

Run the interactive dashboard server for any experiment:

```bash
# Basic usage
uv run python src/visualize/transaction_network_explorer/dashboard.py --experiment tutorial_demo

# Custom port
uv run python src/visualize/transaction_network_explorer/dashboard.py --experiment tutorial_demo --port 8080

# Custom project root
uv run python src/visualize/transaction_network_explorer/dashboard.py \
    --experiment my_experiment \
    --project-root /path/to/AMLGentex
```

Then open http://localhost:5006 (or your custom port) in your browser.

### Dashboard Layout

The dashboard is organized into two main sections:

**Left Panel (Scrollable Control Panel):**
- Dataset information (accounts, transactions, patterns)
- Homophily metrics with explanatory tooltips
- Bank dropdown selector
- Show/hide inter-bank transactions toggle
- Bank account statistics table
- SAR Pattern dropdown selector
- Normal Pattern dropdown selector
- Time range slider
- Show only selected patterns toggle
- UMAP Settings (collapsible card at bottom)

**Main View (2 rows of visualizations):**
- **Row 1:** Transaction amounts, In-degree, Out-degree, Transactions per step
- **Row 2:** Active accounts, Burstiness categories, UMAP projection

All visualizations update dynamically when filters are changed.

**Features:**
- **Filtering controls:**
  - Dropdown multi-select for banks and transaction patterns
  - Time range slider for temporal filtering
  - Show/hide inter-bank transactions
  - UMAP settings (collapsible card): adjust neighbors, min distance, and metric
- **Real-time visualizations:**
  - Transaction amount distributions (Normal vs SAR)
  - In-degree and out-degree distributions
  - Transactions per time step
  - Active accounts over time
  - Burstiness analysis (Low/Medium/High categories)
  - UMAP projection of transaction patterns
- **Network statistics:**
  - Dataset summary (accounts, transactions, pattern counts)
  - Homophily metrics (Edge, Node, Class)
  - Bank-level account statistics
- All visualizations update instantly as filters change

### Python API

```python
from src.visualize.transaction_network_explorer import TransactionNetwork

# Load transaction data (supports both parquet and CSV)
tx_net = TransactionNetwork('experiments/10k_accounts/temporal/tx_log.parquet')

# Access network properties
print(f"Banks: {tx_net.BANK_IDS}")
print(f"Accounts: {tx_net.N_ACCOUNTS}")
print(f"Normal transactions: {tx_net.N_LEGIT_TXS}")
print(f"SAR transactions: {tx_net.N_LAUND_TXS}")
print(f"Time steps: {tx_net.START_STEP} to {tx_net.END_STEP}")
print(f"Edge homophily: {tx_net.HOMOPHILY_EDGE:.3f}")
print(f"Node homophily: {tx_net.HOMOPHILY_NODE:.3f}")
```

### Jupyter Notebook

For basic transaction statistics in notebooks, see **Step 6** in the main tutorial: `tutorial.ipynb` at the project root.

For full interactive exploration, use the standalone dashboard (above) rather than embedding widgets in notebooks.

## Transaction Network Class

### Constructor

```python
TransactionNetwork(path: str)
```

Loads and processes transaction data from a CSV file.

**Parameters:**
- `path`: Path to transaction log CSV file

**Attributes set:**
- `df_nodes`: DataFrame with node information (accounts)
- `df_edges`: DataFrame with edge information (transactions)
- `BANK_IDS`: List of unique bank identifiers
- `N_ACCOUNTS`: Total number of accounts
- `N_LEGIT_TXS`: Count of normal transactions
- `N_LAUND_TXS`: Count of SAR transactions
- `START_STEP`, `END_STEP`, `N_STEPS`: Temporal range
- `HOMOPHILY_EDGE`, `HOMOPHILY_NODE`, `HOMOPHILY_CLASS`: Network homophily metrics

### Key Methods

#### Data Loading and Processing
- `load_data(path)` - Load transaction CSV, exclude CASH transactions
- `format_data(df)` - Convert transactions to node/edge format
- `spread_nodes(df)` - Calculate spatial layout with bank clustering

#### Network Selection
- `get_all_nodes()` - Get all network nodes
- `select_nodes(banks, laundering_models, legitimate_models, steps, x_range, y_range)` - Filter nodes
- `select_edges(banks, laundering_models, legitimate_models, steps, x_range, y_range)` - Filter edges

#### Analysis
- `calc_homophily()` - Calculate network homophily metrics
- `get_balances(index)` - Track account balances over time
- `get_amount_hist(df, bins)` - Transaction amount distribution
- `get_indegree_hist(df, bins)` - In-degree distribution
- `get_outdegree_hist(df, bins)` - Out-degree distribution
- `get_n_payments_hist(df)` - Number of transactions per time step
- `get_volume_hist(df, interval)` - Transaction volume over time
- `get_n_users_hist(df, interval)` - Active accounts over time
- `get_burstiness_plot(df)` - Burstiness analysis with Wasserstein distance categorization
  - Calculates temporal clustering of transactions within patterns
  - Categorizes as Low (0-0.15), Medium (0.15-0.35), High (>0.35)
  - Excludes patterns with <3 transactions (insufficient for analysis)
  - Returns normalized proportions for Normal vs Alert patterns

## Transaction Models

### Legitimate Transaction Models
- `single` - Single transactions
- `fan-out` - One-to-many transactions
- `fan-in` - Many-to-one transactions
- `forward` - Chain transactions
- `mutual` - Bidirectional transactions
- `periodical` - Recurring transactions

### Money Laundering Models (SAR)
- `fan-out` (ID: 21) - Layering: split funds
- `fan-in` (ID: 22) - Integration: collect funds
- `cycle` (ID: 23) - Circular transactions
- `bipartite` (ID: 24) - Two-group exchange
- `stacked` (ID: 25) - Layered structure
- `random` (ID: 26) - Random patterns
- `scatter-gather` (ID: 27) - Scatter then gather
- `gather-scatter` (ID: 28) - Gather then scatter

## Node Layout Algorithm

Nodes are positioned using a hybrid approach:
1. **Random base position**: Initial random radial placement
2. **Bank clustering**: Nodes from same bank are grouped together
3. **Spiral arrangement**: Banks are arranged in a spiral pattern
4. **Radial scaling**: Bank clusters scale outward with increasing bank count

This creates an interpretable layout where:
- Intra-bank transactions are visually short edges
- Inter-bank transactions are longer edges crossing clusters
- SAR patterns can be visually identified by edge density/patterns

## Homophily Metrics

**Edge Homophily**: Fraction of edges connecting nodes of the same type (normal-normal or SAR-SAR)

**Node Homophily**: For each node, fraction of neighbors with same label, averaged across all nodes

**Class Homophily**: Separate homophily calculations for normal vs SAR nodes

High homophily indicates strong clustering of SAR transactions, making them easier to detect.

## Burstiness Analysis

**What is Burstiness?**
Burstiness measures how temporally clustered transactions are within a pattern. High burstiness means transactions occur in concentrated bursts rather than evenly distributed over time.

**Measurement:**
- Uses Wasserstein distance (Earth Mover's Distance) between the pattern's transaction timing and a uniform distribution
- Distance of 0 = perfectly uniform (evenly spaced)
- Higher distance = more bursty (concentrated in time)

**Categories:**
- **Low (0-0.15)**: Nearly uniform distribution, transactions evenly spread
- **Medium (0.15-0.35)**: Moderate temporal clustering
- **High (>0.35)**: Strong clustering, highly bursty behavior

**Interpretation:**
- Alert patterns are typically configured with higher burstiness (level 4: β=0.1, α=5.0)
- Normal patterns use lower burstiness (level 1-3)
- Patterns with <3 transactions are excluded (need at least 2 intervals for meaningful analysis)
- Dashboard shows normalized proportions to compare Alert vs Normal distributions

**Note:** Discretization to integer timesteps reduces observable burstiness, so configured parameters use extreme values to achieve visible differences.

## UMAP Projection

**Purpose:**
UMAP (Uniform Manifold Approximation and Projection) provides a 2D visualization of pattern similarity based on transaction features.

**Features Used:**
- Transaction amounts
- Transaction counts
- Temporal features
- Network topology features

**Settings (Collapsible Card):**
- **n_neighbors**: Number of neighbors to consider (default: 15)
  - Lower values emphasize local structure
  - Higher values emphasize global structure
- **min_dist**: Minimum distance between points (default: 0.1)
  - Lower values create tighter clusters
  - Higher values create more spread-out distributions
- **metric**: Distance metric (default: euclidean)
  - Options: euclidean, manhattan, cosine, etc.

**Interpretation:**
- Points close together have similar transaction patterns
- Alert vs Normal patterns should form distinct clusters if features are discriminative
- Useful for identifying which patterns are harder to distinguish

## Performance Notes

- **Datashader**: Enables visualization of millions of transactions
- **CPU-optimized**: Runs on CPU without GPU acceleration
- **Interactive updates**: Sub-second response for filtering operations
- **Memory efficient**: Streaming computation for large datasets

## Example Workflow

### Using the Dashboard (Recommended)

1. **Launch dashboard**:
   ```bash
   uv run python src/visualize/transaction_network_explorer/dashboard.py --experiment tutorial_demo
   ```

2. **Filter and explore**:
   - Use bank dropdown to select specific banks
   - Use pattern dropdowns to focus on specific SAR or Normal patterns
   - Adjust time range slider to examine temporal evolution
   - Expand UMAP settings card to adjust projection parameters

3. **Analyze patterns**:
   - Check homophily metrics in the info panel
   - Examine burstiness distribution (Low/Medium/High categories)
   - Compare transaction amounts between Normal and Alert patterns
   - Review degree distributions to identify hub accounts
   - Track transactions per step to identify activity peaks

### Using the Python API

1. **Load network**:
   ```python
   tx_net = TransactionNetwork('path/to/tx_log.parquet')
   ```

2. **Explore properties**:
   ```python
   print(f"Edge homophily: {tx_net.HOMOPHILY_EDGE:.3f}")
   print(f"Node homophily: {tx_net.HOMOPHILY_NODE:.3f}")
   print(f"Class homophily: {tx_net.HOMOPHILY_CLASS:.3f}")
   ```

3. **Filter network**:
   ```python
   nodes = tx_net.select_nodes(
       banks=['swedbank', 'nordea'],
       laundering_models=[23, 24],  # cycle, bipartite
       legitimate_models=[0, 1, 2],
       steps=(0, 100),
       x_range=(-1, 1),
       y_range=(-1, 1)
   )
   ```

4. **Generate visualizations**:
   ```python
   amount_hist = tx_net.get_amount_hist(filtered_edges)
   degree_hist = tx_net.get_indegree_hist(filtered_edges)
   burstiness = tx_net.get_burstiness_plot(filtered_edges)
   ```

5. **Track balances**:
   ```python
   balances = tx_net.get_balances(account_indices)
   ```

## Limitations

- Requires transaction log in specific parquet or CSV format
- Layout algorithm assumes reasonable number of banks (< 50)
- Interactive updates may slow with extremely large selections (>1M edges)
- CASH transactions are excluded from network visualization (included in burstiness analysis)
- Burstiness calculation requires patterns with ≥3 transactions
- UMAP projection recalculates on filter changes, may be slow for large datasets

## Future Enhancements

- GPU acceleration support via cuGraph/cuDF for even larger networks
- Additional layout algorithms (force-directed, hierarchical, geographic)
- Export filtered subgraphs for further analysis
- Real-time streaming data support
- Community detection visualization
- Anomaly scoring overlays
- Pattern-level timeline visualization
- Network evolution animation over time steps

## Citation

If you use this tool in research, please cite the AMLGentex framework.

## License

Same license as AMLGentex project.
