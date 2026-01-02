# AMLGentex Images Guide

This directory contains all figures used in the main README. Below is a description of each figure and what it should visualize.

---

## Figure 1: `degree_distribution.png` (Optional)

**Location in README:** Key Features → Scale-Free Transaction Networks

**Description:** Visualization showing how the three parameters (gamma, loc, average_degree) control the scale-free network degree distribution.

**Should show:**
- X-axis: Node degree (d)
- Y-axis: Probability P(degree = d)
- Power-law behavior: P(d) ∝ d^(-gamma) for large d
- Effect of loc and average_degree parameters on small degrees
- Typical values: gamma=2.0, loc=1.0, average_degree=2.0

**Note:** The scale parameter is computed automatically from average_degree using the Riemann zeta function (see Appendix E.1 of AMLGentex paper).

---

## Figure 2: `truncatgaussians.png`

**Location in README:** Key Features → Realistic Transaction Modeling

**Description:** Truncated Gaussian distributions used to sample transaction amounts for normal and SAR transactions.

**Should show:**
- Two overlapping distributions (normal vs SAR)
- Truncation at zero and maximum balance
- Different mean/std for normal (mean=100, std=10) vs SAR (mean=643, std=320)
- Illustrate how SAR transactions tend to be larger

---

## Figure 3: `in-out-flows.png`

**Location in README:** Key Features → Account Behavior Modeling

**Description:** Visualization of temporal dynamics - how money flows in (salary) and out (spending) of accounts over time.

**Should show:**
- Time series of account balance over simulation period
- In-flows: Regular salary deposits (income transactions)
- Out-flows: Spending transactions based on balance history
- Illustrate the sliding window mechanism and spending probability calculation

---

## Figure 4: `laundering_stages.png`

**Location in README:** Key Features → Money Laundering Typologies

**Description:** Two main money laundering approaches supported by AMLGentex.

**Should show:**
- **Left panel:** Transfer-based laundering
  - Placement: Initial deposit into network
  - Layering: Complex series of transfers through multiple accounts
  - Integration: Final extraction from network
- **Right panel:** Cash-based laundering
  - Placement: Cash injection into an account
  - Layering: (minimal or none)
  - Integration: Withdrawal or spending

---

## Figure 5: `alert_patterns_and_normal_patterns.png`

**Location in README:** Key Features → Money Laundering Typologies

**Description:** Visual catalog of all transaction pattern types.

**Should show:**
- **Top half:** Normal patterns
  - Single: One-to-one transactions
  - Fan-out: One sender to multiple receivers
  - Fan-in: Multiple senders to one receiver
  - Forward: Chain of sequential transactions
  - Mutual: Bidirectional transactions between two accounts
  - Periodical: Repeated transactions over time

- **Bottom half:** Alert patterns (SAR)
  - Fan-out: Rapid dispersion from one account
  - Fan-in: Rapid collection into one account
  - Cycle: Circular flow of funds
  - Bipartite: Two groups exchanging funds
  - Stack: Layered sequential transfers
  - Random: Irregular pattern of transfers
  - Gather-scatter: Collection followed by dispersion
  - Scatter-gather: Dispersion followed by collection

---

## Figure 6: `radar_plot.png`

**Location in README:** Modeling Real-World Challenges

**Description:** Expert-assessed severity of key challenges in AML transaction monitoring that AMLGentex addresses.

**Should show:**
- Radar/spider chart with multiple axes (6-8 dimensions)
- Axes represent different challenges:
  - Label noise
  - Feature drift
  - Class imbalance
  - Temporal dynamics
  - Network complexity
  - Cross-bank patterns
  - Multiple laundering events
  - etc.
- Severity scale (0-5 or 0-10)
- Based on expert opinions from Swedbank and Handelsbanken

---

## Figure 7: `noise.png`

**Location in README:** Modeling Real-World Challenges

**Description:** Different types of noise and complexity affecting AML detection.

**Should show:**
- Multiple panels or subplots illustrating:
  - **Label noise:** Mislabeled transactions (false positives/negatives)
  - **Feature drift:** Distribution shift over time
  - **Concept drift:** Changing definitions of suspicious behavior
  - **Measurement error:** Noisy or missing features
  - **Temporal irregularity:** Varying transaction frequencies
- Visual examples of how each noise type manifests in data

---

## Figure 8: `data_generation_procedure.png`

**Location in README:** Data Generation Pipeline → Spatial Graph Generation

**Description:** Step-by-step network creation process from degree distribution to final spatial graph.

**Should show:**
1. **Step 1:** Degree distribution (blueprint)
   - Bar chart showing count vs in-degree/out-degree

2. **Step 2:** Initial scale-free network
   - Graph visualization with nodes colored by degree

3. **Step 3:** Normal pattern injection
   - Highlight inserted normal patterns in different color
   - Show how patterns respect network constraints

4. **Step 4:** Alert pattern injection
   - Overlay SAR patterns on top
   - Different color for SAR accounts/edges

5. **Final graph:** Complete spatial network
   - Mixed normal and SAR patterns
   - Legend showing pattern types

---

## Figure 9: `workflow.png`

**Location in README:** Data Generation Pipeline → Temporal Transaction Generation

**Description:** Complete end-to-end workflow from spatial graph to temporal transactions.

**Should show:**
- Flow diagram with boxes and arrows:
  1. Configuration files (data.yaml, degree.csv, accounts.csv, etc.)
  2. ↓
  3. Spatial graph generation
  4. ↓
  5. Temporal simulation engine
  6. ↓
  7. Transaction log output (Parquet)

- Include sample data snippets at each stage
- Show both spatial (network topology) and temporal (time-series) dimensions

---

## Figure 10: `blueprint.png`

**Location in README:** Data Generation Pipeline → Feature Engineering

**Description:** Feature engineering pipeline transforming raw transactions into ML-ready windowed features.

**Should show:**
- Three connected panels:

  1. **Spatial Graph (t=0)**
     - Network topology with accounts and edges

  2. **Transaction Log (t=1...T)**
     - Time-series table with columns: timestamp, sender, receiver, amount, is_sar
     - Show multiple timesteps

  3. **Windowed Graph Features**
     - Multiple temporal snapshots (t-3*28d, t-2*28d, t-1*28d, t)
     - For each window: aggregated node features (transaction count, volume, etc.)
     - Final dataset ready for GNN input

---

## Figure 11: `autotuning.png`

**Location in README:** Quick Start → Optimize Data Generation

**Description:** Comparison of data-informed vs no-data-informed Bayesian optimization.

**Should show:**
- Side-by-side comparison or single plot with two curves:

  - **X-axis:** Number of optimization trials
  - **Y-axis:** Objective value (FPR + feature importance error)

  - **Curve 1 (No-data-informed):** Model-level optimization only
    - Flat or slowly improving
    - Cannot overcome bad data configuration

  - **Curve 2 (Data-informed):** Two-level optimization
    - Faster convergence
    - Better final performance
    - Lower objective value

- Highlight the benefit of optimizing data generation parameters
- Include Pareto front showing trade-off between FPR and feature importance

---

## Image Specifications

**Recommended format:** PNG (for lossless quality and transparency support)

**Recommended dimensions:**
- Width: 500-800 pixels (will be scaled in README)
- DPI: 150-300 for crisp rendering
- Aspect ratio: 4:3 or 16:9 depending on content

**Style guidelines:**
- Use consistent color scheme across figures
- Clear labels and legends
- Readable font sizes (will be scaled down in README)
- High contrast for visibility
- Avoid red-green color combinations (colorblind-friendly)

---

## Creating Figures

Figures can be generated from:
- Simulation output data
- Real experiments
- Illustrative diagrams (using tools like draw.io, Inkscape, or Python plotting)

Most figures can be generated programmatically using the visualization scripts in `src/visualize/`.

---

## Notes

- All image paths in the main README use relative paths: `assets/images/<filename>.png`
- Ensure filenames match exactly (including underscores vs hyphens)
- If adding new figures, update both this guide and the main README
