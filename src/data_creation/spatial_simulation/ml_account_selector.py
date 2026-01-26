"""
Money Laundering Account Selector

Biases selection of accounts for money laundering typologies based on:
1. Graph structural properties (degree, betweenness, PageRank)
2. KYC-driven locality via seeded probabilistic propagation toward target regions
"""

import networkx as nx
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

logger = logging.getLogger(__name__)


class MoneyLaunderingAccountSelector:
    """
    Selector for choosing accounts for money laundering typologies based on
    structural and KYC-based features.

    Propagation creates "locality fields" that spread from seeds throughout
    the graph structure. Nodes that are graph-structurally close to target
    regions get higher weights, regardless of their own KYC attributes.
    """

    def __init__(self, graph: nx.DiGraph, config: dict, acct_to_bank: dict, seed: int = None):
        """
        Initialize the ML account selector.

        Args:
            graph: Transaction graph (baseline, before ML injection)
            config: Configuration dictionary with ml_selector section
            acct_to_bank: Mapping from account ID to bank ID
            seed: Random seed for reproducibility (overrides config if provided)
        """
        self.g = graph
        # Handle None config (YAML parses empty section as None)
        self.config = config.get('ml_selector') or {}
        self.acct_to_bank = acct_to_bank

        # Initialize reproducible RNG
        rng_seed = seed if seed is not None else self.config.get('seed', 0)
        self.rng = np.random.default_rng(rng_seed)

        # Configuration parameters with defaults
        self.n_seeds_per_target = self.config.get('n_seeds_per_target', 10)
        self.n_target_labels = self.config.get('n_target_labels', 5)
        self.restart_alpha = self.config.get('restart_alpha', 0.15)
        self.seed_strategy = self.config.get('seed_strategy', 'degree')

        # =================================================================
        # PARTICIPATION DECAY
        # =================================================================
        # Controls how often accounts can participate in multiple alert patterns.
        # After each selection, account weight is multiplied by this factor.
        # Combined with structure_weights, this determines participation distribution.
        #
        # How it works with structure weights:
        #   If betweenness gives one account 100x weight of median, with decay=0.3:
        #   - Selection 1: weight 100 → 30
        #   - Selection 2: weight 30 → 9
        #   - Selection 3: weight 9 → 2.7
        #   - Selection 4: weight 2.7 → 0.81 (now below median, unlikely to be picked)
        #
        # Tuning guide:
        #   - 0.1 (aggressive): ~2-3 max participations. Patterns spread widely.
        #   - 0.3 (moderate): ~3-4 max participations. Realistic repeat participants.
        #   - 0.5 (mild): ~5-6 max participations. More concentration allowed.
        #   - 1.0 (none): No decay. Highest-weight accounts dominate all patterns.
        #
        # If seeing too much concentration, either:
        #   1. Lower participation_decay (more aggressive)
        #   2. Lower betweenness weight (it's highly skewed)
        #   3. Increase other weights to balance the distribution
        self.participation_decay = self.config.get('participation_decay', 0.3)

        # =================================================================
        # STRUCTURAL WEIGHTS
        # =================================================================
        # These control which graph positions are favored for ML selection.
        # All metrics are z-score normalized before weighting.
        #
        # degree: Number of connections (in + out).
        #   - High degree = hub accounts with many counterparties.
        #   - Favors: exchanges, money service businesses, active traders.
        #   - ML rationale: More connections = more layering opportunities.
        #   - Typical range: 0.2-0.5. Higher values concentrate on hubs.
        #
        # betweenness: How often node lies on shortest paths between others.
        #   - High betweenness = bridge/broker accounts connecting communities.
        #   - Favors: intermediaries that link otherwise separate groups.
        #   - ML rationale: Good for layering through "neutral" middlemen.
        #   - Typical range: 0.2-0.4. Can create very skewed distributions.
        #   - Warning: Betweenness is highly skewed - a few nodes dominate.
        #     Use lower weights (0.1-0.2) or combine with aggressive decay.
        #
        # pagerank: Importance based on incoming links from important nodes.
        #   - High PageRank = receives from other important accounts.
        #   - Favors: accounts that accumulate from multiple sources.
        #   - ML rationale: Collection points in gather/fan-in patterns.
        #   - Typical range: 0.2-0.4. More evenly distributed than betweenness.
        self.structure_weights = self.config.get('structure_weights', {
            'degree': 0.4,
            'betweenness': 0.2,  # Lower default due to high skew
            'pagerank': 0.4
        })

        # =================================================================
        # PROPAGATION WEIGHTS (Locality Fields)
        # =================================================================
        # Controls geographic/categorical clustering of ML accounts.
        # Uses Personalized PageRank from seed nodes to spread "ML affinity"
        # through the graph structure.
        #
        # city (or other categorical attribute): Weight for locality propagation.
        #   - Higher values make ML accounts cluster in certain regions.
        #   - 0.0 = no geographic clustering, purely structural selection.
        #   - 1.0 = strong clustering around seed locations.
        #   - ML rationale: Real ML networks often have geographic concentration.
        self.propagation_weights = self.config.get('propagation_weights', {
            'city': 0.5
        })

        # =================================================================
        # KYC WEIGHTS
        # =================================================================
        # Controls selection bias based on account KYC attributes.
        # All attributes are z-score normalized (log-transformed for skewed ones).
        #
        # init_balance: Account balance at simulation start.
        #   - Higher balance = capacity for larger transactions.
        #   - ML rationale: Need sufficient funds to move meaningful amounts.
        #   - Typical range: 0.0-0.3. Set to 0 if balance shouldn't matter.
        #
        # salary: Monthly income (from demographics).
        #   - Higher salary = transactions appear more legitimate.
        #   - ML rationale: High earners can justify larger transfers.
        #   - Typical range: 0.0-0.2. Correlated with balance.
        #
        # age: Account holder age in years.
        #   - Can bias toward middle-aged (peak earning) or any age group.
        #   - ML rationale: Varies by typology. Mules often younger.
        #   - Typical range: 0.0-0.1. Usually less important than structure.
        self.kyc_weights = self.config.get('kyc_weights', {
            'init_balance': 0.1,
            'salary': 0.0,
            'age': 0.0
        })

        # Labels to propagate (each creates a global locality field)
        self.propagate_labels = self.config.get('propagate_labels', ['city'])

        # Storage for computed features
        self.structural_features = {}
        self.propagation_scores = {}  # "label_type_global" -> {node: score}

        # Final selection weights (relative, not probabilities)
        # These are softmax-style weights: exp(score - max_score) for numerical stability.
        # They're normalized on-the-fly in weighted_choice() for candidate sampling.
        # To get global probabilities: probs = {n: w/sum(ml_weights.values()) for n, w in ml_weights.items()}
        self.ml_weights = {}  # node -> relative weight
        self.ml_weights_by_bank = defaultdict(dict)  # bank -> {node: relative weight}

        # Target labels and their seeds
        self.target_labels = defaultdict(list)  # label_type -> [target_values]
        self.seeds_by_target = defaultdict(list)  # "label_type:target_value" -> [seed_nodes]

        logger.info(f"Initialized ML Account Selector with config: {self.config}")

    def prepare(self):
        """
        Main preparation method that computes all features and weights.
        Call this once after the baseline graph is built and before alert injection.
        """
        logger.info("Preparing ML account selector...")

        # Step 1: Precompute structural metrics
        self._compute_structural_metrics()

        # Step 2: Select target labels and seeds
        self._select_seeds()

        # Step 3: Probabilistic label propagation toward targets
        self._propagate_labels()

        # Step 4: Combine into final weights
        self._compute_final_weights()

        # Step 5: Cache bank-specific weights
        self._cache_bank_weights()

        logger.info("ML account selector preparation complete")
        self._log_statistics()

    def _compute_structural_metrics(self):
        """
        Compute structural graph metrics: degree, betweenness, PageRank, etc.
        """
        logger.info("Computing structural metrics...")

        nodes = list(self.g.nodes())
        n_nodes = len(nodes)

        # Degree (in + out for directed graph)
        degree_dict = dict(self.g.degree())

        # PageRank (use standard nx implementation)
        try:
            pagerank_dict = nx.pagerank(self.g, alpha=0.85, max_iter=100)
        except:
            logger.warning("PageRank computation failed, using uniform values")
            pagerank_dict = {n: 1.0 / n_nodes for n in nodes}

        # Betweenness (approximate for large graphs using sampling)
        # Uses directed paths by default; pass seed for reproducibility
        try:
            if n_nodes > 5000:
                # Use approximation for large graphs with reproducible seed
                k = min(500, n_nodes // 10)
                btw_seed = int(self.rng.integers(1e9))
                betweenness_dict = nx.betweenness_centrality(
                    self.g, k=k, seed=btw_seed, normalized=True
                )
                logger.info(f"Computed approximate betweenness with k={k} samples, seed={btw_seed}")
            else:
                betweenness_dict = nx.betweenness_centrality(self.g, normalized=True)
        except Exception as e:
            logger.warning(f"Betweenness computation failed: {e}, using degree as proxy")
            # Fallback: use degree as proxy for betweenness
            max_deg = max(degree_dict.values()) if degree_dict else 1
            betweenness_dict = {n: d / max_deg for n, d in degree_dict.items()}

        # Store all structural features
        for node in nodes:
            self.structural_features[node] = {
                'degree': degree_dict.get(node, 0),
                'betweenness': betweenness_dict.get(node, 0),
                'pagerank': pagerank_dict.get(node, 0)
            }

        logger.info(f"Computed structural metrics for {len(nodes)} nodes")

    def _select_seeds(self):
        """
        Select target label values and seeds within each target.

        For each label type (e.g., 'city'):
        1. Identify all unique values in the graph
        2. Sample n_target_labels target values (e.g., target cities)
        3. For each target, select n_seeds_per_target seeds from nodes with that value
        """
        logger.info(f"Selecting target labels and seeds...")

        nodes = list(self.g.nodes())

        for label_type in self.propagate_labels:
            # Build mapping: label_value -> [nodes with that value]
            nodes_by_value = defaultdict(list)
            for node in nodes:
                value = self.g.nodes[node].get(label_type)
                if value is not None:
                    nodes_by_value[value].append(node)

            if not nodes_by_value:
                logger.warning(f"No nodes have attribute '{label_type}', skipping")
                continue

            # Sample target values
            all_values = list(nodes_by_value.keys())
            n_targets = min(self.n_target_labels, len(all_values))

            # Weight target selection by number of nodes (larger pools more likely)
            value_weights = np.array([len(nodes_by_value[v]) for v in all_values])
            value_probs = value_weights / value_weights.sum()

            target_values = self.rng.choice(
                all_values,
                size=n_targets,
                replace=False,
                p=value_probs
            )
            self.target_labels[label_type] = list(target_values)

            logger.info(f"Selected {n_targets} target {label_type}s: {target_values[:5]}{'...' if n_targets > 5 else ''}")

            # For each target value, select seeds from nodes with that value
            for target_value in target_values:
                candidate_nodes = nodes_by_value[target_value]
                n_seeds = min(self.n_seeds_per_target, len(candidate_nodes))

                if n_seeds == 0:
                    continue

                # Select seeds based on strategy
                if self.seed_strategy == 'random':
                    seeds = list(self.rng.choice(candidate_nodes, size=n_seeds, replace=False))

                elif self.seed_strategy == 'degree':
                    # Bias towards high-degree nodes, but softened with sqrt to avoid
                    # concentrating all seeds on top hubs
                    degrees = np.array([self.structural_features[n]['degree'] for n in candidate_nodes])
                    weights = np.sqrt(degrees + 1.0)  # +1 so degree-0 nodes have small chance
                    probs = weights / weights.sum()
                    seeds = list(self.rng.choice(candidate_nodes, size=n_seeds,
                                                 replace=False, p=probs))

                elif self.seed_strategy == 'betweenness':
                    # Bias towards high-betweenness nodes, softened with sqrt
                    betweenness = np.array([self.structural_features[n]['betweenness'] for n in candidate_nodes])
                    weights = np.sqrt(betweenness + 1e-9)  # small epsilon for zeros
                    probs = weights / weights.sum()
                    seeds = list(self.rng.choice(candidate_nodes, size=n_seeds,
                                                 replace=False, p=probs))
                else:
                    logger.warning(f"Unknown seed strategy: {self.seed_strategy}, using random")
                    seeds = list(self.rng.choice(candidate_nodes, size=n_seeds, replace=False))

                label_key = f"{label_type}:{target_value}"
                self.seeds_by_target[label_key] = seeds
                logger.debug(f"Selected {len(seeds)} seeds for {label_key}")

        total_seeds = sum(len(s) for s in self.seeds_by_target.values())
        logger.info(f"Selected {total_seeds} total seeds across {len(self.seeds_by_target)} target groups")

    def _propagate_labels(self):
        """
        Perform probabilistic label propagation (Personalized PageRank) toward targets.

        For each label type:
        1. Compute PPR from seeds of each target value
        2. Combine all target PPR scores into a single global locality field

        Result: Nodes structurally close to ANY target get high propagation scores,
        regardless of their own attribute values.
        """
        logger.info("Propagating labels via Personalized PageRank toward targets...")

        nodes = list(self.g.nodes())

        for label_type in self.propagate_labels:
            target_values = self.target_labels.get(label_type, [])

            if not target_values:
                logger.warning(f"No targets for label type '{label_type}', skipping propagation")
                continue

            # Compute PPR for each target and collect scores
            target_ppr_scores = {}  # target_value -> {node: score}

            for target_value in target_values:
                label_key = f"{label_type}:{target_value}"
                seeds = self.seeds_by_target.get(label_key, [])

                if not seeds:
                    continue

                # Create personalization vector (uniform over seeds for this target)
                # Only include seeds - nx.pagerank treats missing keys as 0
                personalization = {seed: 1.0 / len(seeds) for seed in seeds}

                try:
                    # Compute Personalized PageRank toward this target
                    # Use undirected graph for proximity (neighborhood regardless of flow direction)
                    undirected_g = self.g.to_undirected(as_view=True)
                    ppr_scores = nx.pagerank(
                        undirected_g,
                        alpha=1.0 - self.restart_alpha,  # nx uses alpha = 1 - restart_prob
                        personalization=personalization,
                        max_iter=100
                    )
                    target_ppr_scores[target_value] = ppr_scores
                    logger.debug(f"Computed PPR for target '{target_value}' from {len(seeds)} seeds")

                except Exception as e:
                    logger.warning(f"PPR computation failed for {label_key}: {e}")
                    # Fallback: uniform scores
                    target_ppr_scores[target_value] = {n: 1.0 / len(nodes) for n in nodes}

            # Combine target PPR scores into single global locality field
            # q(n) = sum_c (pi_c * q_c(n)) where pi_c is uniform (or could be weighted)
            if target_ppr_scores:
                global_key = f"{label_type}_global"
                combined_scores = {n: 0.0 for n in nodes}

                # Uniform weighting across targets (could be made configurable)
                n_targets = len(target_ppr_scores)
                target_weight = 1.0 / n_targets

                for target_value, ppr_scores in target_ppr_scores.items():
                    for node in nodes:
                        combined_scores[node] += target_weight * ppr_scores.get(node, 0.0)

                self.propagation_scores[global_key] = combined_scores
                logger.info(f"Combined {n_targets} target PPR fields into '{global_key}'")

    def _compute_final_weights(self):
        """
        Combine structural features, propagation scores, and KYC features into final weights.

        Uses log + z-score normalization for heavy-tailed features (degree, PageRank, PPR).
        Final score = β·z_struct + γ·z_ppr + δ·z_kyc, then softmax via exp().

        Propagation scores are global locality fields - a node's own attribute value
        doesn't matter, only its structural proximity to target regions.
        """
        logger.info("Computing final ML selection weights...")

        nodes = list(self.g.nodes())

        # Z-score normalize structural features (with log transform for heavy-tailed)
        z_structural = self._normalize_structural_features()

        # Z-score normalize propagation scores (with log transform)
        z_propagation = self._normalize_propagation_scores()

        # Z-score normalize KYC features if needed
        # Note: age uses no log (not heavy-tailed), salary/balance use log
        z_kyc = {}
        for kyc_feature, weight in self.kyc_weights.items():
            if weight != 0:
                kyc_dict = {}
                for node in nodes:
                    kyc_value = self.g.nodes[node].get(kyc_feature, 0.0)
                    kyc_value = float(kyc_value) if kyc_value else 0.0
                    kyc_dict[node] = kyc_value

                # Age is not heavy-tailed, don't apply log
                apply_log = kyc_feature != 'age'
                z_kyc[kyc_feature] = self._zscore_dict(kyc_dict, apply_log=apply_log)

        # Compute final score for each node: linear combination of z-scores
        scores = {}
        for node in nodes:
            score = 0.0

            # Add structural component (z-scored)
            for feature, weight in self.structure_weights.items():
                if weight != 0 and feature in z_structural.get(node, {}):
                    score += weight * z_structural[node][feature]

            # Add propagation component (z-scored global locality fields)
            for label_type in self.propagate_labels:
                global_key = f"{label_type}_global"
                prop_weight = self.propagation_weights.get(label_type, 0.0)

                if prop_weight != 0 and global_key in z_propagation:
                    score += prop_weight * z_propagation[global_key].get(node, 0.0)

            # Add direct KYC component (z-scored)
            for kyc_feature, weight in self.kyc_weights.items():
                if weight != 0 and kyc_feature in z_kyc:
                    score += weight * z_kyc[kyc_feature].get(node, 0.0)

            scores[node] = score

        # Convert to softmax-style weights with numerical stability
        # Subtract max for stability: exp(score - max) prevents overflow
        score_max = max(scores.values())
        for node in nodes:
            self.ml_weights[node] = np.exp(scores[node] - score_max)

        logger.info(f"Computed final weights for {len(nodes)} nodes")

    def _cache_bank_weights(self):
        """
        Cache weights split by bank ID for efficient bank-constrained sampling.
        """
        logger.info("Caching bank-specific weights...")

        for node, weight in self.ml_weights.items():
            bank_id = self.acct_to_bank.get(node)
            if bank_id is not None:
                self.ml_weights_by_bank[bank_id][node] = weight

        logger.info(f"Cached weights for {len(self.ml_weights_by_bank)} banks")

    def _zscore_dict(self, values_dict: Dict, apply_log: bool = False) -> Dict:
        """
        Z-score normalize a dictionary of values.

        Args:
            values_dict: {node: value}
            apply_log: If True, apply log1p transform before z-scoring (for heavy-tailed features)

        Returns:
            Z-scored dictionary {node: z_value}
        """
        nodes = list(values_dict.keys())
        vals = np.array([values_dict[n] for n in nodes], dtype=float)

        if apply_log:
            # For heavy-tailed: log1p handles zeros gracefully
            vals = np.log1p(vals)

        mu = vals.mean()
        sd = vals.std() + 1e-12  # Avoid division by zero

        return {nodes[i]: (vals[i] - mu) / sd for i in range(len(nodes))}

    def _normalize_structural_features(self) -> Dict:
        """
        Normalize structural features using log + z-score for heavy-tailed distributions.

        Returns:
            {node: {feature: z_value}}
        """
        nodes = list(self.structural_features.keys())
        normalized = {n: {} for n in nodes}

        # Extract each feature into a dict
        degree_dict = {n: self.structural_features[n]['degree'] for n in nodes}
        betweenness_dict = {n: self.structural_features[n]['betweenness'] for n in nodes}
        pagerank_dict = {n: self.structural_features[n]['pagerank'] for n in nodes}

        # Z-score normalization
        # - degree, pagerank: log + z-score (heavy-tailed)
        # - betweenness: z-score only (already in [0,1], many zeros, log collapses range)
        z_degree = self._zscore_dict(degree_dict, apply_log=True)
        z_betweenness = self._zscore_dict(betweenness_dict, apply_log=False)
        z_pagerank = self._zscore_dict(pagerank_dict, apply_log=True)

        for n in nodes:
            normalized[n] = {
                'degree': z_degree[n],
                'betweenness': z_betweenness[n],
                'pagerank': z_pagerank[n]
            }

        return normalized

    def _normalize_propagation_scores(self) -> Dict:
        """
        Normalize propagation scores using log + z-score.

        Returns:
            {global_key: {node: z_value}}
        """
        normalized = {}

        for global_key, scores in self.propagation_scores.items():
            # PPR scores are heavy-tailed, use log + z-score
            normalized[global_key] = self._zscore_dict(scores, apply_log=True)

        return normalized

    def weighted_choice(self, candidates: List) -> any:
        """
        Choose a single account from candidates using weighted sampling.
        Applies participation decay after selection to spread future selections.

        Args:
            candidates: List of candidate account IDs

        Returns:
            Selected account ID
        """
        if not candidates:
            raise ValueError("Cannot choose from empty candidate list")

        weights = np.array([self.ml_weights.get(c, 1.0) for c in candidates])

        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(candidates)) / len(candidates)

        choice = self.rng.choice(candidates, p=probs)
        self._apply_decay(choice)
        return choice

    def weighted_choice_bank(self, candidates: List, bank_id: str) -> any:
        """
        Choose a single account from candidates constrained to a bank.
        Applies participation decay after selection.

        Args:
            candidates: List of candidate account IDs
            bank_id: Bank ID constraint

        Returns:
            Selected account ID
        """
        if not candidates:
            raise ValueError("Cannot choose from empty candidate list")

        if bank_id in self.ml_weights_by_bank:
            weight_dict = self.ml_weights_by_bank[bank_id]
        else:
            weight_dict = self.ml_weights

        weights = np.array([weight_dict.get(c, 1.0) for c in candidates])

        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(candidates)) / len(candidates)

        choice = self.rng.choice(candidates, p=probs)
        self._apply_decay(choice)
        return choice

    def _apply_decay(self, account):
        """Apply participation decay to an account's weight after selection."""
        if account in self.ml_weights:
            self.ml_weights[account] *= self.participation_decay

        bank_id = self.acct_to_bank.get(account)
        if bank_id and bank_id in self.ml_weights_by_bank:
            if account in self.ml_weights_by_bank[bank_id]:
                self.ml_weights_by_bank[bank_id][account] *= self.participation_decay

    def _log_statistics(self):
        """
        Log statistics about the selector for validation.
        """
        logger.info("=" * 60)
        logger.info("ML Account Selector Statistics")
        logger.info("=" * 60)

        # Weight distribution
        weights = list(self.ml_weights.values())
        logger.info(f"Weight statistics: min={min(weights):.4f}, max={max(weights):.4f}, "
                   f"mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")

        # Structural feature distribution
        degrees = [f['degree'] for f in self.structural_features.values()]
        logger.info(f"Degree statistics: min={min(degrees)}, max={max(degrees)}, "
                   f"mean={np.mean(degrees):.2f}")

        betweenness = [f['betweenness'] for f in self.structural_features.values()]
        logger.info(f"Betweenness statistics: min={min(betweenness):.6f}, "
                   f"max={max(betweenness):.6f}, mean={np.mean(betweenness):.6f}")

        # Target labels
        for label_type in self.propagate_labels:
            targets = self.target_labels.get(label_type, [])
            logger.info(f"Target {label_type}s ({len(targets)}): {targets[:5]}{'...' if len(targets) > 5 else ''}")

        # Propagation score statistics
        for global_key, scores in self.propagation_scores.items():
            score_values = list(scores.values())
            logger.info(f"Propagation '{global_key}': min={min(score_values):.6f}, "
                       f"max={max(score_values):.6f}, mean={np.mean(score_values):.6f}")

        # Seed distribution
        total_seeds = sum(len(s) for s in self.seeds_by_target.values())
        logger.info(f"Total seeds: {total_seeds} across {len(self.seeds_by_target)} target groups")
        for label_key, seeds in list(self.seeds_by_target.items())[:10]:
            logger.info(f"  {label_key}: {len(seeds)} seeds")
        if len(self.seeds_by_target) > 10:
            logger.info(f"  ... and {len(self.seeds_by_target) - 10} more")

        logger.info("=" * 60)

    def plot_ml_selection_analysis(self, output_dir: str):
        """
        Generate plots analyzing how ML selection correlates with structural/KYC components.

        Creates a multi-panel figure comparing distributions of selected (SAR) vs
        non-selected accounts across all selection components.

        Args:
            output_dir: Directory to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping ML selection plots")
            return

        # Get SAR vs non-SAR accounts from graph
        sar_accounts = set()
        non_sar_accounts = set()
        for node in self.g.nodes():
            if self.g.nodes[node].get('is_sar', False):
                sar_accounts.add(node)
            else:
                non_sar_accounts.add(node)

        if len(sar_accounts) == 0:
            logger.warning("No SAR accounts found, skipping ML selection plots")
            return

        logger.info(f"Generating ML selection analysis plots: {len(sar_accounts)} SAR, {len(non_sar_accounts)} non-SAR accounts")

        # Collect data for plotting
        plot_data = {}

        # Structural features
        for feature in ['degree', 'betweenness', 'pagerank']:
            sar_vals = [self.structural_features[n][feature] for n in sar_accounts if n in self.structural_features]
            non_sar_vals = [self.structural_features[n][feature] for n in non_sar_accounts if n in self.structural_features]
            plot_data[feature] = {'SAR': sar_vals, 'Non-SAR': non_sar_vals}

        # KYC features
        for kyc_feature in ['init_balance', 'salary', 'age']:
            sar_vals = [self.g.nodes[n].get(kyc_feature, 0) for n in sar_accounts]
            non_sar_vals = [self.g.nodes[n].get(kyc_feature, 0) for n in non_sar_accounts]
            # Filter out None/0 values for cleaner plots
            sar_vals = [v for v in sar_vals if v is not None and v > 0]
            non_sar_vals = [v for v in non_sar_vals if v is not None and v > 0]
            if sar_vals and non_sar_vals:
                plot_data[kyc_feature] = {'SAR': sar_vals, 'Non-SAR': non_sar_vals}

        # Propagation scores
        for global_key, scores in self.propagation_scores.items():
            sar_vals = [scores.get(n, 0) for n in sar_accounts]
            non_sar_vals = [scores.get(n, 0) for n in non_sar_accounts]
            plot_data[global_key] = {'SAR': sar_vals, 'Non-SAR': non_sar_vals}

        # Final selection weights
        sar_weights = [self.ml_weights.get(n, 0) for n in sar_accounts]
        non_sar_weights = [self.ml_weights.get(n, 0) for n in non_sar_accounts]
        plot_data['selection_weight'] = {'SAR': sar_weights, 'Non-SAR': non_sar_weights}

        # Create figure
        n_plots = len(plot_data)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, (feature_name, data) in enumerate(plot_data.items()):
            ax = axes[idx]

            sar_vals = np.array(data['SAR'])
            non_sar_vals = np.array(data['Non-SAR'])

            # Use log scale for heavy-tailed features
            use_log = feature_name in ['degree', 'betweenness', 'pagerank', 'init_balance',
                                       'salary', 'selection_weight'] or 'global' in feature_name

            if use_log and len(sar_vals) > 0 and len(non_sar_vals) > 0:
                # Filter positive values for log scale
                sar_vals = sar_vals[sar_vals > 0]
                non_sar_vals = non_sar_vals[non_sar_vals > 0]
                if len(sar_vals) > 0 and len(non_sar_vals) > 0:
                    # Create log-spaced bins
                    all_vals = np.concatenate([sar_vals, non_sar_vals])
                    bins = np.logspace(np.log10(all_vals.min()), np.log10(all_vals.max()), 30)
                    ax.hist(non_sar_vals, bins=bins, alpha=0.6, label=f'Non-SAR (n={len(non_sar_vals)})', density=True)
                    ax.hist(sar_vals, bins=bins, alpha=0.6, label=f'SAR (n={len(sar_vals)})', density=True)
                    ax.set_xscale('log')
            else:
                ax.hist(non_sar_vals, bins=30, alpha=0.6, label=f'Non-SAR (n={len(non_sar_vals)})', density=True)
                ax.hist(sar_vals, bins=30, alpha=0.6, label=f'SAR (n={len(sar_vals)})', density=True)

            ax.set_xlabel(feature_name.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

            # Add median lines and ratio
            if len(sar_vals) > 0 and len(non_sar_vals) > 0:
                sar_median = np.median(sar_vals)
                non_sar_median = np.median(non_sar_vals)
                ratio = sar_median / non_sar_median if non_sar_median > 0 else float('inf')
                ax.axvline(sar_median, color='C1', linestyle='--', alpha=0.8)
                ax.axvline(non_sar_median, color='C0', linestyle='--', alpha=0.8)
                ax.set_title(f'{feature_name}\nSAR/Non-SAR median ratio: {ratio:.2f}')

        # Hide unused axes
        for idx in range(len(plot_data), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save plot
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'ml_selection_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ML selection analysis plot to {plot_path}")

        # Log summary statistics
        logger.info("ML Selection Analysis Summary:")
        for feature_name, data in plot_data.items():
            sar_vals = np.array(data['SAR'])
            non_sar_vals = np.array(data['Non-SAR'])
            if len(sar_vals) > 0 and len(non_sar_vals) > 0:
                sar_median = np.median(sar_vals)
                non_sar_median = np.median(non_sar_vals)
                ratio = sar_median / non_sar_median if non_sar_median > 0 else float('inf')
                logger.info(f"  {feature_name}: SAR median={sar_median:.4f}, Non-SAR median={non_sar_median:.4f}, ratio={ratio:.2f}")


def weighted_choice_simple(candidates: List, weights: List) -> any:
    """
    Simple weighted choice helper (for standalone use).

    Args:
        candidates: List of items to choose from
        weights: List of weights (same length as candidates)

    Returns:
        Selected item
    """
    if not candidates:
        raise ValueError("Cannot choose from empty list")

    weights_array = np.array(weights)
    if weights_array.sum() > 0:
        probs = weights_array / weights_array.sum()
    else:
        probs = np.ones(len(candidates)) / len(candidates)

    return np.random.choice(candidates, p=probs)
