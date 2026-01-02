"""
AML Pattern execution for suspicious activity patterns
"""
import numpy as np
from collections import defaultdict
from .utils import TruncatedNormal


class PatternScheduler:
    """Handles scheduling of transactions within a pattern using burstiness levels"""

    # Beta distribution parameters for each burstiness level
    # Level 1 → (1, 1) = uniform → evenly spread transactions
    # Level 2 → (2, 2) = slightly peaked → moderately spread
    # Level 3 → (0.5, 3) = right-skewed → moderate clustering
    # Level 4 → (0.1, 5) = very right-skewed → extreme clustering/highly bursty
    #   Note: Discretization to integer steps reduces observable burstiness,
    #   so we use very extreme parameters for level 4 to achieve visible burstiness
    BETA_PARAMS = {
        1: (1.0, 1.0),
        2: (2.0, 2.0),
        3: (0.5, 3.0),
        4: (0.1, 5.0)  # Changed from (0.5, 2.0) for much stronger burstiness
    }

    @staticmethod
    def compute_burstiness_probs(bias):
        """
        Compute probability distribution over burstiness levels from bias parameter.

        Args:
            bias: Burstiness bias parameter.
                  bias < 0: favors less bursty (level 1)
                  bias = 0: equal probabilities
                  bias > 0: favors more bursty (level 4)

        Returns:
            dict mapping level (1-4) to probability
        """
        # Fixed scores for each level
        scores = np.array([0, 1, 2, 3], dtype=float)

        # Multiply by bias and exponentiate
        weighted_scores = np.exp(bias * scores)

        # Normalize to probabilities
        probs = weighted_scores / weighted_scores.sum()

        # Return as dict
        return {i+1: probs[i] for i in range(4)}

    @staticmethod
    def sample_burstiness_level(bias, random_state):
        """
        Sample a burstiness level (1-4) based on bias parameter.

        Args:
            bias: Burstiness bias parameter
            random_state: numpy RandomState for reproducibility

        Returns:
            int: Burstiness level (1, 2, 3, or 4)
        """
        probs = PatternScheduler.compute_burstiness_probs(bias)
        levels = list(probs.keys())
        probabilities = list(probs.values())
        return random_state.choice(levels, p=probabilities)

    @staticmethod
    def get_transaction_steps(start_step, end_step, num_transactions, burstiness_level, random_state):
        """
        Generate transaction steps using beta distribution based on burstiness level.

        Args:
            start_step: Starting step of the pattern period
            end_step: Ending step of the pattern period
            num_transactions: Number of transactions to schedule
            burstiness_level: Burstiness level (1-4)
            random_state: numpy RandomState for reproducibility

        Returns:
            list of int: Transaction steps sorted in ascending order
        """
        if num_transactions <= 0:
            return []

        # Ensure start_step < end_step
        if start_step >= end_step:
            return [start_step] * num_transactions  # All at start_step if no valid range

        if num_transactions == 1:
            # Single transaction: place randomly in period
            return [random_state.randint(start_step, end_step)]

        # Get beta parameters for this burstiness level
        alpha, beta = PatternScheduler.BETA_PARAMS[burstiness_level]

        # Sample fractions from beta distribution
        fractions = random_state.beta(alpha, beta, num_transactions)

        # Sort fractions to maintain temporal ordering
        fractions = np.sort(fractions)

        # Map fractions to actual steps in [start_step, end_step)
        period = end_step - start_step
        steps = start_step + (fractions * period).astype(int)

        # Ensure all steps are within bounds
        steps = np.clip(steps, start_step, end_step - 1)

        return steps.tolist()


class AlertPattern:
    """Represents an AML alert pattern"""

    def __init__(self, alert_id, pattern_type, accounts, model_id, mean_amount=300, std_amount=100,
                 min_amount=100, max_amount=500, start_step=0, end_step=100, burstiness_level=1, source_type='SAR',
                 random_state=None, phase_layers=None, margin_ratio=0.1):
        self.alert_id = alert_id
        self.pattern_type = pattern_type
        self.accounts = accounts  # List of (account, is_main) tuples
        self.model_id = model_id
        self.mean_amount = mean_amount
        self.std_amount = std_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.start_step = start_step
        self.end_step = end_step
        self.burstiness_level = burstiness_level
        self.source_type = source_type
        self.random_state = random_state
        self.margin_ratio = margin_ratio  # Fee/margin ratio for amount reduction

        # Separate main and member accounts
        self.main_accounts = [acc for acc, is_main in accounts if is_main]
        self.member_accounts = [acc for acc, is_main in accounts if not is_main]

        # Phase information from spatial graph structure
        # phase_layers is a dict: {phase_num: [accounts]}
        # For scatter_gather: phase 0=main, phase 1=scatter members, phase 2=gather members
        # For gather_scatter: phase 0=main, phase 1=gather members, phase 2=scatter members
        # For stack: phase 0=layer 0, phase 1=layer 1, ...
        self.phase_layers = phase_layers

        # Generate transaction schedule
        self.transaction_schedule = self._generate_schedule()

    def _sample_amount(self):
        """Sample transaction amount from truncated normal distribution"""
        tn = TruncatedNormal(
            mean=self.mean_amount,
            std=self.std_amount,
            lower_bound=self.min_amount,
            upper_bound=self.max_amount
        )
        return tn.sample(self.random_state)

    def _generate_schedule(self):
        """Generate the transaction schedule for this pattern"""
        schedule = []

        if self.pattern_type == "fan_out":
            schedule = self._generate_fan_out()
        elif self.pattern_type == "fan_in":
            schedule = self._generate_fan_in()
        elif self.pattern_type == "cycle":
            schedule = self._generate_cycle()
        elif self.pattern_type == "bipartite":
            schedule = self._generate_bipartite()
        elif self.pattern_type == "stack":
            schedule = self._generate_stack()
        elif self.pattern_type == "scatter_gather":
            schedule = self._generate_scatter_gather()
        elif self.pattern_type == "gather_scatter":
            schedule = self._generate_gather_scatter()
        elif self.pattern_type == "random":
            schedule = self._generate_random()
        else:
            # Unknown pattern type
            pass

        return schedule

    def _generate_fan_out(self):
        """Fan-out: main account sends to multiple members"""
        if not self.main_accounts or not self.member_accounts:
            return []

        main = self.main_accounts[0]
        num_txs = len(self.member_accounts)

        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, num_txs, self.burstiness_level, self.random_state
        )

        schedule = []
        for step, member in zip(steps, self.member_accounts):
            amount = self._sample_amount()
            schedule.append({
                'step': step,
                'from': main,
                'to': member,
                'amount': amount,
                'type': 'TRANSFER'  # Always TRANSFER, source_type only affects fund injection
            })

        return schedule

    def _generate_fan_in(self):
        """Fan-in: multiple members send to main account"""
        if not self.main_accounts or not self.member_accounts:
            return []

        main = self.main_accounts[0]
        num_txs = len(self.member_accounts)

        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, num_txs, self.burstiness_level, self.random_state
        )

        schedule = []
        for step, member in zip(steps, self.member_accounts):
            amount = self._sample_amount()
            schedule.append({
                'step': step,
                'from': member,
                'to': main,
                'amount': amount,
                'type': 'TRANSFER'  # Always TRANSFER, source_type only affects fund injection
            })

        return schedule

    def _generate_cycle(self):
        """Cycle: transactions form a cycle through all members"""
        if not self.main_accounts:
            return []

        all_accounts = self.main_accounts + self.member_accounts
        num_txs = len(all_accounts)

        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, num_txs, self.burstiness_level, self.random_state
        )

        schedule = []
        for i, step in enumerate(steps):
            from_acc = all_accounts[i]
            to_acc = all_accounts[(i + 1) % len(all_accounts)]
            amount = self._sample_amount()
            schedule.append({
                'step': step,
                'from': from_acc,
                'to': to_acc,
                'amount': amount,
                'type': 'TRANSFER'  # Always TRANSFER, source_type only affects fund injection
            })

        return schedule

    def _generate_bipartite(self):
        """Bipartite: two groups, transactions flow between groups"""
        if not self.main_accounts or not self.member_accounts:
            return []

        # Split members into two groups
        mid = len(self.member_accounts) // 2
        group1 = [self.main_accounts[0]] + self.member_accounts[:mid]
        group2 = self.member_accounts[mid:]

        num_txs = len(group1) * len(group2)

        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, num_txs, self.burstiness_level, self.random_state
        )

        schedule = []
        step_idx = 0
        for from_acc in group1:
            for to_acc in group2:
                if step_idx < len(steps):
                    amount = self._sample_amount()
                    schedule.append({
                        'step': steps[step_idx],
                        'from': from_acc,
                        'to': to_acc,
                        'amount': amount,
                        'type': 'TRANSFER'  # Always TRANSFER, source_type only affects fund injection
                    })
                    step_idx += 1

        return schedule

    def _generate_stack(self):
        """Stack: stacked bipartite layers

        Uses phase information from spatial generation to create layered bipartite structure.
        Each phase is a layer, and transactions flow from layer i to layer i+1.
        Phase 0 = layer 0, Phase 1 = layer 1, etc.
        """
        if not self.phase_layers:
            # Fallback to simple chain if no phase information available
            all_accounts = self.main_accounts + self.member_accounts
            num_txs = len(all_accounts) - 1

            steps = PatternScheduler.get_transaction_steps(
                self.start_step, self.end_step, num_txs, self.burstiness_level, self.random_state
            )

            schedule = []
            for i, step in enumerate(steps):
                from_acc = all_accounts[i]
                to_acc = all_accounts[i + 1]
                amount = self._sample_amount()
                schedule.append({
                    'step': step,
                    'from': from_acc,
                    'to': to_acc,
                    'amount': amount,
                    'type': 'TRANSFER'
                })
            return schedule

        # Get layers sorted by phase number
        sorted_phases = sorted(self.phase_layers.keys())
        num_layers = len(sorted_phases)

        if num_layers < 2:
            return []  # Need at least 2 layers for transactions

        period = self.end_step - self.start_step
        schedule = []

        # Track incoming amounts and steps for each account (for amount splitting)
        account_incoming = defaultdict(list)  # account -> [(from_acc, amount, step)]

        # Generate transactions between adjacent layers with amount splitting
        for i in range(num_layers - 1):
            layer_from = self.phase_layers[sorted_phases[i]]
            layer_to = self.phase_layers[sorted_phases[i + 1]]

            # Calculate time period for this layer transition
            layer_start = self.start_step + (i * period // (num_layers - 1))
            layer_end = self.start_step + ((i + 1) * period // (num_layers - 1))

            # Ensure layer_end > layer_start (need at least 1 step for transactions)
            if layer_end <= layer_start:
                layer_end = layer_start + 1
                # If we exceed total period, skip this layer transition
                if layer_end > self.end_step:
                    continue

            # For first layer, sample initial amounts
            if i == 0:
                # First layer: each account sends to 1-3 accounts in next layer
                for from_acc in layer_from:
                    num_targets = min(len(layer_to), self.random_state.randint(1, 4))
                    targets = self.random_state.choice(layer_to, size=num_targets, replace=False)

                    # Sample amount and split among targets
                    total_amount = self._sample_amount()
                    proportions = self.random_state.dirichlet(np.ones(num_targets))

                    for to_acc, proportion in zip(targets, proportions):
                        amount = total_amount * proportion
                        step = self.random_state.randint(layer_start, layer_end)
                        schedule.append({
                            'step': step,
                            'from': from_acc,
                            'to': to_acc,
                            'amount': amount,
                            'type': 'TRANSFER'
                        })
                        account_incoming[to_acc].append((from_acc, amount, step))
            else:
                # Subsequent layers: split received amounts among outgoing connections
                for from_acc in layer_from:
                    if from_acc not in account_incoming:
                        continue  # Skip accounts with no incoming

                    # Calculate total received (with margin for fees)
                    incoming = account_incoming[from_acc]
                    total_received = sum(amt for _, amt, _ in incoming) * (1 - self.margin_ratio)
                    latest_receive_step = max(step for _, _, step in incoming)

                    # Send to 1-3 accounts in next layer
                    num_targets = min(len(layer_to), self.random_state.randint(1, 4))
                    targets = self.random_state.choice(layer_to, size=num_targets, replace=False)

                    # Split total_received among targets
                    proportions = self.random_state.dirichlet(np.ones(num_targets))

                    for to_acc, proportion in zip(targets, proportions):
                        amount = total_received * proportion
                        send_start = max(layer_start, latest_receive_step + 1)
                        # Ensure send_start < layer_end
                        if send_start >= layer_end:
                            send_start = layer_end - 1
                        if send_start < layer_start:
                            send_start = layer_start
                        step = self.random_state.randint(send_start, layer_end)
                        schedule.append({
                            'step': step,
                            'from': from_acc,
                            'to': to_acc,
                            'amount': amount,
                            'type': 'TRANSFER'
                        })
                        account_incoming[to_acc].append((from_acc, amount, step))

        return schedule

    def _generate_scatter_gather(self):
        """Scatter-gather: fan-out then fan-in with overlapping but distinct member sets

        Uses phase_layers to determine which accounts participate in each phase.
        Generates random connections between phases and implements amount splitting.
        Phase 0: originators, Phase 1: intermediaries, Phase 2: beneficiaries
        Pattern: originators → intermediaries → beneficiaries
        """
        if not self.phase_layers:
            # Fallback to simple pattern if no phase information
            if not self.main_accounts:
                return []
            originators = [self.main_accounts[0]]
            intermediaries = self.member_accounts
            beneficiaries = [self.main_accounts[0]]
        else:
            originators = self.phase_layers.get(0, self.main_accounts)
            intermediaries = self.phase_layers.get(1, [])
            beneficiaries = self.phase_layers.get(2, [])

            if not intermediaries:
                intermediaries = self.member_accounts

        period = self.end_step - self.start_step
        schedule = []

        # Phase 1: Generate random connections from originators to intermediaries
        # Each intermediary receives from 1-N originators (random)
        intermediary_connections = {}  # intermediary -> [(originator, amount, step)]

        for intermediary in intermediaries:
            # Each intermediary receives from 1-3 random originators
            n_sources = min(len(originators), self.random_state.randint(1, 4))
            sources = self.random_state.choice(originators, size=n_sources, replace=False)

            intermediary_connections[intermediary] = []
            for source in sources:
                amount = self._sample_amount()
                step = self.random_state.randint(self.start_step, self.start_step + period // 2)
                schedule.append({
                    'step': step,
                    'from': source,
                    'to': intermediary,
                    'amount': amount,
                    'type': 'TRANSFER'
                })
                intermediary_connections[intermediary].append((source, amount, step))

        # Phase 2: Generate random connections from intermediaries to beneficiaries
        # Each intermediary splits its total received among 1-M beneficiaries
        for intermediary, incoming in intermediary_connections.items():
            # Calculate total received (with margin for fees)
            total_received = sum(amt for _, amt, _ in incoming) * (1 - self.margin_ratio)
            latest_receive_step = max(step for _, _, step in incoming)

            # Each intermediary sends to 1-3 random beneficiaries
            n_targets = min(len(beneficiaries), self.random_state.randint(1, 4))
            targets = self.random_state.choice(beneficiaries, size=n_targets, replace=False)

            # Split total_received randomly among targets using Dirichlet distribution
            proportions = self.random_state.dirichlet(np.ones(n_targets))

            for target, proportion in zip(targets, proportions):
                amount = total_received * proportion
                step = self.random_state.randint(latest_receive_step + 1, self.end_step)
                schedule.append({
                    'step': step,
                    'from': intermediary,
                    'to': target,
                    'amount': amount,
                    'type': 'TRANSFER'
                })

        return schedule

    def _generate_gather_scatter(self):
        """Gather-scatter: fan-in then fan-out with overlapping but distinct member sets

        Uses phase_layers to determine which accounts participate in each phase.
        Generates random connections between phases and implements amount splitting.
        Phase 0: hubs, Phase 1: sources, Phase 2: targets
        Pattern: sources → hubs → targets
        """
        if not self.phase_layers:
            # Fallback to simple pattern
            if not self.main_accounts:
                return []
            hubs = [self.main_accounts[0]]
            sources = self.member_accounts
            targets = self.member_accounts
        else:
            hubs = self.phase_layers.get(0, self.main_accounts)
            sources = self.phase_layers.get(1, [])
            targets = self.phase_layers.get(2, [])

            if not sources:
                sources = self.member_accounts
            if not targets:
                targets = self.member_accounts

        period = self.end_step - self.start_step
        schedule = []

        # Phase 1: Generate random connections from sources to hubs (gather)
        # Each hub receives from 1-N sources (random)
        hub_connections = {}  # hub -> [(source, amount, step)]

        for hub in hubs:
            # Each hub receives from 1-3 random sources
            n_sources = min(len(sources), self.random_state.randint(1, 4))
            senders = self.random_state.choice(sources, size=n_sources, replace=False)

            hub_connections[hub] = []
            for sender in senders:
                amount = self._sample_amount()
                step = self.random_state.randint(self.start_step, self.start_step + period // 2)
                schedule.append({
                    'step': step,
                    'from': sender,
                    'to': hub,
                    'amount': amount,
                    'type': 'TRANSFER'
                })
                hub_connections[hub].append((sender, amount, step))

        # Phase 2: Generate random connections from hubs to targets (scatter)
        # Each hub splits its total received among 1-M targets
        for hub, incoming in hub_connections.items():
            # Calculate total received (with margin for fees)
            total_received = sum(amt for _, amt, _ in incoming) * (1 - self.margin_ratio)
            latest_receive_step = max(step for _, _, step in incoming)

            # Each hub sends to 1-3 random targets
            n_receivers = min(len(targets), self.random_state.randint(1, 4))
            receivers = self.random_state.choice(targets, size=n_receivers, replace=False)

            # Split total_received randomly among receivers using Dirichlet distribution
            proportions = self.random_state.dirichlet(np.ones(n_receivers))

            for receiver, proportion in zip(receivers, proportions):
                amount = total_received * proportion
                step = self.random_state.randint(latest_receive_step + 1, self.end_step)
                schedule.append({
                    'step': step,
                    'from': hub,
                    'to': receiver,
                    'amount': amount,
                    'type': 'TRANSFER'
                })

        return schedule

    def _generate_random(self):
        """Random: random transactions among all members"""
        if len(self.main_accounts) + len(self.member_accounts) < 2:
            return []

        all_accounts = self.main_accounts + self.member_accounts
        num_txs = len(all_accounts) * 2  # Arbitrary number

        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, num_txs, self.burstiness_level, self.random_state
        )

        schedule = []
        for step in steps:
            from_acc = self.random_state.choice(all_accounts)
            to_acc = self.random_state.choice(all_accounts)
            if from_acc != to_acc:
                amount = self._sample_amount()
                schedule.append({
                    'step': step,
                    'from': from_acc,
                    'to': to_acc,
                    'amount': amount,
                    'type': 'TRANSFER'  # Always TRANSFER, source_type only affects fund injection
                })

        return schedule

    def get_transactions_for_step(self, step):
        """Get transactions scheduled for a specific step"""
        return [tx for tx in self.transaction_schedule if tx['step'] == step]

    def inject_illicit_funds(self):
        """
        Inject illicit funds into accounts based on source_type.
        For CASH patterns: adds cash_balance to the MAIN account only (matches Java behavior)
        For TRANSFER patterns: no action needed (funds come from regular balance)
        """
        if self.source_type != "CASH":
            return

        # Only inject cash to the main account (matches Java's depositCash to 'acct')
        if not self.main_accounts:
            return

        main_account = self.main_accounts[0]

        # Calculate total amount that will flow through this pattern
        total_amount = sum(tx['amount'] for tx in self.transaction_schedule)

        # Inject cash to the main account (add safety margin)
        main_account.cash_balance = max(total_amount * 1.5, main_account.balance * 0.5)
