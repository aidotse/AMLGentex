"""
Main AML Simulation class
"""
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from .account import Account
from .alert_patterns import AlertPattern
from .normal_models import (
    SingleModel, FanInModel, FanOutModel,
    ForwardModel, MutualModel, PeriodicalModel
)
from src.utils.pattern_types import SAR_PATTERN_TYPES, NORMAL_PATTERN_TYPES
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AMLSimulator:
    """Main simulator for anti-money laundering transaction generation"""

    def __init__(self, config):
        """
        Initialize simulator with configuration dict.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Extract configuration
        self.sim_name = self.config['general']['simulation_name']
        self.random_seed = self.config['general']['random_seed']
        self.total_steps = self.config['general']['total_steps']
        self.base_date = datetime.strptime(self.config['general']['base_date'], '%Y-%m-%d')

        # Set random seed
        np.random.seed(self.random_seed)
        self.random = np.random.RandomState(self.random_seed)

        # Paths
        self.spatial_dir = self.config['spatial']['directory']
        self.output_dir = self.config['output']['directory']

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Data structures
        self.accounts = {}  # account_id -> Account object
        self.transactions = []  # List of transaction records
        self.normal_model_objects = []  # Normal model pattern objects
        self.alert_members = []  # SAR alert memberships
        self.alert_patterns = []  # AlertPattern objects for AML typologies

        # Pattern ID assignment - sequential counter to avoid overlap between SAR and normal patterns
        # Assigns unique patternID for each pattern instance in transaction log
        self.next_pattern_id = 0
        self.csv_to_pattern_id = {}  # Maps (pattern_type, csv_modelID) -> assigned patternID

        # Extract margin ratio for amount reduction in patterns
        self.margin_ratio = self.config['default'].get('margin_ratio', 0.1)

        logger.info(f"Initialized AMLSimulator: {self.sim_name}")
        logger.info(f"Random seed: {self.random_seed}")
        logger.info(f"Total steps: {self.total_steps}")

    @staticmethod
    def linear_to_lognormal_params(mean_linear, std_linear):
        """
        Convert linear-space mean and std to log-space parameters for lognormal distribution.

        For a lognormal distribution with log-space parameters μ and σ:
        - Linear mean = exp(μ + σ²/2)
        - Linear variance = [exp(σ²) - 1] × exp(2μ + σ²)

        This function inverts these relationships to get μ and σ from desired linear statistics.

        Args:
            mean_linear: Desired mean in linear space
            std_linear: Desired standard deviation in linear space

        Returns:
            tuple: (mu, sigma) - log-space parameters for np.random.lognormal()
        """
        variance_linear = std_linear ** 2
        mean_squared = mean_linear ** 2

        # Convert to log-space parameters
        mu = np.log(mean_squared / np.sqrt(mean_squared + variance_linear))
        sigma = np.sqrt(np.log(1 + variance_linear / mean_squared))

        return mu, sigma

    def load_accounts(self):
        """Load accounts from CSV file"""
        account_file = os.path.join(self.spatial_dir, self.config['spatial']['accounts'])
        logger.info(f"Loading accounts from: {account_file}")

        df = pd.read_csv(account_file)

        # Get behavior parameters from config
        default_conf = self.config['default']

        for _, row in df.iterrows():
            account = Account(
                account_id=row['ACCOUNT_ID'],
                customer_id=row['CUSTOMER_ID'],
                initial_balance=row['INIT_BALANCE'],
                is_sar=row['IS_SAR'],
                bank_id=row['BANK_ID'],
                random_state=self.random_seed + row['ACCOUNT_ID'],  # Unique seed per account
                n_steps_balance_history=default_conf.get('n_steps_balance_history', 28),
                salary=row['SALARY'],  # Monthly salary from demographics
                age=row['AGE']  # Age from demographics
            )

            # Set behavior parameters
            account.set_parameters(
                prob_income=default_conf['prob_income'],
                mean_income=default_conf['mean_income'],
                std_income=default_conf['std_income'],
                prob_income_sar=default_conf['prob_income_sar'],
                mean_income_sar=default_conf['mean_income_sar'],
                std_income_sar=default_conf['std_income_sar'],
                mean_outcome=default_conf['mean_outcome'],
                std_outcome=default_conf['std_outcome'],
                mean_outcome_sar=default_conf['mean_outcome_sar'],
                std_outcome_sar=default_conf['std_outcome_sar'],
                prob_spend_cash=default_conf['prob_spend_cash'],
                mean_phone_change_frequency=default_conf['mean_phone_change_frequency'],
                std_phone_change_frequency=default_conf['std_phone_change_frequency'],
                mean_phone_change_frequency_sar=default_conf['mean_phone_change_frequency_sar'],
                std_phone_change_frequency_sar=default_conf['std_phone_change_frequency_sar'],
                mean_bank_change_frequency=default_conf['mean_bank_change_frequency'],
                std_bank_change_frequency=default_conf['std_bank_change_frequency'],
                mean_bank_change_frequency_sar=default_conf['mean_bank_change_frequency_sar'],
                std_bank_change_frequency_sar=default_conf['std_bank_change_frequency_sar']
            )

            self.accounts[row['ACCOUNT_ID']] = account

        logger.info(f"Loaded {len(self.accounts)} accounts")

    def load_transactions(self):
        """Load transaction network from CSV file"""
        tx_file = os.path.join(self.spatial_dir, self.config['spatial']['transactions'])
        logger.info(f"Loading transactions from: {tx_file}")

        df = pd.read_csv(tx_file)

        for _, row in df.iterrows():
            src = self.accounts.get(row['src'])
            dst = self.accounts.get(row['dst'])

            if src and dst:
                src.add_beneficiary(dst)
                dst.add_originator(src)
                src.tx_types[dst.account_id] = row['ttype']

        logger.info(f"Loaded {len(df)} transaction connections")

    def load_normal_models(self):
        """Load normal transaction models and create model objects"""
        models_file = os.path.join(self.spatial_dir, self.config['spatial']['normal_models'])
        if not os.path.exists(models_file):
            logger.info(f"Normal models file not found: {models_file}")
            return

        logger.info(f"Loading normal models from: {models_file}")
        df = pd.read_csv(models_file)

        # Get amount parameters from config
        default_conf = self.config['default']
        min_amount = default_conf['min_amount']
        max_amount = default_conf['max_amount']
        mean_amount = default_conf['mean_amount']
        std_amount = default_conf['std_amount']

        # Get duration and burstiness parameters for normal models
        mean_duration_normal_linear = default_conf['mean_duration_normal']
        std_duration_normal_linear = default_conf['std_duration_normal']
        burstiness_bias_normal = default_conf['burstiness_bias_normal']

        # Convert linear-space duration parameters to log-space for lognormal sampling
        mu_normal, sigma_normal = self.linear_to_lognormal_params(
            mean_duration_normal_linear, std_duration_normal_linear
        )

        # Normal pattern type to ID mapping (10-15 to avoid overlap with SAR patterns 1-8)
        # Use centralized pattern type mappings
        normal_type_to_id = NORMAL_PATTERN_TYPES

        # Group by model ID to create model objects
        model_groups = defaultdict(list)
        model_info = {}

        for _, row in df.iterrows():
            model_id = row['modelID']
            account_id = row['accountID']

            if account_id in self.accounts:
                account = self.accounts[account_id]
                is_main = str(row['isMain']).lower() == 'true'
                model_groups[model_id].append((account, is_main))

                # Store model metadata
                if model_id not in model_info:
                    pattern_type = row['type']
                    model_info[model_id] = {
                        'type': pattern_type,
                        'type_id': normal_type_to_id.get(pattern_type, 10)  # Pattern type ID (10-15)
                    }

        # Create model objects based on type with assigned sequential patternIDs
        from .alert_patterns import PatternScheduler

        for model_id, accounts in model_groups.items():
            if model_id not in model_info:
                continue

            info = model_info[model_id]
            model_type = info['type']
            type_id = info['type_id']

            # Assign new unique patternID (continuing from where SAR patterns left off)
            pattern_id = self.next_pattern_id
            self.next_pattern_id += 1
            self.csv_to_pattern_id[('normal', model_id)] = pattern_id

            random_state = np.random.RandomState(self.random_seed + model_id)

            # Sample duration from lognormal distribution (using log-space parameters)
            duration = int(np.round(random_state.lognormal(mu_normal, sigma_normal)))
            duration = max(2, min(duration, self.total_steps - 1))  # Ensure at least 2 steps

            # Sample start time uniformly from [0, T-D-1]
            max_start = max(0, self.total_steps - duration - 1)
            start_step = random_state.randint(0, max_start + 1) if max_start > 0 else 0
            end_step = start_step + duration

            # Sample burstiness level from bias-based distribution
            burstiness_level = PatternScheduler.sample_burstiness_level(burstiness_bias_normal, random_state)

            if model_type == 'single':
                model = SingleModel(
                    model_id=pattern_id,  # Assigned unique pattern instance ID
                    accounts=accounts,
                    start_step=start_step,
                    end_step=end_step,
                    burstiness_level=burstiness_level,
                    mean_amount=mean_amount,
                    std_amount=std_amount,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    random_state=random_state
                )
                model.type_id = type_id  # Add type_id attribute
                self.normal_model_objects.append(model)

            elif model_type == 'fan_in':
                model = FanInModel(
                    model_id=pattern_id,  # Assigned unique pattern instance ID
                    accounts=accounts,
                    start_step=start_step,
                    end_step=end_step,
                    burstiness_level=burstiness_level,
                    mean_amount=mean_amount,
                    std_amount=std_amount,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    random_state=random_state
                )
                model.type_id = type_id  # Add type_id attribute
                self.normal_model_objects.append(model)

            elif model_type == 'fan_out':
                model = FanOutModel(
                    model_id=pattern_id,  # Assigned unique pattern instance ID
                    accounts=accounts,
                    start_step=start_step,
                    end_step=end_step,
                    burstiness_level=burstiness_level,
                    mean_amount=mean_amount,
                    std_amount=std_amount,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    random_state=random_state
                )
                model.type_id = type_id  # Add type_id attribute
                self.normal_model_objects.append(model)

            elif model_type == 'forward':
                model = ForwardModel(
                    model_id=pattern_id,  # Assigned unique pattern instance ID
                    accounts=accounts,
                    start_step=start_step,
                    end_step=end_step,
                    burstiness_level=burstiness_level,
                    mean_amount=mean_amount,
                    std_amount=std_amount,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    random_state=random_state
                )
                model.type_id = type_id  # Add type_id attribute
                self.normal_model_objects.append(model)

            elif model_type == 'mutual':
                model = MutualModel(
                    model_id=pattern_id,  # Assigned unique pattern instance ID
                    accounts=accounts,
                    start_step=start_step,
                    end_step=end_step,
                    burstiness_level=burstiness_level,
                    mean_amount=mean_amount,
                    std_amount=std_amount,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    random_state=random_state
                )
                model.type_id = type_id  # Add type_id attribute
                self.normal_model_objects.append(model)

            elif model_type == 'periodical':
                model = PeriodicalModel(
                    model_id=pattern_id,  # Assigned unique pattern instance ID
                    accounts=accounts,
                    start_step=start_step,
                    end_step=end_step,
                    burstiness_level=burstiness_level,
                    mean_amount=mean_amount,
                    std_amount=std_amount,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    random_state=random_state
                )
                model.type_id = type_id  # Add type_id attribute
                self.normal_model_objects.append(model)

        logger.info(f"Loaded {len(df)} normal model entries")
        logger.info(f"Created {len(self.normal_model_objects)} normal model objects")

    def load_alert_members(self):
        """Load alert model assignments and create AlertPattern objects"""
        alert_file = os.path.join(self.spatial_dir, self.config['spatial']['alert_members'])
        if not os.path.exists(alert_file):
            logger.info(f"Alert models file not found: {alert_file}")
            return

        logger.info(f"Loading alert models from: {alert_file}")
        df = pd.read_csv(alert_file)

        # Get SAR amount parameters from config
        default_conf = self.config['default']
        min_amount = default_conf['min_amount']
        max_amount = default_conf['max_amount']
        mean_amount_sar = default_conf['mean_amount_sar']
        std_amount_sar = default_conf['std_amount_sar']

        # Get duration and burstiness parameters for alerts
        mean_duration_alert_linear = default_conf['mean_duration_alert']
        std_duration_alert_linear = default_conf['std_duration_alert']
        burstiness_bias_alert = default_conf['burstiness_bias_alert']

        # Convert linear-space duration parameters to log-space for lognormal sampling
        mu_alert, sigma_alert = self.linear_to_lognormal_params(
            mean_duration_alert_linear, std_duration_alert_linear
        )

        # Pattern type to model ID mapping (must match spatial simulation)
        # Use centralized pattern type mappings
        alert_type_to_id = SAR_PATTERN_TYPES

        # Group by alert ID to create patterns
        alert_groups = defaultdict(list)
        alert_info = {}
        alert_phases = defaultdict(lambda: defaultdict(list))  # alert_id -> phase -> [accounts]

        for _, row in df.iterrows():
            account_id = row['accountID']
            model_id = row['modelID']  # Unified column name (was alertID)

            if account_id in self.accounts:
                account = self.accounts[account_id]
                is_main = str(row['isMain']).lower() == 'true'
                phase = row.get('phase', 0)  # Get phase from CSV, default to 0 if not present

                # Add to account's alert list
                if model_id not in account.alerts:
                    account.alerts.append(model_id)

                # Group accounts by model ID
                alert_groups[model_id].append((account, is_main))

                # Group accounts by phase within each model
                alert_phases[model_id][phase].append(account)

                # Store alert metadata (same for all members of this pattern)
                if model_id not in alert_info:
                    pattern_type = row['type']  # Unified column name (was reason)
                    alert_info[model_id] = {
                        'pattern_type': pattern_type,
                        'type_id': alert_type_to_id.get(pattern_type, 0),  # Pattern type ID (1-8)
                        'source_type': row['sourceType']
                    }

        # Create AlertPattern objects with assigned sequential patternIDs
        for model_id, accounts in alert_groups.items():
            if model_id in alert_info:
                info = alert_info[model_id]
                pattern_type = info['pattern_type']

                # Assign new unique patternID
                pattern_id = self.next_pattern_id
                self.next_pattern_id += 1
                self.csv_to_pattern_id[('sar', model_id)] = pattern_id

                # Create random state for this pattern
                random_state = np.random.RandomState(self.random_seed + model_id)

                # Sample duration from lognormal distribution (using log-space parameters)
                duration = int(np.round(random_state.lognormal(mu_alert, sigma_alert)))
                duration = max(2, min(duration, self.total_steps - 1))  # Ensure at least 2 steps

                # Sample start time uniformly from [0, T-D-1]
                max_start = max(0, self.total_steps - duration - 1)
                start_step = random_state.randint(0, max_start + 1) if max_start > 0 else 0
                end_step = start_step + duration

                # Sample burstiness level from bias-based distribution
                from .alert_patterns import PatternScheduler
                burstiness_level = PatternScheduler.sample_burstiness_level(burstiness_bias_alert, random_state)

                # Get phase layers for all patterns
                # Convert defaultdict to regular dict, or None if empty
                phase_layers = dict(alert_phases[model_id]) if alert_phases[model_id] else None

                pattern = AlertPattern(
                    alert_id=pattern_id,  # Assigned unique pattern instance ID
                    pattern_type=pattern_type,
                    accounts=accounts,
                    model_id=info['type_id'],  # Pattern type ID (1-8 for SAR patterns)
                    mean_amount=mean_amount_sar,
                    std_amount=std_amount_sar,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    start_step=start_step,
                    end_step=end_step,
                    burstiness_level=burstiness_level,
                    source_type=info['source_type'],
                    random_state=random_state,
                    phase_layers=phase_layers,
                    margin_ratio=self.margin_ratio
                )
                self.alert_patterns.append(pattern)

        logger.info(f"Loaded {len(df)} alert memberships")
        logger.info(f"Created {len(self.alert_patterns)} alert patterns")

    def execute_normal_transactions(self, step):
        """Execute normal transactions based on model patterns"""
        transactions = []

        # Execute transactions for each normal model object
        for model in self.normal_model_objects:
            tx = model.get_transaction(step)
            if tx:
                from_account = tx['from']
                to_account = tx['to']
                amount = tx['amount']
                tx_type = tx['type']

                # Attempt the transaction
                success = from_account.make_transaction(to_account, amount, tx_type)

                if success:
                    # Add transaction with unified structure
                    transactions.append({
                        'step': step,
                        'type': tx_type,
                        'amount': amount,
                        'nameOrig': from_account.account_id,
                        'nameDest': to_account.account_id,
                        'isSAR': 0,  # Normal model transactions are never SAR
                        'patternID': model.model_id,  # Pattern instance ID (modelID from normal_models.csv)
                        'modelType': model.type_id  # Pattern type ID (10-15 for normal patterns)
                    })

        return transactions

    def execute_alert_transactions(self, step):
        """Execute AML alert pattern transactions"""
        transactions = []

        for pattern in self.alert_patterns:
            # Get transactions scheduled for this step
            scheduled_txs = pattern.get_transactions_for_step(step)

            for tx in scheduled_txs:
                from_account = tx['from']
                to_account = tx['to']
                tx_type = tx['type']

                # Handle dependent transactions (scatter phase of gather-scatter)
                # These have amount=None and reference incoming transactions
                if tx.get('amount') is None and '_incoming_refs' in tx:
                    # Compute amount based on actual successful incoming transactions
                    incoming_refs = tx['_incoming_refs']
                    proportion = tx['_proportion']
                    margin_ratio = tx['_margin_ratio']

                    # Sum the actual amounts from successful incoming transactions
                    total_incoming = sum(
                        ref.get('_actual_amount', 0)
                        for ref in incoming_refs
                        if ref.get('_success', False)
                    )

                    # Compute outgoing amount: total * (1 - margin) * proportion
                    amount = total_incoming * (1 - margin_ratio) * proportion

                    if amount <= 0:
                        # No successful incoming, skip this outgoing transaction
                        continue
                else:
                    amount = tx['amount']

                # Attempt the transaction
                success = from_account.make_transaction(to_account, amount, tx_type)

                # Track success and actual amount for dependent transactions
                tx['_success'] = success
                tx['_actual_amount'] = amount if success else 0

                if success:
                    transactions.append({
                        'step': step,
                        'type': tx_type,
                        'amount': amount,
                        'nameOrig': from_account.account_id,
                        'nameDest': to_account.account_id,
                        'isSAR': 1,  # Alert transactions are always SAR
                        'patternID': pattern.alert_id,  # Pattern instance ID (modelID from alert_models.csv)
                        'modelType': pattern.model_id  # Pattern type ID (1-8 for SAR patterns)
                    })

        return transactions

    def run(self):
        """Run the simulation"""
        logger.info(f"\nStarting AMLSim for {self.total_steps} steps...")
        logger.info("=" * 60)

        # Get list of unique banks for bank switching behavior
        available_banks = list(set(account.bank_id for account in self.accounts.values()))
        logger.info(f"Available banks: {len(available_banks)}")

        all_transactions = []

        # Add initial balance transactions
        for account_id, account in self.accounts.items():
            all_transactions.append({
                'step': 0,
                'type': 'INITALBALANCE',
                'amount': account.balance,
                'nameOrig': -2,  # Source
                'nameDest': account_id,
                'bankOrig': 'source',
                'bankDest': account.bank_id,
                'daysInBankOrig': 0,
                'phoneChangesOrig': 0,
                'oldbalanceOrig': 0.0,
                'newbalanceOrig': 0.0,
                'daysInBankDest': 0,
                'phoneChangesDest': 0,
                'oldbalanceDest': account.balance,
                'newbalanceDest': account.balance * 2,  # Initial double for compatibility
                'isSAR': 0,  # Initial balance transactions are not SAR
                'patternID': -1,
                'modelType': 0  # Generic (not a pattern)
            })

        # Run simulation steps
        for step in range(self.total_steps):
            if step % 100 == 0:
                logger.info(f"Step {step}/{self.total_steps}")

            # Handle account behaviors (income/outcome)
            for account in self.accounts.values():
                txs = account.handle_step(step, all_transactions, available_banks)
                for tx in txs:
                    # Enrich transaction with additional info
                    orig = self.accounts.get(tx['nameOrig'], None) if tx['nameOrig'] >= 0 else None
                    dest = self.accounts.get(tx['nameDest'], None) if tx['nameDest'] >= 0 else None

                    all_transactions.append({
                        'step': tx['step'],
                        'type': tx['type'],
                        'amount': tx['amount'],
                        'nameOrig': tx['nameOrig'],
                        'nameDest': tx['nameDest'],
                        'bankOrig': orig.bank_id if orig else 'source',
                        'bankDest': dest.bank_id if dest else 'sink',
                        'daysInBankOrig': orig.days_in_bank if orig else 0,
                        'phoneChangesOrig': orig.phone_changes if orig else 0,
                        'oldbalanceOrig': (orig.balance + tx['amount']) if (orig and tx['nameOrig'] == orig.account_id) else 0.0,
                        'newbalanceOrig': orig.balance if orig else 0.0,
                        'daysInBankDest': dest.days_in_bank if dest else 0,
                        'phoneChangesDest': dest.phone_changes if dest else 0,
                        'oldbalanceDest': (dest.balance - tx['amount']) if (dest and tx['nameDest'] == dest.account_id) else 0.0,
                        'newbalanceDest': dest.balance if dest else 0.0,
                        'isSAR': tx['isSAR'],
                        'patternID': tx['patternID'],
                        'modelType': tx['modelType']
                    })

            # Execute normal transaction patterns
            pattern_txs = self.execute_normal_transactions(step)
            for tx in pattern_txs:
                orig = self.accounts[tx['nameOrig']]
                dest = self.accounts[tx['nameDest']]

                all_transactions.append({
                    'step': tx['step'],
                    'type': tx['type'],
                    'amount': tx['amount'],
                    'nameOrig': tx['nameOrig'],
                    'nameDest': tx['nameDest'],
                    'bankOrig': orig.bank_id,
                    'bankDest': dest.bank_id,
                    'daysInBankOrig': orig.days_in_bank,
                    'phoneChangesOrig': orig.phone_changes,
                    'oldbalanceOrig': orig.balance + tx['amount'],
                    'newbalanceOrig': orig.balance,
                    'daysInBankDest': dest.days_in_bank,
                    'phoneChangesDest': dest.phone_changes,
                    'oldbalanceDest': dest.balance - tx['amount'],
                    'newbalanceDest': dest.balance,
                    'isSAR': tx['isSAR'],
                    'patternID': tx['patternID'],
                    'modelType': tx['modelType']
                })

            # Inject illicit funds for alert patterns at their start step
            for pattern in self.alert_patterns:
                if step == pattern.start_step:
                    pattern.inject_illicit_funds()

            # Execute AML alert pattern transactions
            alert_txs = self.execute_alert_transactions(step)
            for tx in alert_txs:
                orig = self.accounts[tx['nameOrig']]
                dest = self.accounts[tx['nameDest']]

                all_transactions.append({
                    'step': tx['step'],
                    'type': tx['type'],
                    'amount': tx['amount'],
                    'nameOrig': tx['nameOrig'],
                    'nameDest': tx['nameDest'],
                    'bankOrig': orig.bank_id,
                    'bankDest': dest.bank_id,
                    'daysInBankOrig': orig.days_in_bank,
                    'phoneChangesOrig': orig.phone_changes,
                    'oldbalanceOrig': orig.balance + tx['amount'],
                    'newbalanceOrig': orig.balance,
                    'daysInBankDest': dest.days_in_bank,
                    'phoneChangesDest': dest.phone_changes,
                    'oldbalanceDest': dest.balance - tx['amount'],
                    'newbalanceDest': dest.balance,
                    'isSAR': tx['isSAR'],
                    'patternID': tx['patternID'],
                    'modelType': tx['modelType']
                })

        logger.info("=" * 60)
        logger.info(f"Simulation complete! Generated {len(all_transactions)} transactions")

        self.transactions = all_transactions
        return all_transactions

    def write_output(self):
        """Write transaction log to Parquet file with optimized dtypes"""
        output_file = os.path.join(self.output_dir, self.config['output']['transaction_log'])
        logger.info(f"Writing output to: {output_file}")

        df = pd.DataFrame(self.transactions)

        # Optimize data types for compression and performance
        # Use safe ranges that can scale to larger experiments
        dtype_map = {
            'step': 'int32',              # Safe for very long simulations
            'type': 'category',           # Only 3 values: INITALBALANCE, TRANSFER, CASH
            'amount': 'float64',          # Keep precision for monetary values
            'nameOrig': 'int32',          # Safe for millions of accounts
            'nameDest': 'int32',          # Safe for millions of accounts
            'bankOrig': 'category',       # Limited number of banks
            'bankDest': 'category',       # Limited number of banks
            'daysInBankOrig': 'int32',    # Safe for very long periods
            'phoneChangesOrig': 'int16',  # Safe up to 32k changes
            'oldbalanceOrig': 'float64',  # Keep precision for monetary values
            'newbalanceOrig': 'float64',  # Keep precision for monetary values
            'daysInBankDest': 'int32',    # Safe for very long periods
            'phoneChangesDest': 'int16',  # Safe up to 32k changes
            'oldbalanceDest': 'float64',  # Keep precision for monetary values
            'newbalanceDest': 'float64',  # Keep precision for monetary values
            'isSAR': 'int8',              # Boolean: 0 or 1
            'patternID': 'int32',          # Pattern instance ID (was alertID)
            'modelType': 'int16'           # Pattern type ID (1-8 SAR, 10-15 normal),         # Safe for large number of typology instances
        }

        df = df.astype(dtype_map)

        # Write as Parquet with snappy compression (good balance of speed and compression)
        df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)

        logger.info(f"Wrote {len(self.transactions)} transactions to {output_file}")
        return output_file
