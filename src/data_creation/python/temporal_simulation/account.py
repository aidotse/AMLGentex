"""
Account class for AML simulation
"""
import numpy as np
from collections import deque
from .utils import TruncatedNormal, sigmoid
from .salary_distribution import SalaryDistribution


class Account:
    """Represents a bank account in the simulation"""

    def __init__(self, account_id, customer_id, initial_balance,
                 is_sar, bank_id, random_state=None, n_steps_balance_history=28):
        self.account_id = account_id
        self.customer_id = customer_id
        self.balance = initial_balance
        self.is_sar = is_sar
        self.bank_id = bank_id
        self.cash_balance = 0.0

        # Random state for reproducibility
        self.random = np.random.RandomState(random_state)

        # Transaction tracking
        self.beneficiaries = []  # Accounts this account sends money to
        self.originators = []  # Accounts that send money to this account
        self.tx_types = {}  # Transaction types for specific beneficiaries

        # Balance history for spending behavior (configurable window)
        self.balance_history = deque(maxlen=n_steps_balance_history)
        self.balance_history.append(initial_balance)

        # Cache for running mean (optimization)
        self._balance_sum = initial_balance
        self._balance_count = 1

        # Behavior parameters (will be set by simulator)
        self.prob_income = 0.0
        self.mean_income = 0.0
        self.std_income = 0.0
        self.prob_income_sar = 0.0
        self.mean_income_sar = 0.0
        self.std_income_sar = 0.0
        self.mean_outcome = 0.0
        self.std_outcome = 0.0
        self.mean_outcome_sar = 0.0
        self.std_outcome_sar = 0.0
        self.prob_spend_cash = 0.0

        # Monthly income/outcome (will be computed)
        self.monthly_income = 0.0
        self.monthly_income_sar = 0.0
        self.monthly_outcome = 0.0
        self.monthly_outcome_sar = 0.0
        self.step_monthly_outcome = 25  # Default day for monthly expense

        # Phone and bank change parameters
        self.mean_phone_change_frequency = 0.0
        self.std_phone_change_frequency = 0.0
        self.mean_bank_change_frequency = 0.0
        self.std_bank_change_frequency = 0.0

        # Phone and bank change tracking
        self.days_in_bank = 0
        self.phone_changes = 0
        self.days_until_phone_change = 0
        self.days_until_bank_change = 0

        # Bounds for phone/bank changes (from Java implementation)
        self.LB_PHONE = 1.0  # At least 1 day until phone change
        self.UB_PHONE = 365.0 * 10.0  # Max 10 years
        self.LB_BANK = 0.0
        self.UB_BANK = 365.0 * 10.0  # Max 10 years

        # Alert tracking
        self.alerts = []  # List of alert IDs this account participates in

    def set_parameters(self, prob_income, mean_income, std_income,
                       prob_income_sar, mean_income_sar, std_income_sar,
                       mean_outcome, std_outcome, mean_outcome_sar, std_outcome_sar,
                       prob_spend_cash,
                       mean_phone_change_frequency, std_phone_change_frequency,
                       mean_phone_change_frequency_sar, std_phone_change_frequency_sar,
                       mean_bank_change_frequency, std_bank_change_frequency,
                       mean_bank_change_frequency_sar, std_bank_change_frequency_sar):
        """Set behavioral parameters for this account"""
        self.prob_income = prob_income
        self.mean_income = mean_income
        self.std_income = std_income
        self.prob_income_sar = prob_income_sar
        self.mean_income_sar = mean_income_sar
        self.std_income_sar = std_income_sar
        self.mean_outcome = mean_outcome
        self.std_outcome = std_outcome
        self.mean_outcome_sar = mean_outcome_sar
        self.std_outcome_sar = std_outcome_sar
        self.prob_spend_cash = prob_spend_cash

        # Set phone/bank change parameters based on SAR status
        if self.is_sar:
            self.mean_phone_change_frequency = mean_phone_change_frequency_sar
            self.std_phone_change_frequency = std_phone_change_frequency_sar
            self.mean_bank_change_frequency = mean_bank_change_frequency_sar
            self.std_bank_change_frequency = std_bank_change_frequency_sar
        else:
            self.mean_phone_change_frequency = mean_phone_change_frequency
            self.std_phone_change_frequency = std_phone_change_frequency
            self.mean_bank_change_frequency = mean_bank_change_frequency
            self.std_bank_change_frequency = std_bank_change_frequency

        # Sample initial days until phone/bank change
        self.days_until_phone_change = self._sample_days_until_phone_change()
        self.days_until_bank_change = self._sample_days_until_bank_change()

        # Sample monthly salary from realistic Swedish salary distribution
        # Matches Java implementation: SalaryDistribution().sample()
        salary_dist = SalaryDistribution()
        self.monthly_income = salary_dist.sample(self.random)
        self.monthly_income_sar = salary_dist.sample(self.random)

        # Compute monthly outcome as function of monthly income
        # Matches Java: TruncatedNormal(0.5 * income, 0.1 * income, 0.1 * income, 0.9 * income)
        tn_outcome = TruncatedNormal(
            mean=0.5 * self.monthly_income,
            std=0.1 * self.monthly_income,
            lower_bound=0.1 * self.monthly_income,
            upper_bound=0.9 * self.monthly_income
        )
        self.monthly_outcome = tn_outcome.sample(self.random)

        tn_outcome_sar = TruncatedNormal(
            mean=0.5 * self.monthly_income_sar,
            std=0.1 * self.monthly_income_sar,
            lower_bound=0.1 * self.monthly_income_sar,
            upper_bound=0.9 * self.monthly_income_sar
        )
        self.monthly_outcome_sar = tn_outcome_sar.sample(self.random)

    def add_beneficiary(self, beneficiary):
        """Add a beneficiary account"""
        if beneficiary not in self.beneficiaries:
            self.beneficiaries.append(beneficiary)

    def add_originator(self, originator):
        """Add an originator account"""
        if originator not in self.originators:
            self.originators.append(originator)

    def receive_income(self, amount, tx_type="TRANSFER"):
        """Receive income (from external source or salary)"""
        if tx_type == "TRANSFER":
            self.balance += amount
        elif tx_type == "CASH":
            self.cash_balance += amount
        return True

    def make_payment(self, amount, tx_type="TRANSFER"):
        """Make a payment (to external sink or expense)"""
        if tx_type == "TRANSFER":
            if self.balance >= amount and amount > 0:
                self.balance -= amount
                return True
        elif tx_type == "CASH":
            if self.cash_balance >= amount and amount > 0:
                self.cash_balance -= amount
                return True
        return False

    def make_transaction(self, beneficiary, amount, tx_type="TRANSFER"):
        """Make a transaction to another account"""
        if tx_type == "TRANSFER":
            if self.balance >= amount and amount > 0:
                self.balance -= amount
                beneficiary.balance += amount
                return True
        elif tx_type == "CASH":
            # Cash transaction: sender loses cash, receiver gains cash
            if self.cash_balance >= amount and amount > 0:
                self.cash_balance -= amount
                beneficiary.cash_balance += amount
                return True
        return False

    def update_balance_history(self):
        """Update the balance history for spending behavior modeling"""
        # If deque is at max capacity, we need to subtract the oldest value
        if len(self.balance_history) == self.balance_history.maxlen:
            oldest = self.balance_history[0]
            self._balance_sum -= oldest

        self.balance_history.append(self.balance)
        self._balance_sum += self.balance
        self._balance_count = len(self.balance_history)

    def get_spending_probability(self):
        """Calculate spending probability based on balance history"""
        if self._balance_count == 0:
            return 0.0

        # Use cached mean balance (optimization: avoid np.mean call)
        mean_balance = self._balance_sum / self._balance_count

        # Avoid division by zero
        if mean_balance <= 100.0:
            mean_balance = 1000.0 if self.is_sar else 1000.0

        # Calculate deviation from mean
        current_balance = self.balance + (self.cash_balance if self.is_sar else 0.0)
        x = (current_balance - mean_balance) / mean_balance

        # Apply sigmoid to get probability
        return sigmoid(x)

    def _sample_days_until_phone_change(self):
        """Sample days until next phone change using truncated normal distribution"""
        if self.mean_phone_change_frequency <= 0:
            return int(self.UB_PHONE)  # Never change if mean is 0

        tn = TruncatedNormal(
            mean=self.mean_phone_change_frequency,
            std=self.std_phone_change_frequency,
            lower_bound=self.LB_PHONE,
            upper_bound=self.UB_PHONE
        )
        return int(tn.sample(self.random))

    def _sample_days_until_bank_change(self):
        """Sample days until next bank change using truncated normal distribution"""
        if self.mean_bank_change_frequency <= 0:
            return int(self.UB_BANK)  # Never change if mean is 0

        tn = TruncatedNormal(
            mean=self.mean_bank_change_frequency,
            std=self.std_bank_change_frequency,
            lower_bound=self.LB_BANK,
            upper_bound=self.UB_BANK
        )
        return int(tn.sample(self.random))

    def _get_new_bank(self, available_banks):
        """
        Select a new bank different from current bank.
        Matches Java AccountBehaviour.getNewBank() method.

        Args:
            available_banks: List of available bank IDs

        Returns:
            New bank ID (different from current), or current bank if no alternatives
        """
        if not available_banks:
            return self.bank_id

        # Filter out current bank
        other_banks = [b for b in available_banks if b != self.bank_id]

        if not other_banks:
            return self.bank_id

        # Randomly select a new bank
        return other_banks[self.random.randint(0, len(other_banks))]

    def update_behaviour(self, available_banks):
        """
        Update phone and bank change counters.
        Matches Java AccountBehaviour.update() method.

        If bank changes, updates bank_id, resets phone_changes and days_in_bank.
        Otherwise, decrements counters and increments phone_changes when due.

        Args:
            available_banks: List of available bank IDs for bank switching
        """
        if self.days_until_bank_change == 0:
            # Bank change: select new bank and reset counters
            self.bank_id = self._get_new_bank(available_banks)
            self.days_until_bank_change = self._sample_days_until_bank_change()
            self.phone_changes = 0
            self.days_in_bank = 0
        else:
            # No bank change: count down days
            self.days_until_bank_change -= 1
            self.days_in_bank += 1
            self.days_until_phone_change -= 1

            # Check for phone change
            if self.days_until_phone_change == 0:
                self.phone_changes += 1
                self.days_until_phone_change = self._sample_days_until_phone_change()

    def handle_step(self, step, transaction_log, available_banks=None):
        """Handle account behavior for a single time step

        Args:
            step: Current simulation step
            transaction_log: List to append transactions to
            available_banks: List of available bank IDs for bank switching (optional)
        """
        transactions = []

        # Update phone/bank change behavior
        if available_banks is None:
            available_banks = []
        self.update_behaviour(available_banks)

        # Update balance history
        self.update_balance_history()

        # Choose parameters based on SAR status
        if not self.is_sar:
            prob_income = self.prob_income
            mean_income = self.mean_income
            std_income = self.std_income
            mean_outcome = self.mean_outcome
            std_outcome = self.std_outcome
            monthly_income = self.monthly_income
            monthly_outcome = self.monthly_outcome
        else:
            prob_income = self.prob_income_sar
            mean_income = self.mean_income_sar
            std_income = self.std_income_sar
            mean_outcome = self.mean_outcome_sar
            std_outcome = self.std_outcome_sar
            monthly_income = self.monthly_income_sar
            monthly_outcome = self.monthly_outcome_sar

        # Handle monthly salary (25th of month, using 28-day cycle)
        if step % 28 == 25:
            self.receive_income(monthly_income, "TRANSFER")
            transactions.append({
                'step': step,
                'type': 'TRANSFER',
                'amount': monthly_income,
                'nameOrig': -2,  # Source
                'nameDest': self.account_id,
                'isSAR': 0,  # Income transactions are never SAR
                'patternID': -1,
                'modelType': 0  # Monthly income (generic, not a pattern)
            })

        # Handle random income
        if self.random.random() < prob_income:
            tn = TruncatedNormal(mean_income, std_income, 0, 1000000)
            amount = tn.sample(self.random)
            self.receive_income(amount, "TRANSFER")
            transactions.append({
                'step': step,
                'type': 'TRANSFER',
                'amount': amount,
                'nameOrig': -2,  # Source
                'nameDest': self.account_id,
                'isSAR': 0,  # Income transactions are never SAR
                'patternID': -1,
                'modelType': 0  # Random income
            })

        # Handle monthly outcome (26th-28th of month)
        if step == self.step_monthly_outcome:
            if self.make_payment(monthly_outcome, "TRANSFER"):
                transactions.append({
                    'step': step,
                    'type': 'TRANSFER',
                    'amount': monthly_outcome,
                    'nameOrig': self.account_id,
                    'nameDest': -1,  # Sink
                    'isSAR': 0,  # Outcome transactions are never SAR
                    'patternID': -1,
                    'modelType': 0  # Monthly expense (generic, not a pattern)
                })
            # Update next monthly outcome step
            diff = (self.step_monthly_outcome % 28) - 25
            diff = 3 if diff < 0 else diff
            self.step_monthly_outcome = self.step_monthly_outcome + 28 - diff + self.random.randint(0, 4)

        # Handle random spending based on balance history
        spending_prob = self.get_spending_probability()
        if self.random.random() < spending_prob:
            max_amount = 0.9 * self.balance
            if max_amount > 0:
                tn = TruncatedNormal(mean_outcome, std_outcome, 0.0, max_amount)
                amount = tn.sample(self.random)

                if amount > 0 and self.balance >= amount and self.balance >= 100.0:
                    # For SAR accounts, randomly choose between cash and transfer
                    use_cash = self.is_sar and self.random.random() < self.prob_spend_cash
                    tx_type = "CASH" if use_cash else "TRANSFER"

                    if use_cash and self.cash_balance >= amount:
                        if self.make_payment(amount, tx_type):
                            transactions.append({
                                'step': step,
                                'type': tx_type,
                                'amount': amount,
                                'nameOrig': self.account_id,
                                'nameDest': -1,  # Sink
                                'isSAR': 0,  # Spending transactions are never SAR
                                'patternID': -1,
                                'modelType': 0  # Random spending
                            })
                    elif not use_cash:
                        if self.make_payment(amount, tx_type):
                            transactions.append({
                                'step': step,
                                'type': tx_type,
                                'amount': amount,
                                'nameOrig': self.account_id,
                                'nameDest': -1,  # Sink
                                'isSAR': 0,  # Spending transactions are never SAR
                                'patternID': -1,
                                'modelType': 0  # Random spending
                            })

        return transactions

    def __repr__(self):
        return f"Account(id={self.account_id}, balance={self.balance:.2f}, is_sar={self.is_sar})"
