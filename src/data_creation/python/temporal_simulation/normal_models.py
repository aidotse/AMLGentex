"""
Normal transaction model patterns
Implements: single, fan_in, fan_out, forward, mutual, and periodical transaction patterns
"""
import numpy as np
from .utils import TruncatedNormal
from .alert_patterns import PatternScheduler


class SingleModel:
    """
    Single model: Simple bilateral transaction
    Account A sends to Account B once
    """
    def __init__(self, model_id, accounts, start_step, end_step, burstiness_level,
                 mean_amount=300, std_amount=100, min_amount=100, max_amount=500, random_state=None):
        self.model_id = model_id
        self.accounts = accounts  # List of (account, is_main) tuples
        self.start_step = start_step
        self.end_step = end_step
        self.burstiness_level = burstiness_level
        self.mean_amount = mean_amount
        self.std_amount = std_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.random_state = random_state

        # Get main account and partner
        self.main_account = None
        self.partner = None
        for acc, is_main in accounts:
            if is_main:
                self.main_account = acc
            else:
                self.partner = acc

        # If no main specified, use first account
        if self.main_account is None and len(accounts) >= 2:
            self.main_account = accounts[0][0]
            self.partner = accounts[1][0]

        # Transaction happens once at a scheduled time
        self.transaction_step = self._calculate_step()
        self.executed = False

    def _calculate_step(self):
        """Calculate when the transaction should occur"""
        # Use PatternScheduler to get a single step
        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, 1, self.burstiness_level, self.random_state
        )
        return steps[0] if steps else self.start_step

    def get_transaction(self, step):
        """Get transaction for this step if scheduled"""
        if self.executed or self.main_account is None or self.partner is None:
            return None

        if step == self.transaction_step:
            # Sample amount from truncated normal, capped by sender balance
            max_amount = self.main_account.balance * 0.9
            if max_amount <= 0:
                return None

            tn = TruncatedNormal(
                mean=self.mean_amount,
                std=self.std_amount,
                lower_bound=self.min_amount,
                upper_bound=min(self.max_amount, max_amount)
            )
            amount = tn.sample(self.random_state)
            self.executed = True

            return {
                'step': step,
                'from': self.main_account,
                'to': self.partner,
                'amount': amount,
                'type': 'TRANSFER'
            }

        return None


class FanInModel:
    """
    Fan-in model: Multiple accounts send to one main account
    Members send to main account at different times
    """
    def __init__(self, model_id, accounts, start_step, end_step, burstiness_level,
                 mean_amount=300, std_amount=100, min_amount=100, max_amount=500, random_state=None):
        self.model_id = model_id
        self.accounts = accounts
        self.start_step = start_step
        self.end_step = end_step
        self.burstiness_level = burstiness_level
        self.mean_amount = mean_amount
        self.std_amount = std_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.random_state = random_state

        # Get main account (receiver) and members (senders)
        self.main_account = None
        self.members = []
        for acc, is_main in accounts:
            if is_main:
                self.main_account = acc
            else:
                self.members.append(acc)

        if self.main_account is None and len(accounts) > 0:
            # First account is main, rest are members
            self.main_account = accounts[0][0]
            self.members = [acc for acc, _ in accounts[1:]]

        # Schedule transactions from each member
        self.transaction_schedule = self._generate_schedule()
        self.current_index = 0

    def _generate_schedule(self):
        """Generate when each member sends to main"""
        if not self.members:
            return []

        n_members = len(self.members)
        schedule = []

        # Use PatternScheduler to generate steps for all members
        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, n_members, self.burstiness_level, self.random_state
        )

        # Pair each step with a member
        for step, member in zip(steps, self.members):
            schedule.append((step, member))

        return schedule

    def get_transaction(self, step):
        """Get transaction for this step if scheduled"""
        if self.current_index >= len(self.transaction_schedule):
            return None

        if self.main_account is None:
            return None

        scheduled_step, member = self.transaction_schedule[self.current_index]

        if step == scheduled_step:
            # Sample amount from truncated normal, capped by sender balance
            max_amount = member.balance * 0.9

            # Move to next transaction even if this one fails
            self.current_index += 1

            if max_amount <= self.min_amount:
                # Insufficient balance, skip this transaction
                return None

            tn = TruncatedNormal(
                mean=self.mean_amount,
                std=self.std_amount,
                lower_bound=self.min_amount,
                upper_bound=min(self.max_amount, max_amount)
            )
            amount = tn.sample(self.random_state)

            return {
                'step': step,
                'from': member,
                'to': self.main_account,
                'amount': amount,
                'type': 'TRANSFER'
            }

        return None


class FanOutModel:
    """
    Fan-out model: One main account sends to multiple accounts
    Main account distributes to members at different times
    """
    def __init__(self, model_id, accounts, start_step, end_step, burstiness_level,
                 mean_amount=300, std_amount=100, min_amount=100, max_amount=500, random_state=None):
        self.model_id = model_id
        self.accounts = accounts
        self.start_step = start_step
        self.end_step = end_step
        self.burstiness_level = burstiness_level
        self.mean_amount = mean_amount
        self.std_amount = std_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.random_state = random_state

        # Get main account (sender) and members (receivers)
        self.main_account = None
        self.members = []
        for acc, is_main in accounts:
            if is_main:
                self.main_account = acc
            else:
                self.members.append(acc)

        if self.main_account is None and len(accounts) > 0:
            # First account is main, rest are members
            self.main_account = accounts[0][0]
            self.members = [acc for acc, _ in accounts[1:]]

        # Schedule transactions to each member
        self.transaction_schedule = self._generate_schedule()
        self.current_index = 0

    def _generate_schedule(self):
        """Generate when main sends to each member"""
        if not self.members:
            return []

        n_members = len(self.members)
        schedule = []

        # Use PatternScheduler to generate steps for all members
        steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, n_members, self.burstiness_level, self.random_state
        )

        # Pair each step with a member
        for step, member in zip(steps, self.members):
            schedule.append((step, member))

        return schedule

    def get_transaction(self, step):
        """Get transaction for this step if scheduled"""
        if self.current_index >= len(self.transaction_schedule):
            return None

        if self.main_account is None:
            return None

        scheduled_step, member = self.transaction_schedule[self.current_index]

        if step == scheduled_step:
            # Sample amount from truncated normal, capped by sender balance
            max_amount = self.main_account.balance * 0.9

            # Move to next transaction even if this one fails
            self.current_index += 1

            if max_amount <= self.min_amount:
                # Insufficient balance, skip this transaction
                return None

            tn = TruncatedNormal(
                mean=self.mean_amount,
                std=self.std_amount,
                lower_bound=self.min_amount,
                upper_bound=min(self.max_amount, max_amount)
            )
            amount = tn.sample(self.random_state)

            return {
                'step': step,
                'from': self.main_account,
                'to': member,
                'amount': amount,
                'type': 'TRANSFER'
            }

        return None


class ForwardModel:
    """
    Forward model: Sequential chain where money is received and forwarded
    Account A sends to B, then B sends to C
    """
    def __init__(self, model_id, accounts, start_step, end_step, burstiness_level,
                 mean_amount=300, std_amount=100, min_amount=100, max_amount=500, random_state=None):
        self.model_id = model_id
        self.accounts = accounts  # List of (account, is_main) tuples
        self.start_step = start_step
        self.end_step = end_step
        self.burstiness_level = burstiness_level
        self.mean_amount = mean_amount
        self.std_amount = std_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.random_state = random_state

        # Get main account and member accounts
        self.main_account = None
        self.members = []
        for acc, is_main in accounts:
            if is_main:
                self.main_account = acc
            self.members.append(acc)

        if self.main_account is None and len(self.members) > 0:
            self.main_account = self.members[0]

        # Generate transaction steps (2 transactions)
        self.transaction_steps = self._generate_steps()
        self.current_index = 0

    def _generate_steps(self):
        """Generate 2 transaction steps based on burstiness level"""
        return PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, 2, self.burstiness_level, self.random_state
        )

    def get_transaction(self, step):
        """Get transaction for this step if scheduled"""
        if self.current_index >= len(self.transaction_steps):
            return None

        if step == self.transaction_steps[self.current_index]:
            # Determine destination based on index
            if len(self.members) < 3:
                return None

            # Forward pattern: A → B → C
            from_account = self.members[0] if self.current_index == 0 else self.members[1]
            to_account = self.members[1] if self.current_index == 0 else self.members[2]

            # Sample amount from truncated normal, capped by sender balance
            max_amount = from_account.balance * 0.9

            # Move to next transaction even if this one fails
            self.current_index += 1

            if max_amount <= self.min_amount:
                # Insufficient balance, skip this transaction
                return None

            tn = TruncatedNormal(
                mean=self.mean_amount,
                std=self.std_amount,
                lower_bound=self.min_amount,
                upper_bound=min(self.max_amount, max_amount)
            )
            amount = tn.sample(self.random_state)

            return {
                'step': step,
                'from': from_account,
                'to': to_account,
                'amount': amount,
                'type': 'TRANSFER'
            }

        return None


class MutualModel:
    """
    Mutual model: Loan and repayment pattern
    Main account lends to a member, then member repays
    """
    def __init__(self, model_id, accounts, start_step, end_step, burstiness_level,
                 mean_amount=300, std_amount=100, min_amount=100, max_amount=500, random_state=None):
        self.model_id = model_id
        self.accounts = accounts
        self.start_step = start_step
        self.end_step = end_step
        self.burstiness_level = burstiness_level
        self.mean_amount = mean_amount
        self.std_amount = std_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.random_state = random_state

        # Get lender (main) and debtors (members)
        self.lender = None
        self.debtors = []
        for acc, is_main in accounts:
            if is_main:
                self.lender = acc
            else:
                self.debtors.append(acc)

        if self.lender is None and len(accounts) > 0:
            self.lender = accounts[0][0]
            self.debtors = [acc for acc, _ in accounts[1:]]

        # Choose debtor and set interval
        if len(self.debtors) > 0:
            self.debtor = self.random_state.choice(self.debtors)
        else:
            self.debtor = None

        # Generate 2 transaction steps: loan and repayment
        self.transaction_steps = PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, 2, self.burstiness_level, self.random_state
        )

        self.debt_amount = 0
        self.loan_sent = False
        self.repayment_sent = False

    def get_transaction(self, step):
        """Get transaction for this step"""
        if self.debtor is None or self.lender is None or len(self.transaction_steps) < 2:
            return None

        # First transaction: lender sends to debtor
        if step == self.transaction_steps[0] and not self.loan_sent:
            # Sample amount from truncated normal, capped by lender balance
            max_amount = self.lender.balance * 0.9
            if max_amount <= 0:
                return None

            tn = TruncatedNormal(
                mean=self.mean_amount,
                std=self.std_amount,
                lower_bound=self.min_amount,
                upper_bound=min(self.max_amount, max_amount)
            )
            self.debt_amount = tn.sample(self.random_state)
            self.loan_sent = True
            return {
                'step': step,
                'from': self.lender,
                'to': self.debtor,
                'amount': self.debt_amount,
                'type': 'TRANSFER'
            }

        # Second transaction: debtor repays lender
        if step == self.transaction_steps[1] and not self.repayment_sent:
            # Repay full amount or partial based on balance
            repay_amount = min(self.debt_amount, self.debtor.balance * 0.9)
            self.repayment_sent = True
            return {
                'step': step,
                'from': self.debtor,
                'to': self.lender,
                'amount': repay_amount,
                'type': 'TRANSFER'
            }

        return None


class PeriodicalModel:
    """
    Periodical model: Sends to beneficiaries at regular intervals
    """
    def __init__(self, model_id, accounts, start_step, end_step, burstiness_level,
                 mean_amount=300, std_amount=100, min_amount=100, max_amount=500, random_state=None):
        self.model_id = model_id
        self.accounts = accounts
        self.start_step = start_step
        self.end_step = end_step
        self.burstiness_level = burstiness_level
        self.mean_amount = mean_amount
        self.std_amount = std_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.random_state = random_state

        # Get main account
        self.main_account = None
        for acc, is_main in accounts:
            if is_main:
                self.main_account = acc
                break

        if self.main_account is None and len(accounts) > 0:
            self.main_account = accounts[0][0]

        # Generate periodic transaction steps
        # Sample number of transactions (5-15 over the duration)
        period = self.end_step - self.start_step
        num_transactions = self.random_state.randint(5, min(16, max(6, period // 2)))

        # Use PatternScheduler to place transactions
        self.transaction_steps = set(PatternScheduler.get_transaction_steps(
            self.start_step, self.end_step, num_transactions, self.burstiness_level, self.random_state
        ))

        self.amount = None

    def is_valid_step(self, step):
        """Check if this step should execute a transaction"""
        return step in self.transaction_steps

    def get_transaction(self, step):
        """Get transaction for this step if valid"""
        if not self.is_valid_step(step):
            return None

        if self.main_account is None or not self.main_account.beneficiaries:
            return None

        # Sample amount once from truncated normal
        if self.amount is None:
            tn = TruncatedNormal(
                mean=self.mean_amount,
                std=self.std_amount,
                lower_bound=self.min_amount,
                upper_bound=self.max_amount
            )
            self.amount = tn.sample(self.random_state)

        # Find beneficiary that is also a member of this account group
        member_accounts = [acc for acc, _ in self.accounts]
        valid_beneficiaries = [b for b in self.main_account.beneficiaries if b in member_accounts]

        if not valid_beneficiaries:
            return None

        dest = valid_beneficiaries[0]

        # Adjust amount if balance is low
        max_amount = self.main_account.balance * 0.9
        if max_amount <= 0:
            return None
        actual_amount = min(self.amount, max_amount)

        return {
            'step': step,
            'from': self.main_account,
            'to': dest,
            'amount': actual_amount,
            'type': 'TRANSFER'
        }
