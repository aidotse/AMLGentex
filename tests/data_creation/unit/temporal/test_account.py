"""
Unit tests for Account class
"""
import pytest
import numpy as np


@pytest.mark.unit
@pytest.mark.temporal
class TestAccount:
    """Tests for Account class"""

    def test_account_initialization(self, sample_account):
        """Test account is properly initialized"""
        assert sample_account.account_id == 1
        assert sample_account.customer_id == 100
        assert sample_account.balance == 10000.0
        assert sample_account.is_sar is False
        assert sample_account.bank_id == 'BANK001'
        assert sample_account.cash_balance == 0.0

    def test_sar_account_initialization(self, sample_sar_account):
        """Test SAR account is properly initialized"""
        assert sample_sar_account.is_sar is True
        assert sample_sar_account.balance == 50000.0

    def test_receive_income_transfer(self, sample_account):
        """Test receiving income via transfer"""
        initial_balance = sample_account.balance
        amount = 1000.0

        success = sample_account.receive_income(amount, "TRANSFER")

        assert success is True
        assert sample_account.balance == initial_balance + amount
        assert sample_account.cash_balance == 0.0

    def test_receive_income_cash(self, sample_account):
        """Test receiving income via cash"""
        initial_cash = sample_account.cash_balance
        amount = 500.0

        success = sample_account.receive_income(amount, "CASH")

        assert success is True
        assert sample_account.cash_balance == initial_cash + amount
        assert sample_account.balance == 10000.0  # Regular balance unchanged

    def test_make_payment_transfer_success(self, sample_account):
        """Test making a transfer payment with sufficient funds"""
        initial_balance = sample_account.balance
        amount = 500.0

        success = sample_account.make_payment(amount, "TRANSFER")

        assert success is True
        assert sample_account.balance == initial_balance - amount

    def test_make_payment_transfer_insufficient_funds(self, sample_account):
        """Test making a transfer payment with insufficient funds"""
        initial_balance = sample_account.balance
        amount = 20000.0  # More than balance

        success = sample_account.make_payment(amount, "TRANSFER")

        assert success is False
        assert sample_account.balance == initial_balance  # Unchanged

    def test_make_payment_cash_success(self, sample_account):
        """Test making a cash payment with sufficient funds"""
        # First add cash
        sample_account.receive_income(1000.0, "CASH")
        initial_cash = sample_account.cash_balance
        amount = 500.0

        success = sample_account.make_payment(amount, "CASH")

        assert success is True
        assert sample_account.cash_balance == initial_cash - amount

    def test_make_transaction_success(self, sample_account, sample_sar_account):
        """Test transaction between two accounts"""
        sender_balance = sample_account.balance
        receiver_balance = sample_sar_account.balance
        amount = 1000.0

        success = sample_account.make_transaction(sample_sar_account, amount, "TRANSFER")

        assert success is True
        assert sample_account.balance == sender_balance - amount
        assert sample_sar_account.balance == receiver_balance + amount

    def test_make_transaction_cash_success(self, sample_account, sample_sar_account):
        """Test cash transaction between two accounts"""
        # Setup: give sender some cash
        sample_account.receive_income(2000.0, "CASH")
        sender_cash = sample_account.cash_balance
        receiver_cash = sample_sar_account.cash_balance
        amount = 500.0

        success = sample_account.make_transaction(sample_sar_account, amount, "CASH")

        assert success is True
        assert sample_account.cash_balance == sender_cash - amount
        assert sample_sar_account.cash_balance == receiver_cash + amount

    def test_beneficiary_management(self, sample_account, sample_sar_account):
        """Test adding beneficiaries"""
        sample_account.add_beneficiary(sample_sar_account)

        assert sample_sar_account in sample_account.beneficiaries
        assert len(sample_account.beneficiaries) == 1

        # Adding same beneficiary again shouldn't duplicate
        sample_account.add_beneficiary(sample_sar_account)
        assert len(sample_account.beneficiaries) == 1

    def test_originator_management(self, sample_account, sample_sar_account):
        """Test adding originators"""
        sample_account.add_originator(sample_sar_account)

        assert sample_sar_account in sample_account.originators
        assert len(sample_account.originators) == 1

    def test_balance_history_update(self, sample_account):
        """Test balance history tracking"""
        initial_len = len(sample_account.balance_history)

        sample_account.balance = 15000.0
        sample_account.update_balance_history()

        assert len(sample_account.balance_history) == initial_len + 1
        assert sample_account.balance_history[-1] == 15000.0

    def test_balance_history_max_length(self, sample_account):
        """Test balance history respects max length"""
        # Fill up the history
        for i in range(30):
            sample_account.balance = 10000.0 + i * 100
            sample_account.update_balance_history()

        assert len(sample_account.balance_history) <= 28

    def test_spending_probability(self, configured_account):
        """Test spending probability calculation"""
        # Update balance history with some values
        for i in range(10):
            configured_account.balance = 10000.0
            configured_account.update_balance_history()

        prob = configured_account.get_spending_probability()

        assert 0.0 <= prob <= 1.0

    def test_set_parameters(self, sample_account):
        """Test setting behavioral parameters"""
        sample_account.set_parameters(
            prob_income=0.15,
            mean_income=1500,
            std_income=150,
            prob_income_sar=0.25,
            mean_income_sar=2500,
            std_income_sar=250,
            mean_outcome=750,
            std_outcome=150,
            mean_outcome_sar=1500,
            std_outcome_sar=300,
            prob_spend_cash=0.2,
            mean_phone_change_frequency=1460,
            std_phone_change_frequency=365,
            mean_phone_change_frequency_sar=1304,
            std_phone_change_frequency_sar=370,
            mean_bank_change_frequency=1460,
            std_bank_change_frequency=365,
            mean_bank_change_frequency_sar=1305,
            std_bank_change_frequency_sar=412
        )

        assert sample_account.prob_income == 0.15
        assert sample_account.mean_income == 1500
        assert sample_account.monthly_income > 0  # Should be sampled from distribution
        assert sample_account.monthly_outcome > 0  # Should be computed
