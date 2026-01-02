"""
Unit tests for Account parameter configuration
Tests that all config parameters are properly used in temporal simulation
"""
import pytest
from src.data_creation.python.temporal_simulation.account import Account


@pytest.mark.unit
@pytest.mark.temporal
class TestAccountParameters:
    """Tests for Account parameter initialization and configuration"""

    def test_balance_history_window_configurable(self):
        """Test that balance history window size is configurable"""
        # Test default
        account_default = Account(
            account_id=1,
            customer_id=100,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42
        )
        assert account_default.balance_history.maxlen == 28, "Default should be 28"

        # Test custom value
        account_custom = Account(
            account_id=2,
            customer_id=200,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42,
            n_steps_balance_history=14
        )
        assert account_custom.balance_history.maxlen == 14, "Should use custom value"

    def test_phone_change_parameters_normal_account(self):
        """Test that phone change parameters are set correctly for normal accounts"""
        account = Account(
            account_id=1,
            customer_id=100,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42
        )

        account.set_parameters(
            prob_income=0.1,
            mean_income=1000.0,
            std_income=200.0,
            prob_income_sar=0.2,
            mean_income_sar=2000.0,
            std_income_sar=400.0,
            mean_outcome=500.0,
            std_outcome=100.0,
            mean_outcome_sar=1000.0,
            std_outcome_sar=200.0,
            prob_spend_cash=0.3,
            mean_phone_change_frequency=1460,
            std_phone_change_frequency=365,
            mean_phone_change_frequency_sar=730,
            std_phone_change_frequency_sar=180,
            mean_bank_change_frequency=1825,
            std_bank_change_frequency=365,
            mean_bank_change_frequency_sar=1095,
            std_bank_change_frequency_sar=180
        )

        # Normal account should use non-SAR parameters
        assert account.mean_phone_change_frequency == 1460
        assert account.std_phone_change_frequency == 365
        assert account.mean_bank_change_frequency == 1825
        assert account.std_bank_change_frequency == 365

    def test_phone_change_parameters_sar_account(self):
        """Test that phone change parameters are set correctly for SAR accounts"""
        account = Account(
            account_id=2,
            customer_id=200,
            initial_balance=10000.0,
            is_sar=True,
            bank_id='BANK001',
            random_state=42
        )

        account.set_parameters(
            prob_income=0.1,
            mean_income=1000.0,
            std_income=200.0,
            prob_income_sar=0.2,
            mean_income_sar=2000.0,
            std_income_sar=400.0,
            mean_outcome=500.0,
            std_outcome=100.0,
            mean_outcome_sar=1000.0,
            std_outcome_sar=200.0,
            prob_spend_cash=0.3,
            mean_phone_change_frequency=1460,
            std_phone_change_frequency=365,
            mean_phone_change_frequency_sar=730,
            std_phone_change_frequency_sar=180,
            mean_bank_change_frequency=1825,
            std_bank_change_frequency=365,
            mean_bank_change_frequency_sar=1095,
            std_bank_change_frequency_sar=180
        )

        # SAR account should use SAR-specific parameters
        assert account.mean_phone_change_frequency == 730
        assert account.std_phone_change_frequency == 180
        assert account.mean_bank_change_frequency == 1095
        assert account.std_bank_change_frequency == 180

    def test_days_until_change_initialized(self):
        """Test that days until change are sampled on initialization"""
        account = Account(
            account_id=1,
            customer_id=100,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42
        )

        account.set_parameters(
            prob_income=0.0,
            mean_income=0.0,
            std_income=0.0,
            prob_income_sar=0.0,
            mean_income_sar=0.0,
            std_income_sar=0.0,
            mean_outcome=500.0,
            std_outcome=100.0,
            mean_outcome_sar=1000.0,
            std_outcome_sar=200.0,
            prob_spend_cash=0.3,
            mean_phone_change_frequency=1460,
            std_phone_change_frequency=365,
            mean_phone_change_frequency_sar=730,
            std_phone_change_frequency_sar=180,
            mean_bank_change_frequency=1825,
            std_bank_change_frequency=365,
            mean_bank_change_frequency_sar=1095,
            std_bank_change_frequency_sar=180
        )

        # Should have sampled initial values
        assert account.days_until_phone_change > 0
        assert account.days_until_bank_change >= 0
        assert account.days_until_phone_change <= account.UB_PHONE
        assert account.days_until_bank_change <= account.UB_BANK

    def test_phone_change_increments_counter(self):
        """Test that phone changes increment the counter"""
        account = Account(
            account_id=1,
            customer_id=100,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42
        )

        account.set_parameters(
            prob_income=0.0,
            mean_income=0.0,
            std_income=0.0,
            prob_income_sar=0.0,
            mean_income_sar=0.0,
            std_income_sar=0.0,
            mean_outcome=500.0,
            std_outcome=100.0,
            mean_outcome_sar=1000.0,
            std_outcome_sar=200.0,
            prob_spend_cash=0.3,
            mean_phone_change_frequency=10,  # Short frequency for testing
            std_phone_change_frequency=2,
            mean_phone_change_frequency_sar=10,
            std_phone_change_frequency_sar=2,
            mean_bank_change_frequency=1000,  # Long so no bank change
            std_bank_change_frequency=100,
            mean_bank_change_frequency_sar=1000,
            std_bank_change_frequency_sar=100
        )

        initial_phone_changes = account.phone_changes
        available_banks = ['BANK001', 'BANK002']

        # Run until we see a phone change
        for _ in range(50):
            account.update_behaviour(available_banks)
            if account.phone_changes > initial_phone_changes:
                break

        assert account.phone_changes > initial_phone_changes, "Phone changes should increment"

    def test_bank_change_updates_bank_id(self):
        """Test that bank changes update the bank_id"""
        account = Account(
            account_id=1,
            customer_id=100,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42
        )

        account.set_parameters(
            prob_income=0.0,
            mean_income=0.0,
            std_income=0.0,
            prob_income_sar=0.0,
            mean_income_sar=0.0,
            std_income_sar=0.0,
            mean_outcome=500.0,
            std_outcome=100.0,
            mean_outcome_sar=1000.0,
            std_outcome_sar=200.0,
            prob_spend_cash=0.3,
            mean_phone_change_frequency=1000,  # Long so no phone change
            std_phone_change_frequency=100,
            mean_phone_change_frequency_sar=1000,
            std_phone_change_frequency_sar=100,
            mean_bank_change_frequency=10,  # Short frequency for testing
            std_bank_change_frequency=2,
            mean_bank_change_frequency_sar=10,
            std_bank_change_frequency_sar=2
        )

        initial_bank_id = account.bank_id
        available_banks = ['BANK001', 'BANK002', 'BANK003']

        # Run until we see a bank change
        bank_changed = False
        for _ in range(50):
            account.update_behaviour(available_banks)
            if account.bank_id != initial_bank_id:
                bank_changed = True
                break

        assert bank_changed, "Bank should change with short frequency"
        assert account.bank_id != initial_bank_id, "Bank ID should be different"
        assert account.bank_id in available_banks, "New bank should be from available banks"

    def test_bank_change_resets_phone_counter(self):
        """Test that bank changes reset phone_changes counter"""
        account = Account(
            account_id=1,
            customer_id=100,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42
        )

        account.set_parameters(
            prob_income=0.0,
            mean_income=0.0,
            std_income=0.0,
            prob_income_sar=0.0,
            mean_income_sar=0.0,
            std_income_sar=0.0,
            mean_outcome=500.0,
            std_outcome=100.0,
            mean_outcome_sar=1000.0,
            std_outcome_sar=200.0,
            prob_spend_cash=0.3,
            mean_phone_change_frequency=5,  # Very short
            std_phone_change_frequency=1,
            mean_phone_change_frequency_sar=5,
            std_phone_change_frequency_sar=1,
            mean_bank_change_frequency=20,  # Moderate
            std_bank_change_frequency=2,
            mean_bank_change_frequency_sar=20,
            std_bank_change_frequency_sar=2
        )

        available_banks = ['BANK001', 'BANK002', 'BANK003']

        # Run until we get some phone changes
        for _ in range(15):
            account.update_behaviour(available_banks)

        # Should have accumulated some phone changes
        phone_changes_before = account.phone_changes
        assert phone_changes_before > 0, "Should have some phone changes"

        initial_bank = account.bank_id

        # Continue until bank changes
        for _ in range(50):
            account.update_behaviour(available_banks)
            if account.bank_id != initial_bank:
                break

        # Phone changes should be reset to 0 after bank change
        assert account.phone_changes == 0, "Phone changes should reset on bank change"
        assert account.days_in_bank == 0, "Days in bank should reset on bank change"

    def test_zero_frequency_prevents_changes(self):
        """Test that zero frequency prevents phone/bank changes"""
        account = Account(
            account_id=1,
            customer_id=100,
            initial_balance=10000.0,
            is_sar=False,
            bank_id='BANK001',
            random_state=42
        )

        account.set_parameters(
            prob_income=0.0,
            mean_income=0.0,
            std_income=0.0,
            prob_income_sar=0.0,
            mean_income_sar=0.0,
            std_income_sar=0.0,
            mean_outcome=500.0,
            std_outcome=100.0,
            mean_outcome_sar=1000.0,
            std_outcome_sar=200.0,
            prob_spend_cash=0.3,
            mean_phone_change_frequency=0,  # Disabled
            std_phone_change_frequency=0,
            mean_phone_change_frequency_sar=0,
            std_phone_change_frequency_sar=0,
            mean_bank_change_frequency=0,  # Disabled
            std_bank_change_frequency=0,
            mean_bank_change_frequency_sar=0,
            std_bank_change_frequency_sar=0
        )

        # With zero frequency, days should be set to max
        assert account.days_until_phone_change == int(account.UB_PHONE)
        assert account.days_until_bank_change == int(account.UB_BANK)

        initial_phone = account.phone_changes
        initial_bank = account.bank_id
        available_banks = ['BANK001', 'BANK002']

        # Run many steps - nothing should change
        for _ in range(1000):
            account.update_behaviour(available_banks)

        assert account.phone_changes == initial_phone, "No phone changes with zero frequency"
        assert account.bank_id == initial_bank, "No bank changes with zero frequency"
