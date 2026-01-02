"""
Integration tests for temporal simulation pipeline
"""
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

from src.data_creation.python.temporal_simulation.simulator import AMLSimulator


@pytest.mark.integration
@pytest.mark.temporal
@pytest.mark.requires_data
class TestTemporalPipeline:
    """Integration tests for temporal simulation pipeline"""

    def test_load_accounts(self, project_root, test_config):
        """Test loading accounts from CSV"""
        # Change to project root
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()

            assert len(simulator.accounts) > 0
            assert all(hasattr(acc, 'account_id') for acc in simulator.accounts.values())
            assert all(hasattr(acc, 'balance') for acc in simulator.accounts.values())

        finally:
            os.chdir(original_dir)

    def test_load_transactions(self, project_root, test_config):
        """Test loading transaction connections"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()
            simulator.load_transactions()

            # Should have accounts with beneficiaries/originators loaded
            # Check that some accounts have beneficiaries or originators
            has_beneficiaries = any(len(acc.beneficiaries) > 0 for acc in simulator.accounts.values())
            has_originators = any(len(acc.originators) > 0 for acc in simulator.accounts.values())
            assert has_beneficiaries or has_originators

        finally:
            os.chdir(original_dir)

    def test_load_normal_models(self, project_root, test_config):
        """Test loading normal transaction models"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()

            assert len(simulator.normal_model_objects) > 0

        finally:
            os.chdir(original_dir)

    def test_load_alert_members(self, project_root, test_config):
        """Test loading alert patterns"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()
            simulator.load_alert_members()

            assert len(simulator.alert_patterns) > 0

        finally:
            os.chdir(original_dir)

    @pytest.mark.slow
    def test_short_simulation_run(self, project_root, test_config):
        """Test running a short simulation"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()
            simulator.load_alert_members()

            # Run simulation for just 10 steps
            simulator.total_steps = 10
            simulator.run()

            assert len(simulator.transactions) > 0
            # Verify transactions have required fields
            if len(simulator.transactions) > 0:
                first_tx = simulator.transactions[0]
                assert 'step' in first_tx
                assert 'amount' in first_tx
                assert 'nameOrig' in first_tx
                assert 'nameDest' in first_tx

        finally:
            os.chdir(original_dir)

    def test_account_behavior_over_steps(self, project_root, test_config):
        """Test that accounts exhibit behavior over multiple steps"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()

            # Get a sample account
            account = list(simulator.accounts.values())[0]
            initial_balance = account.balance

            # Simulate receiving income
            account.receive_income(1000.0, "TRANSFER")
            assert account.balance == initial_balance + 1000.0

            # Simulate making payment
            account.make_payment(500.0, "TRANSFER")
            assert account.balance == initial_balance + 500.0

        finally:
            os.chdir(original_dir)
