"""
End-to-end tests for complete simulation pipeline
"""
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

from src.data_creation.python.temporal_simulation.simulator import AMLSimulator


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_data
class TestFullSimulation:
    """End-to-end tests for complete AMLSim pipeline"""

    def test_full_100_accounts_simulation(self, project_root, test_config):
        """Test complete simulation with 100 accounts"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            # Load and run simulation
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()
            simulator.load_alert_members()

            # Run for limited steps for faster testing
            original_steps = simulator.total_steps
            simulator.total_steps = min(50, original_steps)  # Run for 50 steps max

            simulator.run()

            # Verify outputs
            assert len(simulator.transactions) > 0
            assert len(simulator.accounts) > 0

            # Verify transaction data quality
            for tx in simulator.transactions[:100]:  # Check first 100
                assert tx['step'] >= 0
                assert tx['amount'] > 0
                assert 'type' in tx
                assert 'nameOrig' in tx
                assert 'nameDest' in tx
                assert 'isSAR' in tx
                assert 'modelType' in tx

            # Verify accounts have been updated
            for account in list(simulator.accounts.values())[:10]:  # Check first 10
                # Balance history should have been updated
                assert len(account.balance_history) > 0

        finally:
            os.chdir(original_dir)

    def test_simulation_reproducibility(self, project_root, test_config):
        """Test that simulation is reproducible with same seed"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            # Run first simulation
            sim1 = AMLSimulator(test_config)
            sim1.load_accounts()
            sim1.load_transactions()
            sim1.load_normal_models()
            sim1.load_alert_members()
            sim1.total_steps = 10
            sim1.run()
            tx_count_1 = len(sim1.transactions)

            # Run second simulation with same seed
            sim2 = AMLSimulator(test_config)
            sim2.load_accounts()
            sim2.load_transactions()
            sim2.load_normal_models()
            sim2.load_alert_members()
            sim2.total_steps = 10
            sim2.run()
            tx_count_2 = len(sim2.transactions)

            # Should generate same number of transactions
            assert tx_count_1 == tx_count_2

        finally:
            os.chdir(original_dir)

    def test_sar_transactions_generated(self, project_root, test_config):
        """Test that SAR (suspicious activity) transactions are generated"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()
            simulator.load_alert_members()
            simulator.total_steps = 50
            simulator.run()

            # Should have some SAR transactions
            sar_transactions = [tx for tx in simulator.transactions if tx['isSAR'] == 1]

            # With alert patterns loaded, should have SAR transactions
            assert len(sar_transactions) > 0

        finally:
            os.chdir(original_dir)

    def test_transaction_types_coverage(self, project_root, test_config):
        """Test that various transaction types are generated"""
        original_dir = os.getcwd()
        os.chdir(project_root)

        try:
            simulator = AMLSimulator(test_config)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()
            simulator.load_alert_members()
            simulator.total_steps = 50
            simulator.run()

            # Collect unique transaction types
            tx_types = set(tx['type'] for tx in simulator.transactions)

            # Should have at least TRANSFER type
            assert 'TRANSFER' in tx_types

            # Collect unique model types
            model_types = set(tx['modelType'] for tx in simulator.transactions)

            # Should have various model types
            assert len(model_types) > 1

        finally:
            os.chdir(original_dir)
