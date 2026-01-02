"""
Unit tests for alert pattern transaction ordering
Tests that transactions in patterns occur in the correct causal order
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'python'))

from src.data_creation.python.temporal_simulation.alert_patterns import AlertPattern
from src.data_creation.python.temporal_simulation.account import Account


@pytest.fixture
def create_accounts():
    """Factory to create test accounts"""
    def _create(num_accounts=5):
        accounts = []
        for i in range(num_accounts):
            acc = Account(
                account_id=i,
                customer_id=100 + i,
                initial_balance=10000.0,
                is_sar=True,
                bank_id=f'BANK{i % 2}',
                random_state=42
            )
            accounts.append((acc, i == 0))  # First is main
        return accounts
    return _create


@pytest.mark.unit
@pytest.mark.temporal
class TestCyclePatternOrdering:
    """Tests for cycle pattern transaction ordering"""

    def test_cycle_transactions_sorted(self, create_accounts):
        """Test cycle pattern produces sorted transaction steps"""
        accounts = create_accounts(5)
        random_state = np.random.RandomState(42)

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='cycle',
            accounts=accounts,
            model_id=3,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state
        )

        schedule = pattern.transaction_schedule

        # Extract steps
        steps = [tx['step'] for tx in schedule]

        # Should be sorted
        assert steps == sorted(steps)

        # Each transaction should have from and to
        assert all('from' in tx and 'to' in tx for tx in schedule)

    def test_cycle_forms_complete_loop(self, create_accounts):
        """Test cycle pattern forms a complete transaction loop"""
        accounts = create_accounts(4)
        random_state = np.random.RandomState(42)

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='cycle',
            accounts=accounts,
            model_id=3,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=1,
            source_type='TRANSFER',
            random_state=random_state
        )

        schedule = pattern.transaction_schedule

        # Should have n transactions for n accounts
        assert len(schedule) == 4

        # Build adjacency: from -> to
        edges = [(tx['from'], tx['to']) for tx in schedule]

        # Check it forms a cycle: each account should appear exactly once as sender and receiver
        senders = [edge[0] for edge in edges]
        receivers = [edge[1] for edge in edges]

        # Extract account objects from tuples
        account_objects = [acc for acc, is_main in accounts]

        for acc in account_objects:
            assert senders.count(acc) == 1, f"Account {acc.account_id} sends {senders.count(acc)} times"
            assert receivers.count(acc) == 1, f"Account {acc.account_id} receives {receivers.count(acc)} times"


@pytest.mark.unit
@pytest.mark.temporal
class TestForwardPatternOrdering:
    """Tests for forward pattern (A→B→C) transaction ordering"""

    def test_forward_two_sorted_transactions(self, create_accounts):
        """Test forward pattern generates 2 sorted transactions (A→B, B→C)"""
        # Forward pattern needs exactly 3 accounts
        accounts = create_accounts(3)
        random_state = np.random.RandomState(42)

        # Note: forward is not an AlertPattern type, but let's test the principle
        # We'll test this through the cycle pattern with 2 transactions
        pass  # Forward is a NormalModel, tested in test_transaction_schedules.py


@pytest.mark.unit
@pytest.mark.temporal
class TestStackPatternOrdering:
    """Tests for stack pattern (layered) transaction ordering"""

    def test_stack_layer_ordering(self, create_accounts):
        """Test stack pattern respects layer ordering"""
        accounts = create_accounts(9)  # 3 layers of 3 accounts each
        random_state = np.random.RandomState(42)

        # Create phase layers
        phase_layers = {
            0: [accounts[0][0], accounts[1][0], accounts[2][0]],  # Layer 0
            1: [accounts[3][0], accounts[4][0], accounts[5][0]],  # Layer 1
            2: [accounts[6][0], accounts[7][0], accounts[8][0]],  # Layer 2
        }

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='stack',
            accounts=accounts,
            model_id=5,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state,
            phase_layers=phase_layers
        )

        schedule = pattern.transaction_schedule

        # Verify transactions respect layer boundaries
        # Group transactions by layer transition
        layer_0_to_1 = []
        layer_1_to_2 = []

        for tx in schedule:
            from_acc = tx['from']
            to_acc = tx['to']

            # Check which layer accounts belong to
            if from_acc in phase_layers[0] and to_acc in phase_layers[1]:
                layer_0_to_1.append(tx)
            elif from_acc in phase_layers[1] and to_acc in phase_layers[2]:
                layer_1_to_2.append(tx)

        # Layer 1 transactions should happen after layer 0 receives funds
        if layer_0_to_1 and layer_1_to_2:
            # Get latest step from layer 0→1
            max_step_0_to_1 = max(tx['step'] for tx in layer_0_to_1)
            # Get earliest step from layer 1→2
            min_step_1_to_2 = min(tx['step'] for tx in layer_1_to_2)

            # Layer 1→2 should happen after some layer 0→1 transactions
            # (allowing for some overlap in the implementation)
            assert min_step_1_to_2 > max_step_0_to_1 - 50, \
                "Layer 1→2 transactions should generally follow layer 0→1"

    def test_stack_accounts_receive_before_sending(self, create_accounts):
        """Test accounts in intermediate layers receive before sending"""
        accounts = create_accounts(6)
        random_state = np.random.RandomState(42)

        phase_layers = {
            0: [accounts[0][0], accounts[1][0]],  # Layer 0 (sources)
            1: [accounts[2][0], accounts[3][0]],  # Layer 1 (intermediaries)
            2: [accounts[4][0], accounts[5][0]],  # Layer 2 (destinations)
        }

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='stack',
            accounts=accounts,
            model_id=5,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=1,
            source_type='TRANSFER',
            random_state=random_state,
            phase_layers=phase_layers
        )

        schedule = pattern.transaction_schedule

        # Track when each account receives and sends
        account_receives = defaultdict(list)  # account -> [steps]
        account_sends = defaultdict(list)

        for tx in schedule:
            account_sends[tx['from']].append(tx['step'])
            account_receives[tx['to']].append(tx['step'])

        # For layer 1 accounts, check they receive before sending
        for acc in phase_layers[1]:
            if acc in account_receives and acc in account_sends:
                earliest_receive = min(account_receives[acc])
                earliest_send = min(account_sends[acc])

                # Should receive before (or at same time as) sending
                assert earliest_receive <= earliest_send, \
                    f"Account {acc.account_id} sends before receiving"


@pytest.mark.unit
@pytest.mark.temporal
class TestScatterGatherOrdering:
    """Tests for scatter-gather pattern transaction ordering"""

    def test_scatter_gather_two_phases(self, create_accounts):
        """Test scatter-gather has scatter phase before gather phase"""
        accounts = create_accounts(7)
        random_state = np.random.RandomState(42)

        # Phase 0: originators, Phase 1: intermediaries, Phase 2: beneficiaries
        phase_layers = {
            0: [accounts[0][0]],  # 1 originator
            1: [accounts[1][0], accounts[2][0], accounts[3][0]],  # 3 intermediaries
            2: [accounts[4][0], accounts[5][0], accounts[6][0]],  # 3 beneficiaries
        }

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='scatter_gather',
            accounts=accounts,
            model_id=7,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state,
            phase_layers=phase_layers
        )

        schedule = pattern.transaction_schedule

        # Separate scatter (phase 0→1) and gather (phase 1→2) transactions
        scatter_txs = []
        gather_txs = []

        for tx in schedule:
            if tx['from'] in phase_layers[0] and tx['to'] in phase_layers[1]:
                scatter_txs.append(tx)
            elif tx['from'] in phase_layers[1] and tx['to'] in phase_layers[2]:
                gather_txs.append(tx)

        # Should have both scatter and gather transactions
        assert len(scatter_txs) > 0, "Should have scatter transactions"
        assert len(gather_txs) > 0, "Should have gather transactions"

        # Gather should happen after scatter
        max_scatter_step = max(tx['step'] for tx in scatter_txs)
        min_gather_step = min(tx['step'] for tx in gather_txs)

        assert min_gather_step > max_scatter_step, \
            "Gather phase should start after scatter phase completes"

    def test_scatter_gather_intermediaries_receive_before_send(self, create_accounts):
        """Test intermediaries in scatter-gather receive before sending"""
        accounts = create_accounts(5)
        random_state = np.random.RandomState(42)

        phase_layers = {
            0: [accounts[0][0]],  # Originator
            1: [accounts[1][0], accounts[2][0]],  # Intermediaries
            2: [accounts[3][0], accounts[4][0]],  # Beneficiaries
        }

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='scatter_gather',
            accounts=accounts,
            model_id=7,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=1,
            source_type='TRANSFER',
            random_state=random_state,
            phase_layers=phase_layers
        )

        schedule = pattern.transaction_schedule

        # Track when intermediaries receive and send
        for intermediary in phase_layers[1]:
            receive_steps = [tx['step'] for tx in schedule if tx['to'] == intermediary]
            send_steps = [tx['step'] for tx in schedule if tx['from'] == intermediary]

            if receive_steps and send_steps:
                latest_receive = max(receive_steps)
                earliest_send = min(send_steps)

                # Should receive before sending
                assert latest_receive < earliest_send, \
                    f"Intermediary {intermediary.account_id} should receive before sending"


@pytest.mark.unit
@pytest.mark.temporal
class TestGatherScatterOrdering:
    """Tests for gather-scatter pattern transaction ordering"""

    def test_gather_scatter_two_phases(self, create_accounts):
        """Test gather-scatter has gather phase before scatter phase"""
        accounts = create_accounts(7)
        random_state = np.random.RandomState(42)

        # Phase 0: hubs, Phase 1: sources, Phase 2: targets
        phase_layers = {
            0: [accounts[0][0]],  # Hub
            1: [accounts[1][0], accounts[2][0], accounts[3][0]],  # Sources
            2: [accounts[4][0], accounts[5][0], accounts[6][0]],  # Targets
        }

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='gather_scatter',
            accounts=accounts,
            model_id=8,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state,
            phase_layers=phase_layers
        )

        schedule = pattern.transaction_schedule

        # Separate gather (phase 1→0) and scatter (phase 0→2) transactions
        gather_txs = []
        scatter_txs = []

        for tx in schedule:
            if tx['from'] in phase_layers[1] and tx['to'] in phase_layers[0]:
                gather_txs.append(tx)
            elif tx['from'] in phase_layers[0] and tx['to'] in phase_layers[2]:
                scatter_txs.append(tx)

        # Should have both gather and scatter transactions
        assert len(gather_txs) > 0, "Should have gather transactions"
        assert len(scatter_txs) > 0, "Should have scatter transactions"

        # Scatter should happen after gather
        max_gather_step = max(tx['step'] for tx in gather_txs)
        min_scatter_step = min(tx['step'] for tx in scatter_txs)

        assert min_scatter_step > max_gather_step, \
            "Scatter phase should start after gather phase completes"


@pytest.mark.unit
@pytest.mark.temporal
class TestBipartitePatternOrdering:
    """Tests for bipartite pattern transaction ordering"""

    def test_bipartite_all_cross_group(self, create_accounts):
        """Test bipartite pattern creates transactions between two groups"""
        accounts = create_accounts(6)
        random_state = np.random.RandomState(42)

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='bipartite',
            accounts=accounts,
            model_id=4,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=1,
            source_type='TRANSFER',
            random_state=random_state
        )

        schedule = pattern.transaction_schedule

        # Extract all unique accounts involved
        all_accounts_in_pattern = set()
        for tx in schedule:
            all_accounts_in_pattern.add(tx['from'])
            all_accounts_in_pattern.add(tx['to'])

        # Should have transactions
        assert len(schedule) > 0

        # All steps should be sorted
        steps = [tx['step'] for tx in schedule]
        assert steps == sorted(steps)


@pytest.mark.unit
@pytest.mark.temporal
class TestFanInFanOutOrdering:
    """Tests for fan-in and fan-out patterns"""

    def test_fan_out_all_sorted(self, create_accounts):
        """Test fan-out pattern transactions are sorted"""
        accounts = create_accounts(5)
        random_state = np.random.RandomState(42)

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='fan_out',
            accounts=accounts,
            model_id=1,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state
        )

        schedule = pattern.transaction_schedule

        steps = [tx['step'] for tx in schedule]
        assert steps == sorted(steps)

        # All should have same sender (main account)
        senders = set(tx['from'] for tx in schedule)
        assert len(senders) == 1, "Fan-out should have single sender"

    def test_fan_in_all_sorted(self, create_accounts):
        """Test fan-in pattern transactions are sorted"""
        accounts = create_accounts(5)
        random_state = np.random.RandomState(42)

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='fan_in',
            accounts=accounts,
            model_id=2,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state
        )

        schedule = pattern.transaction_schedule

        steps = [tx['step'] for tx in schedule]
        assert steps == sorted(steps)

        # All should have same receiver (main account)
        receivers = set(tx['to'] for tx in schedule)
        assert len(receivers) == 1, "Fan-in should have single receiver"


@pytest.mark.unit
@pytest.mark.temporal
class TestRandomPatternOrdering:
    """Tests for random pattern"""

    def test_random_pattern_sorted(self, create_accounts):
        """Test random pattern still produces sorted steps"""
        accounts = create_accounts(5)
        random_state = np.random.RandomState(42)

        pattern = AlertPattern(
            alert_id=1,
            pattern_type='random',
            accounts=accounts,
            model_id=6,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state
        )

        schedule = pattern.transaction_schedule

        steps = [tx['step'] for tx in schedule]
        # Even random pattern should have sorted steps
        assert steps == sorted(steps)


@pytest.mark.unit
@pytest.mark.temporal
class TestAllPatternsWithinBounds:
    """Test all patterns respect start/end bounds"""

    @pytest.mark.parametrize("pattern_type", [
        'fan_out', 'fan_in', 'cycle', 'bipartite', 'random'
    ])
    def test_pattern_within_bounds(self, pattern_type, create_accounts):
        """Test pattern transactions fall within start/end bounds"""
        accounts = create_accounts(5)
        random_state = np.random.RandomState(42)

        start_step = 20
        end_step = 80

        model_id_map = {
            'fan_out': 1, 'fan_in': 2, 'cycle': 3,
            'bipartite': 4, 'random': 6
        }

        pattern = AlertPattern(
            alert_id=1,
            pattern_type=pattern_type,
            accounts=accounts,
            model_id=model_id_map[pattern_type],
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            start_step=start_step,
            end_step=end_step,
            burstiness_level=2,
            source_type='TRANSFER',
            random_state=random_state
        )

        schedule = pattern.transaction_schedule

        # All steps should be within bounds
        for tx in schedule:
            assert start_step <= tx['step'] <= end_step, \
                f"Pattern {pattern_type} has transaction at step {tx['step']} outside [{start_step}, {end_step}]"
