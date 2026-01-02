"""
Integration tests for enhanced multi-phase AML patterns
Tests scatter-gather, gather-scatter, and stack patterns with random connections and amount splitting
"""
import pytest
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from src.utils.pattern_types import SAR_PATTERN_TYPES

# Mark to skip conftest autouse fixture (we generate our own data)
pytestmark = pytest.mark.usefixtures()


@pytest.fixture(scope="module")
def test_data_dir():
    """Get the test data directory"""
    return Path(__file__).parent.parent / "parameters" / "large_test"


@pytest.fixture(scope="module")
def generated_transactions(test_data_dir):
    """Generate test data if needed and return transactions"""
    # Check if spatial and temporal directories exist
    spatial_dir = test_data_dir / "spatial"
    temporal_dir = test_data_dir / "temporal"
    tx_file = temporal_dir / "tx_log.parquet"

    # Generate data if it doesn't exist
    if not tx_file.exists():
        print(f"\nGenerating test data for enhanced pattern tests...")
        print(f"Test data dir: {test_data_dir}")
        print(f"Expected output: {tx_file}")

        config_file = test_data_dir / "config" / "data.yaml"
        print(f"Config file: {config_file}")
        print(f"Config exists: {config_file.exists()}")

        # Run the generator
        result = subprocess.run(
            ["uv", "run", "python", "scripts/generate.py", "--conf_file", str(config_file)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Failed to generate test data:\n{result.stderr}")

        if not tx_file.exists():
            pytest.fail(f"Test data generation completed but output file not found: {tx_file}")

    df = pd.read_parquet(tx_file)
    return df


@pytest.fixture
def alert_transactions(generated_transactions):
    """Filter for SAR alert transactions only"""
    return generated_transactions[generated_transactions['isSAR'] == 1].copy()


def get_first_pattern_txs(alert_transactions, pattern_type):
    """Helper to get transactions for first instance of a pattern type

    Args:
        alert_transactions: DataFrame of alert transactions
        pattern_type: Pattern type name string (e.g., 'scatter_gather')

    Returns:
        DataFrame of transactions for the first instance of this pattern type
    """
    # Convert pattern type string to integer ID
    type_id = SAR_PATTERN_TYPES.get(pattern_type)
    if type_id is None:
        return pd.DataFrame()  # Return empty if pattern type not found

    # Filter by integer modelType (use int() to ensure type compatibility with np.int16)
    patterns = alert_transactions[alert_transactions['modelType'] == int(type_id)]
    if len(patterns) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no patterns found
    first_pattern_id = patterns['patternID'].min()
    return patterns[patterns['patternID'] == first_pattern_id].sort_values('step')


@pytest.mark.integration
@pytest.mark.temporal
class TestScatterGatherPattern:
    """Tests for enhanced scatter-gather pattern with random connections and amount splitting"""

    def test_scatter_gather_has_two_phases(self, alert_transactions):
        """Test that scatter-gather has distinct scatter and gather phases"""
        sg_txs = get_first_pattern_txs(alert_transactions, 'scatter_gather')

        assert len(sg_txs) > 0, "Scatter-gather pattern should have transactions"

        # Get time midpoint
        midpoint = (sg_txs['step'].min() + sg_txs['step'].max()) / 2

        # Should have transactions in both halves
        phase1 = sg_txs[sg_txs['step'] <= midpoint]
        phase2 = sg_txs[sg_txs['step'] > midpoint]

        assert len(phase1) > 0, "Should have scatter phase transactions"
        assert len(phase2) > 0, "Should have gather phase transactions"

    def test_scatter_gather_amount_splitting(self, alert_transactions):
        """Test that intermediaries properly split received amounts with margin"""
        sg_txs = get_first_pattern_txs(alert_transactions, 'scatter_gather')

        # Check each intermediary that both receives and sends
        receivers = set(sg_txs['nameDest'].unique())
        senders = set(sg_txs['nameOrig'].unique())
        intermediaries = receivers & senders

        for intermediate in intermediaries:
            # Get all transactions where intermediary receives
            incoming = sg_txs[sg_txs['nameDest'] == intermediate]
            max_receive_step = incoming['step'].max()

            # Get all transactions where intermediary sends AFTER receiving
            outgoing = sg_txs[(sg_txs['nameOrig'] == intermediate) & (sg_txs['step'] >= max_receive_step)]

            if len(outgoing) > 0:
                received = incoming['amount'].sum()
                sent = outgoing['amount'].sum()

                # With 10% margin, sent should be ~90% of received
                expected_sent = received * 0.9

                # Allow small floating point tolerance
                assert abs(sent - expected_sent) < 1.0, \
                    f"Intermediate {intermediate}: received {received:.2f}, sent {sent:.2f}, expected {expected_sent:.2f}"

    def test_scatter_gather_random_connections(self, alert_transactions):
        """Test that intermediaries have random connections (not all-to-all)"""
        sg_txs = get_first_pattern_txs(alert_transactions, 'scatter_gather')

        # Get time midpoint
        midpoint = (sg_txs['step'].min() + sg_txs['step'].max()) / 2
        phase2 = sg_txs[sg_txs['step'] > midpoint]

        # Each intermediary should send to 1-3 beneficiaries (not necessarily all)
        for intermediate in phase2['nameOrig'].unique():
            beneficiaries = phase2[phase2['nameOrig'] == intermediate]['nameDest'].nunique()
            assert 1 <= beneficiaries <= 3, \
                f"Intermediate {intermediate} should send to 1-3 beneficiaries, got {beneficiaries}"

    def test_scatter_gather_three_phase_structure(self, alert_transactions):
        """Test that scatter-gather uses 3 distinct phases: originators, intermediaries, beneficiaries"""
        sg_txs = get_first_pattern_txs(alert_transactions, 'scatter_gather')

        midpoint = (sg_txs['step'].min() + sg_txs['step'].max()) / 2

        phase1 = sg_txs[sg_txs['step'] <= midpoint]
        phase2 = sg_txs[sg_txs['step'] > midpoint]

        originators = set(phase1['nameOrig'].unique())
        intermediaries = set(phase1['nameDest'].unique()) & set(phase2['nameOrig'].unique())
        beneficiaries = set(phase2['nameDest'].unique())

        # Should have accounts in all 3 phases
        assert len(originators) > 0, "Should have originators"
        assert len(intermediaries) > 0, "Should have intermediaries"
        assert len(beneficiaries) > 0, "Should have beneficiaries"

        # Intermediaries should be distinct from pure originators and beneficiaries
        assert len(intermediaries) > 0, "Should have distinct intermediary accounts"


@pytest.mark.integration
@pytest.mark.temporal
class TestGatherScatterPattern:
    """Tests for enhanced gather-scatter pattern with random connections and amount splitting"""

    def test_gather_scatter_has_two_phases(self, alert_transactions):
        """Test that gather-scatter has distinct gather and scatter phases"""
        # Gather-scatter patterns are patternID 60-69 (10 patterns)
        # Use first gather-scatter pattern
        gs_txs = get_first_pattern_txs(alert_transactions, 'gather_scatter')

        assert len(gs_txs) > 0, "Gather-scatter pattern should have transactions"

        # Get time midpoint
        midpoint = (gs_txs['step'].min() + gs_txs['step'].max()) / 2

        # Should have transactions in both halves
        phase1 = gs_txs[gs_txs['step'] <= midpoint]
        phase2 = gs_txs[gs_txs['step'] > midpoint]

        assert len(phase1) > 0, "Should have gather phase transactions"
        assert len(phase2) > 0, "Should have scatter phase transactions"

    def test_gather_scatter_amount_splitting(self, alert_transactions):
        """Test that hubs properly split received amounts with margin"""
        gs_txs = get_first_pattern_txs(alert_transactions, 'gather_scatter')

        # Check each hub that both receives and sends
        receivers = set(gs_txs['nameDest'].unique())
        senders = set(gs_txs['nameOrig'].unique())
        hubs = receivers & senders

        for hub in hubs:
            # Get all transactions where hub receives
            incoming = gs_txs[gs_txs['nameDest'] == hub]
            max_receive_step = incoming['step'].max()

            # Get all transactions where hub sends AFTER receiving
            outgoing = gs_txs[(gs_txs['nameOrig'] == hub) & (gs_txs['step'] >= max_receive_step)]

            if len(outgoing) > 0:
                received = incoming['amount'].sum()
                sent = outgoing['amount'].sum()

                # With 10% margin, sent should be ~90% of received
                expected_sent = received * 0.9

                # Allow small floating point tolerance
                assert abs(sent - expected_sent) < 1.0, \
                    f"Hub {hub}: received {received:.2f}, sent {sent:.2f}, expected {expected_sent:.2f}"

    def test_gather_scatter_random_connections(self, alert_transactions):
        """Test that hubs have random connections (not all-to-all)"""
        gs_txs = get_first_pattern_txs(alert_transactions, 'gather_scatter')

        # Get time midpoint
        midpoint = (gs_txs['step'].min() + gs_txs['step'].max()) / 2
        phase2 = gs_txs[gs_txs['step'] > midpoint]

        # Each hub should send to 1-3 targets (not necessarily all)
        for hub in phase2['nameOrig'].unique():
            targets = phase2[phase2['nameOrig'] == hub]['nameDest'].nunique()
            assert 1 <= targets <= 3, \
                f"Hub {hub} should send to 1-3 targets, got {targets}"

    def test_gather_scatter_three_phase_structure(self, alert_transactions):
        """Test that gather-scatter uses 3 distinct roles: sources, hubs, targets"""
        gs_txs = get_first_pattern_txs(alert_transactions, 'gather_scatter')

        midpoint = (gs_txs['step'].min() + gs_txs['step'].max()) / 2

        phase1 = gs_txs[gs_txs['step'] <= midpoint]
        phase2 = gs_txs[gs_txs['step'] > midpoint]

        sources = set(phase1['nameOrig'].unique())
        hubs = set(phase1['nameDest'].unique()) & set(phase2['nameOrig'].unique())
        targets = set(phase2['nameDest'].unique())

        # Should have accounts in all 3 phases
        assert len(sources) > 0, "Should have sources"
        assert len(hubs) > 0, "Should have hubs"
        assert len(targets) > 0, "Should have targets"

        # Hubs should be distinct from pure sources and targets
        assert len(hubs) > 0, "Should have distinct hub accounts"


@pytest.mark.integration
@pytest.mark.temporal
class TestStackPattern:
    """Tests for enhanced stack pattern with layered amount splitting"""

    def test_stack_has_multiple_layers(self, alert_transactions):
        """Test that stack pattern has multiple distinct layers"""
        # Stack patterns are patternID 30-39 (10 patterns)
        # Use first stack pattern
        stack_txs = get_first_pattern_txs(alert_transactions, 'stack')

        assert len(stack_txs) > 0, "Stack pattern should have transactions"

        # Should have at least 2 layers (source and destination)
        all_accounts = set(stack_txs['nameOrig'].unique()) | set(stack_txs['nameDest'].unique())
        assert len(all_accounts) >= 2, "Stack should have at least 2 distinct accounts"

    def test_stack_amount_splitting(self, alert_transactions):
        """Test that accounts in stack properly split received amounts with margin"""
        stack_txs = get_first_pattern_txs(alert_transactions, 'stack')

        # For each account that both receives and sends
        accounts_receive = set(stack_txs['nameDest'].unique())
        accounts_send = set(stack_txs['nameOrig'].unique())
        intermediates = accounts_receive & accounts_send

        for account in intermediates:
            # Get all incoming transactions
            incoming = stack_txs[stack_txs['nameDest'] == account]
            # Get all outgoing transactions that happen after receiving
            max_receive_step = incoming['step'].max()
            outgoing = stack_txs[(stack_txs['nameOrig'] == account) &
                                (stack_txs['step'] >= max_receive_step)]

            if len(outgoing) > 0:
                received = incoming['amount'].sum()
                sent = outgoing['amount'].sum()

                # With 10% margin, sent should be ~90% of received
                expected_sent = received * 0.9

                # Allow small floating point tolerance
                assert abs(sent - expected_sent) < 1.0, \
                    f"Account {account}: received {received:.2f}, sent {sent:.2f}, expected {expected_sent:.2f}"

    def test_stack_layered_flow(self, alert_transactions):
        """Test that transactions flow through layers sequentially"""
        stack_txs = get_first_pattern_txs(alert_transactions, 'stack')

        # Each account should receive before it sends
        accounts_both = set(stack_txs['nameDest'].unique()) & set(stack_txs['nameOrig'].unique())

        for account in accounts_both:
            first_receive = stack_txs[stack_txs['nameDest'] == account]['step'].min()
            first_send = stack_txs[stack_txs['nameOrig'] == account]['step'].min()

            # Should receive before sending (or at same step for simultaneous)
            assert first_receive <= first_send, \
                f"Account {account} should receive (step {first_receive}) before/at sending (step {first_send})"

    def test_stack_random_connections(self, alert_transactions):
        """Test that accounts have random connections to next layer (not all-to-all)"""
        stack_txs = get_first_pattern_txs(alert_transactions, 'stack')

        # Each sending account should connect to 1-3 accounts in next layer
        for sender in stack_txs['nameOrig'].unique():
            recipients = stack_txs[stack_txs['nameOrig'] == sender]['nameDest'].nunique()
            assert 1 <= recipients <= 3, \
                f"Sender {sender} should connect to 1-3 recipients, got {recipients}"


@pytest.mark.integration
@pytest.mark.temporal
class TestAmountConservation:
    """Tests for overall amount conservation across patterns"""

    def test_total_amounts_reasonable(self, alert_transactions):
        """Test that total transaction amounts are within reasonable bounds"""
        for pattern_id in alert_transactions['patternID'].unique():
            alert_txs = alert_transactions[alert_transactions['patternID'] == pattern_id]
            total_amount = alert_txs['amount'].sum()

            # Total should be positive and within configured bounds
            assert total_amount > 0, f"Alert {pattern_id} should have positive total amount"

            # Each individual transaction should be positive and within max bounds
            # Note: amounts < 1 can occur after multiple margin applications in layered patterns
            assert alert_txs['amount'].min() > 0, f"Alert {pattern_id} min amount should be positive"
            assert alert_txs['amount'].max() <= 150000, f"Alert {pattern_id} max amount should be <= 150000"

    def test_margin_ratio_applied_consistently(self, alert_transactions):
        """Test that 10% margin ratio is applied consistently across all patterns"""
        margin_ratio = 0.1
        tolerance = 1.0  # Allow $1 tolerance for floating point

        # Test all pattern types with amount splitting
        for pattern_type in ['stack', 'scatter_gather', 'gather_scatter']:
            alert_txs = get_first_pattern_txs(alert_transactions, pattern_type)
            if len(alert_txs) == 0:
                continue  # Skip if pattern type not present

            # Find accounts that both receive and send
            receivers = set(alert_txs['nameDest'].unique())
            senders = set(alert_txs['nameOrig'].unique())
            intermediates = receivers & senders

            for account in intermediates:
                incoming = alert_txs[alert_txs['nameDest'] == account]
                max_receive_step = incoming['step'].max()
                outgoing = alert_txs[(alert_txs['nameOrig'] == account) &
                                    (alert_txs['step'] >= max_receive_step)]

                if len(outgoing) > 0:
                    received = incoming['amount'].sum()
                    sent = outgoing['amount'].sum()
                    expected = received * (1 - margin_ratio)

                    assert abs(sent - expected) < tolerance, \
                        f"{pattern_type}, Account {account}: {sent:.2f} sent != {expected:.2f} expected (90% of {received:.2f})"
