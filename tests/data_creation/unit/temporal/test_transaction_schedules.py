"""
Unit tests for transaction scheduling and timing in temporal simulation
Tests beta distribution-based burstiness levels and pattern ordering
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'python'))

from src.data_creation.python.temporal_simulation.alert_patterns import PatternScheduler, AlertPattern
from src.data_creation.python.temporal_simulation.normal_models import ForwardModel, MutualModel, PeriodicalModel


@pytest.mark.unit
@pytest.mark.temporal
class TestBurstinessLevelSampling:
    """Tests for burstiness level probability distribution"""

    def test_compute_burstiness_probs_zero_bias(self):
        """Test zero bias gives equal probabilities"""
        probs = PatternScheduler.compute_burstiness_probs(bias=0.0)

        assert len(probs) == 4
        assert set(probs.keys()) == {1, 2, 3, 4}

        # All probabilities should be equal (1/4 = 0.25)
        for level, prob in probs.items():
            assert abs(prob - 0.25) < 0.01

        # Should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_compute_burstiness_probs_negative_bias(self):
        """Test negative bias favors lower levels"""
        probs = PatternScheduler.compute_burstiness_probs(bias=-2.0)

        # Level 1 should have highest probability
        assert probs[1] > probs[2]
        assert probs[2] > probs[3]
        assert probs[3] > probs[4]

        # Should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_compute_burstiness_probs_positive_bias(self):
        """Test positive bias favors higher levels"""
        probs = PatternScheduler.compute_burstiness_probs(bias=2.0)

        # Level 4 should have highest probability
        assert probs[4] > probs[3]
        assert probs[3] > probs[2]
        assert probs[2] > probs[1]

        # Should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_sample_burstiness_level_distribution(self):
        """Test sampled levels follow the distribution"""
        random_state = np.random.RandomState(42)
        bias = 1.0
        num_samples = 1000

        samples = [PatternScheduler.sample_burstiness_level(bias, random_state)
                   for _ in range(num_samples)]

        # All samples should be valid levels
        assert all(level in [1, 2, 3, 4] for level in samples)

        # With positive bias, level 4 should be most common
        counts = {i: samples.count(i) for i in range(1, 5)}
        assert counts[4] > counts[1]

    def test_sample_burstiness_level_reproducibility(self):
        """Test same seed gives same samples"""
        samples1 = [PatternScheduler.sample_burstiness_level(0.5, np.random.RandomState(42))
                    for _ in range(10)]
        samples2 = [PatternScheduler.sample_burstiness_level(0.5, np.random.RandomState(42))
                    for _ in range(10)]

        assert samples1 == samples2


@pytest.mark.unit
@pytest.mark.temporal
class TestBetaDistributionScheduler:
    """Tests for PatternScheduler with beta distributions"""

    def test_get_transaction_steps_basic(self):
        """Test basic step generation"""
        random_state = np.random.RandomState(42)
        start_step = 0
        end_step = 100
        num_transactions = 5
        burstiness_level = 1  # Uniform distribution

        steps = PatternScheduler.get_transaction_steps(
            start_step=start_step,
            end_step=end_step,
            num_transactions=num_transactions,
            burstiness_level=burstiness_level,
            random_state=random_state
        )

        # Should have correct number of transactions
        assert len(steps) == num_transactions

        # All steps should be within range
        assert all(start_step <= s < end_step for s in steps)

        # Steps should be sorted
        assert steps == sorted(steps)

    def test_get_transaction_steps_single_transaction(self):
        """Test single transaction placement"""
        random_state = np.random.RandomState(42)

        steps = PatternScheduler.get_transaction_steps(
            start_step=0,
            end_step=100,
            num_transactions=1,
            burstiness_level=2,
            random_state=random_state
        )

        assert len(steps) == 1
        assert 0 <= steps[0] < 100

    def test_get_transaction_steps_zero_transactions(self):
        """Test empty transaction list"""
        random_state = np.random.RandomState(42)

        steps = PatternScheduler.get_transaction_steps(
            start_step=0,
            end_step=100,
            num_transactions=0,
            burstiness_level=1,
            random_state=random_state
        )

        assert len(steps) == 0

    def test_burstiness_level_1_uniform(self):
        """Test level 1 (uniform) produces spread-out transactions"""
        random_state = np.random.RandomState(42)
        num_samples = 10

        all_steps = []
        for _ in range(num_samples):
            steps = PatternScheduler.get_transaction_steps(
                start_step=0,
                end_step=100,
                num_transactions=10,
                burstiness_level=1,  # Beta(1, 1) = uniform
                random_state=np.random.RandomState(np.random.randint(10000))
            )
            all_steps.extend(steps)

        # Should have good coverage across the range
        # Check that we have transactions in first and last quartiles
        assert any(s < 25 for s in all_steps)
        assert any(s > 75 for s in all_steps)

    def test_burstiness_level_4_clustered(self):
        """Test level 4 (bimodal) can produce clustered transactions"""
        random_state = np.random.RandomState(42)

        # Generate many samples to test distribution
        all_steps = []
        for _ in range(20):
            steps = PatternScheduler.get_transaction_steps(
                start_step=0,
                end_step=100,
                num_transactions=10,
                burstiness_level=4,  # Beta(0.3, 0.3) = bimodal
                random_state=np.random.RandomState(np.random.randint(10000))
            )
            all_steps.extend(steps)

        # With bimodal distribution, should have more extreme values
        # (near 0 or near 100)
        extreme_count = sum(1 for s in all_steps if s < 20 or s > 80)
        middle_count = sum(1 for s in all_steps if 40 <= s <= 60)

        # Should have more extreme than middle (with high probability)
        # This is a statistical test, so we use a threshold
        assert extreme_count > middle_count * 0.5

    def test_reproducibility_with_seed(self):
        """Test same seed produces same steps"""
        steps1 = PatternScheduler.get_transaction_steps(
            start_step=0, end_step=100, num_transactions=10,
            burstiness_level=2, random_state=np.random.RandomState(42)
        )

        steps2 = PatternScheduler.get_transaction_steps(
            start_step=0, end_step=100, num_transactions=10,
            burstiness_level=2, random_state=np.random.RandomState(42)
        )

        assert steps1 == steps2

    def test_sorted_output(self):
        """Test steps are always sorted"""
        for level in [1, 2, 3, 4]:
            steps = PatternScheduler.get_transaction_steps(
                start_step=0, end_step=100, num_transactions=20,
                burstiness_level=level, random_state=np.random.RandomState(42)
            )

            assert steps == sorted(steps), f"Level {level} produced unsorted steps"

    def test_boundary_conditions(self):
        """Test edge cases in step generation"""
        random_state = np.random.RandomState(42)

        # Very small period
        steps = PatternScheduler.get_transaction_steps(
            start_step=0, end_step=2, num_transactions=1,
            burstiness_level=1, random_state=random_state
        )
        assert len(steps) == 1
        assert 0 <= steps[0] < 2

        # Large number of transactions
        steps = PatternScheduler.get_transaction_steps(
            start_step=0, end_step=1000, num_transactions=500,
            burstiness_level=2, random_state=np.random.RandomState(42)
        )
        assert len(steps) == 500
        assert all(0 <= s < 1000 for s in steps)


@pytest.mark.unit
@pytest.mark.temporal
class TestForwardModelWithBurstiness:
    """Tests for ForwardModel with burstiness levels"""

    def test_forward_two_transactions(self):
        """Test ForwardModel generates 2 sorted transactions"""
        accounts = [(f"acc_{i}", i == 0) for i in range(3)]
        random_state = np.random.RandomState(42)

        model = ForwardModel(
            model_id=1,
            accounts=accounts,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            random_state=random_state
        )

        # Should have 2 transaction steps
        assert len(model.transaction_steps) == 2

        # Both should be within range
        assert all(0 <= s <= 100 for s in model.transaction_steps)

        # Should be sorted (ensuring A→B happens before B→C)
        assert model.transaction_steps[0] < model.transaction_steps[1]

    def test_forward_transaction_ordering(self, sample_account, sample_sar_account):
        """Test ForwardModel maintains proper transaction sequence"""
        from src.data_creation.python.temporal_simulation.account import Account

        # Create third account
        account_3 = Account(
            account_id=3, customer_id=300, initial_balance=10000.0,
            is_sar=False, bank_id='BANK001', random_state=42
        )

        accounts = [(sample_account, True), (sample_sar_account, False), (account_3, False)]
        random_state = np.random.RandomState(42)

        model = ForwardModel(
            model_id=1,
            accounts=accounts,
            start_step=10,
            end_step=50,
            burstiness_level=1,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            random_state=random_state
        )

        transactions_generated = []

        # Simulate steps
        for step in range(0, 60):
            tx = model.get_transaction(step)
            if tx:
                transactions_generated.append(tx)

        # Should generate exactly 2 transactions
        assert len(transactions_generated) == 2

        # First transaction step < second transaction step
        assert transactions_generated[0]['step'] < transactions_generated[1]['step']


@pytest.mark.unit
@pytest.mark.temporal
class TestMutualModelWithBurstiness:
    """Tests for MutualModel with burstiness levels"""

    def test_mutual_two_sorted_steps(self):
        """Test MutualModel generates 2 sorted steps for loan and repayment"""
        accounts = [("lender", True), ("debtor", False)]
        random_state = np.random.RandomState(42)

        model = MutualModel(
            model_id=1,
            accounts=accounts,
            start_step=0,
            end_step=100,
            burstiness_level=2,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            random_state=random_state
        )

        # Should have 2 steps
        assert len(model.transaction_steps) == 2

        # Loan should happen before repayment
        assert model.transaction_steps[0] < model.transaction_steps[1]

    def test_mutual_loan_before_repayment(self, sample_account, sample_sar_account):
        """Test loan always happens before repayment"""
        accounts = [(sample_account, True), (sample_sar_account, False)]
        random_state = np.random.RandomState(42)

        model = MutualModel(
            model_id=1,
            accounts=accounts,
            start_step=0,
            end_step=100,
            burstiness_level=3,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            random_state=random_state
        )

        transactions = []

        # Simulate steps
        for step in range(0, 110):
            tx = model.get_transaction(step)
            if tx:
                transactions.append(tx)

        # Should have exactly 2 transactions
        assert len(transactions) == 2

        # First should be loan (lender -> debtor)
        assert transactions[0]['from'] == sample_account
        assert transactions[0]['to'] == sample_sar_account

        # Second should be repayment (debtor -> lender)
        assert transactions[1]['from'] == sample_sar_account
        assert transactions[1]['to'] == sample_account

        # Loan step < repayment step
        assert transactions[0]['step'] < transactions[1]['step']


@pytest.mark.unit
@pytest.mark.temporal
class TestPeriodicalModelWithBurstiness:
    """Tests for PeriodicalModel with burstiness levels"""

    def test_periodical_generates_multiple_transactions(self):
        """Test PeriodicalModel generates multiple transactions"""
        accounts = [("sender", True), ("receiver", False)]
        random_state = np.random.RandomState(42)

        model = PeriodicalModel(
            model_id=1,
            accounts=accounts,
            start_step=0,
            end_step=100,
            burstiness_level=1,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            random_state=random_state
        )

        # Should have multiple transaction steps (5-15 range)
        assert 5 <= len(model.transaction_steps) <= 15

        # All within range
        assert all(0 <= s <= 100 for s in model.transaction_steps)

    def test_periodical_steps_within_period(self):
        """Test all periodical steps are within specified period"""
        accounts = [("sender", True), ("receiver", False)]
        random_state = np.random.RandomState(42)

        start_step = 50
        end_step = 150

        model = PeriodicalModel(
            model_id=1,
            accounts=accounts,
            start_step=start_step,
            end_step=end_step,
            burstiness_level=2,
            mean_amount=300,
            std_amount=100,
            min_amount=100,
            max_amount=500,
            random_state=random_state
        )

        # All steps should be within period
        for step in model.transaction_steps:
            assert start_step <= step <= end_step


@pytest.mark.unit
@pytest.mark.temporal
class TestScheduleReproducibility:
    """Tests for schedule reproducibility with random seeds"""

    def test_same_seed_same_schedule(self):
        """Test same random seed produces same schedule"""
        accounts = [("acc_0", True), ("acc_1", False), ("acc_2", False)]

        # Create two models with same seed
        model1 = ForwardModel(
            model_id=1, accounts=accounts, start_step=0, end_step=100,
            burstiness_level=2, mean_amount=300, std_amount=100,
            min_amount=100, max_amount=500, random_state=np.random.RandomState(42)
        )

        model2 = ForwardModel(
            model_id=1, accounts=accounts, start_step=0, end_step=100,
            burstiness_level=2, mean_amount=300, std_amount=100,
            min_amount=100, max_amount=500, random_state=np.random.RandomState(42)
        )

        # Should have identical schedules
        assert model1.transaction_steps == model2.transaction_steps

    def test_different_seed_different_schedule(self):
        """Test different random seed produces different schedule"""
        accounts = [("acc_0", True), ("acc_1", False), ("acc_2", False)]

        model1 = ForwardModel(
            model_id=1, accounts=accounts, start_step=0, end_step=100,
            burstiness_level=2, mean_amount=300, std_amount=100,
            min_amount=100, max_amount=500, random_state=np.random.RandomState(42)
        )

        model2 = ForwardModel(
            model_id=1, accounts=accounts, start_step=0, end_step=100,
            burstiness_level=2, mean_amount=300, std_amount=100,
            min_amount=100, max_amount=500, random_state=np.random.RandomState(99)
        )

        # Should have different schedules (with high probability)
        assert model1.transaction_steps != model2.transaction_steps

    def test_different_burstiness_different_distribution(self):
        """Test different burstiness levels produce different distributions"""
        accounts = [("acc_0", True), ("acc_1", False)]

        # Level 1 (uniform)
        model1 = MutualModel(
            model_id=1, accounts=accounts, start_step=0, end_step=100,
            burstiness_level=1, mean_amount=300, std_amount=100,
            min_amount=100, max_amount=500, random_state=np.random.RandomState(42)
        )

        # Level 4 (bimodal)
        model2 = MutualModel(
            model_id=1, accounts=accounts, start_step=0, end_step=100,
            burstiness_level=4, mean_amount=300, std_amount=100,
            min_amount=100, max_amount=500, random_state=np.random.RandomState(42)
        )

        # Different burstiness should produce different steps (with high probability)
        assert model1.transaction_steps != model2.transaction_steps
