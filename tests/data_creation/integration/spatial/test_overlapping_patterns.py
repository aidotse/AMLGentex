"""
Integration test for accounts participating in multiple alert patterns during graph generation
"""
import pytest
from collections import defaultdict


@pytest.mark.integration
@pytest.mark.spatial
class TestOverlappingAlertPatterns:
    """Test that accounts can be assigned to multiple alert typologies"""

    def test_accounts_participate_in_multiple_alert_patterns(self, large_graph):
        """Test that the probabilistic assignment creates some accounts in multiple alert patterns"""
        txg = large_graph

        # Track which accounts appear in which alert patterns
        account_to_alerts = defaultdict(set)

        for alert_id, alert_graph in txg.alert_groups.items():
            for node in alert_graph.nodes():
                account_to_alerts[node].add(alert_id)

        # Count accounts in multiple patterns
        overlap_counts = defaultdict(int)
        for acc, alerts in account_to_alerts.items():
            num_patterns = len(alerts)
            overlap_counts[num_patterns] += 1

        # With large graph (10 patterns of each type, 10-20 accounts each),
        # probabilistic assignment should create some overlaps
        accounts_in_multiple = sum(count for num, count in overlap_counts.items() if num > 1)

        print(f"\nAlert pattern participation:")
        for num_patterns in sorted(overlap_counts.keys()):
            count = overlap_counts[num_patterns]
            print(f"  {num_patterns} pattern(s): {count} accounts")

        # Verify that accounts can participate in multiple patterns
        assert accounts_in_multiple > 0, \
            "Expected some accounts to participate in multiple alert patterns due to probabilistic assignment"

        # Verify overlapping accounts have valid memberships
        for acc, alerts in account_to_alerts.items():
            if len(alerts) > 1:
                # Each alert should be distinct
                assert len(alerts) == len(set(alerts))

                # Verify account actually exists in each alert graph
                for alert_id in alerts:
                    alert_graph = txg.alert_groups[alert_id]
                    assert acc in alert_graph.nodes()
