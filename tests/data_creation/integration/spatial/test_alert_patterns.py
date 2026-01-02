"""
Unit tests for alert pattern generation in spatial simulation
Tests fan_in, fan_out, bipartite, stack, cycle, scatter_gather, gather_scatter patterns
"""
import pytest
import networkx as nx


ALERT_TYPES = ["fan_in", "fan_out", "bipartite", "stack", "cycle", "scatter_gather", "gather_scatter"]


def perform_alert_test(alert_models,
                       expected_no_of_models,
                       expected_schedule_id,
                       min_accts,
                       max_accts,
                       min_period,
                       max_period,
                       source_type):
    """Helper function to validate alert pattern properties"""

    for i, alert_type in enumerate(ALERT_TYPES):
        assert len(alert_models[alert_type]) == expected_no_of_models

        for nm in alert_models[alert_type]:
            assert nm.graph.get("scheduleID") == expected_schedule_id
            assert len(nm.nodes()) >= min_accts
            assert len(nm.nodes()) <= max_accts
            assert nm.graph.get("source_type") == source_type[i]

            # Ensure all transactions are within the expected range
            for node, neighbors in nm.adjacency():
                for neighbor, edge_data in neighbors.items():
                    assert edge_data["date"] >= min_period
                    assert edge_data["date"] <= max_period

            if alert_type in ["cycle"]:
                # Ensure there is a cycle in the graph
                assert len(list(nx.simple_cycles(nm))) > 0
            else:
                # Ensure there is no cycle in the graph
                assert len(list(nx.simple_cycles(nm))) == 0

            # Ensure incoming transactions are done before outgoing in each layer (for stack)
            if alert_type in ["stack"]:
                for node in nm.nodes():
                    largest_in_date = -1
                    smallest_out_date = 1e9

                    for pred in nm.predecessors(node):
                        pred_value = nm.get_edge_data(pred, node).get("date")
                        largest_in_date = max(largest_in_date, pred_value)

                    for succ in nm.successors(node):
                        succ_value = nm.get_edge_data(node, succ).get("date")
                        smallest_out_date = min(smallest_out_date, succ_value)

                    if largest_in_date != -1 and smallest_out_date != 1e9:
                        assert largest_in_date < smallest_out_date


@pytest.mark.integration
@pytest.mark.spatial
class TestAlertPatterns:
    """Tests for alert pattern generation"""

    def test_alert_small_graph(self, small_graph):
        """Test alert patterns in small graph"""
        txg = small_graph
        alert_models = dict()
        for alert_type in ALERT_TYPES:
            alert_models[alert_type] = [nm for nm in txg.alert_groups.values()
                                        if nm.graph["reason"] == alert_type]

        expected_no_of_models = 1
        expected_schedule_id = 2
        min_accts = 3
        max_accts = 4
        min_period = 1
        max_period = 20
        source_type = ["TRANSFER", "TRANSFER", "CASH", "CASH", "TRANSFER", "TRANSFER", "TRANSFER"]

        perform_alert_test(alert_models,
                          expected_no_of_models,
                          expected_schedule_id,
                          min_accts,
                          max_accts,
                          min_period,
                          max_period,
                          source_type)

    @pytest.mark.slow
    def test_alert_large_graph(self, large_graph):
        """Test alert patterns in large graph"""
        txg = large_graph
        alert_models = dict()
        for alert_type in ALERT_TYPES:
            alert_models[alert_type] = [nm for nm in txg.alert_groups.values()
                                        if nm.graph["reason"] == alert_type]

        expected_no_of_models = 10
        expected_schedule_id = 1
        min_accts = 10
        max_accts = 20
        min_period = 1
        max_period = 100
        source_type = ["TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER"]

        perform_alert_test(alert_models,
                          expected_no_of_models,
                          expected_schedule_id,
                          min_accts,
                          max_accts,
                          min_period,
                          max_period,
                          source_type)
