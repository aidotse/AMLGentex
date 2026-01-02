"""
Unit tests for normal transaction model generation in spatial simulation
Tests single, forward, mutual, periodical, fan_in, fan_out models
"""
import pytest


NORMAL_MODEL_TYPES = ["single", "forward", "mutual", "periodical", "fan_in", "fan_out"]


@pytest.mark.integration
@pytest.mark.spatial
class TestNormalModels:
    """Tests for normal transaction model generation"""

    def test_small_graph(self, small_clean_graph):
        """Test normal models in small graph"""
        # Ensure correct number of models are created
        # For forward, we try to create 1000 models but the graph only supports 60
        # Each n1 -> n2 -> x has 3 different combinations
        # Each n1 -> x -> n3 has 4 different combinations
        # Hence, each n1 -> x -> y has 12 combinations
        # Each 5 nodes have 12 possible forward patterns - 60 in total
        txg = small_clean_graph
        expected_no_of_models = [2, 60, 2, 2, 2, 2]
        normal_models = dict()

        for model_type, expected_num in zip(NORMAL_MODEL_TYPES, expected_no_of_models):
            normal_models[model_type] = [nm for nm in txg.normal_models if nm.type == model_type]
            assert len(normal_models[model_type]) == expected_num

        # Ensure fan patterns have correct number of nodes
        expected_no_of_nodes = [3, 5]
        for nm, expected_num in zip(normal_models["fan_in"], expected_no_of_nodes):
            assert len(nm.node_ids) == expected_num
        for nm, expected_num in zip(normal_models["fan_out"], expected_no_of_nodes):
            assert len(nm.node_ids) == expected_num

    @pytest.mark.slow
    def test_large_graph(self, large_clean_graph):
        """Test normal models in large graph"""
        txg = large_clean_graph

        # Pick out normal models
        normal_models = dict()
        for model_type in NORMAL_MODEL_TYPES:
            normal_models[model_type] = [nm for nm in txg.normal_models if nm.type == model_type]

        # Check Fan patterns
        num_defined_models = 10000
        max_fan_threshold = {"fan_in": 10 - 1, "fan_out": 10 - 1}
        min_fan_threshold = {"fan_in": 6 - 1, "fan_out": 7 - 1}

        for model_type in ["fan_in", "fan_out"]:
            # Find number of nodes with more than min_in_deg and min_out_deg in graph
            if model_type == "fan_in":
                num_candidates = len([n for n in txg.g.nodes()
                                     if txg.g.in_degree(n) >= min_fan_threshold[model_type]])
            else:
                num_candidates = len([n for n in txg.g.nodes()
                                     if txg.g.out_degree(n) >= min_fan_threshold[model_type]])

            # Check how many nodes are main in multiple patterns
            main_ids = [nm.main_id for nm in normal_models[model_type]]
            counter = {i: main_ids.count(i) - 1 for i in main_ids}
            num_of_replicas = sum([v for _, v in counter.items()])

            # Make sure that all candidates are used
            assert num_candidates < num_defined_models
            assert len(normal_models[model_type]) == (num_candidates + num_of_replicas)

            max_nodes_fan = max([len(nm.node_ids) for nm in normal_models[model_type]])
            min_nodes_fan = min([len(nm.node_ids) for nm in normal_models[model_type]])

            assert max_nodes_fan == (max_fan_threshold[model_type] + 1)
            assert min_nodes_fan == (min_fan_threshold[model_type] + 1)

        # Check forward patterns
        total_forward_patterns = 10000
        num_nodes_in_forward = 3
        assert len(normal_models["forward"]) == total_forward_patterns
        for nm in normal_models["forward"]:
            assert len(nm.node_ids) == num_nodes_in_forward

        # Check single, mutual, and periodical patterns
        total_patterns = 10000
        num_nodes_in_pattern = 2
        for model_type in ["single", "mutual", "periodical"]:
            assert len(normal_models[model_type]) == total_patterns
            for nm in normal_models[model_type]:
                assert len(nm.node_ids) == num_nodes_in_pattern
