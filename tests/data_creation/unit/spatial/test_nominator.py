"""
Unit tests for Nominator class
Tests candidate selection and node nomination logic
"""
import pytest
import networkx as nx
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'python'))

from src.data_creation.spatial_simulation.utils.nominator import Nominator


@pytest.fixture
def simple_graph():
    """Create a simple directed graph for testing"""
    g = nx.DiGraph()

    # Add nodes with normal_models attribute
    for i in range(10):
        g.add_node(i, normal_models=[])

    # Create various patterns:
    # Node 0: fan-out (0 -> 1, 2, 3)
    g.add_edges_from([(0, 1), (0, 2), (0, 3)])

    # Node 4: fan-in (5, 6, 7 -> 4)
    g.add_edges_from([(5, 4), (6, 4), (7, 4)])

    # Node 8: forward (7 -> 8 -> 9)
    g.add_edges_from([(7, 8), (8, 9)])

    # Node 1: single (1 -> 2)
    # Already added above

    return g


@pytest.fixture
def complex_graph():
    """Create a more complex graph for advanced testing"""
    g = nx.DiGraph()

    # Add nodes with normal_models attribute
    for i in range(20):
        g.add_node(i, normal_models=[])

    # Create hub node (0) with many outgoing edges
    for i in range(1, 10):
        g.add_edge(0, i)

    # Create sink node (19) with many incoming edges
    for i in range(10, 18):
        g.add_edge(i, 19)

    # Add some forward chains
    g.add_edges_from([(1, 10), (10, 15), (15, 19)])

    # Add bidirectional edges for mutual
    g.add_edges_from([(2, 3), (3, 2)])

    return g


@pytest.fixture
def nominator(simple_graph):
    """Create a Nominator instance with simple graph"""
    return Nominator(simple_graph)


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorInitialization:
    """Tests for Nominator initialization"""

    def test_initialization(self, simple_graph):
        """Test that Nominator initializes correctly"""
        nom = Nominator(simple_graph)

        assert nom.g == simple_graph
        assert isinstance(nom.remaining_count_dict, dict)
        assert isinstance(nom.used_count_dict, dict)
        assert isinstance(nom.model_params_dict, dict)
        assert isinstance(nom.type_candidates, dict)
        assert isinstance(nom.current_candidate_index, dict)
        assert isinstance(nom.current_type_index, dict)

    def test_initialize_count_new_type(self, nominator):
        """Test initializing count for a new type"""
        nominator.initialize_count(
            type='fan_out',
            count=3,
            schedule_id=1,
            min_accounts=3,
            max_accounts=5,
            min_period=1,
            max_period=100,
            bank_id='bank_1'
        )

        assert nominator.remaining_count_dict['fan_out'] == 3
        assert nominator.used_count_dict['fan_out'] == 0
        assert len(nominator.model_params_dict['fan_out']) == 3
        assert nominator.model_params_dict['fan_out'][0] == (1, 3, 5, 1, 100, 'bank_1')

    def test_initialize_count_existing_type(self, nominator):
        """Test initializing count for an existing type (accumulates)"""
        nominator.initialize_count('fan_out', 2, 1, 3, 5, 1, 100, 'bank_1')
        nominator.initialize_count('fan_out', 3, 2, 4, 6, 10, 200, 'bank_2')

        assert nominator.remaining_count_dict['fan_out'] == 5
        assert len(nominator.model_params_dict['fan_out']) == 5
        assert nominator.model_params_dict['fan_out'][2] == (2, 4, 6, 10, 200, 'bank_2')


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorCandidateSelection:
    """Tests for candidate selection methods"""

    def test_get_single_candidates(self, nominator):
        """Test getting candidates for single pattern"""
        candidates = nominator.get_single_candidates()

        # Should include all nodes with at least one outgoing edge
        # Nodes 0, 5, 6, 7, 8 have outgoing edges (not 1, it only has incoming)
        assert len(candidates) >= 5

        # All candidates should have out_degree >= 1
        for node in candidates:
            assert nominator.g.out_degree(node) >= 1

    def test_get_forward_candidates(self, nominator):
        """Test getting candidates for forward pattern"""
        candidates = nominator.get_forward_candidates()

        # Forward requires at least one incoming and one outgoing edge
        # Node 8 has incoming from 7 and outgoing to 9
        assert 8 in candidates

        # All candidates should have both in and out degree >= 1
        for node in candidates:
            assert nominator.g.in_degree(node) >= 1
            assert nominator.g.out_degree(node) >= 1

    def test_get_fan_out_candidates(self, nominator):
        """Test getting candidates for fan_out pattern"""
        # Initialize with min_accounts=3
        nominator.initialize_count('fan_out', 1, 1, 3, 5, 1, 100, 'bank_1')

        candidates = nominator.get_fan_out_candidates()

        # Should have threshold set
        assert nominator.min_fan_out_threshold == 2  # min_accounts - 1

        # Node 0 has 3 outgoing edges, should be a candidate
        assert 0 in candidates

        # All candidates should have out_degree >= threshold
        for node in candidates:
            assert nominator.g.out_degree(node) >= nominator.min_fan_out_threshold

    def test_get_fan_in_candidates(self, nominator):
        """Test getting candidates for fan_in pattern"""
        # Initialize with min_accounts=3
        nominator.initialize_count('fan_in', 1, 1, 3, 5, 1, 100, 'bank_1')

        candidates = nominator.get_fan_in_candidates()

        # Should have threshold set
        assert nominator.min_fan_in_threshold == 2  # min_accounts - 1

        # Node 4 has 3 incoming edges, should be a candidate
        assert 4 in candidates

        # All candidates should have in_degree >= threshold
        for node in candidates:
            assert nominator.g.in_degree(node) >= nominator.min_fan_in_threshold

    def test_initialize_candidates(self, nominator):
        """Test initializing all candidates"""
        nominator.initialize_count('fan_out', 1, 1, 3, 5, 1, 100, 'bank_1')
        nominator.initialize_count('fan_in', 1, 1, 3, 5, 1, 100, 'bank_1')
        nominator.initialize_count('forward', 1, 1, 3, 5, 1, 100, 'bank_1')
        nominator.initialize_count('single', 1, 1, 2, 3, 1, 100, 'bank_1')

        nominator.initialize_candidates()

        assert 'fan_out' in nominator.type_candidates
        assert 'fan_in' in nominator.type_candidates
        assert 'forward' in nominator.type_candidates
        assert 'single' in nominator.type_candidates

        # Check indices are initialized
        for type_name in ['fan_out', 'fan_in', 'forward', 'single']:
            assert nominator.current_candidate_index[type_name] == 0
            assert nominator.current_type_index[type_name] == 0

    def test_initialize_candidates_mutual_uses_single(self, nominator):
        """Test that mutual type uses single candidates"""
        nominator.initialize_count('mutual', 1, 1, 2, 3, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        # Mutual should use same candidates as single
        assert 'mutual' in nominator.type_candidates
        assert len(nominator.type_candidates['mutual']) > 0

    def test_initialize_candidates_periodical_uses_single(self, nominator):
        """Test that periodical type uses single candidates"""
        nominator.initialize_count('periodical', 1, 1, 2, 3, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        # Periodical should use same candidates as single
        assert 'periodical' in nominator.type_candidates
        assert len(nominator.type_candidates['periodical']) > 0

    def test_initialize_candidates_invalid_type(self, nominator):
        """Test that invalid type raises error"""
        nominator.initialize_count('invalid_type', 1, 1, 2, 3, 1, 100, 'bank_1')

        with pytest.raises(ValueError, match='Invalid type'):
            nominator.initialize_candidates()


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorCountManagement:
    """Tests for count management methods"""

    def test_number_unused(self, nominator):
        """Test counting total unused nodes"""
        nominator.initialize_count('fan_out', 3, 1, 3, 5, 1, 100, 'bank_1')
        nominator.initialize_count('fan_in', 2, 1, 3, 5, 1, 100, 'bank_1')

        assert nominator.number_unused() == 5

    def test_has_more_true(self, nominator):
        """Test has_more returns True when there are unused nodes"""
        nominator.initialize_count('fan_out', 1, 1, 3, 5, 1, 100, 'bank_1')

        assert nominator.has_more() is True

    def test_has_more_false(self, nominator):
        """Test has_more returns False when no unused nodes"""
        nominator.initialize_count('fan_out', 1, 1, 3, 5, 1, 100, 'bank_1')
        nominator.remaining_count_dict['fan_out'] = 0

        assert nominator.has_more() is False

    def test_count(self, nominator):
        """Test counting nodes of a specific type"""
        nominator.initialize_count('fan_out', 5, 1, 3, 5, 1, 100, 'bank_1')

        assert nominator.count('fan_out') == 5

    def test_decrement(self, nominator):
        """Test decrementing count"""
        nominator.initialize_count('fan_out', 3, 1, 3, 5, 1, 100, 'bank_1')
        nominator.decrement('fan_out')

        assert nominator.remaining_count_dict['fan_out'] == 2

    def test_increment_used(self, nominator):
        """Test incrementing used count"""
        nominator.initialize_count('fan_out', 3, 1, 3, 5, 1, 100, 'bank_1')
        nominator.increment_used('fan_out')

        assert nominator.used_count_dict['fan_out'] == 1

    def test_conclude(self, nominator):
        """Test concluding a type (setting remaining to 0)"""
        nominator.initialize_count('fan_out', 5, 1, 3, 5, 1, 100, 'bank_1')
        nominator.conclude('fan_out')

        assert nominator.remaining_count_dict['fan_out'] == 0

    def test_types(self, nominator):
        """Test getting all types"""
        nominator.initialize_count('fan_out', 1, 1, 3, 5, 1, 100, 'bank_1')
        nominator.initialize_count('fan_in', 1, 1, 3, 5, 1, 100, 'bank_1')

        types = list(nominator.types())
        assert 'fan_out' in types
        assert 'fan_in' in types


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorNextMethod:
    """Tests for the next() method"""

    def test_next_single(self, nominator):
        """Test getting next node for single pattern"""
        nominator.initialize_count('single', 1, 1, 2, 3, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        node_id = nominator.next('single')

        assert node_id is not None
        assert nominator.remaining_count_dict['single'] == 0
        assert nominator.used_count_dict['single'] == 1

    def test_next_forward(self, nominator):
        """Test getting next node for forward pattern"""
        nominator.initialize_count('forward', 1, 1, 3, 5, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        node_id = nominator.next('forward')

        # Should return node 8 (has both in and out edges)
        assert node_id is not None
        assert nominator.remaining_count_dict['forward'] == 0

    def test_next_exhausted_type(self, nominator):
        """Test that remaining count reaches 0 after using all nodes"""
        nominator.initialize_count('single', 2, 1, 2, 3, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        # Get first node
        first_node = nominator.next('single')
        assert first_node is not None
        assert nominator.remaining_count_dict['single'] == 1

        # Get second node
        second_node = nominator.next('single')
        assert second_node is not None
        assert nominator.remaining_count_dict['single'] == 0

        # Now type is exhausted
        assert not nominator.has_more()


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorRelationshipChecking:
    """Tests for relationship checking methods"""

    def test_is_in_type_relationship_not_in_relation(self, nominator):
        """Test checking if nodes are in type relationship (not in relation)"""
        # Node 0 has no normal models yet
        result = nominator.is_in_type_relationship('fan_out', 0, {0, 1})

        assert result is False

    def test_is_in_type_relationship_in_relation(self, nominator):
        """Test checking if nodes are in type relationship (in relation)"""
        # Create a mock normal model
        mock_nm = MagicMock()
        mock_nm.type = 'fan_out'
        mock_nm.main_id = 0
        mock_nm.node_ids = {0, 1, 2}

        # Add to node's normal models
        nominator.g.nodes[0]['normal_models'].append(mock_nm)

        # Check if {0, 1} is a subset (should be True)
        result = nominator.is_in_type_relationship('fan_out', 0, {0, 1})

        assert result is True

    def test_nodes_in_type_relation_empty(self, nominator):
        """Test getting nodes in type relation (empty)"""
        nodes = nominator.nodes_in_type_relation('fan_out', 0)

        assert isinstance(nodes, set)
        assert len(nodes) == 0

    def test_nodes_in_type_relation_with_models(self, nominator):
        """Test getting nodes in type relation (with models)"""
        # Create mock normal models
        mock_nm1 = MagicMock()
        mock_nm1.type = 'fan_out'
        mock_nm1.main_id = 0
        mock_nm1.node_ids_without_main = MagicMock(return_value={1, 2})

        mock_nm2 = MagicMock()
        mock_nm2.type = 'fan_out'
        mock_nm2.main_id = 0
        mock_nm2.node_ids_without_main = MagicMock(return_value={2, 3})

        nominator.g.nodes[0]['normal_models'].extend([mock_nm1, mock_nm2])

        nodes = nominator.nodes_in_type_relation('fan_out', 0)

        assert isinstance(nodes, set)
        assert nodes == {1, 2, 3}


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorFindAvailableCandidateNeighbors:
    """Tests for finding available candidate neighbors (fan patterns)"""

    def test_find_available_candidate_neighbors_fan_in(self, nominator):
        """Test finding available neighbors for fan_in"""
        nominator.initialize_count('fan_in', 1, 1, 2, 4, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        # Node 4 has edges from 5, 6, 7 (3 predecessors)
        candidates = nominator.find_available_candidate_neighbors('fan_in', 4)

        assert isinstance(candidates, set)
        # min_threshold = min_accounts - 1 = 2 - 1 = 1
        assert len(candidates) >= 1  # min_threshold
        assert len(candidates) <= 3  # max available
        assert candidates.issubset({5, 6, 7})

    def test_find_available_candidate_neighbors_insufficient(self, nominator):
        """Test error when insufficient candidates"""
        nominator.initialize_count('fan_out', 1, 1, 5, 10, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        # Node 1 only has 1 successor (not enough for min_threshold=4)
        with pytest.raises(ValueError, match='not enough candidate'):
            nominator.find_available_candidate_neighbors('fan_out', 1)


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorPostUpdate:
    """Tests for post_update method"""

    def test_post_update_increments_type_index(self, nominator):
        """Test that post_update increments type index"""
        nominator.initialize_count('fan_out', 2, 1, 2, 4, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        initial_index = nominator.current_type_index['fan_out']

        # Mock is_done to return False (node not exhausted)
        nominator.is_done = MagicMock(return_value=False)

        nominator.post_update(0, 'fan_out')

        assert nominator.current_type_index['fan_out'] == initial_index + 1

    def test_post_update_removes_done_node(self, nominator):
        """Test that post_update removes exhausted nodes"""
        nominator.initialize_count('fan_out', 2, 1, 2, 4, 1, 100, 'bank_1')
        nominator.initialize_candidates()

        initial_len = len(nominator.type_candidates['fan_out'])
        current_node = nominator.type_candidates['fan_out'][0]

        # Mock is_done to return True (node exhausted)
        nominator.is_done = MagicMock(return_value=True)

        nominator.post_update(current_node, 'fan_out')

        # Node should be removed
        assert len(nominator.type_candidates['fan_out']) == initial_len - 1
        assert current_node not in nominator.type_candidates['fan_out']

@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorIsDoneMethods:
    """Tests for is_done checking methods"""

    def test_is_done_single_no_successors(self, simple_graph):
        """Test is_done_single when node has no successors"""
        # Node 9 has no successors
        nom = Nominator(simple_graph)

        result = nom.is_done_single(9, 'single')

        # Should be done (no successors to make singles with)
        assert result is True

    def test_is_done_single_with_available_successors(self, simple_graph):
        """Test is_done_single when node has available successors"""
        # Node 0 has successors 1, 2, 3
        nom = Nominator(simple_graph)

        result = nom.is_done_single(0, 'single')

        # Should not be done (has successors not in relationship)
        assert result is False

    def test_is_done_fan_out_sufficient_neighbors(self, complex_graph):
        """Test is_done_fan_out with sufficient neighbors"""
        nom = Nominator(complex_graph)
        nom.min_fan_out_threshold = 3

        # Node 0 has many successors
        result = nom.is_done_fan_out(0, 'fan_out')

        # Should not be done
        assert result is False

    def test_is_done_fan_in_sufficient_neighbors(self, complex_graph):
        """Test is_done_fan_in with sufficient neighbors"""
        nom = Nominator(complex_graph)
        nom.min_fan_in_threshold = 3

        # Node 19 has many predecessors
        result = nom.is_done_fan_in(19, 'fan_in')

        # Should not be done
        assert result is False


@pytest.mark.unit
@pytest.mark.spatial
class TestNominatorComplexGraph:
    """Tests using complex graph scenarios"""

    def test_complex_graph_fan_out_hub(self, complex_graph):
        """Test fan_out on hub node"""
        nom = Nominator(complex_graph)
        nom.initialize_count('fan_out', 2, 1, 3, 5, 1, 100, 'bank_1')
        nom.initialize_candidates()

        # Node 0 is a hub with 9 outgoing edges
        candidates = nom.get_fan_out_candidates()
        assert 0 in candidates

    def test_complex_graph_fan_in_sink(self, complex_graph):
        """Test fan_in on sink node"""
        nom = Nominator(complex_graph)
        nom.initialize_count('fan_in', 2, 1, 3, 5, 1, 100, 'bank_1')
        nom.initialize_candidates()

        # Node 19 is a sink with many incoming edges
        candidates = nom.get_fan_in_candidates()
        assert 19 in candidates

    def test_multiple_type_initialization(self, complex_graph):
        """Test initializing multiple types simultaneously"""
        nom = Nominator(complex_graph)

        nom.initialize_count('fan_out', 1, 1, 3, 5, 1, 100, 'bank_1')
        nom.initialize_count('fan_in', 1, 1, 3, 5, 1, 100, 'bank_1')
        nom.initialize_count('forward', 2, 1, 3, 5, 1, 100, 'bank_1')
        nom.initialize_count('single', 5, 1, 2, 3, 1, 100, 'bank_1')

        nom.initialize_candidates()

        assert nom.number_unused() == 9
        assert len(nom.types()) == 4
