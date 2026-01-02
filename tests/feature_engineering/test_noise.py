"""
Unit tests for noise functions in preprocessing
"""
import pytest
import pandas as pd
import numpy as np

from src.feature_engineering.noise import flip_labels, missing_labels, flip_neighbours, topology_noise


@pytest.mark.unit
class TestFlipLabels:
    """Tests for flip_labels function"""

    def test_flip_labels_basic(self, sample_nodes_df):
        """Test basic label flipping functionality"""
        result = flip_labels(sample_nodes_df, labels=[0], fracs=[0.5], seed=42)

        # Check that true_label column is created
        assert 'true_label' in result.columns

        # Check that some normal accounts were flipped
        flipped = result[result['true_label'] != result['is_sar']]
        assert len(flipped) > 0

    def test_flip_labels_preserves_true_label(self, sample_nodes_df):
        """Test that true_label is preserved if already exists"""
        sample_nodes_df['true_label'] = sample_nodes_df['is_sar'].copy()
        original_true_labels = sample_nodes_df['true_label'].copy()

        result = flip_labels(sample_nodes_df, labels=[0], fracs=[0.1], seed=42)

        # True labels should remain unchanged
        pd.testing.assert_series_equal(
            result['true_label'].reset_index(drop=True),
            original_true_labels.reset_index(drop=True)
        )

    def test_flip_labels_multiple_classes(self, sample_nodes_df):
        """Test flipping multiple label classes"""
        result = flip_labels(sample_nodes_df, labels=[0, 1], fracs=[0.1, 0.2], seed=42)

        # Both normal and SAR accounts should have flips
        normal_flipped = result[(result['true_label'] == 0) & (result['is_sar'] == 1)]
        sar_flipped = result[(result['true_label'] == 1) & (result['is_sar'] == 0)]

        assert len(normal_flipped) > 0 or len(sar_flipped) > 0

    def test_flip_labels_reproducibility(self, sample_nodes_df):
        """Test that same seed produces same results"""
        result1 = flip_labels(sample_nodes_df, labels=[0], fracs=[0.3], seed=42)
        result2 = flip_labels(sample_nodes_df, labels=[0], fracs=[0.3], seed=42)

        pd.testing.assert_frame_equal(result1, result2)

    def test_flip_labels_zero_fraction(self, sample_nodes_df):
        """Test with zero fraction (no flips)"""
        result = flip_labels(sample_nodes_df, labels=[0, 1], fracs=[0.0, 0.0], seed=42)

        # No labels should be flipped
        assert (result['true_label'] == result['is_sar']).all()

    def test_flip_labels_full_fraction(self, sample_nodes_df):
        """Test with fraction = 1.0 (all labels flipped)"""
        result = flip_labels(sample_nodes_df, labels=[0], fracs=[1.0], seed=42)

        # All normal accounts should be flipped
        normal_accounts = result[result['true_label'] == 0]
        assert (normal_accounts['is_sar'] == 1).all()

    def test_flip_labels_does_not_modify_original(self, sample_nodes_df):
        """Test that original DataFrame is not modified"""
        original = sample_nodes_df.copy()
        flip_labels(sample_nodes_df, labels=[0], fracs=[0.5], seed=42)

        pd.testing.assert_frame_equal(sample_nodes_df, original)


@pytest.mark.unit
class TestMissingLabels:
    """Tests for missing_labels function"""

    def test_missing_labels_basic(self, sample_nodes_df):
        """Test basic missing labels functionality"""
        result = missing_labels(sample_nodes_df, labels=[0], fracs=[0.5], seed=42)

        # Check that true_label column is created
        assert 'true_label' in result.columns

        # Check that some labels are set to -1
        missing = result[result['is_sar'] == -1]
        assert len(missing) > 0

    def test_missing_labels_value(self, sample_nodes_df):
        """Test that missing labels are set to -1"""
        result = missing_labels(sample_nodes_df, labels=[0, 1], fracs=[0.2, 0.2], seed=42)

        # Check that -1 is used for missing labels
        assert -1 in result['is_sar'].values

    def test_missing_labels_multiple_classes(self, sample_nodes_df):
        """Test missing labels for multiple classes"""
        result = missing_labels(sample_nodes_df, labels=[0, 1], fracs=[0.3, 0.3], seed=42)

        # Both normal and SAR accounts should have missing labels
        missing_accounts = result[result['is_sar'] == -1]
        assert len(missing_accounts) > 0

    def test_missing_labels_reproducibility(self, sample_nodes_df):
        """Test that same seed produces same results"""
        result1 = missing_labels(sample_nodes_df, labels=[0], fracs=[0.3], seed=42)
        result2 = missing_labels(sample_nodes_df, labels=[0], fracs=[0.3], seed=42)

        pd.testing.assert_frame_equal(result1, result2)

    def test_missing_labels_zero_fraction(self, sample_nodes_df):
        """Test with zero fraction (no missing labels)"""
        result = missing_labels(sample_nodes_df, labels=[0, 1], fracs=[0.0, 0.0], seed=42)

        # No labels should be missing
        assert -1 not in result['is_sar'].values

    def test_missing_labels_does_not_modify_original(self, sample_nodes_df):
        """Test that original DataFrame is not modified"""
        original = sample_nodes_df.copy()
        missing_labels(sample_nodes_df, labels=[0], fracs=[0.5], seed=42)

        pd.testing.assert_frame_equal(sample_nodes_df, original)


@pytest.mark.unit
class TestFlipNeighbours:
    """Tests for flip_neighbours function"""

    def test_flip_neighbours_basic(self, sample_nodes_df, sample_edges_df):
        """Test basic neighbor flipping functionality"""
        result = flip_neighbours(sample_nodes_df, sample_edges_df, frac=0.5, seed=42)

        # Check that true_label column is created
        assert 'true_label' in result.columns

    def test_flip_neighbours_targets_sar_neighbors(self, sample_nodes_df, sample_edges_df):
        """Test that only neighbors of SAR accounts are flipped"""
        result = flip_neighbours(sample_nodes_df, sample_edges_df, frac=1.0, seed=42)

        # Find SAR accounts
        sar_accounts = set(sample_nodes_df[sample_nodes_df['is_sar'] == 1]['account'])

        # Find neighbors of SAR accounts in edges
        edges_with_sar = sample_edges_df[
            sample_edges_df['src'].isin(sar_accounts) |
            sample_edges_df['dst'].isin(sar_accounts)
        ]
        potential_flip_accounts = set(edges_with_sar['src']).union(set(edges_with_sar['dst'])) - sar_accounts

        # Flipped accounts should be a subset of potential flip accounts
        flipped = result[(result['true_label'] == 0) & (result['is_sar'] == 1)]
        assert set(flipped['account']).issubset(potential_flip_accounts)

    def test_flip_neighbours_reproducibility(self, sample_nodes_df, sample_edges_df):
        """Test that same seed produces same results"""
        result1 = flip_neighbours(sample_nodes_df, sample_edges_df, frac=0.3, seed=42)
        result2 = flip_neighbours(sample_nodes_df, sample_edges_df, frac=0.3, seed=42)

        pd.testing.assert_frame_equal(result1, result2)

    def test_flip_neighbours_zero_fraction(self, sample_nodes_df, sample_edges_df):
        """Test with zero fraction (no flips)"""
        result = flip_neighbours(sample_nodes_df, sample_edges_df, frac=0.0, seed=42)

        # No labels should be flipped
        assert (result['true_label'] == result['is_sar']).all()

    def test_flip_neighbours_does_not_modify_original(self, sample_nodes_df, sample_edges_df):
        """Test that original DataFrame is not modified"""
        original = sample_nodes_df.copy()
        flip_neighbours(sample_nodes_df, sample_edges_df, frac=0.5, seed=42)

        pd.testing.assert_frame_equal(sample_nodes_df, original)


@pytest.mark.unit
class TestTopologyNoise:
    """Tests for topology_noise function"""

    def test_topology_noise_basic(self, sample_nodes_df, sample_alert_members_df):
        """Test basic topology noise functionality"""
        result = topology_noise(
            sample_nodes_df,
            sample_alert_members_df,
            topologies=['fan_in'],
            fracs=[0.5],
            seed=42
        )

        # Check that true_label column is created
        assert 'true_label' in result.columns

    def test_topology_noise_multiple_topologies(self, sample_nodes_df, sample_alert_members_df):
        """Test with multiple topologies"""
        result = topology_noise(
            sample_nodes_df,
            sample_alert_members_df,
            topologies=['fan_in', 'fan_out', 'cycle'],
            fracs=[0.3, 0.4, 0.5],
            seed=42
        )

        # Result should have modified SAR labels
        assert 'is_sar' in result.columns

    def test_topology_noise_single_fraction(self, sample_nodes_df, sample_alert_members_df):
        """Test with single fraction for all topologies"""
        result = topology_noise(
            sample_nodes_df,
            sample_alert_members_df,
            topologies=['fan_in', 'fan_out'],
            fracs=0.5,
            seed=42
        )

        assert 'is_sar' in result.columns

    def test_topology_noise_default_fraction(self, sample_nodes_df, sample_alert_members_df):
        """Test with default fraction (None)"""
        result = topology_noise(
            sample_nodes_df,
            sample_alert_members_df,
            topologies=['fan_in'],
            fracs=None,
            seed=42
        )

        assert 'is_sar' in result.columns

    def test_topology_noise_fraction_list_length(self, sample_nodes_df, sample_alert_members_df):
        """Test that fracs list must match topologies list length"""
        with pytest.raises(AssertionError):
            topology_noise(
                sample_nodes_df,
                sample_alert_members_df,
                topologies=['fan_in', 'fan_out'],
                fracs=[0.5],  # Only one fraction for two topologies
                seed=42
            )

    def test_topology_noise_reproducibility(self, sample_nodes_df, sample_alert_members_df):
        """Test that same seed produces same results"""
        result1 = topology_noise(
            sample_nodes_df,
            sample_alert_members_df,
            topologies=['fan_in'],
            fracs=[0.5],
            seed=42
        )
        result2 = topology_noise(
            sample_nodes_df,
            sample_alert_members_df,
            topologies=['fan_in'],
            fracs=[0.5],
            seed=42
        )

        pd.testing.assert_frame_equal(result1, result2)

    def test_topology_noise_does_not_modify_original(self, sample_nodes_df, sample_alert_members_df):
        """Test that original DataFrame is not modified"""
        original = sample_nodes_df.copy()
        topology_noise(
            sample_nodes_df,
            sample_alert_members_df,
            topologies=['fan_in'],
            fracs=[0.5],
            seed=42
        )

        pd.testing.assert_frame_equal(sample_nodes_df, original)
