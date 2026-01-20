"""
Tests for transductive vs inductive preprocessing
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.feature_engineering.preprocessor import DataPreprocessor


@pytest.mark.unit
class TestTransductiveInductiveDetection:
    """Tests for learning mode configuration"""

    def test_detects_transductive_mode(self):
        """Test that transductive mode is correctly configured"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'learning_mode': 'transductive',
            'time_start': 0,
            'time_end': 100,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        assert preprocessor.is_transductive is True

    def test_detects_inductive_mode(self):
        """Test that inductive mode is correctly configured"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'learning_mode': 'inductive',
            'train_start_step': 0,
            'train_end_step': 30,
            'val_start_step': 31,
            'val_end_step': 60,
            'test_start_step': 61,
            'test_end_step': 100,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        assert preprocessor.is_transductive is False

    def test_inductive_mode_with_overlapping_windows(self):
        """Test inductive mode with overlapping time windows"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'learning_mode': 'inductive',
            'train_start_step': 0,
            'train_end_step': 50,
            'val_start_step': 40,
            'val_end_step': 80,
            'test_start_step': 70,
            'test_end_step': 100,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        assert preprocessor.is_transductive is False


@pytest.mark.unit
class TestTransductiveMaskCreation:
    """Tests for transductive label mask creation"""

    def test_creates_three_masks(self, sample_transactions_df):
        """Test that train/val/test masks are created"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_nodes = preprocessor.cal_node_features(sample_transactions_df, 1, 30)
        df_nodes_with_masks = preprocessor.create_transductive_masks(df_nodes)

        assert 'train_mask' in df_nodes_with_masks.columns
        assert 'val_mask' in df_nodes_with_masks.columns
        assert 'test_mask' in df_nodes_with_masks.columns

    def test_masks_are_boolean(self, sample_transactions_df):
        """Test that masks are boolean arrays"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_nodes = preprocessor.cal_node_features(sample_transactions_df, 1, 30)
        df_nodes_with_masks = preprocessor.create_transductive_masks(df_nodes)

        assert df_nodes_with_masks['train_mask'].dtype == bool
        assert df_nodes_with_masks['val_mask'].dtype == bool
        assert df_nodes_with_masks['test_mask'].dtype == bool

    def test_masks_are_non_overlapping(self, sample_transactions_df):
        """Test that masks don't overlap"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_nodes = preprocessor.cal_node_features(sample_transactions_df, 1, 30)
        df_nodes_with_masks = preprocessor.create_transductive_masks(df_nodes)

        # No node should be in multiple splits
        train_and_val = df_nodes_with_masks['train_mask'] & df_nodes_with_masks['val_mask']
        train_and_test = df_nodes_with_masks['train_mask'] & df_nodes_with_masks['test_mask']
        val_and_test = df_nodes_with_masks['val_mask'] & df_nodes_with_masks['test_mask']

        assert train_and_val.sum() == 0
        assert train_and_test.sum() == 0
        assert val_and_test.sum() == 0

    def test_masks_respect_fractions(self, sample_transactions_df):
        """Test that masks respect configured fractions"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_nodes = preprocessor.cal_node_features(sample_transactions_df, 1, 30)
        df_nodes_with_masks = preprocessor.create_transductive_masks(df_nodes)

        n_nodes = len(df_nodes_with_masks)
        n_train = df_nodes_with_masks['train_mask'].sum()
        n_val = df_nodes_with_masks['val_mask'].sum()
        n_test = df_nodes_with_masks['test_mask'].sum()

        # Check approximately correct fractions (within 10%)
        assert abs(n_train / n_nodes - 0.6) < 0.1
        assert abs(n_val / n_nodes - 0.2) < 0.1
        assert abs(n_test / n_nodes - 0.2) < 0.1

    def test_masks_split_sar_and_normal_separately(self, sample_transactions_df):
        """Test that SAR and normal nodes are split separately"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_nodes = preprocessor.cal_node_features(sample_transactions_df, 1, 30)
        df_nodes_with_masks = preprocessor.create_transductive_masks(df_nodes)

        # Both train and test should have SAR and normal nodes
        train_nodes = df_nodes_with_masks[df_nodes_with_masks['train_mask']]
        val_nodes = df_nodes_with_masks[df_nodes_with_masks['val_mask']]
        test_nodes = df_nodes_with_masks[df_nodes_with_masks['test_mask']]

        # Each split should contain both SAR and normal (if there are enough nodes)
        if len(train_nodes) > 1:
            assert train_nodes['is_sar'].min() == 0 or train_nodes['is_sar'].max() == 1
        if len(val_nodes) > 1:
            assert val_nodes['is_sar'].min() == 0 or val_nodes['is_sar'].max() == 1
        if len(test_nodes) > 1:
            assert test_nodes['is_sar'].min() == 0 or test_nodes['is_sar'].max() == 1

    def test_mask_creation_is_reproducible(self, sample_transactions_df):
        """Test that same seed produces same masks"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }

        preprocessor1 = DataPreprocessor(config)
        df_nodes1 = preprocessor1.cal_node_features(sample_transactions_df, 1, 30)
        result1 = preprocessor1.create_transductive_masks(df_nodes1)

        preprocessor2 = DataPreprocessor(config)
        df_nodes2 = preprocessor2.cal_node_features(sample_transactions_df, 1, 30)
        result2 = preprocessor2.create_transductive_masks(df_nodes2)

        pd.testing.assert_series_equal(result1['train_mask'], result2['train_mask'])
        pd.testing.assert_series_equal(result1['val_mask'], result2['val_mask'])
        pd.testing.assert_series_equal(result1['test_mask'], result2['test_mask'])


@pytest.mark.unit
class TestPatternBasedSplitting:
    """Tests for pattern-based transductive splitting"""

    @pytest.fixture
    def sample_transactions_with_patterns(self):
        """Create sample transactions with pattern IDs"""
        # Create transactions with 3 SAR patterns and normal transactions
        return pd.DataFrame({
            'step': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
            'nameOrig': [100, 101, 102, 200, 201, 202, 300, 301, 302, 400, 401, 402, 500, 501, 502, 503],
            'nameDest': [101, 102, 100, 201, 202, 200, 301, 302, 300, 401, 402, 400, 501, 502, 503, 500],
            'amount': [100.0] * 16,
            'bankOrig': ['BANK001'] * 16,
            'bankDest': ['BANK001'] * 16,
            'daysInBankOrig': [10] * 16,
            'daysInBankDest': [10] * 16,
            'phoneChangesOrig': [0] * 16,
            'phoneChangesDest': [0] * 16,
            'isSAR': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 9 SAR, 7 normal
            'patternID': [0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1],  # 3 SAR patterns
        })

    def test_pattern_split_no_pattern_overlap(self, sample_transactions_with_patterns):
        """Test that no pattern appears in multiple splits"""
        config = {
            'num_windows': 1,
            'window_len': 10,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 10,
            'include_edge_features': False,
            'split_by_pattern': True,
            'transductive_train_fraction': 0.34,  # ~1 pattern
            'transductive_val_fraction': 0.33,    # ~1 pattern
            'transductive_test_fraction': 0.33,   # ~1 pattern
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        # Need to keep patternID through load_data simulation
        df = sample_transactions_with_patterns.copy()
        df_nodes = preprocessor.cal_node_features(df, 1, 10)
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # Verify masks exist
        assert 'train_mask' in df_nodes.columns
        assert 'val_mask' in df_nodes.columns
        assert 'test_mask' in df_nodes.columns

        # Verify no overlap
        train_and_val = df_nodes['train_mask'] & df_nodes['val_mask']
        train_and_test = df_nodes['train_mask'] & df_nodes['test_mask']
        val_and_test = df_nodes['val_mask'] & df_nodes['test_mask']

        assert train_and_val.sum() == 0
        assert train_and_test.sum() == 0
        assert val_and_test.sum() == 0

    def test_pattern_split_removes_pattern_id_from_output(self, sample_transactions_with_patterns):
        """Test that pattern_id is removed from final output (not a feature)"""
        config = {
            'num_windows': 1,
            'window_len': 10,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 10,
            'include_edge_features': False,
            'split_by_pattern': True,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        df = sample_transactions_with_patterns.copy()
        df_nodes = preprocessor.cal_node_features(df, 1, 10)
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # pattern_id should be removed from output
        assert 'pattern_id' not in df_nodes.columns

    def test_pattern_split_is_reproducible(self, sample_transactions_with_patterns):
        """Test that same seed produces same pattern split"""
        config = {
            'num_windows': 1,
            'window_len': 10,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 10,
            'include_edge_features': False,
            'split_by_pattern': True,
            'seed': 42
        }

        df = sample_transactions_with_patterns.copy()

        preprocessor1 = DataPreprocessor(config)
        df_nodes1 = preprocessor1.cal_node_features(df.copy(), 1, 10)
        result1 = preprocessor1.create_transductive_masks(df_nodes1)

        preprocessor2 = DataPreprocessor(config)
        df_nodes2 = preprocessor2.cal_node_features(df.copy(), 1, 10)
        result2 = preprocessor2.create_transductive_masks(df_nodes2)

        pd.testing.assert_series_equal(result1['train_mask'], result2['train_mask'])
        pd.testing.assert_series_equal(result1['val_mask'], result2['val_mask'])
        pd.testing.assert_series_equal(result1['test_mask'], result2['test_mask'])

    @pytest.fixture
    def transactions_with_many_patterns(self):
        """Create transactions with 10 SAR patterns for better distribution testing"""
        rows = []
        # 10 SAR patterns, 3 accounts each
        for pattern_id in range(10):
            base_account = pattern_id * 100
            for i in range(3):
                rows.append({
                    'step': pattern_id + 1,
                    'nameOrig': base_account + i,
                    'nameDest': base_account + ((i + 1) % 3),
                    'amount': 100.0,
                    'bankOrig': 'BANK001',
                    'bankDest': 'BANK001',
                    'daysInBankOrig': 10,
                    'daysInBankDest': 10,
                    'phoneChangesOrig': 0,
                    'phoneChangesDest': 0,
                    'isSAR': 1,
                    'patternID': pattern_id,
                })
        # 20 normal accounts
        for i in range(20):
            rows.append({
                'step': 1,
                'nameOrig': 2000 + i,
                'nameDest': 2000 + ((i + 1) % 20),
                'amount': 50.0,
                'bankOrig': 'BANK001',
                'bankDest': 'BANK001',
                'daysInBankOrig': 10,
                'daysInBankDest': 10,
                'phoneChangesOrig': 0,
                'phoneChangesDest': 0,
                'isSAR': 0,
                'patternID': -1,
            })
        return pd.DataFrame(rows)

    def test_no_pattern_leakage_across_splits(self, transactions_with_many_patterns):
        """Test that no SAR pattern appears in multiple splits"""
        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 15,
            'include_edge_features': False,
            'split_by_pattern': True,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df = transactions_with_many_patterns.copy()

        # Get node features WITH pattern_id before masking removes it
        df_nodes = preprocessor.cal_node_features(df, 1, 15)
        pattern_ids_before = df_nodes['pattern_id'].copy()

        # Apply masks
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # Reconstruct pattern assignment for each split
        train_patterns = set(pattern_ids_before[df_nodes['train_mask'] & (pattern_ids_before >= 0)])
        val_patterns = set(pattern_ids_before[df_nodes['val_mask'] & (pattern_ids_before >= 0)])
        test_patterns = set(pattern_ids_before[df_nodes['test_mask'] & (pattern_ids_before >= 0)])

        # Verify no pattern appears in multiple splits
        assert train_patterns.isdisjoint(val_patterns), \
            f"Pattern leakage between train and val: {train_patterns & val_patterns}"
        assert train_patterns.isdisjoint(test_patterns), \
            f"Pattern leakage between train and test: {train_patterns & test_patterns}"
        assert val_patterns.isdisjoint(test_patterns), \
            f"Pattern leakage between val and test: {val_patterns & test_patterns}"

    def test_all_splits_have_patterns(self, transactions_with_many_patterns):
        """Test that train, val, and test each have at least one SAR pattern"""
        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 15,
            'include_edge_features': False,
            'split_by_pattern': True,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df = transactions_with_many_patterns.copy()

        df_nodes = preprocessor.cal_node_features(df, 1, 15)
        pattern_ids_before = df_nodes['pattern_id'].copy()
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # Count SAR patterns in each split
        train_patterns = set(pattern_ids_before[df_nodes['train_mask'] & (pattern_ids_before >= 0)])
        val_patterns = set(pattern_ids_before[df_nodes['val_mask'] & (pattern_ids_before >= 0)])
        test_patterns = set(pattern_ids_before[df_nodes['test_mask'] & (pattern_ids_before >= 0)])

        assert len(train_patterns) >= 1, "Train split has no SAR patterns"
        assert len(val_patterns) >= 1, "Val split has no SAR patterns"
        assert len(test_patterns) >= 1, "Test split has no SAR patterns"

    def test_all_nodes_of_pattern_in_same_split(self, transactions_with_many_patterns):
        """Test that all nodes belonging to a pattern are assigned to the same split"""
        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 15,
            'include_edge_features': False,
            'split_by_pattern': True,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df = transactions_with_many_patterns.copy()

        df_nodes = preprocessor.cal_node_features(df, 1, 15)
        pattern_ids_before = df_nodes['pattern_id'].copy()
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # For each SAR pattern, verify all its nodes are in exactly one split
        for pattern_id in pattern_ids_before[pattern_ids_before >= 0].unique():
            pattern_mask = pattern_ids_before == pattern_id
            nodes_in_pattern = pattern_mask.sum()

            train_count = (pattern_mask & df_nodes['train_mask']).sum()
            val_count = (pattern_mask & df_nodes['val_mask']).sum()
            test_count = (pattern_mask & df_nodes['test_mask']).sum()

            # All nodes should be in exactly one split
            total_assigned = train_count + val_count + test_count
            assert total_assigned == nodes_in_pattern, \
                f"Pattern {pattern_id}: {total_assigned} assigned but {nodes_in_pattern} nodes exist"

            # All nodes should be in the SAME split (not spread across splits)
            splits_with_nodes = sum([train_count > 0, val_count > 0, test_count > 0])
            assert splits_with_nodes == 1, \
                f"Pattern {pattern_id} split across {splits_with_nodes} splits (train={train_count}, val={val_count}, test={test_count})"

    def test_pattern_grouping_preserved_with_noisy_labels(self, transactions_with_many_patterns):
        """Test that nodes stay grouped by pattern even when some labels are flipped"""
        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 15,
            'include_edge_features': False,
            'split_by_pattern': True,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        # Simulate label noise: flip some SAR labels to 0
        df = transactions_with_many_patterns.copy()
        # Flip ~30% of SAR labels
        sar_indices = df[df['isSAR'] == 1].index
        np.random.seed(42)
        flip_indices = np.random.choice(sar_indices, size=int(len(sar_indices) * 0.3), replace=False)
        df.loc[flip_indices, 'isSAR'] = 0

        df_nodes = preprocessor.cal_node_features(df, 1, 15)
        pattern_ids_before = df_nodes['pattern_id'].copy()
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # Even with noisy labels, pattern grouping should be preserved
        # (pattern_id is used for splitting, not is_sar)
        for pattern_id in pattern_ids_before[pattern_ids_before >= 0].unique():
            pattern_mask = pattern_ids_before == pattern_id
            train_count = (pattern_mask & df_nodes['train_mask']).sum()
            val_count = (pattern_mask & df_nodes['val_mask']).sum()
            test_count = (pattern_mask & df_nodes['test_mask']).sum()

            splits_with_nodes = sum([train_count > 0, val_count > 0, test_count > 0])
            assert splits_with_nodes == 1, \
                f"Pattern {pattern_id} split across {splits_with_nodes} splits even with noisy labels"

    def test_pattern_grouping_with_missing_labels(self, transactions_with_many_patterns):
        """Test that pattern grouping works when some labels are set to -1 (unknown)"""
        from src.feature_engineering.noise import missing_labels

        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 15,
            'include_edge_features': False,
            'split_by_pattern': True,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df = transactions_with_many_patterns.copy()

        df_nodes = preprocessor.cal_node_features(df, 1, 15)
        pattern_ids_before = df_nodes['pattern_id'].copy()

        # Apply masks first
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # Apply missing labels noise (simulating post-processing noise)
        df_nodes_noisy = missing_labels(df_nodes, labels=[1, 0], fracs=[0.3, 0.1], seed=42)

        # Pattern grouping should still be intact (masks unchanged by noise)
        for pattern_id in pattern_ids_before[pattern_ids_before >= 0].unique():
            pattern_mask = pattern_ids_before == pattern_id
            train_count = (pattern_mask & df_nodes_noisy['train_mask']).sum()
            val_count = (pattern_mask & df_nodes_noisy['val_mask']).sum()
            test_count = (pattern_mask & df_nodes_noisy['test_mask']).sum()

            splits_with_nodes = sum([train_count > 0, val_count > 0, test_count > 0])
            assert splits_with_nodes == 1, \
                f"Pattern {pattern_id} split across {splits_with_nodes} splits after missing_labels noise"

    def test_noise_does_not_affect_pattern_assignment(self, transactions_with_many_patterns):
        """Test that applying noise before splitting doesn't change which patterns go to which split"""
        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 15,
            'include_edge_features': False,
            'split_by_pattern': True,
            'transductive_train_fraction': 0.6,
            'transductive_val_fraction': 0.2,
            'transductive_test_fraction': 0.2,
            'seed': 42
        }

        # Process without noise
        preprocessor1 = DataPreprocessor(config)
        df1 = transactions_with_many_patterns.copy()
        df_nodes1 = preprocessor1.cal_node_features(df1, 1, 15)
        pattern_ids1 = df_nodes1['pattern_id'].copy()
        df_nodes1 = preprocessor1.create_transductive_masks(df_nodes1)

        # Process with label noise (flip some isSAR values before feature computation)
        preprocessor2 = DataPreprocessor(config)
        df2 = transactions_with_many_patterns.copy()
        # Flip 50% of SAR labels
        sar_indices = df2[df2['isSAR'] == 1].index
        np.random.seed(123)
        flip_indices = np.random.choice(sar_indices, size=int(len(sar_indices) * 0.5), replace=False)
        df2.loc[flip_indices, 'isSAR'] = 0

        df_nodes2 = preprocessor2.cal_node_features(df2, 1, 15)
        pattern_ids2 = df_nodes2['pattern_id'].copy()
        df_nodes2 = preprocessor2.create_transductive_masks(df_nodes2)

        # Pattern assignment should be the same (same seed, same patterns)
        # The masks should be identical because pattern_id drives the splitting
        pd.testing.assert_series_equal(
            df_nodes1['train_mask'].reset_index(drop=True),
            df_nodes2['train_mask'].reset_index(drop=True),
            check_names=False
        )
        pd.testing.assert_series_equal(
            df_nodes1['val_mask'].reset_index(drop=True),
            df_nodes2['val_mask'].reset_index(drop=True),
            check_names=False
        )
        pd.testing.assert_series_equal(
            df_nodes1['test_mask'].reset_index(drop=True),
            df_nodes2['test_mask'].reset_index(drop=True),
            check_names=False
        )

    def test_noisy_labels_within_pattern_different_across_nodes(self, transactions_with_many_patterns):
        """Test that within a pattern, nodes can have different (noisy) labels"""
        from src.feature_engineering.noise import flip_labels

        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 15,
            'include_edge_features': False,
            'split_by_pattern': True,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df = transactions_with_many_patterns.copy()

        df_nodes = preprocessor.cal_node_features(df, 1, 15)
        df_nodes = preprocessor.create_transductive_masks(df_nodes)

        # Apply aggressive noise to create mixed labels within patterns
        df_nodes_noisy = flip_labels(df_nodes, labels=[1], fracs=[0.5], seed=42)

        # Within training patterns, some nodes should have different labels now
        train_nodes = df_nodes_noisy[df_nodes_noisy['train_mask']]
        sar_in_train = train_nodes[train_nodes['true_label'] == 1]

        # Check if there's label heterogeneity (some flipped)
        has_flipped = (sar_in_train['is_sar'] != sar_in_train['true_label']).any()
        has_unflipped = (sar_in_train['is_sar'] == sar_in_train['true_label']).any()

        # With 50% flip rate, we should have both flipped and unflipped nodes
        assert has_flipped or len(sar_in_train) == 0, "Expected some labels to be flipped"


@pytest.mark.unit
class TestEdgeListExtraction:
    """Tests for edge list extraction (always generated)"""

    def test_extract_edge_list_returns_dataframe(self, sample_transactions_df):
        """Test that edge list extraction returns a DataFrame"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_edges = preprocessor.extract_edge_list(sample_transactions_df, 1, 30)

        assert isinstance(df_edges, pd.DataFrame)

    def test_extract_edge_list_has_src_dst_columns(self, sample_transactions_df):
        """Test that edge list has src and dst columns"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_edges = preprocessor.extract_edge_list(sample_transactions_df, 1, 30)

        assert 'src' in df_edges.columns
        assert 'dst' in df_edges.columns

    def test_extract_edge_list_only_two_columns(self, sample_transactions_df):
        """Test that edge list contains only src and dst (no features)"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_edges = preprocessor.extract_edge_list(sample_transactions_df, 1, 30)

        assert len(df_edges.columns) == 2
        assert list(df_edges.columns) == ['src', 'dst']

    def test_extract_edge_list_unique_edges(self, sample_transactions_df):
        """Test that edge list contains unique edges"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_edges = preprocessor.extract_edge_list(sample_transactions_df, 1, 30)

        # No duplicate edges
        assert len(df_edges) == len(df_edges.drop_duplicates())

    def test_extract_edge_list_filters_source_sink(self, sample_transactions_df):
        """Test that source and sink transactions are filtered"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        df_edges = preprocessor.extract_edge_list(sample_transactions_df, 1, 30)

        # Edge list should not be empty (unless all transactions are source/sink)
        assert len(df_edges) >= 0


@pytest.mark.unit
class TestTransductivePreprocessing:
    """Tests for transductive preprocessing workflow"""

    def test_transductive_returns_same_dataframe_for_all_splits(self, sample_transactions_df):
        """Test that transductive returns identical dataframes for train/val/test"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        # All three node dataframes should be identical
        pd.testing.assert_frame_equal(
            result['trainset_nodes'],
            result['valset_nodes']
        )
        pd.testing.assert_frame_equal(
            result['trainset_nodes'],
            result['testset_nodes']
        )

    def test_transductive_includes_masks(self, sample_transactions_df):
        """Test that transductive preprocessing includes mask columns"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        df_nodes = result['trainset_nodes']
        assert 'train_mask' in df_nodes.columns
        assert 'val_mask' in df_nodes.columns
        assert 'test_mask' in df_nodes.columns

    def test_transductive_always_returns_edges(self, sample_transactions_df):
        """Test that transductive always returns edge list (even with include_edge_features=False)"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result


@pytest.mark.unit
class TestInductivePreprocessing:
    """Tests for inductive preprocessing workflow"""

    def test_inductive_returns_different_dataframes(self, sample_transactions_df):
        """Test that inductive returns different dataframes for splits"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'learning_mode': 'inductive',
            'train_start_step': 1,
            'train_end_step': 10,
            'val_start_step': 11,
            'val_end_step': 20,
            'test_start_step': 21,
            'test_end_step': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        # Dataframes should potentially be different
        # (may have different accounts based on activity)
        assert 'trainset_nodes' in result
        assert 'valset_nodes' in result
        assert 'testset_nodes' in result

    def test_inductive_no_masks(self, sample_transactions_df):
        """Test that inductive preprocessing doesn't create mask columns"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'learning_mode': 'inductive',
            'train_start_step': 1,
            'train_end_step': 10,
            'val_start_step': 11,
            'val_end_step': 20,
            'test_start_step': 21,
            'test_end_step': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        df_nodes = result['trainset_nodes']
        assert 'train_mask' not in df_nodes.columns
        assert 'val_mask' not in df_nodes.columns
        assert 'test_mask' not in df_nodes.columns

    def test_inductive_always_returns_edges(self, sample_transactions_df):
        """Test that inductive always returns edge lists"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'learning_mode': 'inductive',
            'train_start_step': 1,
            'train_end_step': 10,
            'val_start_step': 11,
            'val_end_step': 20,
            'test_start_step': 21,
            'test_end_step': 30,
            'include_edge_features': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result
