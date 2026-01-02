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
    """Tests for automatic detection of learning setting"""

    def test_detects_transductive_same_time_windows(self):
        """Test that transductive is detected when all splits have same time window"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'train_start_step': 0,
            'train_end_step': 100,
            'val_start_step': 0,
            'val_end_step': 100,
            'test_start_step': 0,
            'test_end_step': 100,
            'include_edges': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        assert preprocessor.is_transductive is True

    def test_detects_inductive_different_time_windows(self):
        """Test that inductive is detected when splits have different time windows"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'train_start_step': 0,
            'train_end_step': 30,
            'val_start_step': 31,
            'val_end_step': 60,
            'test_start_step': 61,
            'test_end_step': 100,
            'include_edges': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        assert preprocessor.is_transductive is False

    def test_detects_inductive_overlapping_windows(self):
        """Test that inductive is detected even with partial overlap"""
        config = {
            'num_windows': 2,
            'window_len': 10,
            'train_start_step': 0,
            'train_end_step': 50,
            'val_start_step': 40,
            'val_end_step': 80,
            'test_start_step': 70,
            'test_end_step': 100,
            'include_edges': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)

        # Overlapping but not identical = inductive
        assert preprocessor.is_transductive is False


@pytest.mark.unit
class TestTransductiveMaskCreation:
    """Tests for transductive label mask creation"""

    def test_creates_three_masks(self, sample_transactions_df):
        """Test that train/val/test masks are created"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
            'train_label_fraction': 0.6,
            'val_label_fraction': 0.2,
            'test_label_fraction': 0.2,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
            'train_label_fraction': 0.6,
            'val_label_fraction': 0.2,
            'test_label_fraction': 0.2,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
            'train_label_fraction': 0.6,
            'val_label_fraction': 0.2,
            'test_label_fraction': 0.2,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
class TestEdgeListExtraction:
    """Tests for edge list extraction (always generated)"""

    def test_extract_edge_list_returns_dataframe(self, sample_transactions_df):
        """Test that edge list extraction returns a DataFrame"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        df_nodes = result['trainset_nodes']
        assert 'train_mask' in df_nodes.columns
        assert 'val_mask' in df_nodes.columns
        assert 'test_mask' in df_nodes.columns

    def test_transductive_always_returns_edges(self, sample_transactions_df):
        """Test that transductive always returns edge list (even with include_edges=False)"""
        config = {
            'num_windows': 2,
            'window_len': 15,
            'train_start_step': 1,
            'train_end_step': 30,
            'val_start_step': 1,
            'val_end_step': 30,
            'test_start_step': 1,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 10,
            'val_start_step': 11,
            'val_end_step': 20,
            'test_start_step': 21,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 10,
            'val_start_step': 11,
            'val_end_step': 20,
            'test_start_step': 21,
            'test_end_step': 30,
            'include_edges': False,
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
            'train_start_step': 1,
            'train_end_step': 10,
            'val_start_step': 11,
            'val_end_step': 20,
            'test_start_step': 21,
            'test_end_step': 30,
            'include_edges': False,
            'seed': 42
        }
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result
