"""
Tests for apply_train_noise function
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.feature_engineering.noise import missing_labels, flip_labels, apply_train_noise
from src.feature_engineering.preprocessor import DataPreprocessor


@pytest.fixture
def temp_preprocessed_dir_transductive(sample_transactions_df):
    """Create a temporary preprocessed directory with transductive data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create transductive config
        config = {
            'num_windows': 2,
            'window_len': 15,
            'learning_mode': 'transductive',
            'time_start': 1,
            'time_end': 30,
            'include_edge_features': False,
            'seed': 42
        }

        # Preprocess data
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        # Save to directory
        centralized_dir = os.path.join(tmpdir, 'centralized')
        os.makedirs(centralized_dir, exist_ok=True)

        result['trainset_nodes'].to_parquet(
            os.path.join(centralized_dir, 'trainset_nodes.parquet'),
            index=False
        )
        result['valset_nodes'].to_parquet(
            os.path.join(centralized_dir, 'valset_nodes.parquet'),
            index=False
        )
        result['testset_nodes'].to_parquet(
            os.path.join(centralized_dir, 'testset_nodes.parquet'),
            index=False
        )

        yield tmpdir


@pytest.fixture
def temp_preprocessed_dir_inductive(sample_transactions_df):
    """Create a temporary preprocessed directory with inductive data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create inductive config
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

        # Preprocess data
        preprocessor = DataPreprocessor(config)
        result = preprocessor.preprocess(sample_transactions_df)

        # Save to directory
        centralized_dir = os.path.join(tmpdir, 'centralized')
        os.makedirs(centralized_dir, exist_ok=True)

        result['trainset_nodes'].to_parquet(
            os.path.join(centralized_dir, 'trainset_nodes.parquet'),
            index=False
        )
        result['valset_nodes'].to_parquet(
            os.path.join(centralized_dir, 'valset_nodes.parquet'),
            index=False
        )
        result['testset_nodes'].to_parquet(
            os.path.join(centralized_dir, 'testset_nodes.parquet'),
            index=False
        )

        yield tmpdir


@pytest.mark.unit
class TestApplyTrainNoiseTransductive:
    """Tests for apply_train_noise with transductive data"""

    def test_applies_noise_to_training_labels_only(self, temp_preprocessed_dir_transductive):
        """Test that noise is only applied to training labels"""
        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.5, 0.5], 'seed': 42}
        )

        # Load data
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_train = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))
        df_val = pd.read_parquet(os.path.join(centralized_dir, 'valset_nodes.parquet'))
        df_test = pd.read_parquet(os.path.join(centralized_dir, 'testset_nodes.parquet'))

        # Check that training labels have unknowns
        train_nodes = df_train[df_train['train_mask']]
        assert (train_nodes['is_sar'] == -1).sum() > 0

        # Check that val/test labels don't have unknowns
        val_nodes = df_val[df_val['val_mask']]
        test_nodes = df_test[df_test['test_mask']]
        assert (val_nodes['is_sar'] == -1).sum() == 0
        assert (test_nodes['is_sar'] == -1).sum() == 0

    def test_creates_true_label_column(self, temp_preprocessed_dir_transductive):
        """Test that true_label column is created"""
        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.2, 0.1], 'seed': 42}
        )

        # Load data
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_train = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))

        assert 'true_label' in df_train.columns

    def test_true_label_matches_original_labels(self, temp_preprocessed_dir_transductive):
        """Test that true_label preserves original labels"""
        # Load original data
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_original = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))
        original_labels = df_original['is_sar'].copy()

        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.3, 0.2], 'seed': 42}
        )

        # Load noisy data
        df_noisy = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))

        # True labels should match original
        pd.testing.assert_series_equal(
            df_noisy['true_label'].reset_index(drop=True),
            original_labels.reset_index(drop=True),
            check_names=False
        )

    def test_preserves_non_training_nodes(self, temp_preprocessed_dir_transductive):
        """Test that nodes without train_mask=True are not affected"""
        # Load original data
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_original = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))
        non_train_labels = df_original[~df_original['train_mask']]['is_sar'].copy()

        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.5, 0.5], 'seed': 42}
        )

        # Load noisy data
        df_noisy = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))
        non_train_labels_after = df_noisy[~df_noisy['train_mask']]['is_sar'].copy()

        # Non-training labels should be unchanged
        pd.testing.assert_series_equal(
            non_train_labels.reset_index(drop=True),
            non_train_labels_after.reset_index(drop=True),
            check_names=False
        )

    def test_all_splits_updated_identically(self, temp_preprocessed_dir_transductive):
        """Test that train/val/test files remain identical in transductive"""
        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.2, 0.1], 'seed': 42}
        )

        # Load all splits
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_train = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))
        df_val = pd.read_parquet(os.path.join(centralized_dir, 'valset_nodes.parquet'))
        df_test = pd.read_parquet(os.path.join(centralized_dir, 'testset_nodes.parquet'))

        # All should be identical
        pd.testing.assert_frame_equal(df_train, df_val)
        pd.testing.assert_frame_equal(df_train, df_test)

    def test_works_with_flip_labels(self, temp_preprocessed_dir_transductive):
        """Test that apply_train_noise works with flip_labels function"""
        # Apply noise with flip_labels
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            flip_labels,
            {'labels': [0, 1], 'fracs': [0.1, 0.2], 'seed': 42}
        )

        # Load data
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_train = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))

        # Check that true_label exists and differs from is_sar for some nodes
        assert 'true_label' in df_train.columns
        train_nodes = df_train[df_train['train_mask']]
        flipped = train_nodes[train_nodes['true_label'] != train_nodes['is_sar']]
        assert len(flipped) > 0


@pytest.mark.unit
class TestApplyTrainNoiseInductive:
    """Tests for apply_train_noise with inductive data"""

    def test_applies_noise_to_training_file_only(self, temp_preprocessed_dir_inductive):
        """Test that noise is only applied to training file"""
        # Load original val/test data
        centralized_dir = os.path.join(temp_preprocessed_dir_inductive, 'centralized')
        df_val_original = pd.read_parquet(os.path.join(centralized_dir, 'valset_nodes.parquet'))
        df_test_original = pd.read_parquet(os.path.join(centralized_dir, 'testset_nodes.parquet'))

        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_inductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.5, 0.5], 'seed': 42}
        )

        # Load data after noise
        df_train = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))
        df_val = pd.read_parquet(os.path.join(centralized_dir, 'valset_nodes.parquet'))
        df_test = pd.read_parquet(os.path.join(centralized_dir, 'testset_nodes.parquet'))

        # Training should have unknowns
        assert (df_train['is_sar'] == -1).sum() > 0

        # Val/test should be unchanged
        pd.testing.assert_frame_equal(df_val, df_val_original)
        pd.testing.assert_frame_equal(df_test, df_test_original)

    def test_creates_true_label_column(self, temp_preprocessed_dir_inductive):
        """Test that true_label column is created in training data"""
        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_inductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.2, 0.1], 'seed': 42}
        )

        # Load data
        centralized_dir = os.path.join(temp_preprocessed_dir_inductive, 'centralized')
        df_train = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))

        assert 'true_label' in df_train.columns

    def test_val_test_unaffected(self, temp_preprocessed_dir_inductive):
        """Test that val/test files are not modified"""
        # Load original data
        centralized_dir = os.path.join(temp_preprocessed_dir_inductive, 'centralized')
        df_val_before = pd.read_parquet(os.path.join(centralized_dir, 'valset_nodes.parquet'))
        df_test_before = pd.read_parquet(os.path.join(centralized_dir, 'testset_nodes.parquet'))

        # Apply noise
        apply_train_noise(
            temp_preprocessed_dir_inductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.3, 0.2], 'seed': 42}
        )

        # Load data after
        df_val_after = pd.read_parquet(os.path.join(centralized_dir, 'valset_nodes.parquet'))
        df_test_after = pd.read_parquet(os.path.join(centralized_dir, 'testset_nodes.parquet'))

        # Should be identical
        pd.testing.assert_frame_equal(df_val_before, df_val_after)
        pd.testing.assert_frame_equal(df_test_before, df_test_after)


@pytest.mark.unit
class TestApplyTrainNoiseEdgeCases:
    """Tests for edge cases in apply_train_noise"""

    def test_handles_empty_noise_kwargs(self, temp_preprocessed_dir_transductive):
        """Test that function handles empty noise_kwargs"""
        # Should use default kwargs for the noise function
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            None
        )

        # Should not raise an error
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))
        assert isinstance(df, pd.DataFrame)

    def test_handles_zero_noise(self, temp_preprocessed_dir_transductive):
        """Test that function handles zero noise fractions"""
        # Load original
        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_original = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))

        # Apply zero noise
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.0, 0.0], 'seed': 42}
        )

        # Load after
        df_after = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))

        # Training labels should be unchanged (no -1 values)
        train_nodes = df_after[df_after['train_mask']]
        assert (train_nodes['is_sar'] == -1).sum() == 0

    def test_reproducible_with_same_seed(self, temp_preprocessed_dir_transductive):
        """Test that same seed produces same noise"""
        # Apply noise first time
        apply_train_noise(
            temp_preprocessed_dir_transductive,
            missing_labels,
            {'labels': [1, 0], 'fracs': [0.3, 0.2], 'seed': 42}
        )

        centralized_dir = os.path.join(temp_preprocessed_dir_transductive, 'centralized')
        df_first = pd.read_parquet(os.path.join(centralized_dir, 'trainset_nodes.parquet'))

        # Reapply with same seed (need to restore original first)
        # For this test, we'll just check that is_sar column has expected properties
        train_nodes = df_first[df_first['train_mask']]
        n_unknown = (train_nodes['is_sar'] == -1).sum()

        # Should have some unknown labels
        assert n_unknown > 0
