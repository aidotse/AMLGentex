"""
Unit tests for DataPreprocessor class
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.feature_engineering.preprocessor import DataPreprocessor


@pytest.mark.unit
class TestDataPreprocessorInit:
    """Tests for DataPreprocessor initialization"""

    def test_init_basic_config(self, basic_preprocessor_config):
        """Test initialization with basic configuration"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)

        assert preprocessor.num_windows == 2
        assert preprocessor.window_len == 10
        assert preprocessor.train_start_step == 1
        assert preprocessor.train_end_step == 15
        assert preprocessor.bank is None

    def test_init_with_edges(self, preprocessor_config_with_edges):
        """Test initialization with edge features enabled"""
        preprocessor = DataPreprocessor(preprocessor_config_with_edges)

        assert preprocessor.include_edges is True


@pytest.mark.unit
class TestLoadData:
    """Tests for load_data method"""

    def test_load_data_basic(self, basic_preprocessor_config, temp_parquet_file):
        """Test basic data loading"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df = preprocessor.load_data(temp_parquet_file)

        # Check that data was loaded
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_filters_source(self, basic_preprocessor_config, sample_transactions_df):
        """Test that source transactions are filtered out"""
        # Add some source transactions
        sample_transactions_df.loc[0:5, 'bankOrig'] = 'source'

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            sample_transactions_df.to_parquet(tmp.name)
            try:
                preprocessor = DataPreprocessor(basic_preprocessor_config)
                df = preprocessor.load_data(tmp.name)

                # Source transactions should be filtered out
                assert 'source' not in df['bankOrig'].values
            finally:
                Path(tmp.name).unlink()

    def test_load_data_drops_columns(self, basic_preprocessor_config, temp_parquet_file):
        """Test that specified columns are dropped"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df = preprocessor.load_data(temp_parquet_file)

        dropped_columns = ['type', 'oldbalanceOrig', 'oldbalanceDest',
                           'newbalanceOrig', 'newbalanceDest', 'patternID', 'modelType']

        for col in dropped_columns:
            assert col not in df.columns


@pytest.mark.unit
class TestCalNodeFeatures:
    """Tests for cal_node_features method"""

    def test_cal_node_features_basic(self, basic_preprocessor_config, sample_transactions_df):
        """Test basic node feature calculation"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Check that DataFrame is returned
        assert isinstance(df_nodes, pd.DataFrame)
        assert len(df_nodes) > 0

        # Check that essential columns exist
        assert 'account' in df_nodes.columns
        assert 'is_sar' in df_nodes.columns
        assert 'bank' in df_nodes.columns

    def test_cal_node_features_window_features(self, basic_preprocessor_config, sample_transactions_df):
        """Test that window-based features are created"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Check for window-specific feature columns
        window_feature_patterns = ['sum_in', 'mean_in', 'sum_out', 'mean_out',
                                    'sums_spending', 'means_spending']

        for pattern in window_feature_patterns:
            matching_cols = [col for col in df_nodes.columns if pattern in col]
            assert len(matching_cols) > 0, f"No columns found for pattern: {pattern}"

    def test_cal_node_features_no_missing_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no missing values in output"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # No missing values should exist
        assert df_nodes.isnull().sum().sum() == 0

    def test_cal_node_features_no_negative_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no negative values (except bank column)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Check no negative values in numeric columns (excluding bank)
        numeric_cols = df_nodes.select_dtypes(include=[np.number]).columns
        assert (df_nodes[numeric_cols] >= 0).all().all()

    def test_cal_node_features_single_window(self, sample_transactions_df):
        """Test with a single window"""
        config = {
            'num_windows': 1,
            'window_len': 15,
            'train_start_step': 1,
            'train_end_step': 15,
            'val_start_step': 16,
            'val_end_step': 22,
            'test_start_step': 23,
            'test_end_step': 30,
            'include_edges': False
        }
        preprocessor = DataPreprocessor(config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        assert isinstance(df_nodes, pd.DataFrame)
        assert len(df_nodes) > 0

    def test_cal_node_features_window_coverage_error(self, basic_preprocessor_config, sample_transactions_df):
        """Test error when windows don't cover the dataset"""
        config = basic_preprocessor_config.copy()
        config['num_windows'] = 1
        config['window_len'] = 5  # Too small to cover the range

        preprocessor = DataPreprocessor(config)

        with pytest.raises(ValueError, match="do not allow coverage"):
            preprocessor.cal_node_features(
                sample_transactions_df,
                start_step=1,
                end_step=15
            )


@pytest.mark.unit
class TestCalEdgeFeatures:
    """Tests for cal_edge_features method"""

    def test_cal_edge_features_basic(self, basic_preprocessor_config, sample_transactions_df):
        """Test basic edge feature calculation"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # Check that DataFrame is returned
        assert isinstance(df_edges, pd.DataFrame)

        # Check that essential columns exist
        assert 'src' in df_edges.columns
        assert 'dst' in df_edges.columns
        assert 'is_sar' in df_edges.columns

    def test_cal_edge_features_window_features(self, basic_preprocessor_config, sample_transactions_df):
        """Test that window-based edge features are created"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # Check for window-specific feature columns
        window_feature_patterns = ['sums_', 'means_', 'medians_', 'stds_', 'maxs_', 'mins_', 'counts_']

        for pattern in window_feature_patterns:
            matching_cols = [col for col in df_edges.columns if pattern in col]
            assert len(matching_cols) > 0, f"No columns found for pattern: {pattern}"

    def test_cal_edge_features_directional(self, basic_preprocessor_config, sample_transactions_df):
        """Test directional vs non-directional edge features"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)

        df_directional = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        df_undirected = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=False
        )

        # Both should return valid DataFrames
        assert isinstance(df_directional, pd.DataFrame)
        assert isinstance(df_undirected, pd.DataFrame)

    def test_cal_edge_features_no_missing_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no missing values in output"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # No missing values should exist
        assert df_edges.isnull().sum().sum() == 0

    def test_cal_edge_features_no_negative_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no negative values (except src/dst columns)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # Check no negative values in feature columns (excluding src/dst)
        feature_cols = [col for col in df_edges.columns if col not in ['src', 'dst']]
        numeric_cols = df_edges[feature_cols].select_dtypes(include=[np.number]).columns
        assert (df_edges[numeric_cols] >= 0).all().all()

    def test_cal_edge_features_window_coverage_error(self, basic_preprocessor_config, sample_transactions_df):
        """Test error when windows don't cover the dataset"""
        config = basic_preprocessor_config.copy()
        config['num_windows'] = 1
        config['window_len'] = 5  # Too small to cover the range

        preprocessor = DataPreprocessor(config)

        with pytest.raises(ValueError, match="do not allow coverage"):
            preprocessor.cal_edge_features(
                sample_transactions_df,
                start_step=1,
                end_step=15,
                directional=True
            )


@pytest.mark.unit
class TestPreprocess:
    """Tests for preprocess method"""

    def test_preprocess_without_edges(self, basic_preprocessor_config, sample_transactions_df):
        """Test preprocessing without edge features (edges are still generated for graph structure)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        result = preprocessor.preprocess(sample_transactions_df)

        # Check that all node sets are returned
        assert 'trainset_nodes' in result
        assert 'valset_nodes' in result
        assert 'testset_nodes' in result

        # Edge sets are always generated (for graph structure)
        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result

        # Check that DataFrames are valid
        assert isinstance(result['trainset_nodes'], pd.DataFrame)
        assert isinstance(result['valset_nodes'], pd.DataFrame)
        assert isinstance(result['testset_nodes'], pd.DataFrame)

    def test_preprocess_with_edges(self, preprocessor_config_with_edges, sample_transactions_df):
        """Test preprocessing with edge features"""
        preprocessor = DataPreprocessor(preprocessor_config_with_edges)
        result = preprocessor.preprocess(sample_transactions_df)

        # Check that all node and edge sets are returned
        assert 'trainset_nodes' in result
        assert 'valset_nodes' in result
        assert 'testset_nodes' in result
        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result

        # Check that DataFrames are valid
        for key, value in result.items():
            assert isinstance(value, pd.DataFrame), f"{key} is not a DataFrame"


@pytest.mark.unit
class TestCallMethod:
    """Tests for __call__ method"""

    def test_call_method(self, basic_preprocessor_config, temp_parquet_file):
        """Test that __call__ method works end-to-end"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        result = preprocessor(temp_parquet_file)

        # Check that result is a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'trainset_nodes' in result
        assert 'valset_nodes' in result
        assert 'testset_nodes' in result

    def test_call_method_with_edges(self, preprocessor_config_with_edges, temp_parquet_file):
        """Test __call__ method with edge features"""
        preprocessor = DataPreprocessor(preprocessor_config_with_edges)
        result = preprocessor(temp_parquet_file)

        # Check that result includes edges
        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result


@pytest.mark.unit
class TestBankFiltering:
    """Tests for bank-specific filtering"""

    def test_bank_filtering_none(self, basic_preprocessor_config, sample_transactions_df):
        """Test when bank is None (no filtering)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        preprocessor.bank = None

        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Should include accounts from all banks
        assert len(df_nodes) > 0

    def test_bank_filtering_specific_bank(self, basic_preprocessor_config, sample_transactions_df):
        """Test when specific bank is set"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        preprocessor.bank = 'BANK001'

        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Should only include accounts from BANK001
        if len(df_nodes) > 0:
            assert (df_nodes['bank'] == 'BANK001').all()
