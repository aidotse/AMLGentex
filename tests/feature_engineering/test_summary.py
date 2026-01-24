"""
Unit tests for summary module
"""
import pytest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from src.feature_engineering.summary import (
    summarize_dataset,
    _summarize_split,
    _get_dimensions,
    _get_file_sizes,
    _summarize_nodes,
    _summarize_edges,
    _compute_overall_stats,
    _write_markdown_report,
)


class TestGetDimensions:
    """Tests for _get_dimensions function"""

    def test_nodes_only(self):
        """Test dimensions when only nodes dataframe provided"""
        df_nodes = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        dims = _get_dimensions(df_nodes, None)

        assert dims['nodes_shape'] == [3, 2]
        assert dims['nodes_rows'] == 3
        assert dims['nodes_cols'] == 2
        assert dims['edges_shape'] is None
        assert dims['edges_rows'] is None
        assert dims['edges_cols'] is None

    def test_nodes_and_edges(self):
        """Test dimensions with both nodes and edges"""
        df_nodes = pd.DataFrame({'a': [1, 2, 3]})
        df_edges = pd.DataFrame({'src': [1, 2], 'dst': [2, 3], 'weight': [0.5, 0.7]})

        dims = _get_dimensions(df_nodes, df_edges)

        assert dims['nodes_shape'] == [3, 1]
        assert dims['edges_shape'] == [2, 3]
        assert dims['edges_rows'] == 2
        assert dims['edges_cols'] == 3


class TestGetFileSizes:
    """Tests for _get_file_sizes function"""

    def test_file_sizes(self, tmp_path):
        """Test file size calculation"""
        nodes_file = tmp_path / "nodes.parquet"
        edges_file = tmp_path / "edges.parquet"

        # Create test files with known content
        nodes_file.write_bytes(b"x" * 1024)  # 1KB
        edges_file.write_bytes(b"y" * 2048)  # 2KB

        sizes = _get_file_sizes(str(nodes_file), str(edges_file))

        assert sizes['nodes_size_bytes'] == 1024
        assert sizes['edges_size_bytes'] == 2048
        assert sizes['total_size_bytes'] == 3072
        assert "1.00 KB" in sizes['nodes_size']
        assert "2.00 KB" in sizes['edges_size']

    def test_missing_files(self, tmp_path):
        """Test handling of missing files"""
        nodes_file = tmp_path / "nonexistent_nodes.parquet"
        edges_file = tmp_path / "nonexistent_edges.parquet"

        sizes = _get_file_sizes(str(nodes_file), str(edges_file))

        assert sizes['nodes_size_bytes'] == 0
        assert sizes['edges_size_bytes'] == 0
        assert sizes['total_size_bytes'] == 0

    def test_human_readable_sizes(self, tmp_path):
        """Test human-readable size formatting for various sizes"""
        test_file = tmp_path / "test.dat"

        # Test bytes
        test_file.write_bytes(b"x" * 500)
        sizes = _get_file_sizes(str(test_file), str(tmp_path / "none"))
        assert "500.00 B" in sizes['nodes_size']

        # Test KB
        test_file.write_bytes(b"x" * 5000)
        sizes = _get_file_sizes(str(test_file), str(tmp_path / "none"))
        assert "KB" in sizes['nodes_size']

        # Test MB
        test_file.write_bytes(b"x" * (2 * 1024 * 1024))
        sizes = _get_file_sizes(str(test_file), str(tmp_path / "none"))
        assert "MB" in sizes['nodes_size']


class TestSummarizeNodes:
    """Tests for _summarize_nodes function"""

    def test_basic_node_stats(self):
        """Test basic node statistics"""
        df = pd.DataFrame({
            'account': ['A', 'B', 'C', 'D', 'E'],
            'is_sar': [1, 0, 0, 1, 0],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        stats = _summarize_nodes(df)

        assert stats['n_nodes'] == 5
        assert stats['n_sar'] == 2
        assert stats['n_normal'] == 3
        assert stats['sar_ratio'] == 0.4

    def test_no_sar_column(self):
        """Test when is_sar column is missing"""
        df = pd.DataFrame({
            'account': ['A', 'B', 'C'],
            'feature1': [1.0, 2.0, 3.0]
        })

        stats = _summarize_nodes(df)

        assert stats['n_nodes'] == 3
        assert stats['n_sar'] is None
        assert stats['n_normal'] is None
        assert stats['sar_ratio'] is None

    def test_transductive_masks(self):
        """Test node statistics with train/val/test masks"""
        df = pd.DataFrame({
            'account': ['A', 'B', 'C', 'D', 'E'],
            'is_sar': [1, 0, 0, 1, 0],
            'train_mask': [True, True, False, False, False],
            'val_mask': [False, False, True, False, False],
            'test_mask': [False, False, False, True, True]
        })

        stats = _summarize_nodes(df)

        assert 'train_split' in stats
        assert stats['train_split']['n_nodes'] == 2
        assert stats['train_split']['n_sar'] == 1
        assert stats['train_split']['n_normal'] == 1

        assert stats['val_split']['n_nodes'] == 1
        assert stats['test_split']['n_nodes'] == 2

    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        df = pd.DataFrame({'is_sar': []})

        stats = _summarize_nodes(df)

        assert stats['n_nodes'] == 0
        assert stats['sar_ratio'] == 0


class TestSummarizeEdges:
    """Tests for _summarize_edges function"""

    def test_basic_edge_stats(self):
        """Test basic edge statistics"""
        df = pd.DataFrame({
            'src': [1, 1, 2, 3],
            'dst': [2, 3, 3, 4],
            'is_sar': [1, 0, 0, 1],
            'weight': [0.5, 0.3, 0.2, 0.8]
        })

        stats = _summarize_edges(df)

        assert stats['n_edges'] == 4
        assert stats['n_sar'] == 2
        assert stats['n_normal'] == 2
        assert stats['sar_ratio'] == 0.5
        assert stats['n_features'] == 1  # Only 'weight', excluding src, dst, is_sar
        assert stats['unique_nodes'] == 4  # Nodes 1, 2, 3, 4

    def test_degree_calculations(self):
        """Test average degree calculations"""
        # 3 edges, 3 unique nodes
        df = pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'is_sar': [0, 0, 0]
        })

        stats = _summarize_edges(df)

        # 3 edges / 3 nodes = 1.0 average in/out degree
        assert stats['avg_in_degree'] == 1.0
        assert stats['avg_out_degree'] == 1.0
        # 2 * 3 edges / 3 nodes = 2.0 total degree
        assert stats['avg_total_degree'] == 2.0

    def test_no_sar_column(self):
        """Test when is_sar column is missing"""
        df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            'weight': [0.5, 0.3]
        })

        stats = _summarize_edges(df)

        assert stats['n_sar'] is None
        assert stats['n_normal'] is None
        assert stats['sar_ratio'] is None

    def test_no_src_dst_columns(self):
        """Test when src/dst columns are missing"""
        df = pd.DataFrame({
            'is_sar': [0, 1],
            'weight': [0.5, 0.3]
        })

        stats = _summarize_edges(df)

        assert stats['unique_nodes'] is None
        assert stats['avg_in_degree'] is None


class TestComputeOverallStats:
    """Tests for _compute_overall_stats function"""

    def test_returns_empty_dict(self):
        """Test that function returns empty dict (placeholder implementation)"""
        splits = {'trainset': {}, 'valset': {}}

        result = _compute_overall_stats(splits)

        assert result == {}


class TestWriteMarkdownReport:
    """Tests for _write_markdown_report function"""

    def test_writes_basic_report(self, tmp_path):
        """Test basic markdown report generation"""
        summary = {
            'generated_at': '2024-01-01T00:00:00',
            'raw_transactions': None,
            'splits': {
                'trainset': {
                    'nodes': {
                        'n_nodes': 100,
                        'n_sar': 10,
                        'n_normal': 90,
                        'sar_ratio': 0.1
                    },
                    'edges': {
                        'n_edges': 500,
                        'avg_in_degree': 5.0,
                        'avg_out_degree': 5.0,
                        'avg_total_degree': 10.0
                    },
                    'file_sizes': {
                        'nodes_size': '1.00 KB',
                        'edges_size': '2.00 KB',
                        'total_size': '3.00 KB'
                    },
                    'dimensions': {
                        'nodes_rows': 100,
                        'nodes_cols': 10,
                        'edges_rows': 500,
                        'edges_cols': 5
                    }
                }
            }
        }

        output_file = tmp_path / "summary.md"
        _write_markdown_report(summary, str(output_file))

        content = output_file.read_text()
        assert "# Dataset Summary" in content
        assert "2024-01-01" in content
        assert "100" in content  # n_nodes
        assert "Train Set" in content

    def test_writes_raw_transactions(self, tmp_path):
        """Test markdown report with raw transaction stats"""
        summary = {
            'generated_at': '2024-01-01T00:00:00',
            'raw_transactions': {
                'total_transactions': 10000,
                'sar_transactions': 500
            },
            'splits': {}
        }

        output_file = tmp_path / "summary.md"
        _write_markdown_report(summary, str(output_file))

        content = output_file.read_text()
        assert "Raw Transactions" in content
        assert "10,000" in content
        assert "500" in content


class TestSummarizeSplit:
    """Tests for _summarize_split function"""

    def test_summarize_with_edges(self, tmp_path):
        """Test split summary with both nodes and edges"""
        df_nodes = pd.DataFrame({
            'account': ['A', 'B', 'C'],
            'is_sar': [1, 0, 0],
            'feature1': [1.0, 2.0, 3.0]
        })
        df_edges = pd.DataFrame({
            'src': [0, 1],
            'dst': [1, 2],
            'is_sar': [0, 0]
        })

        # Create actual parquet files
        nodes_file = tmp_path / "nodes.parquet"
        edges_file = tmp_path / "edges.parquet"
        df_nodes.to_parquet(nodes_file)
        df_edges.to_parquet(edges_file)

        split_stats = _summarize_split(df_nodes, df_edges, 'trainset', str(nodes_file), str(edges_file))

        assert 'nodes' in split_stats
        assert 'edges' in split_stats
        assert 'file_sizes' in split_stats
        assert 'dimensions' in split_stats
        assert split_stats['nodes']['n_nodes'] == 3
        assert split_stats['edges']['n_edges'] == 2

    def test_summarize_without_edges(self, tmp_path):
        """Test split summary without edges"""
        df_nodes = pd.DataFrame({
            'account': ['A', 'B'],
            'is_sar': [1, 0]
        })

        nodes_file = tmp_path / "nodes.parquet"
        df_nodes.to_parquet(nodes_file)

        split_stats = _summarize_split(df_nodes, None, 'trainset', str(nodes_file), str(tmp_path / "edges.parquet"))

        assert split_stats['edges'] is None


class TestSummarizeDataset:
    """Tests for summarize_dataset function"""

    def test_summarize_full_dataset(self, tmp_path):
        """Test full dataset summarization"""
        # Create centralized directory structure
        centralized_dir = tmp_path / "centralized"
        centralized_dir.mkdir(parents=True)

        # Create sample node data
        df_nodes = pd.DataFrame({
            'account': ['A', 'B', 'C', 'D', 'E'],
            'is_sar': [1, 0, 0, 1, 0],
            'feature1': np.random.randn(5)
        })
        df_edges = pd.DataFrame({
            'src': [0, 1, 2, 3],
            'dst': [1, 2, 3, 4],
            'is_sar': [0, 0, 1, 0],
            'weight': [0.1, 0.2, 0.3, 0.4]
        })

        # Save train/val/test splits
        for split in ['trainset', 'valset', 'testset']:
            df_nodes.to_parquet(centralized_dir / f'{split}_nodes.parquet')
            df_edges.to_parquet(centralized_dir / f'{split}_edges.parquet')

        # Run summarization
        summary = summarize_dataset(str(tmp_path))

        # Verify structure
        assert 'data_directory' in summary
        assert 'generated_at' in summary
        assert 'splits' in summary
        assert 'trainset' in summary['splits']
        assert 'valset' in summary['splits']
        assert 'testset' in summary['splits']

        # Verify output files created
        assert (tmp_path / 'summary.json').exists()
        assert (tmp_path / 'summary.md').exists()

    def test_summarize_with_raw_transactions(self, tmp_path):
        """Test summarization with raw transaction data"""
        # Create centralized directory structure
        centralized_dir = tmp_path / "centralized"
        centralized_dir.mkdir(parents=True)

        # Create sample data
        df_nodes = pd.DataFrame({
            'account': ['A', 'B'],
            'is_sar': [1, 0]
        })
        df_nodes.to_parquet(centralized_dir / 'trainset_nodes.parquet')

        # Create raw transaction data
        df_raw = pd.DataFrame({
            'tx_id': range(1000),
            'isSAR': [1] * 50 + [0] * 950
        })
        raw_file = tmp_path / "tx_log.parquet"
        df_raw.to_parquet(raw_file)

        summary = summarize_dataset(str(tmp_path), raw_data_file=str(raw_file))

        assert summary['raw_transactions'] is not None
        assert summary['raw_transactions']['total_transactions'] == 1000
        assert summary['raw_transactions']['sar_transactions'] == 50

    def test_raises_on_missing_centralized_dir(self, tmp_path):
        """Test that ValueError is raised when centralized dir missing"""
        with pytest.raises(ValueError, match="Centralized data directory not found"):
            summarize_dataset(str(tmp_path))

    def test_custom_output_file(self, tmp_path):
        """Test custom output file path"""
        centralized_dir = tmp_path / "centralized"
        centralized_dir.mkdir(parents=True)

        df_nodes = pd.DataFrame({'account': ['A'], 'is_sar': [0]})
        df_nodes.to_parquet(centralized_dir / 'trainset_nodes.parquet')

        custom_output = tmp_path / "custom" / "output.json"
        summary = summarize_dataset(str(tmp_path), output_file=str(custom_output))

        assert custom_output.exists()
        assert (tmp_path / "custom" / "output.md").exists()

    def test_handles_missing_splits(self, tmp_path):
        """Test handling when some splits are missing"""
        centralized_dir = tmp_path / "centralized"
        centralized_dir.mkdir(parents=True)

        # Only create trainset
        df_nodes = pd.DataFrame({'account': ['A', 'B'], 'is_sar': [1, 0]})
        df_nodes.to_parquet(centralized_dir / 'trainset_nodes.parquet')

        summary = summarize_dataset(str(tmp_path))

        assert 'trainset' in summary['splits']
        assert 'valset' not in summary['splits']
        assert 'testset' not in summary['splits']
