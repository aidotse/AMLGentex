import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def summarize_dataset(preprocessed_data_dir: str, output_file: str = None, raw_data_file: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive statistics about preprocessed dataset.

    Args:
        preprocessed_data_dir: Directory containing preprocessed centralized data
        output_file: Optional path to save summary. If None, saves to preprocessed_data_dir/summary.json
        raw_data_file: Optional path to raw transaction data for SAR transaction count

    Returns:
        Dictionary containing dataset statistics
    """

    # Set default output paths
    if output_file is None:
        json_file = os.path.join(preprocessed_data_dir, 'summary.json')
        markdown_file = os.path.join(preprocessed_data_dir, 'summary.md')
    else:
        json_file = output_file
        markdown_file = output_file.replace('.json', '.md')

    # Check for centralized data
    centralized_dir = os.path.join(preprocessed_data_dir, 'centralized')
    if not os.path.exists(centralized_dir):
        raise ValueError(f"Centralized data directory not found: {centralized_dir}")

    summary = {
        'data_directory': preprocessed_data_dir,
        'generated_at': datetime.now().isoformat(),
        'splits': {}
    }

    # Get raw transaction statistics
    if raw_data_file and os.path.exists(raw_data_file):
        df_raw = pd.read_parquet(raw_data_file)
        summary['raw_transactions'] = {
            'total_transactions': len(df_raw),
            'sar_transactions': int(df_raw['isSAR'].sum()) if 'isSAR' in df_raw.columns else 0
        }
    else:
        summary['raw_transactions'] = None

    # Analyze each split
    for split in ['trainset', 'valset', 'testset']:
        nodes_file = os.path.join(centralized_dir, f'{split}_nodes.parquet')
        edges_file = os.path.join(centralized_dir, f'{split}_edges.parquet')

        if not os.path.exists(nodes_file):
            continue

        # Load data
        df_nodes = pd.read_parquet(nodes_file)
        df_edges = pd.read_parquet(edges_file) if os.path.exists(edges_file) else None

        split_summary = _summarize_split(df_nodes, df_edges, split, nodes_file, edges_file)
        summary['splits'][split] = split_summary

    # Overall statistics across splits
    summary['overall'] = _compute_overall_stats(summary['splits'])

    # Save JSON
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save Markdown report
    _write_markdown_report(summary, markdown_file)

    print(f"\nDataset summary saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Markdown: {markdown_file}")

    return summary


def _summarize_split(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, split_name: str, nodes_file: str, edges_file: str) -> Dict[str, Any]:
    """Generate statistics for a single data split."""

    split_stats = {
        'nodes': _summarize_nodes(df_nodes),
        'edges': _summarize_edges(df_edges) if df_edges is not None else None,
        'file_sizes': _get_file_sizes(nodes_file, edges_file),
        'dimensions': _get_dimensions(df_nodes, df_edges)
    }

    return split_stats


def _get_dimensions(df_nodes: pd.DataFrame, df_edges: pd.DataFrame) -> Dict[str, Any]:
    """Get dimensionality (shape) of dataframes."""

    dims = {
        'nodes_shape': list(df_nodes.shape),
        'nodes_rows': df_nodes.shape[0],
        'nodes_cols': df_nodes.shape[1]
    }

    if df_edges is not None:
        dims.update({
            'edges_shape': list(df_edges.shape),
            'edges_rows': df_edges.shape[0],
            'edges_cols': df_edges.shape[1]
        })
    else:
        dims.update({
            'edges_shape': None,
            'edges_rows': None,
            'edges_cols': None
        })

    return dims


def _get_file_sizes(nodes_file: str, edges_file: str) -> Dict[str, Any]:
    """Get file sizes in human-readable format."""

    def human_readable_size(size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    nodes_size = os.path.getsize(nodes_file) if os.path.exists(nodes_file) else 0
    edges_size = os.path.getsize(edges_file) if os.path.exists(edges_file) else 0

    return {
        'nodes_size_bytes': nodes_size,
        'edges_size_bytes': edges_size,
        'total_size_bytes': nodes_size + edges_size,
        'nodes_size': human_readable_size(nodes_size),
        'edges_size': human_readable_size(edges_size),
        'total_size': human_readable_size(nodes_size + edges_size)
    }


def _summarize_nodes(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate node-level statistics."""

    # Basic counts
    n_nodes = len(df)

    # Label distribution
    if 'is_sar' in df.columns:
        n_sar = int(df['is_sar'].sum())
        n_normal = n_nodes - n_sar
        sar_ratio = n_sar / n_nodes if n_nodes > 0 else 0
    else:
        n_sar = n_normal = sar_ratio = None

    stats = {
        'n_nodes': n_nodes,
        'n_sar': n_sar,
        'n_normal': n_normal,
        'sar_ratio': round(sar_ratio, 4) if sar_ratio is not None else None
    }

    # Check for train/val/test masks (transductive learning)
    if 'train_mask' in df.columns:
        train_nodes = df[df['train_mask']]
        val_nodes = df[df['val_mask']]
        test_nodes = df[df['test_mask']]

        stats['train_split'] = {
            'n_nodes': int(df['train_mask'].sum()),
            'n_sar': int(train_nodes['is_sar'].sum()),
            'n_normal': int((~train_nodes['is_sar'].astype(bool)).sum())
        }
        stats['val_split'] = {
            'n_nodes': int(df['val_mask'].sum()),
            'n_sar': int(val_nodes['is_sar'].sum()),
            'n_normal': int((~val_nodes['is_sar'].astype(bool)).sum())
        }
        stats['test_split'] = {
            'n_nodes': int(df['test_mask'].sum()),
            'n_sar': int(test_nodes['is_sar'].sum()),
            'n_normal': int((~test_nodes['is_sar'].astype(bool)).sum())
        }

    return stats


def _summarize_edges(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate edge-level statistics."""

    n_edges = len(df)

    # Label distribution
    if 'is_sar' in df.columns:
        n_sar = int(df['is_sar'].sum())
        n_normal = n_edges - n_sar
        sar_ratio = n_sar / n_edges if n_edges > 0 else 0
    else:
        n_sar = n_normal = sar_ratio = None

    # Graph properties
    if 'src' in df.columns and 'dst' in df.columns:
        unique_nodes = len(set(df['src'].unique()) | set(df['dst'].unique()))

        # Average degrees for directed graph
        # Each edge contributes 1 to out-degree (src) and 1 to in-degree (dst)
        avg_in_degree = n_edges / unique_nodes if unique_nodes > 0 else 0
        avg_out_degree = n_edges / unique_nodes if unique_nodes > 0 else 0
        avg_total_degree = (2 * n_edges) / unique_nodes if unique_nodes > 0 else 0
    else:
        unique_nodes = avg_in_degree = avg_out_degree = avg_total_degree = None

    # Feature count (excluding identifiers and labels)
    feature_cols = [col for col in df.columns if col not in ['src', 'dst', 'is_sar']]
    n_features = len(feature_cols)

    return {
        'n_edges': n_edges,
        'n_features': n_features,
        'n_sar': n_sar,
        'n_normal': n_normal,
        'sar_ratio': round(sar_ratio, 4) if sar_ratio is not None else None,
        'unique_nodes': unique_nodes,
        'avg_in_degree': round(avg_in_degree, 2) if avg_in_degree is not None else None,
        'avg_out_degree': round(avg_out_degree, 2) if avg_out_degree is not None else None,
        'avg_total_degree': round(avg_total_degree, 2) if avg_total_degree is not None else None
    }


def _compute_overall_stats(splits: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute statistics across all splits."""
    # Not needed for simple summary
    return {}


def _write_markdown_report(summary: Dict[str, Any], output_file: str):
    """Write simple markdown report."""

    lines = []

    # Header
    lines.append("# Dataset Summary")
    lines.append("")
    lines.append(f"**Generated:** {summary['generated_at']}")
    lines.append("")

    # Raw transaction statistics
    if summary.get('raw_transactions'):
        raw = summary['raw_transactions']
        lines.append("## Raw Transactions")
        lines.append("")
        lines.append(f"- **Total transactions:** {raw['total_transactions']:,}")
        lines.append(f"- **SAR transactions:** {raw['sar_transactions']:,} ({100*raw['sar_transactions']/raw['total_transactions']:.2f}%)")
        lines.append("")

    # Split details (simple table)
    for split_name, split_data in summary['splits'].items():
        nodes = split_data['nodes']
        edges = split_data['edges']
        file_sizes = split_data.get('file_sizes', {})
        dimensions = split_data.get('dimensions', {})

        lines.append(f"## {split_name.replace('set', ' Set').title()}")
        lines.append("")
        lines.append(f"- **Nodes:** {nodes['n_nodes']:,}")
        lines.append(f"- **SAR nodes:** {nodes['n_sar']:,} ({nodes['sar_ratio']*100:.1f}%)")

        # Show train/val/test split statistics (transductive learning)
        # Only show the relevant split for each section
        if 'train_split' in nodes:
            # Determine which split to show based on section name
            if 'train' in split_name:
                split_info = nodes['train_split']
                split_label = 'Train'
            elif 'val' in split_name:
                split_info = nodes['val_split']
                split_label = 'Val'
            elif 'test' in split_name:
                split_info = nodes['test_split']
                split_label = 'Test'
            else:
                split_info = None

            if split_info:
                lines.append(f"- **Labeled nodes ({split_label.lower()}):** {split_info['n_nodes']:,} ({split_info['n_sar']} SAR, {split_info['n_normal']} normal)")

        if edges:
            lines.append(f"- **Edges:** {edges['n_edges']:,}")
            lines.append(f"- **Average in-degree:** {edges['avg_in_degree']:.2f}")
            lines.append(f"- **Average out-degree:** {edges['avg_out_degree']:.2f}")
            lines.append(f"- **Average total degree:** {edges['avg_total_degree']:.2f}")

        if dimensions:
            lines.append(f"- **Dimensionality:**")
            lines.append(f"  - Nodes: {dimensions['nodes_rows']:,} rows × {dimensions['nodes_cols']} columns")
            if dimensions.get('edges_rows') is not None:
                lines.append(f"  - Edges: {dimensions['edges_rows']:,} rows × {dimensions['edges_cols']} columns")

        if file_sizes:
            lines.append(f"- **File sizes:**")
            lines.append(f"  - Nodes: {file_sizes['nodes_size']}")
            lines.append(f"  - Edges: {file_sizes['edges_size']}")
            lines.append(f"  - Total: {file_sizes['total_size']}")

        lines.append("")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
