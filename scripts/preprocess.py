import argparse
import os
import yaml
from src.feature_engineering import DataPreprocessor, summarize_dataset
from src.utils.config import load_preprocessing_config
from typing import Dict

def main(config: Dict):

    preprocessor = DataPreprocessor(config)
    datasets = preprocessor(config['raw_data_file'])

    os.makedirs(config['preprocessed_data_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['preprocessed_data_dir'], 'centralized'), exist_ok=True)
    os.makedirs(os.path.join(config['preprocessed_data_dir'], 'clients'), exist_ok=True)

    for name, dataset in datasets.items():
        dataset.to_parquet(os.path.join(config['preprocessed_data_dir'], 'centralized', name+'.parquet'), index=False)

    banks = datasets['trainset_nodes']['bank'].unique()
    for bank in banks:
        bank_str = str(bank)

        os.makedirs(os.path.join(config['preprocessed_data_dir'], 'clients', bank_str), exist_ok=True)

        df_nodes = datasets['trainset_nodes']
        df_nodes[df_nodes['bank'] == bank].to_parquet(os.path.join(config['preprocessed_data_dir'], 'clients', bank_str, 'trainset_nodes.parquet'), index=False)

        if 'trainset_edges' in datasets:
            unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
            df_edges = datasets['trainset_edges']
            df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_parquet(os.path.join(config['preprocessed_data_dir'], 'clients', bank_str, 'trainset_edges.parquet'), index=False)

        df_nodes = datasets['valset_nodes']
        df_nodes[df_nodes['bank'] == bank].to_parquet(os.path.join(config['preprocessed_data_dir'], 'clients', bank_str, 'valset_nodes.parquet'), index=False)

        if 'valset_edges' in datasets:
            unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
            df_edges = datasets['valset_edges']
            df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_parquet(os.path.join(config['preprocessed_data_dir'], 'clients', bank_str, 'valset_edges.parquet'), index=False)

        df_nodes = datasets['testset_nodes']
        df_nodes[df_nodes['bank'] == bank].to_parquet(os.path.join(config['preprocessed_data_dir'], 'clients', bank_str, 'testset_nodes.parquet'), index=False)

        if 'testset_edges' in datasets:
            unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
            df_edges = datasets['testset_edges']
            df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_parquet(os.path.join(config['preprocessed_data_dir'], 'clients', bank_str, 'testset_edges.parquet'), index=False)

    # Generate summary statistics
    summarize_dataset(config['preprocessed_data_dir'], raw_data_file=config['raw_data_file'])

if __name__ == "__main__":

    EXPERIMENT = '10k_accounts'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to preprocessing config file.', default=f'experiments/{EXPERIMENT}/config/preprocessing.yaml')
    args = parser.parse_args()

    # Load config with auto-discovered paths
    config = load_preprocessing_config(args.config)

    main(config)