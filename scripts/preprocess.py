import argparse
import os
import yaml
from flib.preprocess import DataPreprocessor
from typing import Dict

def main(config: Dict):
    
    preprocessor = DataPreprocessor(config['preprocess'])
    datasets = preprocessor(config['preprocess']['raw_data_file'])
    
    os.makedirs(config['preprocess']['preprocessed_data_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['preprocess']['preprocessed_data_dir'], 'centralized'), exist_ok=True)
    os.makedirs(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients'), exist_ok=True)
    
    for name, dataset in datasets.items():
        dataset.to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], 'centralized', name+'.csv'), index=False)
    
    banks = datasets['trainset_nodes']['bank'].unique()
    for bank in banks:
        
        os.makedirs(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients', bank), exist_ok=True)
        
        df_nodes = datasets['trainset_nodes']
        df_nodes[df_nodes['bank'] == bank].to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'trainset_nodes.csv'), index=False)
        unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
        df_edges = datasets['trainset_edges']
        df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'trainset_edges.csv'), index=False)
        
        df_nodes = datasets['valset_nodes']
        df_nodes[df_nodes['bank'] == bank].to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'valset_nodes.csv'), index=False)
        unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
        df_edges = datasets['valset_edges']
        df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'valset_edges.csv'), index=False)
        
        df_nodes = datasets['testset_nodes']
        df_nodes[df_nodes['bank'] == bank].to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'testset_nodes.csv'), index=False)
        unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
        df_edges = datasets['testset_edges']
        df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_csv(os.path.join(config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'testset_edges.csv'), index=False)
        
if __name__ == "__main__":
    
    EXPERIMENT = '10K_accts'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)