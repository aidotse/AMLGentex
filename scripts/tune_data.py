import argparse
import yaml

from flib.sim import DataGenerator
from flib.preprocess import DataPreprocessor
from flib.tune import DataTuner
from time import time

def main():
    
    EXPERIMENT = '12_banks'
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_conf_file', type=str, help='Path to the data config file', default=f'/home/edvin/Desktop/flib/experiments/{EXPERIMENT}/data/param_files/conf.json')
    parser.add_argument('--config', type=str, help='Path to the config file', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--utility', type=str, default='ap')
    parser.add_argument('--bank', type=str, default='bank')
    args = parser.parse_args()
    
    # Create generator, preprocessor, and tuner
    generator = DataGenerator(args.data_conf_file)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    preprocessor = DataPreprocessor(config['preprocess'])
    tuner = DataTuner(data_conf_file=args.data_conf_file, config=config, generator=generator, preprocessor=preprocessor, target=0.01, utility=args.utility, model='DecisionTreeClassifier')
    
    # Tune the temporal sar parameters
    t = time()
    tuner(args.num_trials)
    t = time() - t 
    print(f'\nExec time: {t}\n')

if __name__ == '__main__':
    main()