import argparse
import yaml

from flib.sim import DataGenerator
from flib.preprocess import DataPreprocessor
from flib.tune import DataTuner
from time import time

def main():
    
    EXPERIMENT = 'experiments/10K_accts'
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=False, default=EXPERIMENT)
    parser.add_argument('--num_trials_data', type=int, default=1)
    parser.add_argument('--num_trials_model', type=int, default=100)
    parser.add_argument('--utility', type=str, default='fpr')
    parser.add_argument('--bank', type=str, default='bank')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.data_conf_file = f'{args.experiment_dir}/data/param_files/conf.json'
    args.config = f'{args.experiment_dir}/config.yaml'
    args.bo_dir = f'{args.experiment_dir}/results/BO'
    
    # Create generator, preprocessor, and tuner
    generator = DataGenerator(args.data_conf_file)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    preprocessor = DataPreprocessor(config['preprocess'])
    tuner = DataTuner(data_conf_file=args.data_conf_file, config=config, generator=generator, preprocessor=preprocessor, target=0.01, utility=args.utility, model='DecisionTreeClassifier', bo_dir=args.bo_dir, seed=args.seed, num_trials_model=args.num_trials_model)
    
    # Tune the temporal sar parameters
    t = time()
    tuner(args.num_trials_data)
    t = time() - t 
    print(f'\nExec time: {t}\n')

if __name__ == '__main__':
    main()