from flib.tune import utils
from flib.tune.classifier import Classifier
from flib.tune.optimizer import Optimizer

class DataTuner:
    def __init__(self, data_conf_file, config, generator, preprocessor, target, utility, model, bo_dir, seed, num_trials_model):
        self.data_conf_file = data_conf_file
        self.config = config
        self.generator = generator
        self.preprocessor = preprocessor
        self.target = target
        self.utility = utility
        self.model = model
        self.bo_dir = bo_dir
        self.seed = seed
        self.num_trials_model = num_trials_model
    
    def __call__(self, n_trials):
        
        optimizer = Optimizer(data_conf_file=self.data_conf_file, config=self.config, generator=self.generator, preprocessor=self.preprocessor, target=self.target, utility=self.utility, model=self.model, bo_dir=self.bo_dir, seed=self.seed, num_trials_model=self.num_trials_model)
        best_trials = optimizer.optimize(n_trials=n_trials)
        for trial in best_trials:
            print(f'\ntrial: {trial.number}')
            print(f'values: {trial.values}')
            for param in trial.params:
                print(f'{param}: {trial.params[param]}')
            
        return