import optuna
from flib import clients, models
from flib.tune.classifier import Classifier # TODO: classifiers should be in flib.models
from flib.train.tune_hyperparams import HyperparamTuner
from flib.train import centralized 
import matplotlib.pyplot as plt
import json
import os

class Optimizer():
    def __init__(self, data_conf_file, config, generator, preprocessor, target:float, utility:str, model:str='DecisionTreeClassifier', bank=None, bo_dir:str='tmp', seed:int=0, num_trials_model:int=1):
        self.data_conf_file = data_conf_file
        self.config = config
        self.generator = generator
        self.preprocessor = preprocessor
        self.target = target
        self.utility = utility
        self.model = model
        self.bank = bank
        self.bo_dir = bo_dir
        self.seed = seed
        self.num_trials_model = num_trials_model
    
    def objective(self, trial:optuna.Trial):
        with open(self.data_conf_file, 'r') as f:
            data_config = json.load(f)
        
        for k, v in data_config['optimisation_bounds'].items():
            lower = v[0]
            upper = v[1]
            if type(lower) is int:
                data_config['default'][k] = trial.suggest_int(k, lower, upper)
            elif type(lower) is float:
                data_config['default'][k] = trial.suggest_float(k, lower, upper)
            else:
                raise ValueError(f'Type {type(lower)} in optimisation bounds not recognised, use int or float.')
        
        with open(self.data_conf_file, 'w') as f:
            json.dump(data_config, f, indent=4)
        
        tx_log_file = self.generator(os.path.abspath(self.data_conf_file))
        datasets = self.preprocessor(tx_log_file)
        banks = datasets['trainset_nodes']['bank'].unique()
        for bank in banks:
            os.makedirs(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'clients', bank), exist_ok=True)
            df_nodes = datasets['trainset_nodes']
            df_nodes[df_nodes['bank'] == bank].to_csv(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'trainset_nodes.csv'), index=False)
            unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
            df_edges = datasets['trainset_edges']
            df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_csv(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'trainset_edges.csv'), index=False)
            df_nodes = datasets['valset_nodes']
            df_nodes[df_nodes['bank'] == bank].to_csv(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'valset_nodes.csv'), index=False)
            unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
            df_edges = datasets['valset_edges']
            df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_csv(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'valset_edges.csv'), index=False)
            df_nodes = datasets['testset_nodes']
            df_nodes[df_nodes['bank'] == bank].to_csv(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'testset_nodes.csv'), index=False)
            unique_nodes = df_nodes[df_nodes['bank'] == bank]['account'].unique()
            df_edges = datasets['testset_edges']
            df_edges[(df_edges['src'].isin(unique_nodes)) & (df_edges['dst'].isin(unique_nodes))].to_csv(os.path.join(self.config['preprocess']['preprocessed_data_dir'], 'clients', bank, 'testset_edges.csv'), index=False)
        
        avg_fpr = 0.0
        avg_feature_importances_error = 0.0
        for client in self.config[self.model]['isolated']['clients']:
            os.makedirs(os.path.join(self.bo_dir, f'isolated/{self.model}/clients/{client}'), exist_ok=True)
            storage = None # 'sqlite:///' + os.path.join(self.bo_dir, f'isolated/{self.model}/clients/{client}/hp_study.db')
            params = self.config[self.model]['default'] | self.config[self.model]['isolated']['clients'][client]
            params['save_fpr'] = True
            params['save_feature_importances_error'] = True
            search_space = self.config[self.model]['search_space']
            hyperparamtuner = HyperparamTuner(
                study_name = 'hp_study',
                obj_fn = centralized, # OBS: using centralised here but only with data from one client
                params = params,
                search_space = search_space,
                Client = getattr(clients, self.config[self.model]['default']['client_type']),
                Model = getattr(models, self.model),
                seed = self.seed,
                n_workers = 1,
                storage = storage
            )
            best_trials = hyperparamtuner.optimize(n_trials=self.num_trials_model)
            
            avg_fpr += hyperparamtuner.fpr / len(self.config[self.model]['isolated']['clients'])
            avg_feature_importances_error += hyperparamtuner.feature_importances_error / len(self.config[self.model]['isolated']['clients'])
        
        # classifier = Classifier(dataset, results_dir=self.conf_file.replace('conf.json', ''))
        # model = classifier.train(model=self.model, tune_hyperparameters=True, n_trials=100)
        # score, importances = classifier.evaluate(utility=self.utility)

        # avg_importance = importances.mean()
        # avg_importance_error = abs(avg_importance - importances)
        # sum_avg_importance_error = avg_importance_error.sum()
        
        return abs(avg_fpr-self.target), avg_feature_importances_error
    
    def optimize(self, n_trials:int=10):
        parent_dir = '/'.join(self.data_conf_file.split('/')[:-1])
        storage = 'sqlite:///' + parent_dir + '/amlsim_study.db'
        study = optuna.create_study(storage=storage, sampler=optuna.samplers.TPESampler(multivariate=True), study_name='amlsim_study', directions=['minimize', 'minimize'], load_if_exists=True, pruner=optuna.pruners.HyperbandPruner())
        study.optimize(self.objective, n_trials=n_trials)
        optuna.visualization.matplotlib.plot_pareto_front(study, target_names=[self.utility+'_loss', 'importance_loss'])
        fig_path = parent_dir + '/pareto_front.png'
        plt.savefig(fig_path)
        log_path = parent_dir + '/log.txt'
        with open(log_path, 'w') as f:
            for trial in study.best_trials:
                f.write(f'\ntrial: {trial.number}\n')
                f.write(f'values: {trial.values}\n')
                for param in trial.params:
                    f.write(f'{param}: {trial.params[param]}\n')
        return study.best_trials
    





