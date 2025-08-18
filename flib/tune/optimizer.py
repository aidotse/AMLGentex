import optuna
from flib.tune.classifier import Classifier # TODO: classifiers should be in flib.models
import matplotlib.pyplot as plt
import json
import os

class Optimizer():
    def __init__(self, data_conf_file, config, generator, preprocessor, target:float, utility:str, model:str='DecisionTreeClassifier', bank=None):
        self.data_conf_file = data_conf_file
        self.config = config
        self.generator = generator
        self.preprocessor = preprocessor
        self.target = target
        self.utility = utility
        self.model = model
        self.bank = bank
    
    def objective(self, trial:optuna.Trial):
        with open(self.data_conf_file, 'r') as f:
            data_config = json.load(f)
        
        data_config['default']['mean_amount_sar'] = trial.suggest_int('mean_amount_sar', data_config['optimisation_bounds']['mean_amount_sar'][0], data_config['optimisation_bounds']['mean_amount_sar'][1])
        data_config['default']['std_amount_sar'] = trial.suggest_int('std_amount_sar', data_config['optimisation_bounds']['std_amount_sar'][0], data_config['optimisation_bounds']['std_amount_sar'][1])
        data_config['default']['mean_outcome_sar'] = trial.suggest_int('mean_outcome_sar', data_config['optimisation_bounds']['mean_outcome_sar'][0], data_config['optimisation_bounds']['mean_outcome_sar'][1])
        data_config['default']['std_outcome_sar'] = trial.suggest_int('std_outcome_sar', data_config['optimisation_bounds']['std_outcome_sar'][0], data_config['optimisation_bounds']['std_outcome_sar'][1])
        data_config['default']['prob_spend_cash'] = trial.suggest_float('prob_spend_cash', data_config['optimisation_bounds']['prob_spend_cash'][0], data_config['optimisation_bounds']['prob_spend_cash'][1])
        data_config['default']['mean_phone_change_frequency_sar'] = trial.suggest_int('mean_phone_change_frequency_sar', data_config['optimisation_bounds']['mean_phone_change_frequency_sar'][0], data_config['optimisation_bounds']['mean_phone_change_frequency_sar'][1])
        data_config['default']['std_phone_change_frequency_sar'] = trial.suggest_int('std_phone_change_frequency_sar', data_config['optimisation_bounds']['std_phone_change_frequency_sar'][0], data_config['optimisation_bounds']['std_phone_change_frequency_sar'][1])
        data_config['default']['mean_bank_change_frequency_sar'] = trial.suggest_int('mean_bank_change_frequency_sar', data_config['optimisation_bounds']['mean_bank_change_frequency_sar'][0], data_config['optimisation_bounds']['mean_bank_change_frequency_sar'][1])
        data_config['default']['std_bank_change_frequency_sar'] = trial.suggest_int('std_bank_change_frequency_sar', data_config['optimisation_bounds']['std_bank_change_frequency_sar'][0], data_config['optimisation_bounds']['std_bank_change_frequency_sar'][1])
        data_config['default']['prob_participate_in_multiple_sars'] = trial.suggest_float('prob_participate_in_multiple_sars', data_config['optimisation_bounds']['prob_participate_in_multiple_sars'][0], data_config['optimisation_bounds']['prob_participate_in_multiple_sars'][1])
        
        with open(self.data_conf_file, 'w') as f:
            json.dump(data_config, f, indent=4)
        
        tx_log_file = self.generator(self.data_conf_file)
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
        for client in self.config[model_type]['isolated']['clients']:
            os.makedirs(os.path.join(args.results_dir, f'isolated/{model_type}/clients/{client}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'isolated/{model_type}/clients/{client}/hp_study.db')
            params = config[model_type]['default'] | config[model_type]['isolated']['clients'][client]
            search_space = config[model_type]['search_space']
            hyperparamtuner = HyperparamTuner(
                study_name = 'hp_study',
                obj_fn = centralized, # OBS: using centralised here but only with data from one client
                params = params,
                search_space = search_space,
                Client = getattr(clients, config[model_type]['default']['client_type']),
                Model = getattr(models, model_type),
                seed = args.seed,
                n_workers = args.n_workers,
                storage = storage
            )
            best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
                
        classifier = Classifier(dataset, results_dir=self.conf_file.replace('conf.json', ''))
        model = classifier.train(model=self.model, tune_hyperparameters=True, n_trials=100)
        score, importances = classifier.evaluate(utility=self.utility)

        avg_importance = importances.mean()
        avg_importance_error = abs(avg_importance - importances)
        sum_avg_importance_error = avg_importance_error.sum()
        
        return abs(score-self.target), sum_avg_importance_error
    
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
    





