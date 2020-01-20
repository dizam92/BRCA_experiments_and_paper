# -*- coding: utf-8 -*-
__author__ = 'maoss2'
from experiments.utilities import *
from os.path import join, abspath, dirname, exists
from os import makedirs
from collections import defaultdict
from learners.pyscmGroup import GroupSCM
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import argparse
from functools import partial
import multiprocessing
from multiprocessing import Pool
import logging
logging.getLogger('parso.python.diff').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nb_jobs = 30
cv_fold = KFold(n_splits=5, random_state=42)

#TODO: Create a def analyse to link features to their pathways selected

class LearnGroupTN(object):
    def __init__(self, parameters, learner, saving_dict, saving_file="", rs=42, nb_jobs=nb_jobs,
                 cv=cv_fold, data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna, return_views='all'):
        """
        Learning Class to learn experiment on TN dataset
        Args:
            parameters: dict, parameters for the right learner
            learner: obj, a learner DT, RF or DecisionStumpSCM
            saving_dict: dict, an empty dictionary to save the results to
            saving_file: str, output file where to write the saving_dict
            rs: int, the random seed
            nb_jobs: int, number of cpu to run the class
            cv: int of Kfold object, crossvalidation object
            data_path: str, path to dataset
            return_views: str, for the loader
        """
        self.parameters = parameters
        self.learner = learner
        self.saving_file = saving_file
        self.rs = rs
        self.data_path = data_path
        self.return_views = return_views
        self.saving_dict = saving_dict
        self.nb_jobs = nb_jobs
        self.cv = cv
        self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)

    @staticmethod
    def save_features_selected(classifier, parameters, features_names, dictionary):
        dictionary['rules'].append(classifier.best_estimator_.get_stats())
        dictionary['rules_str'].append([(el.__str__(), features_names[el.feature_idx]) for el in
                                        dictionary['rules'][-1]['Binary_attributes']])
        dictionary['groups_rules'].append(classifier.best_estimator_.groups_rules)
        logger.info('{}'.format(dictionary['rules_str']))

    def fit(self, x, y):
        self.gs_clf.fit(x, y)

    def predict(self, x):
        return self.gs_clf.predict(x)

    def learning(self, features_names, x_train, x_test, y_train, y_test, patients_train, patients_test):
        self.gs_clf.fit(x_train, y_train)
        pred = self.gs_clf.predict(x_test)
        y_train_pred = self.gs_clf.predict(x_train)
        train_metrics = get_metrics(y_test=y_train, predictions_binary=y_train_pred)
        print(self.learner)
        print('*' * 50)
        print('Train metrics', train_metrics)
        metrics = get_metrics(y_test=y_test, predictions_binary=pred)
        print('Test metrics', metrics)
        print()
        cnf_matrix = confusion_matrix(y_test, pred)
        print(cnf_matrix)
        self.saving_dict['metrics'] = metrics
        self.saving_dict['train_metrics'] = train_metrics
        self.saving_dict['cv_results'] = self.gs_clf.cv_results_
        self.saving_dict['best_params'] = self.gs_clf.best_params_
        self.saving_dict['cnf_matrix'] = cnf_matrix
        patients_train = [el.encode("utf-8") for el in patients_train]
        patients_train = [el.decode("utf-8") for el in patients_train]
        patients_test = [el.encode("utf-8") for el in patients_test]
        patients_test = [el.decode("utf-8") for el in patients_test]
        self.saving_dict['patients_train'] = patients_train
        self.saving_dict['patients_test'] = patients_test
        self.saving_dict['importances'] = []
        self.saving_dict['rules_str'] = []
        self.saving_dict['rules'] = []
        self.saving_dict['groups_rules'] = []
        self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                    features_names=features_names, dictionary=self.saving_dict)
        self.saving_file = self.saving_file + '_{}_{}.pck'.format(self.return_views, str(self.rs))
        with open(self.saving_file, 'wb') as f:
            pickle.dump(self.saving_dict, f)

def build_priors_rules_vector(psi_g, psi_r, 
                              dictionnary_for_prior_group=f"{data_repository}/biogrid_pathways_dict.pck", 
                              dictionnary_for_prior_rules=f"{data_repository}/pathways_biogrid_groups.pck"):
    """
    Build the vector of the prior rules integreting the prior on the group/pathways 
    Args:
        psi_g, float, hyperparameter to be in the formula calculating p_g, p_g = exp (- psi_g * |g| ) with 0 <= psi_g < 1.0
        psi_ri, float, hyperparameter to be in the formula calculating p_ri, p_ri = exp ( - psi_r * sum_((g \in g_i) 1 / p_g)) with 0 <= psi_r < 1.0 
        dictionnary_for_prior_group, str, path to the dictionnary for generating . Structure must be: d = {'Group_name1': [gen1, gen100,...],  'Group_nameX': [genXXX,...]}
        dictionnary_for_prior_rules, str, path to the dictionnary. Structure must be: d = {'Feature_name1': [Group_name1, Group_name100,...],  'Feature_nameX': [Group_nameXXX,...]}
    Return:
        prior_values_dict_pr_group, dict
        prior_values_dict_pr_rules, dict
    """
    # Build PriorGroups vector, p_g
    dict_pr_group = pickle.load(open(dictionnary_for_prior_group, 'rb'))
    prior_values_dict_pr_group = {k: np.exp(- psi_g * len(v)) for k, v in dict_pr_group.items()}
    dict_pr_rules = pickle.load(open(dictionnary_for_prior_rules, 'rb'))
    prior_values_dict_pr_rules = {k: np.exp(- psi_r * np.sum([1 / prior_values_dict_pr_group[el] if prior_values_dict_pr_group[el] != 0.0 else 0 for el in v])) for k, v in dict_pr_rules.items()}
    return prior_values_dict_pr_group, prior_values_dict_pr_rules
    
    
def run_experiment(return_views, pathway_file, nb_repetitions, update_method='neg_exp', psi_g=0.0, psi_r=0.0, 
                   data=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,  
                   experiment_name='experiment_group_scm', saving_rep=saving_repository):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
        update_method: str, name of the method to update the prioRules in the GroupSCM model
                'neg_exp': p_ri = p_ri * exp( - | g_i  GR | )  update 1
                'pos_exp': p_ri = p_ri * exp( + | g_i  GR | )  update 2
                'neg_exp_group': p_ri = exp ( - |g_i| ) * exp( - | g_i  GR | )  update 3
                'pos_exp_group': p_ri = exp ( - |g_i| ) * exp( + | g_i  GR | )  update 4
        psi_g, float, hyperparameter to be in the formula calculating p_g, p_g = exp (- psi_g * |g| ) with 0 <= psi_g < 1.0
        psi_ri, float, hyperparameter to be in the formula calculating p_ri, p_ri = exp ( - psi_r * sum_((g \in g_i) 1 / p_g)) with 0 <= psi_r < 1.0 
        pathway_file: str, path to the file containing information on the pathways
        experiment_name: str, experiment name for saving file
        return_views: str, which view to run experiment on
        nb_repetitions: int, number of repetitions
        saving_rep: str, saving repertory
    Return:
       Create a saving repertory and put the pickle results in
    """
    assert nb_repetitions >= 1, 'At least one split'
    saving_dict_scm = defaultdict(dict)
    x, y, features_names, patients_names = load_data(data=data, return_views=return_views)
    features_names = [el.encode("utf-8") for el in features_names]
    features_names = [el.decode("utf-8") for el in features_names]
    random.seed(42)
    random_seeds_list = [random.randint(1, 2000) for _ in range(nb_repetitions)]
    # Parameters for GROUP_SCM
    dict_biogrid_groups = pickle.load(open(pathway_file, 'rb'))
    features_to_index = {idx: name for idx, name in enumerate(features_names)}
    _, prior_values_dict_pr_rules = build_priors_rules_vector(psi_g=psi_g, psi_r=psi_r)
    prior_rules = [prior_values_dict_pr_rules[name] for name in features_names]
    learner_clf = GroupSCM(features_to_index=features_to_index, 
                           prior_rules=prior_rules, 
                           update_method=update_method,
                           groups=dict_biogrid_groups,
                           tiebreaker='', 
                           p=1.0, 
                           model_type='conjunction', 
                           max_rules=10)
    results_repertory = join(saving_rep, experiment_name)
    if not exists(results_repertory): makedirs(results_repertory)
    os.chdir(f"{results_repertory}")
    existing_files_list = [fichier for fichier in glob('*.pck')]
    seeds_already_done = None
    if len(existing_files_list) != 0:
        seeds_already_done = [int(f.split('_')[-1].split('.')[0]) for f in existing_files_list]
    for state in range(nb_repetitions):
        if seeds_already_done is not None and random_seeds_list[state] in seeds_already_done: continue
        clf = LearnGroupTN(parameters=parameters_group_scm,
                            learner=learner_clf,
                            saving_dict=saving_dict_scm,
                            saving_file=experiment_name,
                            rs=random_seeds_list[state],
                            nb_jobs=nb_jobs,
                            cv=cv_fold,
                            data_path=data,
                            return_views=return_views)
        x_train, x_test, y_train, y_test, patients_train, patients_test = \
            train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
        logger.info('Train set shape {}'.format(x_train.shape))
        logger.info('Test set shape {}'.format(x_test.shape))
        clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                        y_train=y_train, y_test=y_test, patients_train=patients_train,
                        patients_test=patients_test)


def run_experiment_1st_algorithm(return_views, pathway_file, nb_repetitions, update_method='neg_exp', prior_initialization_type='exp', 
                   data=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,  
                   experiment_name='experiment_group_scm', saving_rep=saving_repository):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
        update_method: str, name of the method to update the prioRules in the GroupSCM model
                'neg_exp': p_ri = p_ri * exp( - | g_i  GR | )  update 1
                'pos_exp': p_ri = p_ri * exp( + | g_i  GR | )  update 2
                'neg_exp_group': p_ri = exp ( - |g_i| ) * exp( - | g_i  GR | )  update 3
                'pos_exp_group': p_ri = exp ( - |g_i| ) * exp( + | g_i  GR | )  update 4
        prior_initialization_type: str, type of initialization of the prior ['exp', 'normal', 'default'] 
                                'exp': exp ( - |g_i| ); 'normal': 1 / |g_i|; 'default': 1
        pathway_file: str, path to the file containing information on the pathways
        experiment_name: str, experiment name for saving file
        return_views: str, which view to run experiment on
        nb_repetitions: int, number of repetitions
        saving_rep: str, saving repertory
    Return:
       Create a saving repertory and put the pickle results in
    """
    assert nb_repetitions >= 1, 'At least one split'
    saving_dict_scm = defaultdict(dict)
    x, y, features_names, patients_names = load_data(data=data, return_views=return_views)
    features_names = [el.encode("utf-8") for el in features_names]
    features_names = [el.decode("utf-8") for el in features_names]
    random.seed(42)
    random_seeds_list = [random.randint(1, 2000) for _ in range(nb_repetitions)]
    # # ********* I ADD A HACKK HERE TO JUST DO THE BEST SEEDS FOR EACH VIEWS TO COMPARE THAT TO THE BEST SCM ***************#
    # if return_views=='methyl_rna_iso_mirna':
    #     random_seeds_list = [564]
    # if return_views=='methyl_rna_iso_mirna_snp_clinical':
    #     random_seeds_list = [229]
    # if return_views=='methyl_rna_mirna':
    #     random_seeds_list = [1310]
    # if return_views=='methyl_rna_mirna_snp_clinical':
    #     random_seeds_list = [52]
    # if return_views=='all':
    #     random_seeds_list = [1310]    
    # # *********************** TO BE DELETED OR PUT IN COMMENT AFTER *********************************************#
    # Parameters for GROUP_SCM
    dict_biogrid_groups = pickle.load(open(pathway_file, 'rb'))
    features_to_index = {idx: name for idx, name in enumerate(features_names)}
    if prior_initialization_type == 'exp':
        prior_rules = [np.exp(- len(dict_biogrid_groups[name])) for name in features_names]
    elif prior_initialization_type == 'normal':
        prior_rules = [1 / len(dict_biogrid_groups[name]) for name in features_names]
    elif prior_initialization_type == 'default':
        prior_rules = [1. for name in features_names]
    else:
        raise ValueError (f"{prior_initialization_type} must be a str and in ['exp', 'normal', 'default']")
    learner_clf = GroupSCM(features_to_index=features_to_index, 
                           prior_rules=prior_rules, 
                           update_method=update_method,
                           groups=dict_biogrid_groups,
                           tiebreaker='', 
                           p=1.0, 
                           model_type='conjunction', 
                           max_rules=10)
    results_repertory = join(saving_rep, experiment_name)
    if not exists(results_repertory): makedirs(results_repertory)
    os.chdir(f"{results_repertory}")
    existing_files_list = [fichier for fichier in glob('*.pck')]
    seeds_already_done = None
    if len(existing_files_list) != 0:
        seeds_already_done = [int(f.split('_')[-1].split('.')[0]) for f in existing_files_list]
    for state in range(nb_repetitions):
        if seeds_already_done is not None and random_seeds_list[state] in seeds_already_done: continue
        clf = LearnGroupTN(parameters=parameters_group_scm,
                            learner=learner_clf,
                            saving_dict=saving_dict_scm,
                            saving_file=experiment_name,
                            rs=random_seeds_list[state],
                            nb_jobs=nb_jobs,
                            cv=cv_fold,
                            data_path=data,
                            return_views=return_views)
        x_train, x_test, y_train, y_test, patients_train, patients_test = \
            train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
        logger.info('Train set shape {}'.format(x_train.shape))
        logger.info('Test set shape {}'.format(x_test.shape))
        clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                        y_train=y_train, y_test=y_test, patients_train=patients_train,
                        patients_test=patients_test)


def main_old():
    parser = argparse.ArgumentParser(description="Learn Group TN Experiment")
    parser.add_argument('-rt', '--return_views', type=str, default="all")
    parser.add_argument('-nb_r', '--nb_repetitions', type=int, default=1)
    parser.add_argument('-g_dict', '--groups_dict', type=str, default=f"{data_repository}/pathways_biogrid_groups.pck")
    parser.add_argument('-u_m', '--update_method', type=str, default="neg_exp")
    parser.add_argument('-init', '--prior_initialization_type', type=str, default="exp")
    parser.add_argument('-data', '--data', type=str, default=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna)
    parser.add_argument('-exp_name', '--experiment_name', type=str, default="experiment_group_scm")
    parser.add_argument('-o', '--saving_rep', type=str, default=saving_repository)
    args = parser.parse_args()
    run_experiment_1st_algorithm(return_views=args.return_views, 
                                pathway_file=args.groups_dict, 
                                nb_repetitions=args.nb_repetitions,
                                update_method=args.update_method, 
                                prior_initialization_type=args.prior_initialization_type, 
                                data=args.data,  
                                experiment_name=args.experiment_name, 
                                saving_rep=args.saving_rep)    
    
    
def main():
    parser = argparse.ArgumentParser(description="Learn Group TN Experiment")
    parser.add_argument('-rt', '--return_views', type=str, default="all")
    parser.add_argument('-nb_r', '--nb_repetitions', type=int, default=1)
    parser.add_argument('-g_dict', '--groups_dict', type=str, default=f"{data_repository}/pathways_biogrid_groups.pck")
    parser.add_argument('-u_m', '--update_method', type=str, default="neg_exp")
    parser.add_argument('-psi_g', '--psi_g', type=float, default=0.0)
    parser.add_argument('-psi_r', '--psi_r', type=float, default=0.0)
    parser.add_argument('-data', '--data', type=str, default=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna)
    parser.add_argument('-exp_name', '--experiment_name', type=str, default="experiment_group_scm")
    parser.add_argument('-o', '--saving_rep', type=str, default=saving_repository)
    args = parser.parse_args()
    run_experiment(return_views=args.return_views, 
                   pathway_file=args.groups_dict, 
                   nb_repetitions=args.nb_repetitions,
                   update_method=args.update_method, 
                   psi_g=args.psi_g,
                   psi_r=args.psi_r,
                   data=args.data,  
                   experiment_name=args.experiment_name, 
                   saving_rep=args.saving_rep)    


if __name__ == '__main__':
    main()
    # main_run_experiments_new_labels()
