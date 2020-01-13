# -*- coding: utf-8 -*-
__author__ = 'maoss2'
from experiments.utilities import *
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

def run_experiment(data, pathway_file, experiment_name, return_views, nb_repetitions, saving_rep=saving_repository):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
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
    # ********* I ADD A HACKK HERE TO JUST DO THE BEST SEEDS FOR EACH VIEWS TO COMPARE THAT TO THE BEST SCM ***************#
    if return_views=='methyl_rna_iso_mirna':
        random_seeds_list = [564]
    if return_views=='methyl_rna_iso_mirna_snp_clinical':
        random_seeds_list = [229]
    if return_views=='methyl_rna_mirna':
        random_seeds_list = [1310]
    if return_views=='methyl_rna_mirna_snp_clinical':
        random_seeds_list = [52]
    if return_views=='all':
        random_seeds_list = [1310]    
    # *********************** TO BE DELETED OR PUT IN COMMENT AFTER *********************************************#
    # Parameters for GROUP_SCM
    dict_biogrid_groups = pickle.load(open(pathway_file, 'rb'))
    features_to_index = {idx: name for idx, name in enumerate(features_names)}
    prior_rules = [np.exp(- len(dict_biogrid_groups[name])) for name in features_names]
    learner_clf = GroupSCM(features_to_index=features_to_index, prior_rules=prior_rules, groups=dict_biogrid_groups, 
               tiebreaker='', p=1.0, model_type='conjunction', max_rules=10)
    try:
        os.mkdir('{}/group_best_seed_scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        os.chdir('{}/group_best_seed_scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        # os.mkdir('{}/group_scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        # os.chdir('{}/group_scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
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
    except OSError:
        os.chdir('{}/group_best_seed_scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        # os.chdir('{}/group_scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
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
    # os.chdir('/home/maoss2/')


def main_run_experiments_new_labels():
    for view in return_views:
        logger.info('experiment_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna')
        logger.info('{}'.format(view))
        logger.info('--------------------------------------------------------')
        run_experiment(data=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna, 
                    pathway_file=data_repository.format('pathways_biogrid_groups.pck'), 
                    experiment_name='experiment_group_scm_unbalanced_mean_biogrid', 
                    return_views=view, 
                    nb_repetitions=1, 
                    saving_rep=saving_repository)

def main():
    print()

if __name__ == '__main__':
    main_run_experiments_new_labels()
