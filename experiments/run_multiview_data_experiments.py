# -*- coding: utf-8 -*-
__author__ = 'maoss2'
from experiments.utilities import *
from collections import defaultdict
from learners.decisionStumpSCM_learner import DecisionStumpSCMNew
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import logging
import click
import time
import json
import hashlib
import subprocess
import argparse
import sys
import traceback
from pathlib import Path
logging.getLogger('parso.python.diff').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nb_jobs = 30
cv_fold = KFold(n_splits=5, random_state=42)

data_path_graham = '/home/maoss2/project/maoss2/Baptiste_Dataset'
saving_rep = '/home/maoss2/project/maoss2/Baptiste_Dataset'
dataset_baptiste = f'{data_path_graham}/lives_14view_EMF.hdf5'


class LearnMultiViewData(object):
    def __init__(self, parameters, learner, saving_dict, balanced_weights, saving_file="", subsampling=False, 
                 rs=42, nb_jobs=nb_jobs, cv=cv_fold, data_path=dataset_baptiste):
        """
        Learning Class to learn experiment on TN dataset
        Args:
            parameters: dict, parameters for the right learner
            learner: obj, a learner DT, RF or DecisionStumpSCM
            saving_dict: dict, an empty dictionary to save the results to
            balanced_weights: dict, weights to balanced the data
            saving_file: str, output file where to write the saving_dict
            rs: int, the random seed
            nb_jobs: int, number of cpu to run the class
            cv: int of Kfold object, crossvalidation object
            data_path: str, path to dataset
        """
        self.parameters = parameters
        self.learner = learner
        self.saving_file = saving_file
        self.rs = rs
        self.data_path = data_path
        self.saving_dict = saving_dict
        self.nb_jobs = nb_jobs
        self.cv = cv
        self.balanced_weights = balanced_weights
        self.subsampling = subsampling
        self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)

    @staticmethod
    def save_features_selected(classifier, parameters, features_names, dictionary):
        if 'SCM__p' in parameters:
            dictionary['rules'].append(classifier.best_estimator_.named_steps['SCM'].get_stats())
            dictionary['rules_str'].append([(el.__str__(), features_names[el.feature_idx]) for el in
                                            dictionary['rules'][-1]['Binary_attributes']])
            logger.info('{}'.format(dictionary['rules_str']))
        else:
            importances = classifier.best_estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(100):
                if importances[indices[f]] > 0:
                    logger.info("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]],
                                                      features_names[indices[f]]))
            dictionary['importances'].append(importances)
            listes_resultats = [(f + 1, indices[f], importances[indices[f]], features_names[indices[f]]) for f in
                                range(100) if importances[indices[f]] > 0]
            dictionary['rules_str'].append(listes_resultats)

    def fit(self, x, y):
        self.gs_clf.fit(x, y)

    def predict(self, x):
        return self.gs_clf.predict(x)

    def learning(self, features_names, x_train, x_test, y_train, y_test, proteins_ids_train, proteins_ids_test):
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
        proteins_ids_train = proteins_ids_train.astype('U13')
        proteins_ids_test = proteins_ids_test.astype('U13')
        self.saving_dict['proteins_ids_train'] = proteins_ids_train
        self.saving_dict['proteins_ids_test'] = proteins_ids_test
        self.saving_dict['importances'] = []
        self.saving_dict['rules_str'] = []
        self.saving_dict['rules'] = []
        self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                    features_names=features_names, dictionary=self.saving_dict)
        self.saving_file = self.saving_file + '_{}_{}.pck'.format(self.subsampling, str(self.rs))
        with open(self.saving_file, 'wb') as f:
            pickle.dump(self.saving_dict, f)


def run_experiment(nb_repetitions, subsampling=False, data=dataset_baptiste, experiment_name='experiment', saving_rep=saving_rep):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
        subsampling: bool, to subsample or not the data
        experiment_name: str, experiment name for saving file
        return_views: str, which view to run experiment on
        nb_repetitions: int, number of repetitions
        saving_rep: str, saving repertory
    Return:
       Create a saving repertory and put the pickle results in
    """
    assert nb_repetitions >= 1, 'At least one split'
    saving_dict_rf = defaultdict(dict)
    saving_dict_dt = defaultdict(dict)
    saving_dict_scm = defaultdict(dict)
    _,_, _,_, _,_, _,_, _,_, _,_, _,_, _,_, _,_, _,_, _,_, _,_, _,_, _,_, x, _, y, proteins_ids, features_names = load_baptiste_data(dataset=data, subsampling=subsampling) 
    balanced_weights = weighted_sample(y=y, y_target=y)
    balanced_weights = np.unique(balanced_weights)
    balanced_weights = {1: balanced_weights.max() * x.shape[0], -1: balanced_weights.min() * x.shape[0]}
    random.seed(42)
    random_seeds_list = [random.randint(1, 2000) for _ in range(nb_repetitions)]
    try:
        os.mkdir(f'{saving_rep}/{experiment_name}_dt_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        os.chdir(f'{saving_rep}/{experiment_name}_dt_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        for state in range(nb_repetitions):
            clf = LearnMultiViewData(parameters=parameters_dt, 
                                     learner=DecisionTreeClassifier(random_state=42, class_weight=balanced_weights), 
                                     saving_dict=saving_dict_dt, 
                                     balanced_weights=balanced_weights,
                                     saving_file=experiment_name, 
                                     subsampling=subsampling, 
                                     rs=random_seeds_list[state], 
                                     nb_jobs=nb_jobs, 
                                     cv=cv_fold, 
                                     data_path=dataset_baptiste)  
            x_train, x_test, y_train, y_test, proteins_ids_train, proteins_ids_test = \
                train_test_split(x, y, proteins_ids, train_size=0.8, random_state=random_seeds_list[state])
            logger.info('Train set shape {}'.format(x_train.shape))
            logger.info('Test set shape {}'.format(x_test.shape))
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, proteins_ids_train=proteins_ids_train, proteins_ids_test=proteins_ids_test)
    except OSError:
        os.chdir(f'{saving_rep}/{experiment_name}_dt_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        existing_files_list = [fichier for fichier in glob('*.pck')]
        if len(existing_files_list) != 0:
            seeds_already_done = [int(f.split('_')[-1].split('.')[0]) for f in existing_files_list]
        for state in range(nb_repetitions):
            if random_seeds_list[state] in seeds_already_done: continue
            clf = LearnMultiViewData(parameters=parameters_dt, 
                                     learner=DecisionTreeClassifier(random_state=42, class_weight=balanced_weights), 
                                     saving_dict=saving_dict_dt, 
                                     balanced_weights=balanced_weights,
                                     saving_file=experiment_name, 
                                     subsampling=subsampling, 
                                     rs=random_seeds_list[state], 
                                     nb_jobs=nb_jobs, 
                                     cv=cv_fold, 
                                     data_path=dataset_baptiste)  
            x_train, x_test, y_train, y_test, proteins_ids_train, proteins_ids_test = \
                train_test_split(x, y, proteins_ids, train_size=0.8, random_state=random_seeds_list[state])
            logger.info('Train set shape {}'.format(x_train.shape))
            logger.info('Test set shape {}'.format(x_test.shape))
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, proteins_ids_train=proteins_ids_train, proteins_ids_test=proteins_ids_test)
    os.chdir('/home/maoss2/')

    try:
        os.mkdir(f'{saving_rep}/{experiment_name}_scm_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        os.chdir(f'{saving_rep}/{experiment_name}_scm_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        for state in range(nb_repetitions):
            clf = LearnMultiViewData(parameters=parameters_scm, 
                                    learner=Pipeline([('SCM', DecisionStumpSCMNew())]),
                                    saving_dict=saving_dict_scm, 
                                    balanced_weights=balanced_weights,
                                    saving_file=experiment_name, 
                                    subsampling=subsampling, 
                                    rs=random_seeds_list[state], 
                                    nb_jobs=nb_jobs, 
                                    cv=cv_fold, 
                                    data_path=dataset_baptiste)  
            x_train, x_test, y_train, y_test, proteins_ids_train, proteins_ids_test = \
                train_test_split(x, y, proteins_ids, train_size=0.8, random_state=random_seeds_list[state])
            logger.info('Train set shape {}'.format(x_train.shape))
            logger.info('Test set shape {}'.format(x_test.shape))
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, proteins_ids_train=proteins_ids_train, proteins_ids_test=proteins_ids_test)
    except OSError:
        os.chdir(f'{saving_rep}/{experiment_name}_scm_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        existing_files_list = [fichier for fichier in glob('*.pck')]
        if len(existing_files_list) != 0:
            seeds_already_done = [int(f.split('_')[-1].split('.')[0]) for f in existing_files_list]
        for state in range(nb_repetitions):
            if random_seeds_list[state] in seeds_already_done: continue
            clf = LearnMultiViewData(parameters=parameters_scm, 
                                learner=Pipeline([('SCM', DecisionStumpSCMNew())]),
                                saving_dict=saving_dict_scm, 
                                balanced_weights=balanced_weights,
                                saving_file=experiment_name, 
                                subsampling=subsampling, 
                                rs=random_seeds_list[state], 
                                nb_jobs=nb_jobs, 
                                cv=cv_fold, 
                                data_path=dataset_baptiste)
            x_train, x_test, y_train, y_test, proteins_ids_train, proteins_ids_test = \
                train_test_split(x, y, proteins_ids, train_size=0.8, random_state=random_seeds_list[state])
            logger.info('Train set shape {}'.format(x_train.shape))
            logger.info('Test set shape {}'.format(x_test.shape))
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, proteins_ids_train=proteins_ids_train, proteins_ids_test=proteins_ids_test)
    os.chdir('/home/maoss2/')

    try:
        os.mkdir(f'{saving_rep}/{experiment_name}_rf_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        os.chdir(f'{saving_rep}/{experiment_name}_rf_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        for state in range(nb_repetitions):
            clf = LearnMultiViewData(parameters=parameters_rf, 
                                    learner=RandomForestClassifier(random_state=42, class_weight=balanced_weights),
                                    saving_dict=saving_dict_rf,
                                    balanced_weights=balanced_weights,
                                    saving_file=experiment_name, 
                                    subsampling=subsampling, 
                                    rs=random_seeds_list[state], 
                                    nb_jobs=nb_jobs, 
                                    cv=cv_fold, 
                                    data_path=dataset_baptiste)
            x_train, x_test, y_train, y_test, proteins_ids_train, proteins_ids_test = \
                train_test_split(x, y, proteins_ids, train_size=0.8, random_state=random_seeds_list[state])
            logger.info('Train set shape {}'.format(x_train.shape))
            logger.info('Test set shape {}'.format(x_test.shape))
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, proteins_ids_train=proteins_ids_train, proteins_ids_test=proteins_ids_test)
    except OSError:
        os.chdir(f'{saving_rep}/{experiment_name}_rf_subsampling_{subsampling}_nb_repetitions_{nb_repetitions}')
        existing_files_list = [fichier for fichier in glob('*.pck')]
        if len(existing_files_list) != 0:
            seeds_already_done = [int(f.split('_')[-1].split('.')[0]) for f in existing_files_list]
        for state in range(nb_repetitions):
            if random_seeds_list[state] in seeds_already_done: continue
            clf = LearnMultiViewData(parameters=parameters_rf, 
                                    learner=RandomForestClassifier(random_state=42, class_weight=balanced_weights),
                                    saving_dict=saving_dict_rf,
                                    balanced_weights=balanced_weights,
                                    saving_file=experiment_name, 
                                    subsampling=subsampling, 
                                    rs=random_seeds_list[state], 
                                    nb_jobs=nb_jobs, 
                                    cv=cv_fold, 
                                    data_path=dataset_baptiste)
            x_train, x_test, y_train, y_test, proteins_ids_train, proteins_ids_test = \
                train_test_split(x, y, proteins_ids, train_size=0.8, random_state=random_seeds_list[state])
            logger.info('Train set shape {}'.format(x_train.shape))
            logger.info('Test set shape {}'.format(x_test.shape))
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, proteins_ids_train=proteins_ids_train, proteins_ids_test=proteins_ids_test)
    os.chdir('/home/maoss2/')


def main():
    parser = argparse.ArgumentParser(description="Learn Baptiste Experiment")
    parser.add_argument('-subs', '--subsampling', type=bool, default="False")
    parser.add_argument('-nb_r', '--nb_repetitions', type=int, default=1)
    parser.add_argument('-exp_name', '--experiment_name', type=str, default="experiments")
    args = parser.parse_args()
    run_experiment(nb_repetitions=args.nb_repetitions,
                   subsampling=args.subsampling, 
                   data=dataset_baptiste, 
                   experiment_name=args.experiment_name, 
                   saving_rep=saving_rep)
    
if __name__ == "__main__":
    main()