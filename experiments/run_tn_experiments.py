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

saving_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository'
data_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/{}'
data_tn_new_label_balanced_mean = data_repository.format('triple_neg_new_labels_balanced_mean.h5')
data_tn_new_label_balanced_median = data_repository.format('triple_neg_new_labels_balanced_median.h5')
data_tn_new_label_balanced_zero = data_repository.format('triple_neg_new_labels_balanced_zero.h5')

data_tn_new_label_unbalanced_mean = data_repository.format('triple_neg_new_labels_unbalanced_mean.h5')
data_tn_new_label_unbalanced_median = data_repository.format('triple_neg_new_labels_unbalanced_median.h5')
data_tn_new_label_unbalanced_zero = data_repository.format('triple_neg_new_labels_unbalanced_zero.h5')

data_tn_old_label_balanced_mean = data_repository.format('triple_neg_old_labels_balanced_mean.h5')
data_tn_old_label_balanced_median = data_repository.format('triple_neg_old_labels_balanced_median.h5')
data_tn_old_label_balanced_zero = data_repository.format('triple_neg_old_labels_balanced_zero.h5')

data_tn_old_label_unbalanced_mean = data_repository.format('triple_neg_old_labels_unbalanced_mean.h5')
data_tn_old_label_unbalanced_median = data_repository.format('triple_neg_old_labels_unbalanced_median.h5')
data_tn_old_label_unbalanced_zero = data_repository.format('triple_neg_old_labels_unbalanced_zero.h5')

return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']
datasets_new_labels = [data_tn_new_label_unbalanced_mean, data_tn_new_label_unbalanced_median,
                       data_tn_new_label_unbalanced_zero, data_tn_new_label_balanced_mean,
                       data_tn_new_label_balanced_median, data_tn_new_label_balanced_zero]

datasets_old_labels = [data_tn_old_label_unbalanced_mean, data_tn_old_label_unbalanced_median,
                       data_tn_old_label_unbalanced_zero, data_tn_old_label_balanced_mean,
                       data_tn_old_label_balanced_median, data_tn_old_label_balanced_zero]

parameters_dt = {'max_depth': np.arange(1, 7),
                 'min_samples_split': np.arange(2, 9),
                 'criterion': ['gini', 'entropy']
                 }
parameters_rf = {'max_depth': np.arange(1, 7),
                 'min_samples_split': np.arange(2, 9),
                 'criterion': ['gini', 'entropy'],
                 'n_estimators': [100, 200, 500, 1000]
                 }
param_model_type = ['conjunction', 'disjunction']
# param_p = [0.001, 0.1, 0.178, 0.316, 0.562, 1.0, 1.778, 3.162, 5.623, 10.0, 999999.0]
# param_p = [0.001, 0.05, 0.1, 0.178, 0.25, 0.316, 0.45, 0.562, 0.85, 1.0, 1.5, 1.778, 2, 2.5, 3.162, 4.39, 5.623,
#            6.62, 7.623, 8.386, 9.15, 10.0, 11.0]
param_p = [0.001, 0.1, 0.178, 0.316, 0.45, 0.562, 0.85, 1.0, 1.778, 2.5, 3.162, 4.39, 5.623, 7.623, 10.0, 999999.0]
param_max_attributes = np.arange(1, 11, 1)
parameters_scm = {'SCM__model_type': param_model_type,
                  'SCM__p': param_p,
                  'SCM__max_rules': param_max_attributes
                  }
nb_jobs = 40
cv_fold = KFold(n_splits=5, random_state=42)


class LearnTN(object):
    def __init__(self, parameters, learner, saving_dict, balanced_weights, saving_file="", rs=42, nb_jobs=nb_jobs,
                 cv=cv_fold, data_path=data_tn_new_label_unbalanced_mean, return_views='all'):
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
        self.balanced_weights = balanced_weights
        self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)

    @staticmethod
    def save_features_selected(classifier, parameters, features_names, dictionary):
        if 'SCM__p' in parameters:
            dictionary['rules'].append(classifier.best_estimator_.named_steps['SCM'].get_stats())
            print(dictionary['rules'])
            print(type(dictionary['rules']))
            dictionary['rules_str'].append([(el.__str__(), features_names[el.feature_idx]) for el in
                                            dictionary['rules'][-1]['Binary_attributes']])
        else:
            importances = classifier.best_estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(100):
                if importances[indices[f]] > 0:
                    print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]],
                                                      features_names[indices[f]]))

            dictionary['importances'].append(importances)
            listes_resultats = [(f + 1, indices[f], importances[indices[f]], features_names[indices[f]]) for f in
                                range(100) if importances[indices[f]] > 0]
            dictionary['rules_str'].append(listes_resultats)

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
        self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                    features_names=features_names, dictionary=self.saving_dict)
        self.saving_file = self.saving_file + '_{}_{}.pck'.format(self.return_views, str(self.rs))
        with open(self.saving_file, 'wb') as f:
            pickle.dump(self.saving_dict, f)

    def majority_learning(self, y_train, y_test, patients_train, patients_test,
                          x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical, features_names_methyl,
                          features_names_rna,
                          features_names_rna_iso, features_names_mirna, features_names_snp, features_names_clinical,
                          indices_train, indices_test):
        self.saving_dict['y_test'] = y_test
        patients_train = [el.encode("utf-8") for el in patients_train]
        patients_train = [el.decode("utf-8") for el in patients_train]
        patients_test = [el.encode("utf-8") for el in patients_test]
        patients_test = [el.decode("utf-8") for el in patients_test]
        self.saving_dict['patients_train'] = patients_train
        self.saving_dict['patients_test'] = patients_test
        # initialize to an empty list
        self.saving_dict['importances'] = []
        self.saving_dict['rules_str'] = []
        self.saving_dict['rules'] = []
        train_pred = []
        pred = []
        # Train on methyl
        if x_methyl is not None:
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv,
                                       verbose=1)
            self.gs_clf.fit(x_methyl[indices_train], y_train)
            pred_methyl = self.gs_clf.predict(x_methyl[indices_test])
            y_train_pred_methyl = self.gs_clf.predict(x_methyl[indices_train])
            train_metrics_methyl = get_metrics(y_test=y_train, predictions_binary=y_train_pred_methyl)
            print(self.learner)
            print('*' * 50)
            print('Train metrics', train_metrics_methyl)
            metrics_methyl = get_metrics(y_test=y_test, predictions_binary=pred_methyl)
            print('Test metrics', metrics_methyl)
            print()
            cnf_matrix_methyl = confusion_matrix(y_test, pred_methyl)
            print(cnf_matrix_methyl)
            pred.append(pred_methyl)
            train_pred.append(train_metrics_methyl)
            self.saving_dict['y_pred_methyl'] = pred_methyl
            self.saving_dict['metrics_methyl'] = metrics_methyl
            self.saving_dict['train_metrics_methyl'] = train_metrics_methyl
            self.saving_dict['cv_results_methyl'] = self.gs_clf.cv_results_
            self.saving_dict['best_params_methyl'] = self.gs_clf.best_params_
            self.saving_dict['cnf_matrix_methyl'] = cnf_matrix_methyl
            self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                        features_names=features_names_methyl, dictionary=self.saving_dict)
        # Train on rna
        if x_rna is not None:
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv,
                                       verbose=1)
            self.gs_clf.fit(x_rna[indices_train], y_train)
            pred_rna = self.gs_clf.predict(x_rna[indices_test])
            y_train_pred_rna = self.gs_clf.predict(x_rna[indices_train])
            train_metrics_rna = get_metrics(y_test=y_train, predictions_binary=y_train_pred_rna)
            print(self.learner)
            print('*' * 50)
            print('Train metrics', train_metrics_rna)
            metrics_rna = get_metrics(y_test=y_test, predictions_binary=pred_rna)
            print('Test metrics', metrics_rna)
            print()
            cnf_matrix_rna = confusion_matrix(y_test, pred_rna)
            print(cnf_matrix_rna)
            pred.append(pred_rna)
            train_pred.append(train_metrics_rna)
            self.saving_dict['y_pred_rna'] = pred_rna
            self.saving_dict['metrics_rna'] = metrics_rna
            self.saving_dict['train_metrics_rna'] = train_metrics_rna
            self.saving_dict['cv_results_rna'] = self.gs_clf.cv_results_
            self.saving_dict['best_params_rna'] = self.gs_clf.best_params_
            self.saving_dict['cnf_matrix_rna'] = cnf_matrix_rna
            self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                        features_names=features_names_rna, dictionary=self.saving_dict)
        # Train on rna_iso
        if x_rna_iso is not None:
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv,
                                       verbose=1)
            self.gs_clf.fit(x_rna_iso[indices_train], y_train)
            pred_rna_iso = self.gs_clf.predict(x_rna_iso[indices_test])
            y_train_pred_rna_iso = self.gs_clf.predict(x_rna_iso[indices_train])
            train_metrics_rna_iso = get_metrics(y_test=y_train, predictions_binary=y_train_pred_rna_iso)
            print(self.learner)
            print('*' * 50)
            print('Train metrics', train_metrics_rna_iso)
            metrics_rna_iso = get_metrics(y_test=y_test, predictions_binary=pred_rna_iso)
            print('Test metrics', metrics_rna_iso)
            print()
            cnf_matrix_rna_iso = confusion_matrix(y_test, pred_rna_iso)
            print(cnf_matrix_rna_iso)
            pred.append(pred_rna_iso)
            train_pred.append(train_metrics_rna_iso)
            self.saving_dict['y_pred_rna_iso'] = pred_rna_iso
            self.saving_dict['metrics_rna_iso'] = metrics_rna_iso
            self.saving_dict['train_metrics_rna_iso'] = train_metrics_rna_iso
            self.saving_dict['cv_results_rna_iso'] = self.gs_clf.cv_results_
            self.saving_dict['best_params_rna_iso'] = self.gs_clf.best_params_
            self.saving_dict['cnf_matrix_rna_iso'] = cnf_matrix_rna_iso
            self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                        features_names=features_names_rna_iso, dictionary=self.saving_dict)
        # Train on mirna
        if x_mirna is not None:
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv,
                                       verbose=1)
            self.gs_clf.fit(x_mirna[indices_train], y_train)
            pred_mirna = self.gs_clf.predict(x_mirna[indices_test])
            y_train_pred_mirna = self.gs_clf.predict(x_mirna[indices_train])
            train_metrics_mirna = get_metrics(y_test=y_train, predictions_binary=y_train_pred_mirna)
            print(self.learner)
            print('*' * 50)
            print('Train metrics', train_metrics_mirna)
            metrics_mirna = get_metrics(y_test=y_test, predictions_binary=pred_mirna)
            print('Test metrics', metrics_mirna)
            print()
            cnf_matrix_mirna = confusion_matrix(y_test, pred_mirna)
            print(cnf_matrix_mirna)
            pred.append(pred_mirna)
            train_pred.append(train_metrics_mirna)
            self.saving_dict['y_pred_mirna'] = pred_mirna
            self.saving_dict['metrics_mirna'] = metrics_mirna
            self.saving_dict['train_metrics_mirna'] = train_metrics_mirna
            self.saving_dict['cv_results_mirna'] = self.gs_clf.cv_results_
            self.saving_dict['best_params_mirna'] = self.gs_clf.best_params_
            self.saving_dict['cnf_matrix_mirna'] = cnf_matrix_mirna
            self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                        features_names=features_names_mirna, dictionary=self.saving_dict)

        # Train on snp
        if x_snp is not None:
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv,
                                       verbose=1)
            self.gs_clf.fit(x_snp[indices_train], y_train)
            pred_snp = self.gs_clf.predict(x_snp[indices_test])
            y_train_pred_snp = self.gs_clf.predict(x_snp[indices_train])
            train_metrics_snp = get_metrics(y_test=y_train, predictions_binary=y_train_pred_snp)
            print(self.learner)
            print('*' * 50)
            print('Train metrics', train_metrics_snp)
            metrics_snp = get_metrics(y_test=y_test, predictions_binary=pred_snp)
            print('Test metrics', metrics_snp)
            print()
            cnf_matrix_snp = confusion_matrix(y_test, pred_snp)
            print(cnf_matrix_snp)
            pred.append(pred_snp)
            train_pred.append(train_metrics_snp)
            self.saving_dict['y_pred_snp'] = pred_snp
            self.saving_dict['metrics_snp'] = metrics_snp
            self.saving_dict['train_metrics_snp'] = train_metrics_snp
            self.saving_dict['cv_results_snp'] = self.gs_clf.cv_results_
            self.saving_dict['best_params_snp'] = self.gs_clf.best_params_
            self.saving_dict['cnf_matrix_snp'] = cnf_matrix_snp
            self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                        features_names=features_names_snp, dictionary=self.saving_dict)

        # Train on clinical
        if x_clinical is not None:
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.parameters, n_jobs=self.nb_jobs, cv=self.cv,
                                       verbose=1)
            self.gs_clf.fit(x_clinical[indices_train], y_train)
            pred_clinical = self.gs_clf.predict(x_clinical[indices_test])
            y_train_pred_clinical = self.gs_clf.predict(x_clinical[indices_train])
            train_metrics_clinical = get_metrics(y_test=y_train, predictions_binary=y_train_pred_clinical)
            print(self.learner)
            print('*' * 50)
            print('Train metrics', train_metrics_clinical)
            metrics_clinical = get_metrics(y_test=y_test, predictions_binary=pred_clinical)
            print('Test metrics', metrics_clinical)
            print()
            cnf_matrix_clinical = confusion_matrix(y_test, pred_clinical)
            print(cnf_matrix_clinical)
            pred.append(pred_clinical)
            train_pred.append(train_metrics_clinical)
            self.saving_dict['y_pred_clinical'] = pred_clinical
            self.saving_dict['metrics_clinical'] = metrics_clinical
            self.saving_dict['train_metrics_clinical'] = train_metrics_clinical
            self.saving_dict['cv_results_clinical'] = self.gs_clf.cv_results_
            self.saving_dict['best_params_clinical'] = self.gs_clf.best_params_
            self.saving_dict['cnf_matrix_clinical'] = cnf_matrix_clinical
            self.save_features_selected(classifier=self.gs_clf, parameters=self.parameters,
                                        features_names=features_names_clinical, dictionary=self.saving_dict)

        print("MAJORITY VOTE CONSENSUS")
        pred = np.asarray(pred)
        majority_vote_pred = np.zeros(len(pred[0]))
        for i in range(len(pred[0])):
            most_common, num_most_common = Counter(pred[:, i]).most_common(1)[0]
            majority_vote_pred[i] = most_common

        metrics = get_metrics(y_test=y_test, predictions_binary=majority_vote_pred)
        print('Test metrics', metrics)
        print()
        cnf_matrix = confusion_matrix(y_test, majority_vote_pred)
        self.saving_dict['cnf_matrix'] = cnf_matrix
        self.saving_dict['y_pred'] = majority_vote_pred
        self.saving_dict['metrics'] = metrics

        self.saving_file = self.saving_file + '_{}_{}.pck'.format(self.return_views, str(self.rs))
        with open(self.saving_file, 'wb') as f:
            pickle.dump(self.saving_dict, f)


def run_experiment(data, experiment_name, return_views, nb_repetitions, saving_rep=saving_repository):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
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
    x, y, features_names, patients_names = load_data(data=data, return_views=return_views)
    features_names = [el.encode("utf-8") for el in features_names]
    features_names = [el.decode("utf-8") for el in features_names]
    balanced_weights = weighted_sample(y=y, y_target=y)
    balanced_weights = np.unique(balanced_weights)
    balanced_weights = {1: balanced_weights.max() * x.shape[0], -1: balanced_weights.min() * x.shape[0]}
    random.seed(42)
    random_seeds_list = [random.randint(1, 2000) for _ in range(nb_repetitions)]
    # try:
    #     os.mkdir('{}/dt_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
    #     os.chdir('{}/dt_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
    #     for state in range(nb_repetitions):
    #         clf = LearnTN(parameters=parameters_dt,
    #                       learner=DecisionTreeClassifier(random_state=42, class_weight=balanced_weights),
    #                       saving_dict=saving_dict_dt,
    #                       balanced_weights=balanced_weights,
    #                       saving_file=experiment_name,
    #                       rs=random_seeds_list[state],
    #                       nb_jobs=nb_jobs,
    #                       cv=cv_fold,
    #                       data_path=data,
    #                       return_views=return_views)
    #         x_train, x_test, y_train, y_test, patients_train, patients_test = \
    #             train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
    #         clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
    #                      y_train=y_train, y_test=y_test, patients_train=patients_train,
    #                      patients_test=patients_test)
    # except OSError:
    #     os.chdir('{}/dt_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
    #     for state in range(nb_repetitions):
    #         clf = LearnTN(parameters=parameters_dt,
    #                       learner=DecisionTreeClassifier(random_state=42, class_weight=balanced_weights),
    #                       saving_dict=saving_dict_dt,
    #                       balanced_weights=balanced_weights,
    #                       saving_file=experiment_name,
    #                       rs=random_seeds_list[state],
    #                       nb_jobs=nb_jobs,
    #                       cv=cv_fold,
    #                       data_path=data,
    #                       return_views=return_views)
    #         x_train, x_test, y_train, y_test, patients_train, patients_test = \
    #             train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
    #         clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
    #                      y_train=y_train, y_test=y_test, patients_train=patients_train,
    #                      patients_test=patients_test)
    # os.chdir('/home/maoss2/')

    try:
        os.mkdir('{}/scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        os.chdir('{}/scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_scm,
                          learner=Pipeline([('SCM', DecisionStumpSCMNew())]),
                          saving_dict=saving_dict_scm,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test = \
                train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, patients_train=patients_train,
                         patients_test=patients_test)
    except OSError:
        os.chdir('{}/scm_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_scm,
                          learner=Pipeline([('SCM', DecisionStumpSCMNew())]),
                          saving_dict=saving_dict_scm,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test = \
                train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, patients_train=patients_train,
                         patients_test=patients_test)
    os.chdir('/home/maoss2/')

    try:
        os.mkdir('{}/rf_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        os.chdir('{}/rf_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_rf,
                          learner=RandomForestClassifier(random_state=42, class_weight=balanced_weights),
                          saving_dict=saving_dict_rf,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test = \
                train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, patients_train=patients_train,
                         patients_test=patients_test)
    except OSError:
        os.chdir('{}/rf_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_rf,
                          learner=RandomForestClassifier(random_state=42, class_weight=balanced_weights),
                          saving_dict=saving_dict_rf,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test = \
                train_test_split(x, y, patients_names, train_size=0.8, random_state=random_seeds_list[state])
            clf.learning(features_names=features_names, x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test, patients_train=patients_train,
                         patients_test=patients_test)
    os.chdir('/home/maoss2/')


def run_experiment_majority_vote(data, experiment_name, return_views, nb_repetitions, saving_rep=saving_repository):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
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
    x, x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical, y, features_names, \
    features_names_methyl, features_names_rna, features_names_rna_iso, features_names_mirna, \
    features_names_snp, features_names_clinical, patients_names = load_data(data=data, return_views='majority_vote')
    indices = np.arange(len(y))
    features_names = [el.encode("utf-8") for el in features_names]
    features_names = [el.decode("utf-8") for el in features_names]
    features_names_methyl = [el.encode("utf-8") for el in features_names_methyl]
    features_names_methyl = [el.decode("utf-8") for el in features_names_methyl]
    features_names_rna = [el.encode("utf-8") for el in features_names_rna]
    features_names_rna = [el.decode("utf-8") for el in features_names_rna]
    features_names_rna_iso = [el.encode("utf-8") for el in features_names_rna_iso]
    features_names_rna_iso = [el.decode("utf-8") for el in features_names_rna_iso]
    features_names_mirna = [el.encode("utf-8") for el in features_names_mirna]
    features_names_mirna = [el.decode("utf-8") for el in features_names_mirna]
    features_names_snp = [el.encode("utf-8") for el in features_names_snp]
    features_names_snp = [el.decode("utf-8") for el in features_names_snp]
    features_names_clinical = [el.encode("utf-8") for el in features_names_clinical]
    features_names_clinical = [el.decode("utf-8") for el in features_names_clinical]
    balanced_weights = weighted_sample(y=y, y_target=y)
    balanced_weights = np.unique(balanced_weights)
    balanced_weights = {1: balanced_weights.max() * x.shape[0], -1: balanced_weights.min() * x.shape[0]}
    random.seed(42)
    random_seeds_list = [random.randint(1, 2000) for _ in range(nb_repetitions)]

    try:
        os.mkdir('{}/dt_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        os.chdir('{}/dt_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_dt,
                          learner=DecisionTreeClassifier(random_state=42, class_weight=balanced_weights),
                          saving_dict=saving_dict_dt,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test, indices_train, indices_test = \
                train_test_split(x, y, patients_names, indices, train_size=0.8, random_state=random_seeds_list[state])
            if return_views == 'methyl_rna_iso_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_iso_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'all':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
    except OSError:
        os.chdir('{}/dt_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_dt,
                          learner=DecisionTreeClassifier(random_state=42, class_weight=balanced_weights),
                          saving_dict=saving_dict_dt,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test, indices_train, indices_test = \
                train_test_split(x, y, patients_names, indices, train_size=0.8, random_state=random_seeds_list[state])
            if return_views == 'methyl_rna_iso_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_iso_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'all':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
    os.chdir('/home/maoss2/')

    try:
        os.mkdir('{}/scm_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        os.chdir('{}/scm_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_scm,
                          learner=Pipeline([('SCM', DecisionStumpSCMNew())]),
                          saving_dict=saving_dict_scm,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test, indices_train, indices_test = \
                train_test_split(x, y, patients_names, indices, train_size=0.8, random_state=random_seeds_list[state])
            if return_views == 'methyl_rna_iso_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_iso_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'all':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
    except OSError:
        os.chdir('{}/scm_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_scm,
                          learner=Pipeline([('SCM', DecisionStumpSCMNew())]),
                          saving_dict=saving_dict_scm,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test, indices_train, indices_test = \
                train_test_split(x, y, patients_names, indices, train_size=0.8, random_state=random_seeds_list[state])
            if return_views == 'methyl_rna_iso_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_iso_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'all':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
    os.chdir('/home/maoss2/')

    try:
        os.mkdir('{}/rf_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        os.chdir('{}/rf_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_rf,
                          learner=RandomForestClassifier(random_state=42, class_weight=balanced_weights),
                          saving_dict=saving_dict_rf,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test, indices_train, indices_test = \
                train_test_split(x, y, patients_names, indices, train_size=0.8, random_state=random_seeds_list[state])
            if return_views == 'methyl_rna_iso_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_iso_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'all':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
    except OSError:
        os.chdir('{}/rf_majority_vote_{}_{}_{}'.format(saving_rep, experiment_name, return_views, nb_repetitions))
        for state in range(nb_repetitions):
            clf = LearnTN(parameters=parameters_dt,
                          learner=DecisionTreeClassifier(random_state=42, class_weight=balanced_weights),
                          saving_dict=saving_dict_dt,
                          balanced_weights=balanced_weights,
                          saving_file=experiment_name,
                          rs=random_seeds_list[state],
                          nb_jobs=nb_jobs,
                          cv=cv_fold,
                          data_path=data,
                          return_views=return_views)
            x_train, x_test, y_train, y_test, patients_train, patients_test, indices_train, indices_test = \
                train_test_split(x, y, patients_names, indices, train_size=0.8, random_state=random_seeds_list[state])
            if return_views == 'methyl_rna_iso_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_iso_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=None,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=None,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=None,
                                      features_names_clinical=None,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'methyl_rna_mirna_snp_clinical':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=None,
                                      x_mirna=x_mirna,
                                      x_snp=None,
                                      x_clinical=None,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=None,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
            if return_views == 'all':
                clf.majority_learning(y_train=y_train,
                                      y_test=y_test,
                                      patients_train=patients_train,
                                      patients_test=patients_test,
                                      x_methyl=x_methyl,
                                      x_rna=x_rna,
                                      x_rna_iso=x_rna_iso,
                                      x_mirna=x_mirna,
                                      x_snp=x_snp,
                                      x_clinical=x_clinical,
                                      features_names_methyl=features_names_methyl,
                                      features_names_rna=features_names_rna,
                                      features_names_rna_iso=features_names_rna_iso,
                                      features_names_mirna=features_names_mirna,
                                      features_names_snp=features_names_snp,
                                      features_names_clinical=features_names_clinical,
                                      indices_train=indices_train,
                                      indices_test=indices_test)
    os.chdir('/home/maoss2/')


def main_run_experiments_new_labels():
    for view in return_views:
        for dataset in datasets_new_labels:
            if dataset.find("mean") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_new_label_unbalanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_new_label_balanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("median") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_new_label_unbalanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_new_label_balanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("zero") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_new_label_unbalanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_new_label_balanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)


def main_run_experiments_old_labels():
    for view in return_views:
        for dataset in datasets_old_labels:
            if dataset.find("mean") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_old_label_unbalanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_old_label_balanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("median") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_old_label_unbalanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_old_label_balanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("zero") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_old_label_unbalanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_tn_old_label_balanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)


def main_run_experiments_majority_vote_new_labels():
    for view in return_views:
        for dataset in datasets_new_labels:
            if dataset.find("mean") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_new_label_unbalanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_new_label_balanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("median") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_new_label_unbalanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_new_label_balanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("zero") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_new_label_unbalanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_new_label_balanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)


def main_run_experiments_majority_vote_old_labels():
    for view in return_views:
        for dataset in datasets_old_labels:
            if dataset.find("mean") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_old_label_unbalanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_old_label_balanced_mean',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("median") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_old_label_unbalanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_old_label_balanced_median',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
            if dataset.find("zero") != -1:
                if dataset.find('unbalanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_old_label_unbalanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)
                if dataset.find('balanced') != -1:
                    run_experiment(data=dataset,
                                   experiment_name='experiment_majority_vote_old_label_balanced_zero',
                                   return_views=view,
                                   nb_repetitions=15,
                                   saving_rep=saving_repository)


if __name__ == '__main__':
    main_run_experiments_new_labels()
    # main_run_experiments_majority_vote_new_labels()
    # main_run_experiments_old_labels()
    # main_run_experiments_majority_vote_old_labels()
