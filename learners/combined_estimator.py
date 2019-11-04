# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import numpy as np
import h5py

from collections import defaultdict
from copy import deepcopy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV


def in_range(min, max, v):
    """ Interval classification
     Args:
         min, float, min value
         max, float, max value
         v, value to compare
     Return:
         bool, True if the value is in interval or False if not
         """
    if max == 1.:
        return min <= v <= max
    else:
        return min <= v < max


def build_matrix(proba_matrix):
    """ Take a matrix of probabilities and build the expandable matrix of 0/1 of 10 subdivision 0-100.
    Labels: 1 if prob in interval [x, y[ with 0 <=x <y <= 100.
            0 otherwise
    Args: proba_matrix, probability matrix
    Return: new_matrix, a new probability matrix
    """
    new_matrix = np.zeros((proba_matrix.shape[0], proba_matrix.shape[1] * 11), dtype=bool)
    reference_array = np.arange(0, 1.1, 0.1)
    print(reference_array)
    for i, example in enumerate(proba_matrix):
        idx_start = 0
        idx_end = 11
        for j, el in enumerate(example):
            print("el is", el)
            comparaison_array = np.zeros((11, ), dtype=bool)
            k = 0
            while k < comparaison_array.size - 1:
                comparaison_array[k] = in_range(min=reference_array[k], max=reference_array[k+1], v=el)
                k += 1
            print(comparaison_array)
            new_matrix[i:, idx_start:idx_end] = comparaison_array
            idx_start += 11
            idx_end += 11
    return new_matrix


def pre_fitting(pre_clf_list, x, y, i, g):
    """ Do the pre clf fitting on the group
    Args: pre_clf_list, a class object of CombinedEstimator
          x, dataset with features
          y, labels of dataset
          i, index of the classifier
          g, group index
    Return:
        The fit classifier
    """
    pre_clf_list[i].fit(x[:, g], y)


class CombinedEstimator(BaseEstimator, ClassifierMixin):
    """
    This class learn multiple pre classifiers per groups then use the predictions on each groups to build a new X.
    On the new X (the transformed data, we applied a post classifier of our choice, to simulate the group lasso effect
    """
    def __init__(self, pre_clf=None, post_clf=None, pre_params=None, post_params=None, use_proba=False,
                 n_jobs=8, cv=5, groups=[]):
        assert len(groups) != 0
        assert isinstance(groups, list)
        self.pre_clf = pre_clf
        self.post_clf = post_clf
        self.groups = groups
        if pre_params and post_params is not None:
            assert isinstance(pre_params, dict)
            assert isinstance(post_params, dict)
            self.pre_params = pre_params
            self.post_params = post_params
        self.best_params_pre_clf = []
        self.best_estimator_pre_clf = []
        self.best_params_post_clf = []
        self.best_estimator_post_clf = []
        self.pre_clf_list = []
        self.dict_best_params = defaultdict(dict)
        self.use_proba = use_proba
        self.n_jobs = n_jobs
        self.cv = cv

    def fit(self, x, y):
        """Fit the pre_clf on the data, then fit the final classifier on the new x build by the pre_clf_fit.
        """
        if self.use_proba:
            self.fit_x = np.ones((x.shape[0], len(self.groups)), dtype=float)
            self.pre_clf_list = [GridSearchCV(self.pre_clf, self.pre_params, cv=5, n_jobs=self.n_jobs)
                                 for _ in self.groups]
            print("Pre clf Fitting on the group ...")
            for i, g in enumerate(self.groups):
                self.pre_clf_list[i].fit(x[:, g], y)
                self.fit_x[:, i] = self.pre_clf_list[i].predict_proba(x[:, g]).T
            self.new_matrix = build_matrix(self.fit_x)
            print("Final clf fitting ...")
            self.g_s_post_clf = GridSearchCV(self.post_clf, self.post_params, cv=5, n_jobs=self.n_jobs)
            self.g_s_post_clf.fit(self.new_matrix, y)
        else:
            self.fit_x = np.ones((x.shape[0], len(self.groups)), dtype=float)
            self.pre_clf_list = [GridSearchCV(self.pre_clf, self.pre_params, cv=self.cv, n_jobs=self.n_jobs)
                                 for _ in self.groups]
            print("Pre clf Fitting on the group ...")
            for i, g in enumerate(self.groups):
                self.pre_clf_list[i].fit(x[:, g], y)
                self.fit_x[:, i] = self.pre_clf_list[i].predict(x[:, g]).T
            print("Final clf fitting ...")
            self.g_s_post_clf = GridSearchCV(self.post_clf, self.post_params, cv=self.cv, n_jobs=self.n_jobs)
            self.g_s_post_clf.fit(self.fit_x, y)
        # Best params for plotting
        self.best_params_post_clf.append(self.g_s_post_clf.best_params_)
        self.best_estimator_post_clf.append(self.g_s_post_clf.best_estimator_)
        self.dict_best_params["params"]["post_clf"] = self.best_params_post_clf
        self.dict_best_params["estimator"]["post_clf"] = self.best_estimator_post_clf

        # Todo: Is it important to save the best params/estimator for the pre_clf? I dont think so! Comment!
        # self.best_params_pre_clf = [clf.best_params_ for clf in self.pre_clf_list]
        # self.best_estimator_pre_clf = [clf.best_estimator_ for clf in self.pre_clf_list]
        # self.dict_best_params["params"]["pre_clf"] = self.best_params_pre_clf
        # self.dict_best_params["estimator"]["pre_clf"] = self.best_estimator_pre_clf

    def predict(self, x):
        """Build the new x for the predict phase with the pre_clf and
        predict with the post_clf on the new x build by the pre_clf
        """
        assert isinstance(self.groups, list)
        self.pred_x = np.ones((x.shape[0], len(self.groups)), dtype=float)
        if self.use_proba:
            for i, g in enumerate(self.groups):
                self.pred_x[:, i] = self.pre_clf_list[i].predict_proba(x[:, g]).T
            print("Predictions on going ...")
            self.pred = self.g_s_post_clf.predict(self.pred_x)
        else:
            for i, g in enumerate(self.groups):
                self.pred_x[:, i] = self.pre_clf_list[i].predict(x[:, g]).T
            print("Predictions on going ...")
            self.pred = self.g_s_post_clf.predict(self.pred_x)
        return self.pred

    def dump_accuracy_hdf5_file(self, experiences_name, y):
        """ Write the accuracy score to a hdf5 file. This file will be manage after for plotting
        The accuracy_score_predictor.jscon file contain the score for the pre_clf for each group and the last element is
        the score for the post_clf on the transformation
        """
        from sklearn.metrics import accuracy_score
        assert isinstance(experiences_name, str)
        accuracy_data = np.ones((len(self.groups) + 1, ), dtype=float)
        for i in range(accuracy_data.shape[0] - 1):
            accuracy_data[i] = accuracy_score(y_true=y, y_pred=self.pred_x[:, i])
        accuracy_data[-1] = accuracy_score(y_true=y, y_pred=self.pred)
        h5f = h5py.File('accuracy_scores_predictor.h5', 'a')
        h5f.create_dataset('dataset_%s' % experiences_name, data=accuracy_data)
        h5f.close()


class CombinedEstimatorBetaVersion(CombinedEstimator):
    """
    """
    def fit(self, x, y):
        self.fit_x = np.ones((x.shape[0], len(self.groups)), dtype=float)
        self.pre_clf.set_params(**self.pre_params)
        self.pre_clf_list = [deepcopy(self.pre_clf) for _ in self.groups]
        print("Pre clf Fitting on the group ...")
        for i, g in enumerate(self.groups):
            self.pre_clf_list[i].fit(x[:, g], y)
            self.fit_x[:, i] = self.pre_clf_list[i].predict(x[:, g]).T
        print("Final clf fitting ...")
        self.post_clf.set_params(**self.post_params)
        self.post_clf.fit(self.fit_x, y)

    def predict(self, x):
        self.pred_x = np.ones((x.shape[0], len(self.groups)), dtype=float)
        for i, g in enumerate(self.groups):
            self.pred_x[:, i] = self.pre_clf_list[i].predict(x[:, g]).T
        print("Predictions on going ...")
        self.pred = self.post_clf.predict(self.pred_x)
        return self.pred

    def attributes(self, post_clf_name='scm'):
        """ Return 2 informations: the binary attributes if scm or the feature_importance if DT
        Args: post_clf_name: string, the name of the post clf
                2 values: scm or dt; default = scm
        """
        if post_clf_name == 'scm':
            return self.post_clf.get_stats()
        else:
            return self.post_clf.feature_importances_


class CombinedEstimatorDecisionFunction(CombinedEstimatorBetaVersion):
    """ Little bit comment:
        pre_params must be a dictionnary
        post_params must be a dictionnary
    """

    def fit(self, x, y):
        self.fit_x = np.ones((x.shape[0], len(self.groups)), dtype=float)
        self.pre_clf.set_params(**self.pre_params)
        self.pre_clf_list = [deepcopy(self.pre_clf) for _ in self.groups]
        print("Pre clf Fitting on the group ...")
        for i, g in enumerate(self.groups):
            self.pre_clf_list[i].fit(x[:, g], y)
            self.fit_x[:, i] = self.pre_clf_list[i].decision_function(x[:, g]).T
        print("Final clf fitting ...")
        self.post_clf.set_params(**self.post_params)
        self.post_clf.fit(self.fit_x, y)

    def predict(self, x):
        self.pred_x = np.ones((x.shape[0], len(self.groups)), dtype=float)
        for i, g in enumerate(self.groups):
            self.pred_x[:, i] = self.pre_clf_list[i].decision_function(x[:, g]).T
        print("Predictions on going ...")
        self.pred = self.post_clf.predict(self.pred_x)
        return self.pred

    def attributes(self, post_clf_name='scm'):
        """ Return 2 informations: the binary attributes if scm or the feature_importance if DT
        Args: post_clf_name: string, the name of the post clf
                2 values: scm or dt; default = scm
        """
        if post_clf_name == 'scm':
            return self.post_clf.get_stats()
        else:
            return self.post_clf.feature_importances_
