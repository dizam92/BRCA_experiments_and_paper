# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'maoss2'
import logging
import numpy as np

from sklearn.utils.validation import check_X_y, check_random_state
from future.utils import iteritems

from pyscm.scm import BaseSetCoveringMachine
from pyscm._scm_utility import find_max as find_max_utility  # cpp extensions
from pyscm.model import ConjunctionModel, DisjunctionModel
from pyscm.rules import DecisionStump

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(filename='pyscm_group.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


# Dont need the x+ 1 into log no more because by default the new scm is working with features weights of 1
def log_lambda(x, ld):
    """
    Weight function using the log, log(x) / reg
    Args: x, float, value to be weighted
         ld: float, reg value
    Returns:
        The surponderation value
    """
    return (np.log(1 + x) / float(np.log(ld))) + 1


def tanh_lambda(x, ld):
    """
       Weight function using the tanh, tanh(x) / reg
       Args: x, float, value to be weighted
            ld: float, reg value
       Returns:
           The surponderation value
       """
    return np.tanh(ld * x) + 1


def arctan_lambda(x, ld):
    """
       Weight function using the arctan
       Args: x, float, value to be weighted
            ld: float, reg value
       Returns:
           The surponderation value
       """
    return ((2 / float(np.pi)) * np.arctan((2 / float(np.pi)) * x * ld)) + 1


def abs_lambda(x, ld):
    """
       Weight function using the abs
       Args: x, float, value to be weighted
            ld: float, reg value
       Returns:
           The surponderation value
       """
    return (x * ld / float(1 + np.abs(x * ld))) + 1


def softmax(x, ld=1.):
    """
       Weight function using the lsoftmax
       Compute softmax values for each sets of scores in x.
       Args: x, float, value to be weighted
            ld: float, reg value
       Returns:
           The surponderation value
       """
    return np.exp(x / ld) / np.sum(np.exp(x / ld), axis=0)


def my_tiebreaker(model_indexes, groups_ids, new_indexes_list_, opti_N, opti_P_bar):
    """
    A new tiebreaker with a prior on the selected features.
    Args:
        model_indexes, list, list of index already selected in the model
        groups_ids, np.array, list of groups ids
        new_indexes_list_, list, the new list of indexes to be evaluated
    Return:
        index, int,
    """
    logging.debug('{} {}'.format(model_indexes, new_indexes_list_))
    model_indexes_groups_ids = groups_ids[model_indexes]
    keep_idx = -1
    for position, idx in enumerate(new_indexes_list_):
        logging.debug("We enter the tiebreaking on the groups section")
        if groups_ids[idx] in model_indexes_groups_ids:
            keep_idx = position
    if keep_idx == -1:
        logging.debug("The tiebreaking on the groups cannot be done, Please revert to the default tiebreaking")
        training_risk_decrease = 1.0 * opti_N - opti_P_bar
        keep_idx = np.where(training_risk_decrease == training_risk_decrease.max())[0][0]
    return keep_idx


class GroupSetCoveringMachineClassifierClassyVersion(BaseSetCoveringMachine):
    """This version works without features duplication """
    def __init__(self, p=1.0, reg_lambdas=2, model_type="conjunction", max_rules=10, reg_function_name='log_lambda',
                 random_state=None):
        super(GroupSetCoveringMachineClassifierClassyVersion, self).__init__(model_type=model_type, max_rules=max_rules,
                                                                             random_state=random_state)
        self.p = p
        self.reg_lambdas = reg_lambdas
        self.reg_function_name = reg_function_name
        if self.reg_function_name == 'log_lambda':
            self.reg_function = log_lambda
        elif self.reg_function_name == 'tanh_lambda':
            self.reg_function = tanh_lambda
        elif self.reg_function_name == 'arctan_lambda':
            self.reg_function = arctan_lambda
        elif self.reg_function_name == 'abs_lambda':
            self.reg_function = abs_lambda
        elif self.reg_function_name == 'softmax_lambda':
            self.reg_function = softmax

    def fit(self, X, y, groups_ids, groups_ids_weights, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.
        tiebreaker: a function defines by the user, If None, the rule that most
            decreases the training error is selected.
        iteration_callback: function(model)
            A function that is called each time a rule is added to the model.

        Returns:
        --------
        self: object
            Returns self.

        """
        # random_state = check_random_state(self.random_state)

        if self.model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif self.model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        logging.debug("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        logging.debug("Validating the input data")
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=np.double)
        self.classes_, y, total_n_ex_by_class = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        logging.debug("The data contains %d examples. Negative class is %s (n: %d) and positive class is %s (n: %d)." %
                      (len(y), self.classes_[0], total_n_ex_by_class[0], self.classes_[1], total_n_ex_by_class[1]))

        # Invert the classes if we are learning a disjunction
        logging.debug("Preprocessing example labels")
        pos_ex_idx, neg_ex_idx = self._get_example_idx_by_class(y)
        y = np.zeros(len(y), dtype=np.int)
        y[pos_ex_idx] = 1
        y[neg_ex_idx] = 0

        # Presort all the features
        logging.debug("Presorting all features")
        X_argsort_by_feature_T = np.argsort(X, axis=0).T.copy()

        # Create an empty model
        logging.debug("Initializing empty model")
        self.model_ = ConjunctionModel() if self.model_type == "conjunction" else DisjunctionModel()

        logging.debug("Training start")
        remaining_example_idx = np.arange(len(y))
        remaining_negative_example_idx = neg_ex_idx
        groups_ids = np.asarray(groups_ids)
        features_weights = np.ones((groups_ids.shape[0]))  # Initialization of groups weights
        while len(remaining_negative_example_idx) > 0 and len(self.model_) < self.max_rules:
            # iteration_info = {"iteration_number": len(self.model_) + 1}
            # iteration_info['list_feat_idx_selected'] = []
            logging.debug("Finding the optimal rule to add to the model")
            opti_utility, \
            opti_feat_idx, \
            opti_threshold, \
            opti_kind, \
            opti_N, \
            opti_P_bar = self._get_best_utility_rules(X,
                                                      y,
                                                      X_argsort_by_feature_T,
                                                      remaining_example_idx.copy(),
                                                      groups_ids=groups_ids,
                                                      groups_ids_weights=groups_ids_weights,
                                                      features_weights=features_weights,
                                                      model_attributes_idx=[el.feature_idx for el in self.model_.rules],
                                                      **utility_function_additional_args)
            # iteration_info["utility_max"] = opti_utility
            # iteration_info["utility_argmax"] = opti_feat_idx
            # logging.debug("Greatest utility is %.5f" % iteration_info["utility_max"])
            # logging.debug("There are %d rules with the same utility." % len(iteration_info["utility_argmax"]))
            # # Do not select rules that cover no negative examples and make errors on no positive examples
            # best_utility_idx = iteration_info["utility_argmax"][
            #     np.logical_or(opti_N != 0, opti_P_bar != 0)]
            # iteration_info['best_utility_idx'] = best_utility_idx
            logging.debug("Tiebreaking. Found %d optimal rules" % len(opti_feat_idx))
            if tiebreaker is None:
                if len(opti_feat_idx) > 1:
                    trainig_risk_decrease = 1.0 * opti_N - opti_P_bar
                    keep_idx = np.where(trainig_risk_decrease == trainig_risk_decrease.max())[0][0]
                else:
                    keep_idx = 0
            else:
                if len(opti_feat_idx) > 1:
                    list_model_idx = [el.feature_idx for el in self.model_.rules]
                    if list_model_idx == []:
                        keep_idx = 0
                    else:
                        keep_idx = tiebreaker(list_model_idx,
                                              groups_ids, opti_feat_idx, opti_N, opti_P_bar)
                        # the tiebreaker must return an index
                else:
                    keep_idx = 0

            stump = DecisionStump(feature_idx=opti_feat_idx[keep_idx], threshold=opti_threshold[keep_idx],
                                  kind="greater" if opti_kind[keep_idx] == 0 else "less_equal")

            logging.debug("The best rule has utility %.3f" % opti_utility)
            self._add_attribute_to_model(stump)

            logging.debug("Discarding all examples that the rule classifies as negative")
            remaining_example_idx = remaining_example_idx[stump.classify(X[remaining_example_idx])]
            remaining_negative_example_idx = remaining_negative_example_idx[stump.classify(X[remaining_negative_example_idx])]
            logging.debug("There are %d examples remaining (%d negatives)" % (len(remaining_example_idx),
                                                                              len(remaining_negative_example_idx)))

            iteration_callback(self.model_)

        logging.debug("Training completed")

        self.rule_importances_ = []  # TODO: implement rule importances (like its done in Kover)

        return self

    def _get_best_utility_rules(self, X, y, X_argsort_by_feature_T, example_idx, groups_ids, groups_ids_weights,
                                features_weights,  model_attributes_idx):
        if len(model_attributes_idx) == 0:
            return find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx,
                                    self.reg_function(x=features_weights, ld=self.reg_lambdas))
        else:
            regle_groups_id = groups_ids[model_attributes_idx]
            regle_groups_id = np.asarray([item for sublist in regle_groups_id for item in sublist])  # flatten the list
            regles_ids, nb_count = np.unique(regle_groups_id, return_counts=True)
            for idx_in_regles_ids, ids_in_regles_ids in enumerate(regles_ids):
                idx_positions_list = []
                idx_positions_list_weights = []
                for idx, el_in_groups_id in enumerate(groups_ids):
                    if len(np.where(el_in_groups_id == ids_in_regles_ids)[0]) != 0:
                        idx_positions_list.append(idx)
                        idx_positions_list_weights.append(groups_ids_weights[idx])  # recuperer les poids de chacun des groupes
                # features_weights[idx_positions_list] += 1.  # on repondère en ajoutant + 1 à chaque fois
                features_weights[idx_positions_list] += idx_positions_list_weights
                # on repondère en ajoutant la valeur précalculée de la taille pour repondérer le groupe
            return find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx,
                                    self.reg_function(x=features_weights, ld=self.reg_lambdas))

