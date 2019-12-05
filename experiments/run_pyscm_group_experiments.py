# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import logging
from experiments.utilities import *
from collections import defaultdict
from joblib import Parallel, delayed
from glob import glob
from learners.pyscmGroup import GroupSetCoveringMachineClassifierClassyVersion, my_tiebreaker
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import confusion_matrix

logging.getLogger('parso.python.diff').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nb_jobs = 15
cv_fold = KFold(n_splits=3, random_state=42)


def cv_function(x_train, y_train, param, groups_ids=None, groups_ids_weights=None, tiebreaker=None,
                function_name='log_lambda', n_folds=3):
    """
    Cross Validation Function
    Args:
        x_train: the x_train
        y_train: the y_train
        param: supposed to be a ParameterGrid object of the pre_clf parameters and the post_clf parameters
        groups_ids: list, of the groups ids
        groups_ids_weights: np.array, weights of each groups
        tiebreaker: a tiebreaker, it's a function running the tiebreak. Default: None
        function_name: str, the function for the reponderation, must be in ['log_lambda', 'tanh_lambda',
                                            'arctan_lambda', 'abs_lambda', 'softmax']
        n_folds: int, the number of cv folds
    Returns:
        A list of list containing the param and the mean_cv_score
    """
    assert isinstance(param, dict), 'param must be a dictionary'
    kf = KFold(n_splits=n_folds, random_state=42)
    test_cv_risk = []
    test_cv_risk_imbalanced = []
    for train_index, test_index in kf.split(x_train):
        x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        clf = GroupSetCoveringMachineClassifierClassyVersion(p=param['p'],
                                                             reg_lambdas=param['reg_lambdas'],
                                                             model_type=param['model_type'],
                                                             max_rules=param['max_rules'],
                                                             reg_function_name=function_name,
                                                             random_state=42)
        clf.fit(X=x_train_cv, y=y_train_cv, groups_ids=groups_ids, groups_ids_weights=groups_ids_weights,
                tiebreaker=tiebreaker)
        pred = clf.predict(X=x_test_cv)
        test_cv_risk.append(zero_one_loss(y_target=y_test_cv, y_estimate=pred))
        weights = weighted_sample(y=y_train, y_target=y_test_cv)
        test_cv_risk_imbalanced.append((zero_one_loss_imbalanced(y_target=y_test_cv, y_estimate=pred,
                                                                 sample_weight=weights)))
    return [param, np.round(np.mean(test_cv_risk_imbalanced), 6)]


class GenericGroupScmTcga(object):
    def __init__(self, data_path, function_hps, function_name, random_state_list, saving_repertory, saving_file="",
                 return_views='all', n_folds=3):
        """
        Args:
             data_path: str, path to dataset
             saving_file: str, output file where to write the result
             function_hps: list, hyperparameters values for teh corresponding function
             function_name: str, the function for the reponderation, must be in ['log_lambda', 'tanh_lambda',
                                            'arctan_lambda', 'abs_lambda', 'softmax']
             random_state_list: list, list of random seed where to run the class
             saving_repertory: str, saving repertory
             return_views:  str, which view to run experiment on much match the loader
             n_folds: number of folds for the CV
             pathway_file: str, path to the file containing information on the pathways
             remove_inexistant_group: bool, if true, the model should not take the '_INEXISTANT' group into
                                    consideration. Default False. We work on every type of groups
        """
        self.random_state_list = random_state_list
        self.saving_dict = defaultdict(dict)
        self.function_hps = function_hps
        self.function_name = function_name
        self.saving_file = saving_file
        self.n_folds = n_folds
        self.saving_repertory = saving_repertory
        assert return_views != 'majority_vote', 'The majority vote loader is not supported here'
        self.X, self.y, self.features_names, self.patients_names = load_data(data=data_path, return_views=return_views)
        self.groups_ids = []
        self.groups_ids_weights = []
        self.size_groups = []  # total number of groups

    def learning(self):
        for cpt, indices in enumerate(self.random_state_list):
            random.seed(indices)
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=indices)
            params = list(ParameterGrid({'max_rules': param_max_attributes,
                                         'p': param_p,
                                         'model_type': param_model_type,
                                         'reg_lambdas': self.function_hps
                                         }))
            logger.info('Begin the cross-validation')
            mean_cv_list = Parallel(n_jobs=nb_jobs)(delayed(cv_function)(x_train, y_train, param, self.groups_ids,
                                                                         function_name=self.function_name,
                                                                         n_folds=self.n_folds)
                                                    for param in params)
            logger.info('The cvs scores are: {}'.format(mean_cv_list))
            best_hp_idx = np.argmin(np.asarray([mean_cv_list[i][1] for i in range(len(mean_cv_list))]))
            best_cv_param = mean_cv_list[best_hp_idx][0]
            best_cv_score = mean_cv_list[best_hp_idx][1]
            clf = GroupSetCoveringMachineClassifierClassyVersion(p=best_cv_param['p'],
                                                                 reg_lambdas=best_cv_param['reg_lambdas'],
                                                                 model_type=best_cv_param['model_type'],
                                                                 max_rules=best_cv_param['max_rules'],
                                                                 reg_function_name=self.function_name,
                                                                 random_state=42)
            clf.fit(X=x_train, y=y_train, groups_ids=self.groups_ids, groups_ids_weights=self.groups_ids_weights,
                    tiebreaker=my_tiebreaker)
            pred = clf.predict(X=x_train)
            train_metrics = get_metrics(y_test=y_train, predictions_binary=pred)
            pred = clf.predict(X=x_test)
            test_metrics = get_metrics(y_test=y_test, predictions_binary=pred)
            logger.info('train metrics: {}'.format(train_metrics))
            logger.info('test metrics: {}'.format(test_metrics))
            logger.info('best params: {}'.format(best_cv_param))
            self.saving_dict[indices]['cv_infos'] = [best_cv_param, best_cv_score, mean_cv_list]
            self.saving_dict[indices]['train_metrics'] = train_metrics
            self.saving_dict[indices]['test_metrics'] = test_metrics
            self.saving_dict[indices]['learner_attributes'] = [clf.model_.rules]
            self.saving_dict[indices]['groups_ids_weights'] = self.groups_ids_weights

        self.saving_file = self.saving_file + '_{}.pck'.format(return_views) + time.strftime("%Y%m%d-%H%M%S")
        # TODO: COme back on this one
        with open(os.path.join(self.saving_repertory, self.saving_file), 'wb') as f:
            pickle.dump(self.saving_dict, f)

    def make_group(self):
        pass


class LearnFromMsigGroups(GenericGroupScmTcga):
    def __init__(self, pathway_file, data_path, function_hps, function_name, random_state_list, saving_repertory,
                 saving_file="", return_views='all', n_folds=3, remove_inexistant_group=False):
        """
        Args:
             data_path: str, path to dataset
             saving_file: str, output file where to write the result
             function_hps: list, hyperparameters values for teh corresponding function
             function_name: str, the function for the reponderation, must be in ['log_lambda', 'tanh_lambda',
                                            'arctan_lambda', 'abs_lambda', 'softmax']
             random_state_list: list, list of random seed where to run the class
             saving_repertory: str, saving repertory
             return_views:  str, which view to run experiment on much match the loader
             n_folds: number of folds for the CV
             pathway_file: str, path to the file containing information on the pathways
             remove_inexistant_group: bool, if true, the model should not take the '_INEXISTANT' group into
                                    consideration. Default False. We work on every type of groups
        """
        super(LearnFromMsigGroups, self).__init__(data_path, function_hps, function_name, random_state_list,
                                                  saving_repertory, saving_file, return_views, n_folds)
        self.pathway_file = pathway_file
        self.remove_inexistant_group = remove_inexistant_group

    def make_group(self):
        logger.info('Begin the Group section')
        pathway_data = pd.read_table(self.pathway_file, dtype=str)
        pathway_data_features_name_unique = np.unique(pathway_data['IDS'].values)
        if self.remove_inexistant_group:
            pathway_data_features_name_unique = np.asarray([el for el in pathway_data_features_name_unique
                                                            if el.find('INEXISTANT') == -1])
        features_names_index_to_selected = [i for i, el in enumerate(self.features_names)
                                            if el in pathway_data_features_name_unique]
        self.features_names = self.features_names[features_names_index_to_selected]
        self.X = self.X[:, features_names_index_to_selected]
        pathway_data_features_name_all = pathway_data['IDS'].values
        pathway_data_groups_name_all = pathway_data['G'].values

        self.groups_ids = [[] for _ in range(len(self.features_names))]
        self.groups_ids_weights = [[] for _ in range(len(self.features_names))]
        # To create a list of list structure that will manage the overlapping groups status
        for indexes, ids in enumerate(self.features_names):
            position_ids = np.where(pathway_data_features_name_all == ids)[0]
            for el in pathway_data_groups_name_all[position_ids]:
                self.groups_ids[indexes].append(el)

        groups_ids_choosen, groups_ids_choosen_count = np.unique(
            [el for liste_ids in self.groups_ids for el in liste_ids], return_counts=True)
        dico_number_of_element_in_group = dict(zip(groups_ids_choosen, groups_ids_choosen_count))
        dico_weights_groups = {cle: valeurs / len(groups_ids_choosen) for cle, valeurs in
                               dico_number_of_element_in_group.items()}
        for idx, gps_list in enumerate(self.groups_ids):
            for el in gps_list:
                self.groups_ids_weights[idx].append(dico_weights_groups[el])
        # Je fais l'assumption pour les groupes overlaper de faire l'addition des weights calculés pour ces features
        # donc ca donne le truc de flatten et de faire la somme sur chaque valeur
        self.groups_ids_weights = [np.sum(el) for el in self.groups_ids_weights]
        # Faire flatten la liste et faire la somme dessus pour que pyscmgroup  comme c'est fait actuellement aille
        # juste chercher la bonne affaire à la bonne place straight up
        # TODO: peut changer j'ai pris pour l'instant taille du groupe / nbre total de groupe

    def learning(self):
        self.make_group()
        super(LearnFromMsigGroups, self).learning()


class LearnFromBiogridGroup(GenericGroupScmTcga):
    def __init__(self, pathway_file, data_path, function_hps, function_name, random_state_list, saving_repertory,
                 saving_file="", return_views='all', n_folds=3, remove_inexistant_group=False):
        """
        Args:
             data_path: str, path to dataset
             saving_file: str, output file where to write the result
             function_hps: list, hyperparameters values for teh corresponding function
             function_name: str, the function for the reponderation, must be in ['log_lambda', 'tanh_lambda',
                                            'arctan_lambda', 'abs_lambda', 'softmax']
             random_state_list: list, list of random seed where to run the class
             saving_repertory: str, saving repertory
             return_views:  str, which view to run experiment on much match the loader
             n_folds: number of folds for the CV
             pathway_file: str, path to the file containing information on the pathways
             remove_inexistant_group: bool, if true, the model should not take the '_INEXISTANT' group into
                                    consideration. Default False. We work on every type of groups
        """
        super(LearnFromBiogridGroup, self).__init__(data_path, function_hps, function_name, random_state_list,
                                                    saving_repertory, saving_file, return_views, n_folds)
        self.pathway_file = pathway_file
        self.dict_biogrid_groups = pickle.load(open(self.pathway_file, 'rb'))
        self.remove_inexistant_group = remove_inexistant_group

    def make_group(self):
        logger.info('Begin the Group section')
        logger.info('The assumption is made that each features is a group')
        pathway_data_features_name_unique = np.asarray(list(self.dict_biogrid_groups.keys()))
        if self.remove_inexistant_group:
            pathway_data_features_name_unique = np.asarray([el for el in pathway_data_features_name_unique
                                                            if el.find('INEXISTANT') == -1])
        features_names_index_to_selected = [i for i, el in enumerate(self.features_names)
                                            if el in pathway_data_features_name_unique]
        self.features_names = self.features_names[features_names_index_to_selected]
        self.X = self.X[:, features_names_index_to_selected]

        self.groups_ids = [[] for _ in range(len(self.features_names))]
        self.groups_ids_weights = [[] for _ in range(len(self.features_names))]
        # To create a list of list structure that will manage the overlapping groups status
        for indexes, feature in enumerate(self.features_names):
            self.groups_ids[indexes].extend(self.dict_biogrid_groups[feature])
        groups_ids_choosen, groups_ids_choosen_count = np.unique(
            [el for liste_ids in self.groups_ids for el in liste_ids], return_counts=True)
        dico_number_of_element_in_group = dict(zip(groups_ids_choosen, groups_ids_choosen_count))
        dico_weights_groups = {cle: valeurs / len(groups_ids_choosen) for cle, valeurs in
                               dico_number_of_element_in_group.items()}
        for idx, gps_list in enumerate(self.groups_ids):
            for el in gps_list:
                self.groups_ids_weights[idx].append(dico_weights_groups[el])
        # Je fais l'assumption pour les groupes overlaper de faire l'addition des weights calculés pour ces features
        # donc ca donne le truc de flatten et de faire la somme sur chaque valeur
        self.groups_ids_weights = [np.sum(el) for el in self.groups_ids_weights]
        del self.dict_biogrid_groups
        # Faire flatten la liste et faire la somme dessus pour que pyscmgroup  comme c'est fait actuellement aille
        # juste chercher la bonne affaire à la bonne place straight up
        # TODO: peut changer j'ai pris pour l'instant taille du groupe / nbre total de groupe

    def learning(self):
        self.make_group()
        super(LearnFromBiogridGroup, self).learning()


def run_pyscm_group_experiment_msig(data, experiment_name, pathway_file_name, return_views, nb_of_splits_seeds=5,
                                    function_name='tanh_lambda', remove_inexistant_group=False,
                                    saving_rep=saving_repository):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
        pathway_file_name: str, pathway file
        return_views: str, for the loader
        function_name: str, name of the function
        experiment_name: str, experiment name for saving file
        return_views: str, which view to run experiment on
        saving_rep: str, saving repertory
        nb_of_splits_seeds: int, number of splits seeds to run the pyscmGroup
    Return:
       Create a saving repertory and put the pickle results in
    """
    random.seed(42)
    random_seeds_list = [random.randint(1, 2000) for _ in range(nb_of_splits_seeds)]
    if function_name == 'tanh':
        logger.info('Experimentation of: {}'.format(experiment_name.format(function_name)))
        try:
            os.mkdir('{}/group_scm_{}_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                          function_name, nb_of_splits_seeds))
            os.chdir('{}/group_scm_{}_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                          function_name, nb_of_splits_seeds))
            learner = LearnFromMsigGroups(pathway_file=pathway_file_name,
                                          data_path=data,
                                          function_hps=param_lambdas_tanh,
                                          function_name=function_name,
                                          random_state_list=random_seeds_list,
                                          saving_repertory=saving_rep,
                                          saving_file=experiment_name,
                                          return_views=return_views,
                                          n_folds=3,
                                          remove_inexistant_group=remove_inexistant_group)
            learner.learning()
        except OSError:
            os.chdir('{}/group_scm_{}_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                          function_name, nb_of_splits_seeds))
            learner = LearnFromMsigGroups(pathway_file=pathway_file_name,
                                          data_path=data,
                                          function_hps=param_lambdas_tanh,
                                          function_name=function_name,
                                          random_state_list=random_seeds_list,
                                          saving_repertory=saving_rep,
                                          saving_file=experiment_name,
                                          return_views=return_views,
                                          n_folds=3,
                                          remove_inexistant_group=remove_inexistant_group)
            learner.learning()

    if function_name == 'softmax':
        logger.info('Experimentation of: {}'.format(experiment_name.format(function_name)))
        try:
            os.mkdir('{}/group_scm_{}_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                          function_name, nb_of_splits_seeds))
            os.chdir('{}/group_scm_{}_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                          function_name, nb_of_splits_seeds))
            learner = LearnFromMsigGroups(pathway_file=pathway_file_name,
                                          data_path=data,
                                          function_hps=param_softmax,
                                          function_name=function_name,
                                          random_state_list=random_seeds_list,
                                          saving_repertory=saving_rep,
                                          saving_file=experiment_name,
                                          return_views=return_views,
                                          n_folds=3,
                                          remove_inexistant_group=remove_inexistant_group)
            learner.learning()
        except OSError:
            os.chdir('{}/group_scm_{}_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                          function_name, nb_of_splits_seeds))
            learner = LearnFromMsigGroups(pathway_file=pathway_file_name,
                                          data_path=data,
                                          function_hps=param_softmax,
                                          function_name=function_name,
                                          random_state_list=random_seeds_list,
                                          saving_repertory=saving_rep,
                                          saving_file=experiment_name,
                                          return_views=return_views,
                                          n_folds=3,
                                          remove_inexistant_group=remove_inexistant_group)
            learner.learning()


# # Creer une liste ou une variable pour les pathway et les mettre dans la boucle
# groups_pathway_file = []
#
#
# # TODO: COmplete this
# def main_run_experiments_new_labels():
#     for view in return_views:
#         for dataset in datasets_new_labels:
#             for pathway_name_file in groups_pathway_file:
#                 if dataset.find("mean") != -1:
#                     if dataset.find('unbalanced') != -1:
#                         run_pyscm_group_experiment(data=dataset,
#                                                    experiment_name='expe_group_scm_',
#                                                    pathway_file_name=pathway_name_file,
#                                                    return_views='',
#                                                    nb_of_splits_seeds=5,
#                                                    function_name='tanh_lambda',
#                                                    remove_inexistant_group=False,
#                                                    saving_rep=saving_repository)
#                     if dataset.find('balanced') != -1:
#                         run_pyscm_group_experiment(data=dataset,
#                                                    experiment_name='',
#                                                    pathway_file_name=pathway_name_file,
#                                                    return_views='',
#                                                    nb_of_splits_seeds=5,
#                                                    function_name='tanh_lambda',
#                                                    remove_inexistant_group=False,
#                                                    saving_rep=saving_repository)
#                 if dataset.find("median") != -1:
#                     if dataset.find('unbalanced') != -1:
#                         run_pyscm_group_experiment(data=dataset,
#                                                    experiment_name='',
#                                                    pathway_file_name=pathway_name_file,
#                                                    return_views='',
#                                                    nb_of_splits_seeds=5,
#                                                    function_name='tanh_lambda',
#                                                    remove_inexistant_group=False,
#                                                    saving_rep=saving_repository)
#                     if dataset.find('balanced') != -1:
#                         run_pyscm_group_experiment(data=dataset,
#                                                    experiment_name='',
#                                                    pathway_file_name=pathway_name_file,
#                                                    return_views='',
#                                                    nb_of_splits_seeds=5,
#                                                    function_name='tanh_lambda',
#                                                    remove_inexistant_group=False,
#                                                    saving_rep=saving_repository)
#                 if dataset.find("zero") != -1:
#                     if dataset.find('unbalanced') != -1:
#                         run_pyscm_group_experiment(data=dataset,
#                                                    experiment_name='',
#                                                    pathway_file_name=pathway_name_file,
#                                                    return_views='',
#                                                    nb_of_splits_seeds=5,
#                                                    function_name='tanh_lambda',
#                                                    remove_inexistant_group=False,
#                                                    saving_rep=saving_repository)
#                     if dataset.find('balanced') != -1:
#                         run_pyscm_group_experiment(data=dataset,
#                                                    experiment_name='',
#                                                    pathway_file_name=pathway_name_file,
#                                                    return_views='',
#                                                    nb_of_splits_seeds=5,
#                                                    function_name='tanh_lambda',
#                                                    remove_inexistant_group=False,
#                                                    saving_rep=saving_repository)


def run_pyscm_group_experiment_biogrid(data, experiment_name, pathway_file_name, return_views, nb_of_splits_seeds=5,
                                       function_name='tanh_lambda', remove_inexistant_group=False,
                                       saving_rep=saving_repository):
    """
    Utility function to run experiment on specific data and with specific wiew. To be called in a loop in a main
    Args:
        data: str, data path
        pathway_file_name: str, pathway file
        return_views: str, for the loader
        function_name: str, name of the function
        experiment_name: str, experiment name for saving file
        return_views: str, which view to run experiment on
        saving_rep: str, saving repertory
        nb_of_splits_seeds: int, number of splits seeds to run the pyscmGroup
    Return:
       Create a saving repertory and put the pickle results in
    """
    random.seed(42)
    random_seeds_list = [random.randint(1, 2000) for _ in range(nb_of_splits_seeds)]
    if function_name == 'tanh_lambda':
        logger.info('Experimentation of: {}'.format(experiment_name.format(function_name)))
        try:
            os.mkdir('{}/group_scm_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                       nb_of_splits_seeds))
            os.chdir('{}/group_scm_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                       nb_of_splits_seeds))
            learner = LearnFromBiogridGroup(pathway_file=pathway_file_name,
                                            data_path=data,
                                            function_hps=param_lambdas_tanh,
                                            function_name=function_name,
                                            random_state_list=random_seeds_list,
                                            saving_repertory=saving_rep,
                                            saving_file=experiment_name,
                                            return_views=return_views,
                                            n_folds=3,
                                            remove_inexistant_group=remove_inexistant_group)
            learner.learning()
        except OSError:
            os.chdir('{}/group_scm_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                       nb_of_splits_seeds))
            learner = LearnFromBiogridGroup(pathway_file=pathway_file_name,
                                            data_path=data,
                                            function_hps=param_lambdas_tanh,
                                            function_name=function_name,
                                            random_state_list=random_seeds_list,
                                            saving_repertory=saving_rep,
                                            saving_file=experiment_name,
                                            return_views=return_views,
                                            n_folds=3,
                                            remove_inexistant_group=remove_inexistant_group)
            learner.learning()

    if function_name == 'softmax_lambda':
        logger.info('Experimentation of: {}'.format(experiment_name.format(function_name)))
        try:
            os.mkdir('{}/group_scm_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                       nb_of_splits_seeds))
            os.chdir('{}/group_scm_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                       nb_of_splits_seeds))
            learner = LearnFromMsigGroups(pathway_file=pathway_file_name,
                                          data_path=data,
                                          function_hps=param_softmax,
                                          function_name=function_name,
                                          random_state_list=random_seeds_list,
                                          saving_repertory=saving_rep,
                                          saving_file=experiment_name,
                                          return_views=return_views,
                                          n_folds=3,
                                          remove_inexistant_group=remove_inexistant_group)
            learner.learning()
        except OSError:
            os.chdir('{}/group_scm_{}_{}_{}_{}'.format(saving_rep, experiment_name, function_name, return_views,
                                                       nb_of_splits_seeds))
            learner = LearnFromMsigGroups(pathway_file=pathway_file_name,
                                          data_path=data,
                                          function_hps=param_softmax,
                                          function_name=function_name,
                                          random_state_list=random_seeds_list,
                                          saving_repertory=saving_rep,
                                          saving_file=experiment_name,
                                          return_views=return_views,
                                          n_folds=3,
                                          remove_inexistant_group=remove_inexistant_group)
            learner.learning()

# ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
#                 'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']


def main_unbalanced_all():
    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='all',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='all',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='all',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='all',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)


def main_unbalanced_methyl_rna_iso_mirna():
    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)


def main_unbalanced_methyl_rna_iso_mirna_snp_clinical():
    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_iso_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)


def main_unbalanced_methyl_rna_mirna():
    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)


def main_unbalanced_methyl_rna_mirna_snp_clinical():
    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='tanh_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=False,
                                       saving_rep=saving_repository)

    run_pyscm_group_experiment_biogrid(data=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       experiment_name='experiment_group_scm_unbalanced_mean_biogrid_remove_inexistant',
                                       pathway_file_name=data_repository.format('pathways_biogrid.pck'),
                                       return_views='methyl_rna_mirna_snp_clinical',
                                       nb_of_splits_seeds=5,
                                       function_name='softmax_lambda',
                                       remove_inexistant_group=True,
                                       saving_rep=saving_repository)


if __name__ == '__main__':
    # main_unbalanced_all()
    main_unbalanced_methyl_rna_iso_mirna()
    # main_unbalanced_methyl_rna_iso_mirna_snp_clinical()
    # main_unbalanced_methyl_rna_mirna()
    # main_unbalanced_methyl_rna_mirna_snp_clinical()

