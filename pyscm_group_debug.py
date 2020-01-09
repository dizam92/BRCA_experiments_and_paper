from experiments.utilities import *
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from learners.pyscmGroup import GroupSetCoveringMachineClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from six import iteritems
data = '/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5'
groups = dict_groups_biogrid = pickle.load(open('/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/pathways_biogrid_groups.pck', 'rb'))
x, y, features_names, _ = load_data(data=data, return_views='all')
x = x[:, 0:100]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
features_names = features_names[:100]
features_to_index = {idx: name for idx, name in enumerate(features_names)}
prior_rules = [np.exp(- len(groups[name])) for name in features_names]

class GroupSCM(BaseEstimator, ClassifierMixin):
    """
    A hands on class of SCM using decision stump, built with sklearn format in order to use sklearn function on SCM like
    CV, gridsearch, and so on ...
    """

    def __init__(self, features_to_index, prior_rules, groups, tiebreaker='', model_type='conjunction', p=0.1, max_rules=10, random_state=42):
        super(GroupSCM, self).__init__()
        self.model_type = model_type
        self.p = p
        self.max_rules = max_rules
        self.random_state = random_state
        self.features_to_index = features_to_index
        self.prior_rules = prior_rules
        self.groups = groups
        self.tiebreaker = tiebreaker
        
    def fit(self, X, y):
        self.clf = GroupSetCoveringMachineClassifier(features_to_index=self.features_to_index, 
                                                                  prior_rules=self.prior_rules, 
                                                                  groups=self.groups, 
                                                                  tiebreaker=self.tiebreaker,
                                                                  model_type=self.model_type, 
                                                                  max_rules=self.max_rules, 
                                                                  p=self.p, 
                                                                  random_state=self.random_state)
        self.clf.fit(X=X, y=y)

    def predict(self, X):
        return self.clf.predict(X)

    def set_params(self, **params):
        for key, value in iteritems(params):
            if key == 'p':
                self.p = value
            if key == 'model_type':
                self.model_type = value
            if key == 'max_rules':
                self.max_rules = value
            if key == 'features_to_index':
                self.features_to_index = value
            if key == 'prior_rules':
                self.prior_rules = value
            if key == 'groups':
                self.groups = value
            if key == 'tiebreaker':
                self.tiebreaker = value
        return self

    def get_stats(self):
        return {"Binary_attributes": self.clf.model_.rules}


list_p = [2,3]
list_model = ['conjunction', 'disjunction']
list_rules = [3]
parameters_scm = {'model_type': list_model,
                  'p': list_p,
                  'max_rules': list_rules
                  }
clf = GroupSCM(features_to_index=features_to_index, prior_rules=prior_rules, groups=groups, 
               tiebreaker='', p=1.0, model_type='conjunction',max_rules=1)
g_clf = GridSearchCV(clf, param_grid=parameters_scm, n_jobs=1, cv=3, verbose=1)
print(g_clf)

g_clf.fit(x_train, y_train)
pred = g_clf.predict(x_test)
metrics = get_metrics(y_test=y_test, predictions_binary=pred)
print(metrics)
