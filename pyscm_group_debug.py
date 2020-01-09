from experiments.utilities import *
from sklearn.model_selection import train_test_split
from learners.pyscmGroup import GroupSetCoveringMachineClassifierClassyVersion, my_tiebreaker
data = '/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5'
groups = dict_groups_biogrid = pickle.load(open('/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/pathways_biogrid_groups.pck', 'rb'))
x, y, features_names, _ = load_data(data=data, return_views='all')
x = x[:, 0:100]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
features_names = features_names[:100]
features_to_index = {idx: name for idx, name in enumerate(features_names)}
prior_rules = [np.exp(- len(groups[name])) for name in features_names]

clf = GroupSetCoveringMachineClassifierClassyVersion(p=5.0,
                                                     model_type='conjunction',
                                                     max_rules=5)
clf.fit(x_train, y_train, 
        features_to_index=features_to_index,
        prior_rules=prior_rules, 
        groups=groups,
        tiebreaker='')

print(clf.model_.rules)
