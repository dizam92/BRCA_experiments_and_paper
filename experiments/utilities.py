# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import random
import os
import h5py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from itertools import combinations
from copy import deepcopy
from collections import defaultdict
from glob import glob
from collections import Counter
goa_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/goa_human_isoform_valid.gaf'
biogrid_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/BIOGRID-ORGANISM-Homo_sapiens-3.5.178.tab.txt'
genesID_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/Results_genes.txt'

c1_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c1.all.v6.1.symbols.gmt'
c2_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c2.all.v6.1.symbols.gmt'
c3_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c3.all.v6.1.symbols.gmt'
c4_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c4.all.v6.1.symbols.gmt'
c5_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c5.all.v6.1.symbols.gmt'
c6_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c6.all.v6.1.symbols.gmt'
c7_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c7.all.v6.1.symbols.gmt'
hall_mark_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/h.all.v6.1.symbols.gmt'

c2_pickle_dictionary = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c2_curated_genes.pck'
c5_pickle_dictionary = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c5_curated_genes.pck'
list_dict = [c2_pickle_dictionary, c5_pickle_dictionary]

saving_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository'
data_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/{}'

data_tn_new_label_balanced_all_views = data_repository.format('triple_neg_new_labels_balanced_all_views.h5')
data_tn_new_label_balanced_cpg_rna_rna_iso_mirna = data_repository.format('triple_neg_new_labels_balanced_cpg_rna_rna_iso_mirna.h5')
data_tn_new_label_unbalanced_all_views = data_repository.format('triple_neg_new_labels_unbalanced_all_views.h5')
data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna = data_repository.format('triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5')

data_tn_old_label_balanced_all_views = data_repository.format('triple_neg_old_labels_balanced_all_views.h5')
data_tn_old_label_balanced_cpg_rna_rna_iso_mirna = data_repository.format('triple_neg_old_labels_balanced_cpg_rna_rna_iso_mirna.h5')
data_tn_old_label_unbalanced_all_views = data_repository.format('triple_neg_old_labels_unbalanced_all_views.h5')
data_tn_old_label_unbalanced_cpg_rna_rna_iso_mirna = data_repository.format('triple_neg_old_labels_unbalanced_cpg_rna_rna_iso_mirna.h5')

return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']

datasets_new_labels = [data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna, data_tn_new_label_unbalanced_all_views,
                       data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,  data_tn_new_label_balanced_all_views]

datasets_old_labels = [data_tn_old_label_unbalanced_cpg_rna_rna_iso_mirna, data_tn_old_label_unbalanced_all_views,
                       data_tn_old_label_balanced_cpg_rna_rna_iso_mirna, data_tn_old_label_balanced_all_views]

# TODO: Old parameters for learning: I'm changing all of them since there is overfitting (especially in the RF)
# parameters_dt = {'max_depth': np.arange(1, 7),
#                  'min_samples_split': np.arange(2, 9),
#                  'criterion': ['gini', 'entropy']
#                  }
# parameters_rf = {'max_depth': np.arange(1, 7),
#                  'min_samples_split': np.arange(2, 9),
#                  'criterion': ['gini', 'entropy'],
#                  'n_estimators': [100, 200, 500, 1000]
#                  }
# param_model_type = ['conjunction', 'disjunction']
# # param_p = [0.001, 0.1, 0.178, 0.316, 0.562, 1.0, 1.778, 3.162, 5.623, 10.0, 999999.0]
# # param_p = [0.001, 0.05, 0.1, 0.178, 0.25, 0.316, 0.45, 0.562, 0.85, 1.0, 1.5, 1.778, 2, 2.5, 3.162, 4.39, 5.623,
# #            6.62, 7.623, 8.386, 9.15, 10.0, 11.0]
# param_p = [0.001, 0.1, 0.178, 0.316, 0.45, 0.562, 0.85, 1.0, 1.778, 2.5, 3.162, 4.39, 5.623, 7.623, 10.0, 999999.0]

parameters_dt = {'max_depth': np.arange(1, 5),  # Moins de profondeur pour toujours eviter l'overfitting
                 'min_samples_split': np.arange(2, 15),  # Eviter les small value pour eviter l'overfitting
                 'criterion': ['gini', 'entropy']
                 }
parameters_rf = {'max_depth': np.arange(1, 5),
                 'min_samples_split': np.arange(2, 15),
                 'criterion': ['gini', 'entropy'],
                 'n_estimators': [25, 50, 75, 100]
                 }
param_model_type = ['conjunction', 'disjunction']
param_p = [0.1, 0.316, 0.45, 0.562, 0.65, 0.85, 1.0, 2.5, 4.39, 5.623, 7.623, 10.0]

param_max_attributes = np.arange(1, 7, 1)
parameters_scm = {'SCM__model_type': param_model_type,
                  'SCM__p': param_p,
                  'SCM__max_rules': param_max_attributes
                  }
param_lambdas_tanh = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
param_softmax = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


#                                              Parameters for Run Group Experiment                                     #

pathway_file_c1_curated_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_c1_positional_genes.tsv'
pathway_file_c2_curated_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_c2_curated_genes.tsv'
pathway_file_c3_curated_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_c3_motif_genes.tsv'
pathway_file_c4_curated_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_c4_computational_genes.tsv'
pathway_file_c5_curated_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_c5_gene_ontology_genes.tsv'
pathway_file_c6_curated_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_c6_oncogenetic_genes.tsv'
pathway_file_c7_curated_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_c7_immunologic_signatures_genes.tsv'
pathway_file_tcga_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_genes_tcga.tsv'
pathway_file_clusters_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_clusters_genes.tsv'
pathway_file_vant_mat_groups = '/home/maoss2/PycharmProjects/breast_cancer/datasets/pathways.txt'
pathway_file_random_groups = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_random_groups.tsv'
pathway_file_views_group = '/home/maoss2/PycharmProjects/breast_cancer/improvisations/group_scm_experimentation/group_types/pathway_file_views_groups.tsv'


def load_data(data, return_views='all'):
    """
    Load the triple neg datasets
    Args:
        data, str, path to the .h5py dataset
        return_views, str, the type of the x we want to return, default 'all'
                - 'methyl_rna_iso_mirna' for methyl + rna_iso + mirna,
                - 'methyl_rna_iso_mirna_snp_clinical' for methyl + rna_iso + mirna + snp + clinical,
                - 'methyl_rna_mirna' for methyl + rna + mirna,
                - 'methyl_rna_mirna_snp_clinical' for methyl + rna_iso + rna + snp + clinical,
                - 'all' for all
                - 'majority_vote' for the majority vote experiments
    Returns:
        x, y, features_names, patients_names
    """
    assert return_views in ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                            'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all', 'majority_vote']
    d = h5py.File(data, 'r')
    x = d['data'][()]
    y = d['target'][()]
    features_names = d['features_names'][()]
    features_names = np.asarray([el.decode("utf-8") for el in features_names])
    patients_names = d['patients_ids'][()]
    patients_names = np.asarray([el.decode("utf-8") for el in patients_names])
    random.seed(42)
    data_x_y_patients_names = list(zip(x, y, patients_names))
    random.shuffle(data_x_y_patients_names)
    x = [el[0] for el in data_x_y_patients_names]
    y = [el[1] for el in data_x_y_patients_names]
    patients_names = [el[2] for el in data_x_y_patients_names]
    x = np.asarray(x)
    y = np.asarray(y)
    patients_names = np.asarray(patients_names)
    # Indices retrieving
    merge_liste = []
    index_methyl = [idx for idx, el in enumerate(features_names) if el.startswith('cg')]
    index_rna_iso = [idx for idx, el in enumerate(features_names) if el.startswith('uc')]
    index_mirna = [idx for idx, el in enumerate(features_names) if el.startswith('hsa')]
    index_rna = [idx for idx, el in enumerate(features_names) if el.find('|') != -1]
    index_snps = []
    for idx, el in enumerate(features_names):
        splitting = el.split('_')
        if len(splitting) == 6:
            try:
                _ = int(splitting[0])
                index_snps.append(idx)
            except ValueError:  # That means it's a string
                if splitting[0] == 'X':
                    index_snps.append(idx)
    merge_liste.extend(index_methyl)
    merge_liste.extend(index_rna_iso)
    merge_liste.extend(index_mirna)
    merge_liste.extend(index_rna)
    if len(index_snps) != 0:
        merge_liste.extend(index_snps)
    index_clinical = [idx for idx in range(len(features_names)) if idx not in merge_liste]
    # Methyl
    x_methyl = x[:, index_methyl]
    features_names_methyl = features_names[index_methyl]
    # RNA ISO
    x_rna_iso = x[:, index_rna_iso]
    features_names_rna_iso = features_names[index_rna_iso]
    # MiRNA
    x_mirna = x[:, index_mirna]
    features_names_mirna = features_names[index_mirna]
    # SNP
    if len(index_snps) != 0:
        x_snp = x[:, index_snps]
        features_names_snp = features_names[index_snps]
    # Clinical
    x_clinical = x[:, index_clinical]
    features_names_clinical = features_names[index_clinical]
    # RNA
    x_rna = x[:, index_rna]
    features_names_rna = features_names[index_rna]
    # Normalization
    x_rna_iso = StandardScaler().fit_transform(x_rna_iso)
    x_mirna = StandardScaler().fit_transform(x_mirna)
    x_rna = StandardScaler().fit_transform(x_rna)
    if return_views == 'methyl_rna_iso_mirna':
        x_methyl_rna_iso_mirna = np.hstack((x_methyl, x_rna_iso, x_mirna))
        features_names_rna_iso_mirna = np.hstack((features_names_methyl, features_names_rna_iso, features_names_mirna))
        x = x_methyl_rna_iso_mirna
        features_names = features_names_rna_iso_mirna
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'methyl_rna_iso_mirna_snp_clinical':
        if len(index_snps) != 0:
            x_methyl_rna_iso_mirna_snp_clinical = np.hstack((x_methyl, x_rna_iso, x_mirna, x_snp, x_clinical))
            features_names_rna_iso_mirna_snp_clinical = np.hstack((features_names_methyl, features_names_rna_iso,
                                                               features_names_mirna, features_names_snp,
                                                               features_names_clinical))
        else:
            x_methyl_rna_iso_mirna_snp_clinical = np.hstack((x_methyl, x_rna_iso, x_mirna, x_clinical))
            features_names_rna_iso_mirna_snp_clinical = np.hstack((features_names_methyl, features_names_rna_iso,
                                                               features_names_mirna, features_names_clinical))
        x = x_methyl_rna_iso_mirna_snp_clinical
        features_names = features_names_rna_iso_mirna_snp_clinical
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'methyl_rna_mirna':
        x_methyl_rna_mirna = np.hstack((x_methyl, x_rna, x_mirna))
        features_names_rna_mirna = np.hstack((features_names_methyl, features_names_rna, features_names_mirna))
        x = x_methyl_rna_mirna
        features_names = features_names_rna_mirna
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'methyl_rna_mirna_snp_clinical':
        if len(index_snps) != 0:
            x_methyl_rna_mirna_snp_clinical = np.hstack((x_methyl, x_rna, x_mirna, x_snp, x_clinical))
            features_names_rna_mirna_snp_clinical = np.hstack((features_names_methyl, features_names_rna,
                                                               features_names_mirna, features_names_snp,
                                                               features_names_clinical))
        else:
            x_methyl_rna_mirna_snp_clinical = np.hstack((x_methyl, x_rna, x_mirna, x_clinical))
            features_names_rna_mirna_snp_clinical = np.hstack((features_names_methyl, features_names_rna,
                                                               features_names_mirna, features_names_clinical))
        x = x_methyl_rna_mirna_snp_clinical
        features_names = features_names_rna_mirna_snp_clinical
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'all':
        if len(index_snps) != 0:
            x_all = np.hstack((x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical))
        else:
            x_all = np.hstack((x_methyl, x_rna, x_rna_iso, x_mirna, x_clinical))
            features_names =  np.hstack((features_names_methyl, features_names_rna,features_names_rna_iso, 
                features_names_mirna, features_names_clinical))
        x = x_all
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'majority_vote':
        if len(index_snps) != 0:
            x_all = np.hstack((x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical))
            x = x_all
            return x, x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical, y, features_names, features_names_methyl, \
                   features_names_rna, features_names_rna_iso, features_names_mirna, features_names_snp, \
                   features_names_clinical, patients_names
        else:
            x_all = np.hstack((x_methyl, x_rna, x_rna_iso, x_mirna, x_clinical))
            x = x_all
            return x, x_methyl, x_rna, x_rna_iso, x_mirna, x_clinical, y, features_names, features_names_methyl, \
                   features_names_rna, features_names_rna_iso, features_names_mirna, features_names_clinical, patients_names


def results_analysis(directory, output_text_file):
    """
    An utility function to run the results analysis and output them in a readable way
    Args:
        directory, str, path to the directory containing the pickle files
        data_path, str, path to the data of interest to be loaded to run the analysis
        output_text_file, str, file where to write the results to
    Returns:
        Write results to text file
    """
    os.chdir('{}'.format(directory))
    metrics_train = []
    metrics_test = []
    features_retenus = []
    model_comptes = []
    cnt = Counter()
    cnt_rf = Counter()
    for fichier in glob("*.pck"):
        f = open(fichier, 'rb')
        d = pickle.load(f)
        metrics_train.append(d['train_metrics'])
        metrics_test.append(d['metrics'])
        features_retenus.append(d['rules_str'])
    accuracy_train = [el['accuracy'] for el in metrics_train]
    accuracy_test = [el['accuracy'] for el in metrics_test]
    f1_score_train = [el['f1_score'] for el in metrics_train]
    f1_score_test = [el['f1_score'] for el in metrics_test]
    precision_train = [el['precision'] for el in metrics_train]
    precision_test = [el['precision'] for el in metrics_test]
    recall_train = [el['recall'] for el in metrics_train]
    recall_test = [el['recall'] for el in metrics_test]
    if directory.find('dt') != -1:
        for model in features_retenus:
            temp = []
            for el in model[0]:
                if el[2] > 0:
                    temp.append(el)
            var = ''
            for i, el in enumerate(temp):
                var += '_{}'.format(el[3])
                if i == 2:
                    break
            model_comptes.append(var)
    if directory.find('rf') != -1:
        for model in features_retenus:
            var = ''
            for el in model[0][:3]:
                var += '_{}'.format(el[3])
            model_comptes.append(var)

    if directory.find('scm') != -1:
        for model in features_retenus:
            temp = []
            for el in model[0]:
                temp.append(el[1])
            var = ''
            for el in temp:
                var += '_{}'.format(el)
            model_comptes.append(var)

    if directory.find('rf') != -1:
        features_retenus_flatten = [el[3] for liste in features_retenus for el in liste[:50]]
        for el in features_retenus_flatten:
            cnt_rf[el] += 1

        # most_common_features_names = np.asarray([el[0] for el in most_common_features])
        # most_common_features_values = np.asarray([el[1] for el in most_common_features])

        # nbResults = len(most_common_features)
        # figKW = {"figsize": (nbResults, 8)}
        # f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        # barWidth = 0.35
        # ax.set_title('{}'.format('RF 50 most common features in the 50 best features for each experiment'))
        # rects = ax.bar(range(nbResults), most_common_features_values, barWidth, color="r")
        # autolabel(rects, ax)
        # # ax.legend(rects[0], 'Counts')
        # # ax.set_ylim(-0.1, 1.1)
        # ax.set_xticks(np.arange(nbResults) + barWidth)
        # ax.set_xticklabels(most_common_features_names, rotation="vertical")
        # plt.tight_layout()
        # f.savefig('RF_50_Most_Common_Features' + time.strftime("%Y%m%d-%H%M%S") + ".png")
        # plt.close()

    with open(output_text_file, 'a+') as f:
        f.write('Repository:{}\n'.format(directory))
        f.write('TRAINING RESULTS\n')
        f.write('-'*50)
        f.write('\n')
        f.write('Training: Accuracy mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(accuracy_train), 4), np.round(np.std(accuracy_train), 4), np.round(np.max(accuracy_train), 4),
            np.round(np.min(accuracy_train), 4), np.round(np.median(accuracy_train), 4)))
        f.write('Training: f1_score mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(f1_score_train), 4), np.round(np.std(f1_score_train), 4), np.round(np.max(f1_score_train), 4),
            np.round(np.min(f1_score_train), 4), np.round(np.median(f1_score_train), 4)))
        f.write('Training: Precision mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(precision_train), 4), np.round(np.std(precision_train), 4),
            np.round(np.max(precision_train), 4),
            np.round(np.min(precision_train), 4), np.round(np.median(precision_train), 4)))
        f.write('Training: Recall mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(recall_train), 4), np.round(np.std(recall_train), 4), np.round(np.max(recall_train), 4),
            np.round(np.min(recall_train), 4), np.round(np.median(recall_train), 4)))
        f.write('TESTS RESULTS\n')
        f.write('-*50\n')
        f.write('Test: Accuracy mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(accuracy_test), 4), np.round(np.std(accuracy_test), 4), np.round(np.max(accuracy_test), 4),
            np.round(np.min(accuracy_test), 4), np.round(np.median(accuracy_test), 4)))
        f.write('Test: f1_score mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(f1_score_test), 4), np.round(np.std(f1_score_test), 4), np.round(np.max(f1_score_test), 4),
            np.round(np.min(f1_score_test), 4), np.round(np.median(f1_score_test), 4)))
        f.write('Test: Precision mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(precision_test), 4), np.round(np.std(precision_test), 4), np.round(np.max(precision_test), 4),
            np.round(np.min(precision_test), 4), np.round(np.median(precision_test), 4)))
        f.write('Test: Recall mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(recall_test), 4), np.round(np.std(recall_test), 4), np.round(np.max(recall_test), 4),
            np.round(np.min(recall_test), 4), np.round(np.median(recall_test), 4)))

        for el in model_comptes:
            cnt[el] += 1
        f.write('Most frequent model\n'.format(cnt.most_common(10))) 
        most_common_features = cnt_rf.most_common(50)
        f.write('Most frequent Features\n'.format(most_common_features)) 
    os.chdir(saving_repository)


def main_run_analysis():
    """ 
    Main utility function to run the run analysis on every repository 

    """
    list_of_directories = os.listdir('./')
    output_text_file = saving_repository + '/results_analysis.txt'
    for directory in list_of_directories:
        if directory not in ['.DS_Store', '._.DS_Store', 'results_analysis.txt']:

            results_analysis(directory=directory, output_text_file=output_text_file)


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, "%.2f" % height, ha='center', va='bottom')


def weighted_sample(y, y_target):
    """ Build a weighted sample array
    Args:
         y: the original target array
         y_target: the y for which we want to build a sample size
    Returns:
        y_samples, weighted
    """
    y = np.asarray(y)
    y_target = np.asarray(y_target)
    y_samples = np.ones((y_target.size,))
    nb_classes = np.unique(y).size
    negative_type = -1
    if 0 in np.unique(y):
        negative_type = 0
    nb_negatives = np.where(y == negative_type)[0].size
    nb_positives = np.where(y == 1)[0].size
    c_n = 1.0 / (nb_classes * nb_negatives)
    c_p = 1.0 / (nb_classes * nb_positives)
    y_samples[np.where(y_target == negative_type)[0]] *= c_n
    y_samples[np.where(y_target == 1)[0]] *= c_p
    return y_samples


def get_metrics(y_test, predictions_binary):
    """Compute the metrics for classifiers predictors
    Args:
        y_test: real labels
        predictions_binary: the predicted labels
    Return: metrics: a dictionnary of the metrics
    """
    y_test = np.asarray(y_test, dtype=np.float)
    predictions_binary = np.asarray(predictions_binary, dtype=np.float)
    metrics = {"accuracy": accuracy_score(y_test, predictions_binary),
               "f1_score": f1_score(y_test, predictions_binary),
               "precision": precision_score(y_test, predictions_binary),
               "recall": recall_score(y_test, predictions_binary)
               }
    return metrics


def get_metrics_balanced(y_test, predictions_binary, weights):
    """Compute the balanced metrics for classifiers predictors
    Args: y_test: real labels
            predictions_binary: the predicted labels
            weights: the weights used for learning
    Return: metrics: a dictionnary of the metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    y_test = np.asarray(y_test, dtype=np.float)
    predictions_binary = np.asarray(predictions_binary, dtype=np.float)

    metrics = {"accuracy": accuracy_score(y_test, predictions_binary, sample_weight=weights),
               "f1_score": f1_score(y_test, predictions_binary),
               "precision": precision_score(y_test, predictions_binary),
               "recall": recall_score(y_test, predictions_binary)
               }
    return metrics


def zero_one_loss(y_target, y_estimate):
    if len(y_target) == 0:
        return 0.0
    return np.mean(y_target != y_estimate)


def zero_one_loss_imbalanced(y_target, y_estimate, sample_weight):
    """ Zero one loss for imbalanced dataset"""
    if len(y_target) == 0:
        return 0.0
    assert len(sample_weight) == len(y_target) == len(y_estimate), 'Sample weight and y_target must have the same shape'
    return np.mean(np.dot((y_target != y_estimate).astype(np.int), sample_weight))


class ExtractGroupsToPickle:
    def __init__(self, fichier, saving_file):
        """
        Extract file to pickle
        Args:
            fichier: str, pathway files
            saving_file: str, name of the saving file
        """
        self.fichier = fichier
        self.saving_file = saving_file

    def extract(self):
        saving_dict = defaultdict(dict)
        # load the genes from TCGA
        temp = pd.read_table(genesID_file)
        tcga_genes_ids = temp.values  # shape (20501, 1)
        tcga_genes_ids = np.asarray([el[0] for el in tcga_genes_ids])
        # load the group
        with open(self.fichier, 'r') as f:
            lignes = f.readlines()
            for l in lignes:
                # split the lines
                splitting_results = re.split('\t', l[:-1])
                pathway_name = splitting_results[0]
                # pathway_website = splitting_results[1]
                genes = splitting_results[2:]
                genes = [el for el in genes if el in tcga_genes_ids]
                saving_dict[pathway_name] = genes

        with open(self.saving_file, 'w') as f:
            pickle.dump(saving_dict, f)


def main_extraction_building():
    extractor = ExtractGroupsToPickle(fichier=c1_file, saving_file='c1_positional_genes.pck')
    extractor.extract()
    extractor = ExtractGroupsToPickle(fichier=c2_file, saving_file='c2_curated_genes.pck')
    extractor.extract()
    extractor = ExtractGroupsToPickle(fichier=c3_file, saving_file='c3_motif_genes.pck')
    extractor.extract()
    extractor = ExtractGroupsToPickle(fichier=c4_file, saving_file='c4_computational_genes.pck')
    extractor.extract()
    extractor = ExtractGroupsToPickle(fichier=c5_file, saving_file='c5_gene_ontology_genes.pck')
    extractor.extract()
    extractor = ExtractGroupsToPickle(fichier=c6_file, saving_file='c6_oncogenetic_genes.pck')
    extractor.extract()
    extractor = ExtractGroupsToPickle(fichier=c7_file, saving_file='c7_immunologic_signatures_genes.pck')
    extractor.extract()
    extractor = ExtractGroupsToPickle(fichier=hall_mark_file, saving_file='hall_mark_genes.pck')
    extractor.extract()


def construction_pathway_gene_groups_tcga(data_path=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                          return_views='all',
                                          output_file_name='pathway_file_genes_tcga'):
    """
    Construct group based on the genes to which the future is related should receive file with future like this
    ('feature_GENES')
    Args:
        data_path: str, data path
        return_views: str, correct view for the group
        output_file_name: str, output file name
    Returns:
        output_file_name
    """
    _, _, features_names, _ = load_data(data=data_path, return_views=return_views)
    groups_dict = {}
    for el in features_names:
        cles = [el.split('_')[-1]]
        if cles[0].find(';') != -1:
            cles = cles[0].split(';')
        for cle in cles:
            if cle not in groups_dict.keys():
                groups_dict[cle] = [el]
            else:
                groups_dict[cle].append(el)
    output_file_name = output_file_name + '_{}.tsv'.format(return_views)
    with open(output_file_name, 'w') as f:
        f.write('G\tIDS\n')
        for item in groups_dict.items():
            for gene in item[1]:
                f.write('{}\t{}\n'.format(item[0], gene))


def construction_pathway_clusters_groups(data_path=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                         model_loaded=False,
                                         return_views='all',
                                         output_file_name='pathway_file_clusters_genes',
                                         model_agglomeration_file_name='feature_agglomeration_model.pck'):
    """
       Construct group based on clustering
       Args:
           data_path: str, data path
           model_loaded: bool, if true, load the model_agglomeration_file_name, if false re-do the fit
           return_views: str, correct view for the group
           output_file_name: str, output file name
           model_agglomeration_file_name: path to .pck file if we already run the feature agglomeration fit
       Returns:
           output_file_name
       """
    from sklearn.cluster import FeatureAgglomeration
    agglo = FeatureAgglomeration(n_clusters=1000)
    x, y, features_names, _ = load_data(data=data_path, return_views=return_views)
    output_file_name = output_file_name + '_{}.tsv'.format(return_views)
    if model_loaded:
        assert model_agglomeration_file_name != '', 'You should give the model agglomeration name file'
        f = open(model_agglomeration_file_name, 'rb')
        agglo = pickle.load(f)
        groups_and_features = list(zip(features_names, agglo.labels_))
        with open(output_file_name, 'w') as f:
            f.write('G\tIDS\n')
            for zip_el in groups_and_features:
                f.write('{}\t{}\n'.format(zip_el[1], zip_el[0]))
    else:
        f = open(model_agglomeration_file_name, 'wb')
        agglo.fit(x)
        pickle.dump(agglo, f)
        groups_and_features = list(zip(features_names, agglo.labels_))
        with open(output_file_name, 'w') as f:
            f.write('G\tIDS\n')
            for zip_el in groups_and_features:
                f.write('{}\t{}\n'.format(zip_el[1], zip_el[0]))


def construction_pathway_random_groups(data_path=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                       nb_of_groups=1000,
                                       return_views='all',
                                       output_file_name='pathway_file_random_groups'):
    """
    Construct group based on random groups
    Args:
        data_path: str, data path
        return_views: str, correct view for the group
        output_file_name: str, output file name
        nb_of_groups: int, nbr of groups we randomly want to build
    Returns:
        output_file_name
    """
    _, _, features_names, _ = load_data(data=data_path, return_views=return_views)
    features_names_copy = deepcopy(features_names)
    random.seed(42)
    np.random.seed(42)
    nbr_element_per_groups = np.arange(2, 2000)  # Nbr of feature in the groups
    elements_choisis = []
    output_file_name = output_file_name + '_{}_{}.tsv'.format(nb_of_groups, return_views)
    with open(output_file_name, 'w') as f:
        f.write('G\tIDS\n')
        for i in range(nb_of_groups):
            taille_groupe = np.random.choice(nbr_element_per_groups)
            group = random.sample(list(features_names), taille_groupe)
            elements_choisis.extend(group)
            for el in group:
                f.write('group_{}\t{}\n'.format(i, el))
            print('The group_{} is done and length is {}'.format(i, taille_groupe))
        elements_restant = np.delete(features_names,
                                     np.where(np.isin(features_names_copy, elements_choisis) == True)[0])
        for el in elements_restant:
            f.write('group_{}\t{}\n'.format(nb_of_groups, el))
        print('The group_{} is done and length is {}'.format(nb_of_groups, len(elements_restant)))


def construction_pathway_views_groups(data_path=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                                      return_views='all',
                                      output_file_name='pathway_file_views_groups'):
    """
       Construct group based on the original views in the dataset
       Args:
           data_path: str, data path
           return_views: str, correct view for the group
           output_file_name: str, output file name
       Returns:
           output_file_name
       """
    _, _, features_names, _ = load_data(data=data_path, return_views=return_views)
    output_file_name = output_file_name + '_{}.tsv'.format(return_views)
    with open(output_file_name, 'w') as f:
        f.write('G\tIDS\n')
        for el in features_names:
            if el.startswith('cg'):
                f.write('group_1\t{}\n'.format(el))
            elif el.startswith('uc'):
                f.write('group_2\t{}\n'.format(el))
            elif el.startswith('hsa'):
                f.write('group_3\t{}\n'.format(el))
            else:
                f.write('group_4\t{}\n'.format(el))


def construction_pathway_file(data_path=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna,
                              return_views='all',
                              dictionnaire='',
                              output_file_name=''):
    """
    Utility function to build pathway file of the groups to be loaded in LearnFromMsigGroups
    Args:
        data_path: str, data path
        return_views: str, correct view for the group
        dictionnaire: path to the correspond dictionary group
        output_file_name: str, output file name
    Return:
        output_file_name, 'G\tIDS\n'
    """
    x, y, features_names, _ = load_data(data=data_path, return_views=return_views)
    features_names = list(features_names)
    features_names_copy = deepcopy(features_names)
    f = open(dictionnaire, 'rb')
    dict_file = pickle.load(f)
    output_file_name = output_file_name + '_{}.tsv'.format(return_views)
    with open(output_file_name, 'w') as f:
        f.write('G\tIDS\n')
        for group in dict_file.items():
            for raw_genes_name in np.asarray(group[1]):
                for feature in features_names:
                    feature_splits = feature.split('_')
                    if len(feature_splits) == 2:
                        if feature_splits[1] == raw_genes_name:
                            f.write('{}\t{}\n'.format(group[0], feature))
                            try:
                                features_names_copy.remove(feature)  # supprimer de la liste
                            except ValueError:  # si l'élément a déja été supprimer dans la liste
                                pass
        for feature_restant in features_names_copy:
            f.write('{}\t{}\n'.format('unknown_group', feature_restant))


def main_construct():
    # for view in return_views:
    #     construction_pathway_gene_groups_tcga(return_views=view)
    #
    # for view in return_views:
    #     construction_pathway_random_groups(return_views=view)
    #
    # for view in return_views:
    #     construction_pathway_views_groups(return_views=view)

    # construction_biogrid_pathway_file(output_file_name='pathways_biogrid')

    for dictionary in list_dict:
        if dictionary == c2_pickle_dictionary:
            output_file_name = 'pathway_file_c2_curated_groups'
        else:
            output_file_name = 'pathway_file_c5_curated_groups'
        for view in return_views:
            construction_pathway_file(return_views=view, dictionnaire=dictionary, output_file_name=output_file_name)

    for view in return_views:
        construction_pathway_clusters_groups(model_loaded=False,
                                             return_views=view,
                                             model_agglomeration_file_name='feature_agglomeration_model_{}.pck'.format(
                                                 view))


def load_go_idmapping():
    """
    Utility function to map go_ids
    """
    res = defaultdict(set)
    with open(goa_file, 'r') as fin:
        for line in fin:
            if line.startswith("UniProtKB"):
                content = line.split("\t")
                geneid = content[2].lower()
                goid = content[4]
                # go_refid = content[5]
                res[geneid].add(goid)  # go_refid
    nb_go_terms = sum([len(r) for i, r in res.items()])
    nb_genes = len(res)
    print("GO: {} genes and {} GO terms".format(nb_genes, nb_go_terms))
    return res


def load_biogrid_network():
    """
    Utility function to load biogrid_network in graph nx
    """
    go_terms = load_go_idmapping()
    G = nx.Graph()
    edges = set()
    with open(biogrid_file, 'r') as fin:
        fin.readline()
        for line in fin:
            content = line.split("\t")
            if len(content) > 1:
                gene_a, gene_b = content[2].strip(" "), content[3].strip(" ")
                if len(gene_a) > 0 and len(gene_b) > 0:
                    edges.add((gene_a.lower(), gene_b.lower()))
        G.add_edges_from(list(edges))
    for node in G.nodes:
        G.nodes[node]["go_terms"] = go_terms[node]
    print("BioGRID: {} genes and {} interactions".format(G.number_of_nodes(), G.number_of_edges()))
    return G, edges


def construction_biogrid_pathway_file(data_path=data_tn_new_label_balanced_cpg_rna_rna_iso_mirna, return_views='all',
                                      output_file_name=''):
    """
    Utility function to build pathway file of the groups to be loaded in LearnFromBiogridGroup
    Args:
        data_path: str, data path
        return_views: str, correct view for the group
        output_file_name: str, output file name
    Return:
        output_file_name, 'G\tIDS\n'
    """
    graph, _ = load_biogrid_network()
    x, y, features_names, _ = load_data(data=data_path, return_views=return_views)
    features_names = list(features_names)
    adjacency_matrix = np.asarray(list(graph.adjacency()))
    nodes = np.asarray(list(graph.nodes))
    # Initialisation dictionnaire
    dico_results = {feature: [] for feature in features_names}
    for feature in features_names:
        if feature.find('_') != -1:
            # I went step by step for the comprehension but remember the gene is always at the end of the feature so
            # use the [-1] access
            split_results = feature.split('_')
            gene_cible = split_results[-1]
            if gene_cible.find(';'):
                gene_cibles = gene_cible.split(';')
                for gene in gene_cibles:
                    pos_temp = np.where(nodes == gene.lower())[0]
                    if len(pos_temp) > 0:
                        pos = pos_temp[0]
                        dico_results[feature].extend(list(adjacency_matrix[pos][1].keys()))
                    else:
                        dico_results[feature].extend(['INEXISTANT'])
            else:
                pos_temp = np.where(nodes == gene_cible.lower())[0]
                if len(pos_temp) > 0:
                    pos = pos_temp[0]
                    dico_results[feature].extend(list(adjacency_matrix[pos][1].keys()))
                else:
                    dico_results[feature] = ['INEXISTANT']
        elif feature.find('|') != -1:
            # Here the gene is the 1st element always since it's directly the RNA view only
            split_results = feature.split('|')
            gene_cible = split_results[0]
            pos_temp = np.where(nodes == gene_cible.lower())[0]
            if len(pos_temp) > 0:
                pos = pos_temp[0]
                dico_results[feature].extend(list(adjacency_matrix[pos][1].keys()))
        elif feature.startswith('hsa'):  # MiRNA View: faire le traitement directement
            dico_results[feature].extend(['miRNA'])
        else:  # Clinical View
            dico_results[feature].extend(['clinical View'])

    with open(output_file_name + '.pck', 'wb') as f:
        pickle.dump(dico_results, f)


# # TODO: To be modified and adapted to the new outing: COme back here soon
# def gen_histogram_results(pattern_to_search='*_unbalanced_*.pck', metric='accuracy', directory='results_analysis',
#                           results_path='/home/maoss2/PycharmProjects/breast_cancer/experimentations/Results'):
#     os.chdir('{}'.format(results_path))
#     assert metric in ['accuracy', 'f1_score', 'precision', 'recall'], 'metric {} is not implemented yet'.format(metric)
#     noms_fichiers = []
#     metric_train = []
#     metric_test = []
#
#     for fichier in glob('{}'.format(pattern_to_search)):
#         noms_fichiers.append(fichier)
#         f = open(fichier, 'r')
#         d = pickle.load(f)
#         if metric == 'accuracy':
#             metric_train.append(d['train_metrics']['accuracy'])
#             metric_test.append(d['metrics']['accuracy'])
#         if metric == 'precision':
#             metric_train.append(d['train_metrics']['precision'])
#             metric_test.append(d['metrics']['precision'])
#         if metric == 'recall':
#             metric_train.append(d['train_metrics']['recall'])
#             metric_test.append(d['metrics']['recall'])
#         if metric == 'f1_score':
#             metric_train.append(d['train_metrics']['f1_score'])
#             metric_test.append(d['metrics']['f1_score'])
#     noms_fichiers = np.asarray(noms_fichiers)
#     metric_test = np.asarray(metric_test)
#     metric_train = np.asarray(metric_train)
#     nbResults = len(metric_train)
#     figKW = {"figsize": (nbResults, 3.0 / 4 * nbResults + 2.0)}
#     f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
#     barWidth = 0.35
#     sorted_indices = np.argsort(metric_test)
#     testScores = metric_test[sorted_indices]
#     trainScores = metric_train[sorted_indices]
#     names = noms_fichiers[sorted_indices]
#     ax.set_title(''.format(metric))
#     rects = ax.bar(range(nbResults), testScores, barWidth, color="r", )
#     rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth, color="0.7", )
#     autolabel(rects, ax)
#     autolabel(rect2, ax)
#     ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
#     ax.set_ylim(-0.1, 1.1)
#     ax.set_xticks(np.arange(nbResults) + barWidth)
#     ax.set_xticklabels(names, rotation="vertical")
#     plt.tight_layout()
#     f.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + '_unbalanced_metric_analysis_{}'.format(metric) + ".png")
#     plt.close()


if __name__ == '__main__':
    main_construct()
