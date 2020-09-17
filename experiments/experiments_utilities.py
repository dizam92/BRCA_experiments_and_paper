# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import os
import pickle
import random
import click
import re
import time
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
from copy import deepcopy
from glob import glob
from itertools import combinations
from os import makedirs
from os.path import abspath, dirname, exists, join
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from datasets.brca_builder import select_features_based_on_mad

c2_pickle_dictionary = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c2_curated_genes.pck'
c5_pickle_dictionary = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c5_curated_genes.pck'
list_dict = [c2_pickle_dictionary, c5_pickle_dictionary]

# saving_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository'
saving_repository = '/home/maoss2/project/maoss2/saving_repository_article/'
histogram_repo = f'{saving_repository}/histograms_repo'
data_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository'
data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna = f"{data_repository}/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5"
data_prad = f"{data_repository}/prad_cancer_metastase_vs_non_metastase.h5"
brca_dictionnary_for_prior_rules=f"{data_repository}/groups2pathwaysTN_biogrid.pck"
prad_dictionnary_for_prior_rules=f"{data_repository}/groups2pathwaysPRAD_biogrid.pck"

return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']

datasets_new_labels = [data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna] 
# The only data i am using now is the one with cpg; rna; rnaIso; miRNA; clinical

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
parameters_group_scm = {'model_type': param_model_type,
                        'p': param_p,
                        'max_rules': param_max_attributes
                        }

# color_pool = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
color_pool = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

########################################################### Loaders Sections ################################################################
def load_data(data, return_views='all', drop_inexistant_features=True, mad_selection=True):
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
    if drop_inexistant_features:
        index_features_with_genes = [idx for idx, el in enumerate(features_names) if el.find('INEXISTANT') == -1]
        features_names = features_names[index_features_with_genes]
        x = x[:, index_features_with_genes]
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
    if mad_selection:
        indices_mad_selected = select_features_based_on_mad(x=x_methyl, nb_features=2000)
        x_methyl = x_methyl[:, indices_mad_selected]
        features_names_methyl = features_names_methyl[indices_mad_selected]
    # RNA ISO
    x_rna_iso = x[:, index_rna_iso]
    features_names_rna_iso = features_names[index_rna_iso]
    if mad_selection:
        indices_mad_selected = select_features_based_on_mad(x=x_rna_iso, nb_features=2000)
        x_rna_iso = x_rna_iso[:, indices_mad_selected]
        features_names_rna_iso = features_names_rna_iso[indices_mad_selected]
    # MiRNA
    x_mirna = x[:, index_mirna]
    features_names_mirna = features_names[index_mirna]
    if mad_selection:
        indices_mad_selected = select_features_based_on_mad(x=x_mirna, nb_features=250)
        x_mirna = x_mirna[:, indices_mad_selected]
        features_names_mirna = features_names_mirna[indices_mad_selected]
    # SNP
    if len(index_snps) != 0:
        x_snp = x[:, index_snps]
        features_names_snp = features_names[index_snps]
        if mad_selection:
            indices_mad_selected = select_features_based_on_mad(x=x_snp, nb_features=2000)
            x_snp = x_snp[:, indices_mad_selected]
            features_names_snp = features_names_snp[indices_mad_selected]
    # Clinical
    x_clinical = x[:, index_clinical]
    features_names_clinical = features_names[index_clinical]
    # RNA
    x_rna = x[:, index_rna]
    features_names_rna = features_names[index_rna]
    if mad_selection:
        indices_mad_selected = select_features_based_on_mad(x=x_rna, nb_features=2000)
        x_rna = x_rna[:, indices_mad_selected]
        features_names_rna = features_names_rna[indices_mad_selected]
    # Normalization
    x_rna_iso = StandardScaler().fit_transform(x_rna_iso)
    x_mirna = StandardScaler().fit_transform(x_mirna)
    x_rna = StandardScaler().fit_transform(x_rna)
    if return_views == 'methyl_rna_iso_mirna': # THATS THE VIEW I SHOULD JUST BE USING: CE QUI CHANGE DE SO FAR J'AI JUSTE PAS LES !* FEATURES DE CLINICAL QUI SE RAJOUTE THATS IT
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
            features_names = np.hstack((features_names_methyl, features_names_rna,features_names_rna_iso,
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


def load_prad_data(data, return_views='all'):
    """
    Load the triple neg datasets
    Args:
        data, str, path to the .h5py dataset
        return_views, str, the type of the x we want to return, default 'all'
                - 'cna'
                - 'mrna'
                - 'majority_vote' for the majority vote experiments
    Returns:
        x, y, features_names, patients_names
    """
    assert return_views in ['cna', 'mrna', 'all', 'majority_vote']
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
    index_cna = [idx for idx, el in enumerate(features_names) if el.startswith('cna')]
    index_rna = [idx for idx, el in enumerate(features_names) if el.startswith('rna')]
    merge_liste.extend(index_cna)
    merge_liste.extend(index_rna)
    # CNA
    x_cna = x[:, index_cna]
    features_names_cna = features_names[index_cna]
    # RNA
    x_rna = x[:, index_rna]
    features_names_rna = features_names[index_rna]
    # Normalization
    # x_rna = StandardScaler().fit_transform(x_rna)
    if return_views == 'cna':
        x = x_cna
        features_names = features_names_cna
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

    if return_views == 'mrna':
        x = x_rna
        features_names = features_names_rna
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
        return x, x_cna, x_rna, y, features_names, features_names_cna, features_names_rna, patients_names


############################################################################################################################################

def eliminate_features_for_scm(x, y, features_names, features_to_be_excluded):
    """ Objectif, enlever les features significatifs obtenus d'une execution du SCM afin de le rerouler.
    Args:
        x, dataset
        y, label (on fait rien avec)
        features_names, features names
        features_to_be_excluded, liste de features (string) à exclure
    Return:
        x, y, features_names
    """
    assert type(features_to_be_excluded) == list, 'Must be a list of string'
    idx_features_retained = [i for i, el in enumerate(features_names) if el not in features_to_be_excluded]
    x = x[:, idx_features_retained]
    return x, y, features_names

########################################################## Analysis Sections  ################################################################

@click.group()
def cli():
    pass

def sub_repo_analysis(directory, output_text_file, recap_table_file, dictionnary_for_prior_rules='', histogram_file_name='temp', plot_hist=True):
    """
    An utility function to run the results analysis and output them in a readable way. Work on the sub repo site
    Args:
        directory, str, path to the directory containing the pickle files
        data_path, str, path to the data of interest to be loaded to run the analysis
        output_text_file, str, file where to write the results to
    Returns:
        Write results to text file
    """
    dict_pr_rules = pickle.load(open(dictionnary_for_prior_rules, 'rb'))
    os.chdir(f'{directory}')
    metrics_train = []
    metrics_test = []
    features_retenus = []
    model_comptes = []
    cnt = Counter()
    cnt_rf = Counter()
    list_fichiers = []
    groups_features = []
    for fichier in glob("*.pck"):
        list_fichiers.append(fichier)
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
    if plot_hist:
        if not exists(histogram_repo): makedirs(histogram_repo)
        generate_histogram(file_name=histogram_file_name, 
                           fig_title='Metrics', 
                           accuracy_train=accuracy_train, 
                           accuracy_test=accuracy_test, 
                           f1_score_train=f1_score_train,
                           f1_score_test=f1_score_test, 
                           precision_train=precision_train, 
                           precision_test=precision_test, 
                           recall_train=recall_train, 
                           recall_test=recall_test)
        plot_errorbar_error(file_name=histogram_file_name, 
                           fig_title='Metrics', 
                           accuracy_train=accuracy_train, 
                           accuracy_test=accuracy_test, 
                           f1_score_train=f1_score_train,
                           f1_score_test=f1_score_test, 
                           precision_train=precision_train, 
                           precision_test=precision_test, 
                           recall_train=recall_train, 
                           recall_test=recall_test)
    # Find the best seed based on the F1-score (since the dataset is unbalanced)
    best_file = list_fichiers[np.argmax(f1_score_test)] # work for t=both unbalanced and balanced dataset
    index_best_experiment_file = np.argmax(f1_score_test)
    if directory.find('dt') != -1:
        for model in features_retenus:
            temp = [el[3] for el in model[0][:3] if el[2] > 0]
            var = ''
            for el in temp:
                var += '&{}'.format(el) 
            groups_features.append([dict_pr_rules[el] for el in temp])
            model_comptes.append(var)

    if directory.find('rf') != -1:
        for model in features_retenus:
            temp = [el[3] for el in model[0][:3] if el[2] > 0]
            var = ''
            for el in temp:
                var += '&{}'.format(el) 
            groups_features.append([dict_pr_rules[el] for el in temp])
            model_comptes.append(var)
        features_retenus_flatten = [el[3] for liste in features_retenus for el in liste[0][:50]]
        for el in features_retenus_flatten:
            cnt_rf[el] += 1
    if directory.find('scm') != -1:
        for model in features_retenus:
            temp = [el[1] for el in model[0]]
            var = ''
            for el in temp:
                var += '&{}'.format(el)
            groups_features.append([dict_pr_rules[el] for el in temp])
            model_comptes.append(var)
            
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
        f.write('-'*50)
        f.write('\n')
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
        # print('model comptes', model_comptes)
        for el in model_comptes:
            cnt[el] += 1
        most_common_model = cnt.most_common(10)
        if most_common_model != []:
            f.write('_-------------_---------_-------_\n')
            f.write('Most frequent model: {}\n'.format(most_common_model))
        most_common_features = cnt_rf.most_common(50)
        if most_common_features != []: 
            f.write('_-------------_---------_-------_\n')
            f.write('Most frequent Features: {}\n'.format(most_common_features))
            f.write('-'*100)
            f.write('\n')
        f.write('Best seed file based on f1 score (argmax): {}\n'.format(best_file))
        f.write('-'*50)
        f.write('\n')
    # Write best results file of the directory to the recap file
    best_acc = np.round(accuracy_test[index_best_experiment_file], 4)
    best_f1 = np.round(f1_score_test[index_best_experiment_file], 4)
    best_pre = np.round(precision_test[index_best_experiment_file], 4)
    best_rec = np.round(recall_test[index_best_experiment_file], 4)
    best_feat = model_comptes[index_best_experiment_file]
    best_groups_feat = groups_features[index_best_experiment_file]
    with open(recap_table_file, 'a+') as f:
        f.write(f"{best_file}\t{best_acc}\t{best_f1}\t{best_pre}\t{best_rec}\t{best_feat}\t{best_groups_feat}\n")
    os.chdir(saving_repository)
    return np.round(np.mean(accuracy_test), 4), np.round(np.mean(f1_score_test), 4), \
           np.round(np.mean(precision_test), 4), np.round(np.mean(recall_test), 4), model_comptes, groups_features


def global_repo_analysis(directory, output_text_file='results_analysis', 
                         type_experiment='normal', 
                         dict_for_prior_rules='', 
                         sous_experiment_types=['dt', 'scm', 'rf'], 
                         plot_hist=True):
    """
    Parcours le grand repertoire contenant tous les resultats pour extraire (par experiences) le meilleur .pck et aussi la meilleure experience au total.
    Work on the global repo site
    Args:
        directory, path to the results directory (groups_PRAD_experiments/ or groups_TN_experiments/ or normal_experiments/)
        type_experiment: string, normal (for dt, scm, and rf); group_scm for pyscmgroup experiment
        sous_experiment_types, list of string: name of the targeted experiments: [dt, scm, rf]; ['methyl_rna_iso_mirna_snp_clinical'](for brca); ['all'] for prad
        plot_hist, bool, if true plot the figure (histogram)
    Returns:
        Write to 2 files: output_text_file_experiment and recap_text_file_experiment
    """
    output_text_file = f"{saving_repository}/{output_text_file}"
    if type_experiment == 'normal':
        for experiment in sous_experiment_types:
            output_text_file_experiment = f"{output_text_file}__{experiment}.txt"
            recap_text_file_experiment = f"{output_text_file}__{experiment}__recap.txt"
            acc_test_list = []
            f1_test_list = []
            precis_test_list = []
            rec_test_list = []
            model_comptes_list = []
            groups_features_list = []
            list_of_directories = os.listdir(directory)
            list_of_directories = [f'{directory}/{repository}' for repository in list_of_directories if repository.startswith(experiment)]
            temp = output_text_file.split('/')[-1]
            file_name = f'{experiment}_{temp}'
            for repository in list_of_directories:
                acc_test, f1_test, precis_test, rec_test, model_comptes, groups_features = sub_repo_analysis(directory=repository,
                                                                                              output_text_file=output_text_file_experiment,
                                                                                              recap_table_file=recap_text_file_experiment,
                                                                                              dictionnary_for_prior_rules=dict_for_prior_rules, 
                                                                                              histogram_file_name=file_name,
                                                                                              plot_hist=plot_hist)
                acc_test_list.append(acc_test)
                f1_test_list.append(f1_test)
                precis_test_list.append(precis_test)
                rec_test_list.append(rec_test)
                model_comptes_list.append(model_comptes)
                groups_features_list.append(groups_features)
            with open(output_text_file_experiment, 'a+') as f:
                f.write('-' * 50)
                f.write('\n')
                f.write('Best model selected based on F1 Score\n')
                best_model_idx = np.argmax(np.asarray(f1_test_list)) # to be modified if i decied to go with the accuracy for PRAD vs f1 for BRCA
                f.write(f'Best Experiment is:{list_of_directories[best_model_idx]}\n')
                f.write(f'Results: Acc: {acc_test_list[best_model_idx]}\t F1: {f1_test_list[best_model_idx]}\t Prec: {precis_test_list[best_model_idx]}\t Rec: {rec_test_list[best_model_idx]} \n')
                f.write(f'Model comptes {model_comptes_list[best_model_idx]}\n')
                f.write(f'Groups Features Best Model {groups_features_list[best_model_idx]}\n')
    if type_experiment == 'group_scm':
        for experiment in sous_experiment_types:
            output_text_file_experiment = f"{output_text_file}__group_scm__{experiment}.txt"
            recap_text_file_experiment = f"{output_text_file}__{experiment}__recap.txt"
            acc_test_list = []
            f1_test_list = []
            precis_test_list = []
            rec_test_list = []
            model_comptes_list = [] 
            groups_features_list = []
            list_of_directories = os.listdir(directory)
            list_of_directories = [f'{directory}/{repository}' for repository in list_of_directories if repository.startswith(experiment)]
            temp = output_text_file.split('/')[-1]
            file_name = f'{experiment}_{temp}'
            for repository in list_of_directories:
                acc_test, f1_test, precis_test, rec_test, model_comptes, groups_features = sub_repo_analysis(directory=repository,
                                                                                                            output_text_file=output_text_file_experiment,
                                                                                                            recap_table_file=recap_text_file_experiment,
                                                                                                            dictionnary_for_prior_rules=dict_for_prior_rules,
                                                                                                            histogram_file_name=file_name,
                                                                                                            plot_hist=plot_hist)
                acc_test_list.append(acc_test)
                f1_test_list.append(f1_test)
                precis_test_list.append(precis_test)
                rec_test_list.append(rec_test)
                model_comptes_list.append(model_comptes)
                groups_features_list.append(groups_features)
            with open(output_text_file_experiment, 'a+') as f:
                f.write('-' * 50)
                f.write('\n')
                f.write('Best model selected based on F1 Score\n')
                best_model_idx = np.argmax(np.asarray(f1_test_list))
                f.write(f'Best Experiment is:{list_of_directories[best_model_idx]}\n')
                f.write(f'Results: Acc: {acc_test_list[best_model_idx]}\t F1: {f1_test_list[best_model_idx]}\t Prec: {precis_test_list[best_model_idx]}\t Rec: {rec_test_list[best_model_idx]} \n')
                f.write(f'Model comptes {model_comptes_list[best_model_idx]}\n')
                f.write(f'Groups Features Best Model {groups_features_list[best_model_idx]}\n')
        

@cli.command(help="Run the analysis results")
@click.option('--directory', type=str, default=None, help="""results path""")
@click.option('--output-text-file', type=str, default='normal_brca_results_analysis', help="""outout name file""")
@click.option('--type-experiment', type=str, default='normal', help="""type of experiment global or group_scm""")
@click.option('--dict-for-prior-rules', type=str, default=None, help="""dictionary path""")
@click.option('--sous-experiment-types', type=str, default='dt scm rf', help="""name of experiment""")
@click.option('--plot-hist/--no-plot-hist', default=False, help="""plot histogram""")
def run_analysis(directory, output_text_file, type_experiment, dict_for_prior_rules, sous_experiment_types, plot_hist):
    if plot_hist:
        global_repo_analysis(directory, output_text_file, type_experiment, dict_for_prior_rules, sous_experiment_types.split(), True)
    else:
        global_repo_analysis(directory, output_text_file, type_experiment, dict_for_prior_rules, sous_experiment_types.split(), False)

############################################################################################################################################

########################################################## Analysis Sections Plot########################################################

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, "%.2f" % height, ha='center', va='bottom')


def generate_histogram(file_name, fig_title, accuracy_train, accuracy_test, f1_score_train, f1_score_test, 
                       precision_train, precision_test, recall_train, recall_test):
    """
    Plot histogram figures
    Args:
    Returns:
        Figures plotted, histogram bar plot (could rethink this and plot better figures huh)
    """
    train_metrics = np.asarray([np.round(np.mean(accuracy_train), 4), np.round(np.mean(f1_score_train), 4),
                               np.round(np.mean(precision_train), 4), np.round(np.mean(recall_train), 4)])
    test_metrics = np.asarray([np.round(np.mean(accuracy_test), 4), np.round(np.mean(f1_score_test), 4),
                               np.round(np.mean(precision_test), 4), np.round(np.mean(recall_test), 4)])
    std_train_metrics = np.asarray([np.round(np.std(accuracy_train), 4), np.round(np.std(f1_score_train), 4),
                               np.round(np.std(precision_train), 4), np.round(np.std(recall_train), 4)])
    std_test_metrics = np.asarray([np.round(np.std(accuracy_test), 4), np.round(np.std(f1_score_test), 4),
                               np.round(np.std(precision_test), 4), np.round(np.std(recall_test), 4)])
    
    nbResults = len(train_metrics)
    # figKW = {"figsize": (nbResults, 8)}
    # f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    f, ax = plt.subplots(nrows=1, ncols=1)
    sns.set_style("darkgrid")
    barWidth = 0.35
    ax.set_title(f"{fig_title}")
    rects = ax.bar(range(nbResults), test_metrics, barWidth, color="r", yerr=std_test_metrics)
    rect2 = ax.bar(np.arange(nbResults) + barWidth, train_metrics, barWidth, color="0.7", yerr=std_train_metrics)
    autolabel(rects, ax)
    autolabel(rect2, ax)
    ax.legend((rects[0], rect2[0]), ('Test', 'Train'), loc='upper right', ncol=2, mode="expand", borderaxespad=0.)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xticks(np.arange(nbResults) + barWidth)
    ax.set_xticklabels(['Acc', 'F1', 'Prec', 'Rec'])
    plt.tight_layout()
    f.savefig(f"{histogram_repo}/{file_name}.png")
    plt.close()


def plot_errorbar_error(file_name, fig_title, accuracy_train, accuracy_test, f1_score_train, f1_score_test, 
                       precision_train, precision_test, recall_train, recall_test):
    """
    Plot Error bar figures
    Args:
    Returns:
        Figures plotted, error bar on the figures (rebranding the previous function)
    pos[0]: accuracy
    pos[1]: f1_score
    pos[2]: precicion
    pos[3]: recall
    """
    train_metrics = np.asarray([np.round(np.mean(accuracy_train), 4), np.round(np.mean(f1_score_train), 4),
                               np.round(np.mean(precision_train), 4), np.round(np.mean(recall_train), 4)])
    test_metrics = np.asarray([np.round(np.mean(accuracy_test), 4), np.round(np.mean(f1_score_test), 4),
                               np.round(np.mean(precision_test), 4), np.round(np.mean(recall_test), 4)])
    std_train_metrics = np.asarray([np.round(np.std(accuracy_train), 4), np.round(np.std(f1_score_train), 4),
                               np.round(np.std(precision_train), 4), np.round(np.std(recall_train), 4)])
    std_test_metrics = np.asarray([np.round(np.std(accuracy_test), 4), np.round(np.std(f1_score_test), 4),
                               np.round(np.std(precision_test), 4), np.round(np.std(recall_test), 4)])
    fig = plt.figure(figsize=(9, 7))
    sns.set_style("darkgrid")
    x = ['Acc', 'F1', 'Prec', 'Rec']
    fig.suptitle(f'{fig_title}_train', fontsize=8)
    plt.errorbar(x, train_metrics, yerr=std_train_metrics, fmt='-', color=color_pool[0], label='train')
    plt.fill_between(x, train_metrics - std_train_metrics, train_metrics + std_train_metrics, facecolor=color_pool[0], alpha=0.5)
    plt.errorbar(x, test_metrics, yerr=std_test_metrics, fmt='-', color=color_pool[1], label='test')
    plt.fill_between(x, test_metrics - std_test_metrics, test_metrics + std_test_metrics, facecolor=color_pool[1], alpha=0.5)
    plt.legend(loc='best', fontsize='x-small', shadow=True)
    plt.xlabel('Metrics', fontsize=10)
    plt.xticks(range(len(x)), x)  # rotation='vertical'
    plt.ylabel(f'Values', fontsize=10)
    plt.yticks(rotation=90)
    fig.savefig(f"{histogram_repo}/{file_name}_error_bars.png")
    plt.close()
 
    
def parcours_one_directory(directory):
    """
    Parcours chaque sub repo et retourne les différentes metrics pour faciliter le plotting
    Args:
        directory, path to a single directory (sub_repo)
    Returns:
        train_metrics, test_metrics, std_train_metrics, std_test_metrics, train_metrics_best_file, test_metrics_best_file, features_retenus
    """
    os.chdir(f'{directory}')
    metrics_train = []
    metrics_test = []
    features_retenus = []
    list_fichiers = []
    for fichier in glob("*.pck"):
        list_fichiers.append(fichier)
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
    best_file_idx = np.argmax(f1_score_test)
    train_metrics = np.asarray([np.round(np.mean(accuracy_train), 4), np.round(np.mean(f1_score_train), 4),
                               np.round(np.mean(precision_train), 4), np.round(np.mean(recall_train), 4)])
    test_metrics = np.asarray([np.round(np.mean(accuracy_test), 4), np.round(np.mean(f1_score_test), 4),
                               np.round(np.mean(precision_test), 4), np.round(np.mean(recall_test), 4)])
    std_train_metrics = np.asarray([np.round(np.std(accuracy_train), 4), np.round(np.std(f1_score_train), 4),
                               np.round(np.std(precision_train), 4), np.round(np.std(recall_train), 4)])
    std_test_metrics = np.asarray([np.round(np.std(accuracy_test), 4), np.round(np.std(f1_score_test), 4),
                               np.round(np.std(precision_test), 4), np.round(np.std(recall_test), 4)])
    train_metrics_best_file = np.asarray([np.round(accuracy_train[best_file_idx], 4), np.round(f1_score_train[best_file_idx], 4),
                                          np.round(precision_train[best_file_idx], 4), np.round(recall_train[best_file_idx], 4)])
    test_metrics_best_file = np.asarray([np.round(accuracy_test[best_file_idx], 4), np.round(f1_score_test[best_file_idx], 4),
                                          np.round(precision_test[best_file_idx], 4), np.round(recall_test[best_file_idx], 4)])
    os.chdir('../')
    return train_metrics, test_metrics, std_train_metrics, std_test_metrics, train_metrics_best_file, test_metrics_best_file, features_retenus


def generate_figures_mean_results(directory, sous_experiment_types, fig_name='', cancer_name='brca', f='exp', type_of_update='inner', random_weights=False):
    """
    Utility function to plot the results for the groups method using the MEAN (Should rethink this probably)
    Args:
        directory,
        experiment, str, experiment name
        f, str, activation function name
        type_of_update, str, 
        random_weights, bool
    """
    x = np.round(np.linspace(0.1, 1, 10), 3)
    os.chdir(f"{directory}")
    list_of_directories = os.listdir('./')
    list_of_directories = [directory for directory in list_of_directories if directory.startswith(sous_experiment_types)] 
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{type_of_update}') != -1]
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{random_weights}') != -1]
    list_of_directories = list(np.sort(list_of_directories)) # garantie que ca va de 0.1 à 1.0 ici (sinon tjrs de min a max value de c)
    train_metrics_list = []; test_metrics_list = []; std_train_metrics_list = []; std_test_metrics_list = []
    for directory in list_of_directories:
        train_metrics, test_metrics, std_train_metrics, std_test_metrics, _, _, _ = parcours_one_directory(directory=directory)
        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        std_train_metrics_list.append(std_train_metrics)
        std_test_metrics_list.append(std_test_metrics)
    train_metrics_list = np.asarray(train_metrics_list)
    test_metrics_list = np.asarray(test_metrics_list)
    std_train_metrics_list = np.asarray(std_train_metrics_list)
    std_test_metrics_list = np.asarray(std_test_metrics_list)
    # Plot the train fig
    # fig_title_train = f'Train mean metrics: Update Function:{f} {type_of_update}_groups random_weights: {random_weights}'
    fig_title_train = 'Train mean metrics'
    if fig_name == '':
        fig_name_train = f'{f}_{cancer_name}_train_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    else:
        fig_name_train = f'{fig_name}_train_mean_metrics_c_values.png'
        # fig_name_train = f'{fig_name}_train_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_train, ax_train = plt.subplots(nrows=1, ncols=1)
    sns.set_style("darkgrid")
    ax_train.set_title(f"{fig_title_train}")
    ax_train.set_xlabel('c values')
    ax_train.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_train.plot(x, train_metrics_list[:, 0], 'o-', color=color_pool[0], label='Acc', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 1], 'o-', color=color_pool[1], label='F1 ', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 2], 'o-', color=color_pool[2], label='Prec', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 3], 'o-', color=color_pool[3], label='Rec', linewidth=2)
    ax_train.legend()
    plt.tight_layout()
    f_train.savefig(f"{histogram_repo}/{fig_name_train}")
    plt.close()
    
    # Plot the Test fig
    # fig_title_test = f'Test mean metrics: {type_of_update}_groups random_weights: {random_weights}'
    fig_title_test = 'Test mean metrics'
    if fig_name == '':
        fig_name_test = f'{f}_{cancer_name}_test_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    else:
        fig_name_test = f'{fig_name}_test_mean_metrics_c_values.png'
        # fig_name_test = f'{fig_name}_test_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_test, ax_test = plt.subplots(nrows=1, ncols=1)
    sns.set_style("darkgrid")
    ax_test.set_title(f"{fig_title_test}")
    ax_test.set_xlabel('c values')
    ax_test.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_test.plot(x, test_metrics_list[:, 0], 'o-', color=color_pool[0], label='Acc', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 1], 'o-', color=color_pool[1], label='F1 ', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 2], 'o-', color=color_pool[2], label='Prec', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 3], 'o-', color=color_pool[3], label='Rec', linewidth=2)
    ax_test.legend()
    plt.tight_layout()
    f_test.savefig(f"{histogram_repo}/{fig_name_test}")
    plt.close()
    os.chdir(f'{saving_repository}')
    

def generate_figures_mean_results_errorbar_error(directory, sous_experiment_types, fig_name='', cancer_name='brca', f='exp', type_of_update='inner', random_weights=False):
    """
    Utility function to plot the results for the groups method using the MEAN (Should rethink this probably)
    Args:
        directory,
        experiment, str, experiment name
        f, str, activation function name
        type_of_update, str, 
        random_weights, bool
    """
    x = np.round(np.linspace(0.1, 1, 10), 3)
    os.chdir(f"{directory}")
    list_of_directories = os.listdir('./')
    list_of_directories = [directory for directory in list_of_directories if directory.startswith(sous_experiment_types)] 
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{type_of_update}') != -1]
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{random_weights}') != -1]
    list_of_directories = list(np.sort(list_of_directories)) # garantie que ca va de 0.1 à 1.0 ici (sinon tjrs de min a max value de c)
    train_metrics_list = []; test_metrics_list = []; std_train_metrics_list = []; std_test_metrics_list = []
    for directory in list_of_directories:
        train_metrics, test_metrics, std_train_metrics, std_test_metrics, _, _, _ = parcours_one_directory(directory=directory)
        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        std_train_metrics_list.append(std_train_metrics)
        std_test_metrics_list.append(std_test_metrics)
    train_metrics_list = np.asarray(train_metrics_list)
    test_metrics_list = np.asarray(test_metrics_list)
    std_train_metrics_list = np.asarray(std_train_metrics_list)
    std_test_metrics_list = np.asarray(std_test_metrics_list)
    # Plot the train fig
    fig_title_train = 'Train mean metrics'
    if fig_name == '':
        fig_name_train = f'{f}_{cancer_name}_train_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    else:
        fig_name_train = f'{fig_name}_train_mean_metrics_c_values_error_bars.png'
        # fig_name_train = f'{fig_name}_train_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_train, ax_train = plt.subplots(nrows=1, ncols=1)
    sns.set_style("darkgrid")
    ax_train.set_title(f"{fig_title_train}")
    ax_train.set_xlabel('c values')
    ax_train.set_ylabel('Metrics values')
    ax_train.errorbar(x, train_metrics_list[:, 0], yerr=std_train_metrics_list[:, 0], fmt='-', color=color_pool[0], label='Acc')
    ax_train.fill_between(x, train_metrics_list[:, 0] - std_train_metrics_list[:, 0], train_metrics_list[:, 0] + std_train_metrics_list[:, 0], facecolor=color_pool[0], alpha=0.5)
    ax_train.errorbar(x, train_metrics_list[:, 1], yerr=std_train_metrics_list[:, 1], fmt='-', color=color_pool[1], label='F1')
    ax_train.fill_between(x, train_metrics_list[:, 1] - std_train_metrics_list[:, 1], train_metrics_list[:, 1] + std_train_metrics_list[:, 1], facecolor=color_pool[1], alpha=0.5)
    ax_train.errorbar(x, train_metrics_list[:, 2], yerr=std_train_metrics_list[:, 2], fmt='-', color=color_pool[2], label='Prec')
    ax_train.fill_between(x, train_metrics_list[:, 2] - std_train_metrics_list[:, 2], train_metrics_list[:, 2] + std_train_metrics_list[:, 2], facecolor=color_pool[2], alpha=0.5)
    ax_train.errorbar(x, train_metrics_list[:, 3], yerr=std_train_metrics_list[:, 3], fmt='-', color=color_pool[3], label='Rec')
    ax_train.fill_between(x, train_metrics_list[:, 3] - std_train_metrics_list[:, 3], train_metrics_list[:, 3] + std_train_metrics_list[:, 3], facecolor=color_pool[3], alpha=0.5) 
    ax_train.legend(loc='best', fontsize='x-small', shadow=True)
    # plt.tight_layout()
    f_train.savefig(f"{histogram_repo}/{fig_name_train}")
    plt.close()
    
    # Plot the Test fig
    # fig_title_test = f'Test mean metrics: {type_of_update}_groups random_weights: {random_weights}'
    fig_title_test = 'Test mean metrics'
    if fig_name == '':
        fig_name_test = f'{f}_{cancer_name}_test_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    else:
        fig_name_test = f'{fig_name}_test_mean_metrics_c_values_error_bars.png'
        # fig_name_test = f'{fig_name}_test_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_test, ax_test = plt.subplots(nrows=1, ncols=1)
    sns.set_style("darkgrid")
    ax_test.set_title(f"{fig_title_test}")
    ax_test.set_xlabel('c values')
    ax_test.set_ylabel('Metrics values')
    ax_test.errorbar(x, test_metrics_list[:, 0], yerr=std_test_metrics_list[:, 0], fmt='-', color=color_pool[0], label='Acc')
    ax_test.fill_between(x, test_metrics_list[:, 0] - std_test_metrics_list[:, 0], test_metrics_list[:, 0] + std_test_metrics_list[:, 0], facecolor=color_pool[0], alpha=0.5)
    ax_test.errorbar(x, test_metrics_list[:, 1], yerr=std_test_metrics_list[:, 1], fmt='-', color=color_pool[1], label='F1')
    ax_test.fill_between(x, test_metrics_list[:, 1] - std_test_metrics_list[:, 1], test_metrics_list[:, 1] + std_test_metrics_list[:, 1], facecolor=color_pool[1], alpha=0.5)
    ax_test.errorbar(x, test_metrics_list[:, 2], yerr=std_test_metrics_list[:, 2], fmt='-', color=color_pool[2], label='Prec')
    ax_test.fill_between(x, test_metrics_list[:, 2] - std_test_metrics_list[:, 2], test_metrics_list[:, 2] + std_test_metrics_list[:, 2], facecolor=color_pool[2], alpha=0.5)
    ax_test.errorbar(x, test_metrics_list[:, 3], yerr=std_test_metrics_list[:, 3], fmt='-', color=color_pool[3], label='Rec')
    ax_test.fill_between(x, test_metrics_list[:, 3] - std_test_metrics_list[:, 3], test_metrics_list[:, 3] + std_test_metrics_list[:, 3], facecolor=color_pool[3], alpha=0.5)
    ax_test.legend(loc='best', fontsize='x-small', shadow=True)
    # plt.tight_layout()
    f_test.savefig(f"{histogram_repo}/{fig_name_test}")
    plt.close()
    os.chdir(f'{saving_repository}')

   
def generate_figures_best_results(directory, sous_experiment_types, fig_name='', cancer_name='brca', f='exp', type_of_update='inner', random_weights=False):
    """
    Utility function to plot the results for the groups method using the BEST scores return (Should rethink this probably)
    Args:
        directory,
        experiment, str, experiment name
        f, str, activation function name
        type_of_update, str, 
        random_weights, bool
    """
    x = np.round(np.linspace(0.1, 1, 10), 3)
    os.chdir(f"{directory}")
    list_of_directories = os.listdir('./')
    list_of_directories = [directory for directory in list_of_directories if directory.startswith(sous_experiment_types)] 
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{type_of_update}') != -1]
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{random_weights}') != -1]
    list_of_directories = list(np.sort(list_of_directories)) # garantie que ca va de 0.1 à 1.0 ici (sinon tjrs de min a max value de c)
    train_metrics_list = []; test_metrics_list = []
    for directory in list_of_directories:
        _, _, _, _, train_metrics_best_file, test_metrics_best_file, _ = parcours_one_directory(directory=directory)
        train_metrics_list.append(train_metrics_best_file)
        test_metrics_list.append(test_metrics_best_file)
    train_metrics_list = np.asarray(train_metrics_list)
    test_metrics_list = np.asarray(test_metrics_list)
    # Plot the train fig
    # fig_title_train = f'Train best metrics: Update Function:{f} {type_of_update}_groups random_weights: {random_weights}'
    fig_title_train = 'Train best metrics'
    if fig_name == '':
        fig_name_train = f'{f}_{cancer_name}_train_best_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    else:
        fig_name_train = f'{fig_name}_train_best_metrics_c_values.png'
        # fig_name_train = f'{fig_name}_train_best_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_train, ax_train = plt.subplots(nrows=1, ncols=1)
    sns.set_style("darkgrid")
    ax_train.set_title(f"{fig_title_train}")
    ax_train.set_xlabel('c values')
    ax_train.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_train.plot(x, train_metrics_list[:, 0], 'o-', color=color_pool[0], label='Acc', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 1], 'o-', color=color_pool[1], label='F1 ', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 2], 'o-', color=color_pool[2], label='Prec', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 3], 'o-', color=color_pool[3], label='Rec', linewidth=2)
    ax_train.legend()
    plt.tight_layout()
    f_train.savefig(f"{histogram_repo}/{fig_name_train}")
    plt.close()
    
    # Plot the Test fig
    # fig_title_test = f'Test best metrics: {type_of_update}_groups random_weights: {random_weights}'
    fig_title_test = 'Test best metrics'
    if fig_name == '':
        fig_name_test = f'{f}_{cancer_name}_test_best_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    else:
        fig_name_test = f'{fig_name}_test_best_metrics_c_values.png'
        # fig_name_test = f'{fig_name}_test_best_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_test, ax_test = plt.subplots(nrows=1, ncols=1)
    sns.set_style("darkgrid")
    ax_test.set_title(f"{fig_title_test}")
    ax_test.set_xlabel('c values')
    ax_test.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_test.plot(x, test_metrics_list[:, 0], 'o-', color=color_pool[0], label='Acc', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 1], 'o-', color=color_pool[1], label='F1 ', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 2], 'o-', color=color_pool[2], label='Prec', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 3], 'o-', color=color_pool[3], label='Rec', linewidth=2)
    ax_test.legend()
    plt.tight_layout()
    f_test.savefig(f"{histogram_repo}/{fig_name_test}")
    plt.close()
    os.chdir(f'{saving_repository}')
 

@cli.command(help="Run the analysis results")
@click.option('--directory', type=str, default=None, help="""results path""")
@click.option('--sous-experiment-types', type=str, default='all', help="""name of experiment in results_views""")
@click.option('--cancer-name', type=str, default='brca', help="""cancer name""")
@click.option('--fig-name', type=str, default='', help="""figure name""")
@click.option('--f', type=str, default='exp', help="""cancer name""")
@click.option('--type-of-update', type=str, default='inner', help="""update type""")
@click.option('--random-weights/--no-random-weights', default=False, help="""random-weights generate or not""")
@click.option('--plot-mean/--no-plot-mean', default=True, help="""plot mean values figures""")
@click.option('--plot-best/--no-plot-best', default=True, help="""plot best values figures""")
def run_plot_groups(directory, sous_experiment_types, fig_name, cancer_name, f, type_of_update, random_weights, plot_mean, plot_best):
    if plot_mean:
        if random_weights:
            generate_figures_mean_results(directory, sous_experiment_types, fig_name, cancer_name, f, type_of_update, True)
        else:
            generate_figures_mean_results(directory, sous_experiment_types, fig_name, cancer_name, f, type_of_update, False)
            generate_figures_mean_results_errorbar_error(directory, sous_experiment_types, fig_name, cancer_name, f, type_of_update, False)
    if plot_best:
        if random_weights:
            generate_figures_best_results(directory, sous_experiment_types, fig_name, cancer_name, f, type_of_update, True)
        else:
            generate_figures_best_results(directory, sous_experiment_types, fig_name, cancer_name, f, type_of_update, False)

############################################################################################################################################
    
########################################################### Metrics Sections ################################################################
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

############################################################################################################################################

if __name__ == "__main__":
    cli()
