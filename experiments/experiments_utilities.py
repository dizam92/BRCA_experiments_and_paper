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


c2_pickle_dictionary = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c2_curated_genes.pck'
c5_pickle_dictionary = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c5_curated_genes.pck'
list_dict = [c2_pickle_dictionary, c5_pickle_dictionary]

# saving_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository'
saving_repository = '/home/maoss2/project/maoss2/saving_repository_article/'
histogram_repo = f'{saving_repository}/histograms_repo'
data_repository = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository'
data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna = f"{data_repository}/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5"
data_prad = f"{data_repository}/prad_cancer_metastase_vs_non_metastase.h5"

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

########################################################### Loaders Sections ################################################################
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


# Facteur de rebalancement : si mult = 1 on aura 1 négatif pour chaque positif, si mult = 2, on aura 2 négtifs pour chaque positif
mult = 1
def load_baptiste_data(dataset, subsampling=False):
    """
    Load the dataset
    Ideal cible: label 1vs2 i.e multi_clustered vs EMF
    Args: 
        dataset, str path to the dataset
        subsampling, bool if true, make a subsampling else do nothing
    Return:
        view_0, view_0_name, view_1, view_1_name, view_2, view_2_name, view_3, view_3_name, 
        view_4, view_4_name, view_5, view_5_name, view_6, view_6_name, view_7, view_7_name, 
        view_8, view_8_name, view_9, view_9_name, view_10, view_10_name, view_11, view_11_name, 
        view_12, view_12_name, view_13, view_13_name, all_data, groups_ids, y, proteins_ids, features_names
    """
    d = h5py.File(dataset, 'r')
    labels = d['Labels'][()]
    labels_emf_idx = np.where(labels == 2)[0] # EMF
    labels_multi_c_idx = np.where(labels == 1)[0] # multi_clustered
    labels_mono_c_idx = np.where(labels == 0)[0] # mono_clustered
    random_state = np.random.RandomState(42)
    if subsampling:
        labels_multi_c_idx = random_state.choice(labels_multi_c_idx, labels_emf_idx.shape[0]*mult, replace=False)
    proteins_ids = d["Metadata"]["example_ids"][()]
    proteins_ids_pos = proteins_ids[labels_emf_idx]
    proteins_ids_neg = proteins_ids[labels_multi_c_idx]
    proteins_ids = np.hstack((proteins_ids_pos, proteins_ids_neg))
    view_0 = d["View0"][()];  view_0_name = d["View0"].attrs["name"] # PPInetwork_topology
    view_1 = d["View1"][()];  view_1_name = d["View1"].attrs["name"] # Subcell_Location
    view_2 = d["View2"][()];  view_2_name = d["View2"].attrs["name"] # Tissue_Expression
    view_3 = d["View3"][()];  view_3_name = d["View3"].attrs["name"] # SDNE_PPInetwork
    view_4 = d["View4"][()];  view_4_name = d["View4"].attrs["name"] # Gene_Ontology_BP
    view_5 = d["View5"][()];  view_5_name = d["View5"].attrs["name"] # Gene_Ontology_CC
    view_6 = d["View6"][()];  view_6_name = d["View6"].attrs["name"] # Gene_Ontology_MF
    view_7 = d["View7"][()];  view_7_name = d["View7"].attrs["name"] # BP_PPInetwork_embed
    view_8 = d["View8"][()];  view_8_name = d["View8"].attrs["name"] # CC_PPInetwork_embed
    view_9 = d["View9"][()];  view_9_name = d["View9"].attrs["name"] # Phenotype_Ontology
    view_10 = d["View10"][()];  view_10_name = d["View10"].attrs["name"] # Protein_Domains
    view_11 = d["View11"][()];  view_11_name = d["View11"].attrs["name"] # PTM
    view_12 = d["View12"][()];  view_12_name = d["View12"].attrs["name"] # 3UTR_Complexes
    view_13 = d["View13"][()];  view_13_name = d["View13"].attrs["name"] # Linear_Motifs

    view_0_pos = view_0[labels_emf_idx]; view_0_neg = view_0[labels_multi_c_idx]
    view_1_pos = view_1[labels_emf_idx]; view_1_neg = view_1[labels_multi_c_idx]
    view_2_pos = view_2[labels_emf_idx]; view_2_neg = view_2[labels_multi_c_idx]
    view_3_pos = view_3[labels_emf_idx]; view_3_neg = view_3[labels_multi_c_idx]
    view_4_pos = view_4[labels_emf_idx]; view_4_neg = view_4[labels_multi_c_idx]
    view_5_pos = view_5[labels_emf_idx]; view_5_neg = view_5[labels_multi_c_idx]
    view_6_pos = view_6[labels_emf_idx]; view_6_neg = view_6[labels_multi_c_idx]
    view_7_pos = view_7[labels_emf_idx]; view_7_neg = view_7[labels_multi_c_idx]
    view_8_pos = view_8[labels_emf_idx]; view_8_neg = view_8[labels_multi_c_idx]
    view_9_pos = view_9[labels_emf_idx]; view_9_neg = view_9[labels_multi_c_idx]
    view_10_pos = view_10[labels_emf_idx]; view_10_neg = view_10[labels_multi_c_idx]
    view_11_pos = view_11[labels_emf_idx]; view_11_neg = view_11[labels_multi_c_idx]
    view_12_pos = view_12[labels_emf_idx]; view_12_neg = view_12[labels_multi_c_idx]
    view_13_pos = view_13[labels_emf_idx]; view_13_neg = view_13[labels_multi_c_idx]
    y_pos = np.ones(labels_emf_idx.shape); y_neg = np.ones(labels_multi_c_idx.shape) * -1
    
    view_0 = np.vstack((view_0_pos, view_0_neg)); groups_0 = ['ppi' for _ in range(view_0.shape[1])]
    view_1 = np.vstack((view_1_pos, view_1_neg)); groups_1 = ['subcell' for _ in range(view_1.shape[1])]
    view_2 = np.vstack((view_2_pos, view_2_neg)); groups_2 = ['tissue_expression' for _ in range(view_2.shape[1])]
    view_3 = np.vstack((view_3_pos, view_3_neg)); groups_3 = ['sdne_ppi' for _ in range(view_3.shape[1])]
    view_4 = np.vstack((view_4_pos, view_4_neg)); groups_4 = ['go_BP' for _ in range(view_4.shape[1])]
    view_5 = np.vstack((view_5_pos, view_5_neg)); groups_5 = ['go_CC' for _ in range(view_5.shape[1])]
    view_6 = np.vstack((view_6_pos, view_6_neg)); groups_6 = ['go_MF' for _ in range(view_6.shape[1])]
    view_7 = np.vstack((view_7_pos, view_7_neg)); groups_7 = ['BP_ppi' for _ in range(view_7.shape[1])]
    view_8 = np.vstack((view_8_pos, view_8_neg)); groups_8 = ['CC_ppi' for _ in range(view_8.shape[1])]
    view_9 = np.vstack((view_9_pos, view_9_neg)); groups_9 = ['phenotype_ontology' for _ in range(view_9.shape[1])]
    view_10 = np.vstack((view_10_pos, view_10_neg)); groups_10 = ['Protein_Domains' for _ in range(view_10.shape[1])]
    view_11 = np.vstack((view_11_pos, view_11_neg)); groups_11 = ['PTM' for _ in range(view_11.shape[1])]
    view_12 = np.vstack((view_12_pos, view_12_neg)); groups_12 = ['3UTR_Complexes' for _ in range(view_12.shape[1])]
    view_13 = np.vstack((view_13_pos, view_13_neg)); groups_13 = ['Linear_Motifs' for _ in range(view_13.shape[1])]
    
    all_data = np.hstack((view_0, view_1, view_2, view_3, view_4, view_5, view_6, view_7, view_8, view_9, view_10, view_11, view_12, view_13)) 
    # (2370, 37325)
    groups_ids = groups_0 + groups_1 + groups_2 + groups_3 + groups_4 + groups_5 + groups_6 + groups_7 + groups_8 + groups_9 + groups_10 + groups_11 + groups_12 + groups_13
    y = np.hstack((y_pos, y_neg))
    features_names = [f'feature_{idx}' for idx in range(all_data.shape[1])]
    return view_0, view_0_name, view_1, view_1_name, view_2, view_2_name, view_3, view_3_name, view_4, view_4_name, view_5, view_5_name, view_6, view_6_name, view_7, view_7_name, view_8, view_8_name, view_9, view_9_name, view_10, view_10_name, view_11, view_11_name, view_12, view_12_name, view_13, view_13_name, all_data, groups_ids, y, proteins_ids, features_names
  

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


def sub_repo_analysis(directory, output_text_file, recap_table_file, cancer_name='brca', plot_hist=True):
    """
    An utility function to run the results analysis and output them in a readable way. Work on the sub repo site
    Args:
        directory, str, path to the directory containing the pickle files
        data_path, str, path to the data of interest to be loaded to run the analysis
        output_text_file, str, file where to write the results to
    Returns:
        Write results to text file
    """
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
        generate_histogram(file_name=f"{directory}", 
                           fig_title='Metrics', 
                           accuracy_train=accuracy_train, 
                           accuracy_test=accuracy_test, 
                           f1_score_train=f1_score_train,
                           f1_score_test=f1_score_test, 
                           precision_train=precision_train, 
                           precision_test=precision_test, 
                           recall_train=recall_train, 
                           recall_test=recall_test)
    # if cancer_name == 'brca':
    #     # Find the best seed based on the F1-score (since the dataset is unbalanced)
    #     best_file = list_fichiers[np.argmax(f1_score_test)]
    # elif cancer_name == 'prad': # this is balanced right?
    #     best_file = list_fichiers[np.argmax(accuracy_test)]
    best_file = list_fichiers[np.argmax(f1_score_test)]
    if directory.find('dt') != -1:
        for model in features_retenus:
            temp = []
            for el in model[0]:
                if el[2] > 0:
                    temp.append(el)
            var = ''
            for i, el in enumerate(temp):
                var += '&{}'.format(el[3])
                if i == 2:
                    break
            model_comptes.append(var)
    if directory.find('rf') != -1:
        for model in features_retenus:
            var = ''
            for el in model[0][:3]:
                var += '&{}'.format(el[3])
            model_comptes.append(var)

        features_retenus_flatten = [el[3] for liste in features_retenus for el in liste[0][:50]]
        for el in features_retenus_flatten:
            cnt_rf[el] += 1
    if directory.find('scm') != -1 and directory.find('group') == -1:
        for model in features_retenus:
            temp = []
            for el in model[0]:
                temp.append(el[1])
            var = ''
            for el in temp:
                var += '&{}'.format(el)
            model_comptes.append(var)
    if directory.find('scm') != -1 and directory.find('group') != -1:
        for fichier in glob("*.pck"):
            f = open(fichier, 'rb')
            d = pickle.load(f)
            groups_features.append(d['groups_rules'])
        for model in features_retenus:
            temp = []
            for el in model[0]:
                temp.append(el[1])
            var = ''
            for el in temp:
                var += '&{}'.format(el)
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
    d_temp = pickle.load(open(best_file, 'rb'))
    best_acc = np.round(d_temp['metrics']['accuracy'], 4)
    best_f1 = np.round(d_temp['metrics']['f1_score'], 4)
    best_pre = np.round(d_temp['metrics']['precision'], 4)
    best_rec = np.round(d_temp['metrics']['recall'], 4)
    best_feat = d_temp['rules_str']
    best_groups_feat = d_temp['groups_rules']
    if directory.find('dt') != -1:
        for model in features_retenus:
            temp = []
            for el in model[0]:
                if el[2] > 0:
                    temp.append(el)
            var = ''
            for i, el in enumerate(temp):
                var += '&{}'.format(el[3])
                if i == 2:
                    break
    if directory.find('rf') != -1:
        for model in features_retenus:
            var = ''
            for el in model[0][:3]:
                var += '&{}'.format(el[3])
    if directory.find('scm') != -1:
        for model in best_feat:
            temp = []
            for el in best_feat[0]:
                temp.append(el[1])
            var = ''
            for el in temp:
                var += '&{}'.format(el)
    var_groups = ''
    if best_groups_feat != []:
        for el in best_groups_feat:
            if type(el) == list:
                for gp in el:
                    var_groups += '&{}'.format(gp)
    with open(recap_table_file, 'a+') as f:
        f.write(f"{best_file}\t{best_acc}\t{best_f1}\t{best_pre}\t{best_rec}\t{var}\t{var_groups}\n")
    os.chdir(saving_repository)
    return np.round(np.mean(accuracy_test), 4), np.round(np.mean(f1_score_test), 4), \
           np.round(np.mean(precision_test), 4), np.round(np.mean(recall_test), 4), model_comptes, groups_features


def global_repo_analysis(directory, output_text_file='results_analysis', type_experiment='normal', sous_experiment_types=['dt', 'scm', 'rf'], plot_hist=True):
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
    # list_of_directories = os.listdir(directory)
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
            list_of_directories = os.listdir(directory)
            list_of_directories = [repository for repository in list_of_directories if repository.startswith(experiment)] 
            for repository in list_of_directories:
                acc_test, f1_test, precis_test, rec_test, model_comptes, _ = sub_repo_analysis(directory=repository,
                                                                                              output_text_file=output_text_file_experiment,
                                                                                              recap_table_file=recap_text_file_experiment,
                                                                                              plot_hist=plot_hist)
                acc_test_list.append(acc_test)
                f1_test_list.append(f1_test)
                precis_test_list.append(precis_test)
                rec_test_list.append(rec_test)
                model_comptes_list.append(model_comptes)
            with open(output_text_file_experiment, 'a+') as f:
                f.write('-' * 50)
                f.write('\n')
                f.write('Best model selected based on F1 Score\n')
                best_model_idx = np.argmax(np.asarray(f1_test_list)) # to be modified if i decied to go with the accuracy for PRAD vs f1 for BRCA
                f.write(f'Best Experiment is:{list_of_directories[best_model_idx]}\n')
                f.write(f'Results: Acc: {acc_test_list[best_model_idx]}\t F1: {f1_test_list[best_model_idx]}\t Prec: {precis_test_list[best_model_idx]}\t Rec: {rec_test_list[best_model_idx]} \n')
                f.write(f'Model comptes {model_comptes_list[best_model_idx]}\n')
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
            list_of_directories = [repository for repository in list_of_directories if repository.startswith(experiment)] 
            for repository in list_of_directories:
                acc_test, f1_test, precis_test, rec_test, model_comptes, groups_features = sub_repo_analysis(directory=repository,
                                                                                                            output_text_file=output_text_file_experiment,
                                                                                                            recap_table_file=recap_text_file_experiment,
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

@cli.command(help="Run the analysis results")
@click.option('--directory', type=str, default=None, help="""results path""")
@click.option('--output-text-file', type=str, default='normal_brca_results_analysis', help="""outout name file""")
@click.option('--type-experiment', type=str, default='normal', help="""type of experiment global or group_scm""")
@click.option('--sous-experiment-types', type=str, default='dt scm rf', help="""name of experiment""")
@click.option('--plot-hist/--no-plot-hist', default=False, help="""plot histogram""")
def run_analysis(directory, output_text_file, type_experiment, sous_experiment_types, plot_hist):
    if plot_hist:
        global_repo_analysis(directory, output_text_file, type_experiment, sous_experiment_types.split(), True)
    else:
        global_repo_analysis(directory, output_text_file, type_experiment, sous_experiment_types.split(), False)

# python experiments/experiments_utilities.py --directory /home/maoss2/project/maoss2/saving_repository_article/normal_experiments_brca --output-text-file normal_brca_results_analysis --type-experiment dt scm rf --plot-hist
# python experiments/experiments_utilities.py --directory /home/maoss2/project/maoss2/saving_repository_article/normal_experiments_prad --output-text-file normal_prad_results_analysis --type-experiment dt scm rf --plot-hist

# python experiments/experiments_utilities.py --directory /home/maoss2/project/maoss2/saving_repository_article/groups_TN_experiments --output-text-file group_scm_brca_results_analysis --type-experiment methyl_rna_iso_mirna_snp_clinical 
# python experiments/experiments_utilities.py --directory /home/maoss2/project/maoss2/saving_repository_article/groups_PRAD_experiments --output-text-file group_scm_prad_results_analysis --type-experiment all 

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
