# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from glob import glob
from itertools import combinations, product
from os import makedirs
from os.path import abspath, dirname, exists, join

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from experiments.experiments_utilities import data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna, load_data, load_prad_data, data_repository, data_prad
from datasets.brca_builder import select_features_based_on_mad
goa_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/goa_human_isoform_valid.gaf'
biogrid_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/BIOGRID-ORGANISM-Homo_sapiens-3.5.178.tab.txt'
genesID_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/Results_genes.txt'
mirna_dat_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/miRNA.dat'

c2_file_canonical_pathways = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c2.cp.v7.1.symbols.gmt'
c3_file_mirna = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/c3.mir.v7.1.symbols.gmt'

#-------------------------------------Old fashion to build the different type of groups ------------------------------------#
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


def construction_pathway_gene_groups_tcga(data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,
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


def construction_pathway_clusters_groups(data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,
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


def construction_pathway_random_groups(data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,
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


def construction_pathway_views_groups(data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,
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


def construction_pathway_file(data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,
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

        
def construction_biogrid_pathway_file(data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna, return_views='all', output_file_name=''):
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
    _, _, features_names, _ = load_data(data=data_path, return_views=return_views)
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


def main_construct():
    print()
    # for view in return_views:
    #     construction_pathway_gene_groups_tcga(return_views=view)
    #
    # for view in return_views:
    #     construction_pathway_random_groups(return_views=view)
    #
    # for view in return_views:
    #     construction_pathway_views_groups(return_views=view)

    # construction_biogrid_pathway_file(output_file_name='pathways_biogrid')

    # for dictionary in list_dict:
    #     if dictionary == c2_pickle_dictionary:
    #         output_file_name = 'pathway_file_c2_curated_groups'
    #     else:
    #         output_file_name = 'pathway_file_c5_curated_groups'
    #     for view in return_views:
    #         construction_pathway_file(return_views=view, dictionnaire=dictionary, output_file_name=output_file_name)

    # for view in return_views:
    #     construction_pathway_clusters_groups(model_loaded=False,
    #                                          return_views=view,
    #                                          model_agglomeration_file_name='feature_agglomeration_model_{}.pck'.format(
    #                                              view))


def load_mirna_dat(data_path=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna, return_views='all'):
    """
    It's a great idea but it is not worth it though. We have much more of the miRNA not related 
    to a cluster at all even though i was using mirDB. Leaving it here to have a  proof of work.
    Utility fonction to read the miRNA dat file and to do the groups(pathways) building for them
    
    """
    _, _, features_names, _ = load_data(data=data_path, return_views=return_views)
    features_names = list(features_names)
    mirna_features_names = [feature for feature in features_names if feature.startswith('hsa')]
    fichier = open(mirna_dat_file, 'r')
    lines = fichier.readlines()
    lines_to_keep = [] 
    for line in lines:
        if  line.startswith('ID') or line.startswith('DR   RFAM'):
            lines_to_keep.append(line.strip('\n'))
    dict_mirna_features_to_cluster = {feature: [] for feature in mirna_features_names}
    for feature in mirna_features_names:
        for idx, line in enumerate(lines_to_keep):
            if line.find(feature) != -1:
                if lines_to_keep[idx + 1].find('RFAM') != -1:
                    valeur = lines_to_keep[idx + 1].split(';')[-1].strip('.')
                    dict_mirna_features_to_cluster[feature].append(valeur)
    return dict_mirna_features_to_cluster

#-------------------------------------Old fashion to build the different type of groups ------------------------------------#
data_to_extract_features_from = f"{data_repository}/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5"
    
def load_go_idmapping():
    """
    Utility function to map go_ids
    """
    res = defaultdict(set)
    map_genes_ids_to_function_name = defaultdict(set)
    with open(goa_file, 'r') as fin:
        lines = fin.readlines()
        lines = lines[19:]
        for line in lines:
            content = line.split('\t')
            geneid = content[2].lower()
            goid = content[4]
            genes_function = content[9]
            synonyms_geneids = content[10].split('|')[:-1]  # exclure le dernier element car c'est une ancienne appelation dnas une bd: hCG_1646942 ca n'a pas rapport aux sites cg du tout
            res[geneid].add(goid)
            map_genes_ids_to_function_name[geneid].add(genes_function)
            for synonym_id in synonyms_geneids:
                res[synonym_id].add(goid)
                map_genes_ids_to_function_name[synonym_id].add(genes_function)
    nb_go_terms = sum([len(r) for i, r in res.items()])
    nb_genes = len(res)
    print("GO: {} genes and {} GO terms".format(nb_genes, nb_go_terms))
    with open(f'{data_repository}/mapping_genes_ids_to_function.pck', 'wb') as f:
        pickle.dump(map_genes_ids_to_function_name, f)
    with open(f'{data_repository}/mapping_genes_ids_to_GO_ids.pck', 'wb') as f:
        pickle.dump(res, f)
    return res

def load_biogrid_network():
    """
    Utility function to load biogrid_network in graph nx
    """
    go_terms = load_go_idmapping()
    G = nx.Graph()
    d = pd.read_csv(biogrid_file, delimiter='\t', skiprows=range(35))
    simple_edges = [(gene_a, d['OFFICIAL_SYMBOL_B'].values[idx]) for idx, gene_a in enumerate(d['OFFICIAL_SYMBOL_A'].values)]
    aliases_for_a = [el.split('|') for el in d['ALIASES_FOR_A'].values]
    aliases_for_b = [el.split('|') for el in d['ALIASES_FOR_B'].values]
    alias_edges = [product(aliases_for_a[idx], aliases_for_b[idx]) for idx in range(len(aliases_for_a))]
    alias_edges_list = []
    for el in alias_edges:
        alias_edges_list.extend(list(el))
    simple_edges.extend(alias_edges_list)
    simple_edges = set(simple_edges)
    G.add_edges_from(list(simple_edges))
    for node in G.nodes:
        G.nodes[node]["go_terms"] = go_terms[node]
    print("BioGRID: {} genes and {} interactions".format(G.number_of_nodes(), G.number_of_edges()))
    with open(f'{data_repository}/graph_interactions_biogrids.pck', 'wb') as f:
        pickle.dump(G, f)
    return G, simple_edges
 
def load_mirna_pathways():
    """
    Utility function to build a dictionnary with the miRNA from msigDB and their target genes:
    All microRNA targets, gene symbols (microRNA targets, 2598 gene sets)
    """
    with open(c3_file_mirna, 'r') as f:
        c3_mirna_lines = f.readlines()
    dico_mirna = defaultdict()
    for line in c3_mirna_lines:
        line = line.strip('\n')
        content = line.split('\t')
        pathways_title = content[0].lower()
        genes_in_pathways = content[2:]
        if pathways_title.startswith('mir'):
            mirna_keys = pathways_title.split('_')
            mirna_keys = [el for el in mirna_keys if el not in ['3p', '5p', '2', '1', '3']]
            for clee in mirna_keys:
                if clee not in dico_mirna.keys():
                    dico_mirna[clee.lower()] = genes_in_pathways
                else:
                    dico_mirna[clee.lower()].extend(genes_in_pathways)
        elif pathways_title.startswith('let'):
            if pathways_title not in dico_mirna.keys():
                dico_mirna[pathways_title] = genes_in_pathways
            else:
                dico_mirna[pathways_title].extend(genes_in_pathways)
        else:
            mirna_keys = pathways_title.split('_')[1:]
            for clee in mirna_keys:
                if clee not in dico_mirna.keys():
                    dico_mirna[clee.lower()] = genes_in_pathways
                else:
                    dico_mirna[clee.lower()].extend(genes_in_pathways)
    with open(f'{data_repository}/mirna_mapping_to_genes_from_msigDB.pck', 'wb') as f:
        pickle.dump(dico_mirna, f)
    return dico_mirna

def load_canonical_pathways():
    """
    Utility function to build a dictionnary with the canonical pathways from msigDB: including Biocarta; KEGG; PID; REACTOME
    (Canonical pathways, 2232 gene sets)
    """
    with open(c2_file_canonical_pathways, 'r') as f:
        c2_cp_lines = f.readlines()
    dico_cp = defaultdict()
    for line in c2_cp_lines:
        line = line.strip('\n')
        content = line.split('\t')
        pathways_title = content[0].lower()
        genes_in_pathways = content[2:]
        dico_cp[pathways_title] = genes_in_pathways
    with open(f'{data_repository}/canonical_pathways_biocarta_kegg_pid_reactome_genes_set.pck', 'wb') as f:
        pickle.dump(dico_cp, f)
    return dico_cp 

def new_function():
    
    graph_biogrid = pickle.load(open(f'{data_repository}/graph_interactions_biogrids.pck', 'rb'))
    canonical_pathways = pickle.load(open(f'{data_repository}/canonical_pathways_biocarta_kegg_pid_reactome_genes_set.pck', 'rb'))
    mirna_pathways = pickle.load(open(f'{data_repository}/mirna_mapping_to_genes_from_msigDB.pck', 'rb'))
    _, _, features_names, _ = load_data(data=data_path, return_views=return_views, drop_inexistant_features=True, mad_selection=True)
    features_names = list(features_names)
    dico_results = {feature: [] for feature in features_names}
    adjacency_matrix = np.asarray(list(graph_biogrid.adjacency()))
    biogrid_pathways = [list(el[1].keys()) + [el[0]] for el in adjacency_matrix] # Obtention des pathways de biogrids 78384
    # biogrid_pathways.append('clinical View') # biogrid_pathways[20170]
    # biogrid_pathways.append('unknown') # biogrid_pathways[20171]
    features_mirna_to_index = {feature: idx for idx, feature in enumerate(features_names) if feature.startswith('hsa')}
    features_rna_to_index = {feature: idx for idx, feature in enumerate(features_names) if feature.find('|') != -1}
    features_iso_to_index = {feature: idx for idx, feature in enumerate(features_names) if feature.startswith('uc')}
    features_cg_to_index = {feature: idx for idx, feature in enumerate(features_names) if feature.startswith('cg')}
    # features_clinical_to_index = {feature: idx for idx, feature in enumerate(features_names) if feature in ['race', 'gender']}
    # len(biogrid_pathways) : 78384
    # len(canonical_pathways.values()): 2232
    # len(mirna_pathways.values()): 1756
    all_pathways_list = [] # len(all_pathways_list): 82372
    all_pathways_list.extend(biogrid_pathways)
    all_pathways_list.extend(list(canonical_pathways.values()))
    all_pathways_list.extend(list(mirna_pathways.values()))
    # miRNA represente les gens TARGETTÉ par les miRNA ca ne veut pas dire que les genes interagissent "necessairement entre eux". 
    # Contrairement aux 2 autres types de liste qui disent clairement "tous les genes dans cette liste soit 
    # 1- ils interagissent ensemble 2- soit ils appartiennent à un meme pathways" . Du coup par defaut tous les miRNA auront leur groupes à part, 
    # et on va rajouter les autres groupes si jamais un gene est dans un autre pathways des 2 premiers. 
    # Maintenant comment on fait ca sans que ce soit le bordel avec les boucles imbriquées et la complexité???? c'est caaaaaa
    groups_idx = [f'G_{idx}' for idx, _ in enumerate(all_pathways_list)]
    groups_idx_biogrid = groups_idx[:78384]
    groups_idx_canonical = groups_idx[78384:80616]
    groups_idx_miRNA = groups_idx[80616:]
    groups_idx_inexistant = 'G_82372'
    hsa_features_original = list(features_mirna_to_index.keys())
    hsa_features = [''.join(el.split('-')[1:3]) for el in hsa_features_original]
    mirna_pathways_keys_to_list = np.asarray(list(mirna_pathways.keys()))
    mirna_pathways_values_to_list = list(mirna_pathways.values())
    canonical_pathways_values_to_list = list(canonical_pathways.values())
    for features_position_in_dico, mirna_id in enumerate(hsa_features):
        pos = np.where(mirna_pathways_keys_to_list == mirna_id)[0]
        if len(pos) != 0:
            dico_results[hsa_features_original[features_position_in_dico]].append(groups_idx_miRNA[pos[0]])
        else:
            dico_results[hsa_features_original[features_position_in_dico]].append(groups_idx_inexistant)
    # I am not sure if this is necessary. Because we have the targets of the miRNAs, it doesnt need to be mixed with
    # the other pathways car ca ne signifie pas la meme chose. Mais au niveau de l'analyse on peut resortir leur 
    # pathways en mode yo that's why u did this or that
    # for features_position_in_dico, mirna_genes_targets_list in enumerate(mirna_pathways_values_to_list):
    #     for mirna_genes_target in mirna_genes_targets_list:
    #         for biogrid_pathways_list_pos, biogrid_pathways_list in enumerate(biogrid_pathways):
    #             if mirna_genes_target in biogrid_pathways_list:
    #                 dico_results[hsa_features_original[features_position_in_dico]].append(groups_idx_biogrid[biogrid_pathways_list_pos])
    #         for canonical_pathways_list_pos, canonical_pathways_list in enumerate(canonical_pathways_values_to_list):
    #             if mirna_genes_target in canonical_pathways_list:
    #                 dico_results[hsa_features_original[features_position_in_dico]].append(groups_idx_canonical[canonical_pathways_list_pos])
    
    all_pathways_list_biogrid_canonical = all_pathways_list[:80616]
    groups_idx_all_pathways_list_biogrid_canonical = groups_idx[:80616]
    
    rna_features_original = list(features_rna_to_index.keys())
    rna_features = [el.split('|')[0] for el in rna_features_original]
    iso_features_original = list(features_iso_to_index.keys())
    iso_features = [el.split('_')[-1] for el in iso_features_original]
    
    cg_features_original = list(features_cg_to_index.keys())
    cg_features = [el.split('_')[-1] for el in cg_features_original]
    
    for pos, gene in enumerate(rna_features):
        for pathway_list_pos, pathway_list in enumerate(all_pathways_list_biogrid_canonical):
            if gene in pathway_list:
                dico_results[rna_features_original[pos]].append(groups_idx_all_pathways_list_biogrid_canonical[pathway_list_pos])
            if iso_features[pos] in pathway_list:
                dico_results[iso_features_original[pos]].append(groups_idx_all_pathways_list_biogrid_canonical[pathway_list_pos])
            if cg_features[pos].find(';') != -1:
                genes = cg_features[pos].split(';')
                for g in genes:
                    if g in pathway_list:
                        dico_results[cg_features_original[pos]].append(groups_idx_all_pathways_list_biogrid_canonical[pathway_list_pos])
            if cg_features[pos].find(';') == -1:
                if cg_features[pos] in pathway_list:
                    dico_results[cg_features_original[pos]].append(groups_idx_all_pathways_list_biogrid_canonical[pathway_list_pos])

    features_not_in_any_pathway = []
    for cle, valeur in dico_results.items():
        if valeur == []:
            dico_results[cle].append(groups_idx_inexistant)  
            features_not_in_any_pathway.append(cle)
     
    with open(f'{data_repository}/groups2genes_biogrid_msigDB.pck', 'wb') as f: # Dict: {'G_number': []} # 
        dict_results = {el: all_pathways_list[idx] for idx, el in enumerate(groups_idx)}
        pickle.dump(dict_results, f)
    with open(f'{data_repository}/featuresNotInAnyPathways.pck', 'wb') as f:
        pickle.dump(features_not_in_any_pathway, f)    
    with open(f'{data_repository}/{output_file_name}', 'wb') as f:
        pickle.dump(dico_results, f)
             
    
def build_dictionnary_groups(data_path=data_to_extract_features_from, return_views='all', output_file_name=''):
    """
    Utility function to build pathway file of the groups to be loaded in LearnFromBiogridGroup
    The pathway will be stored in pickle file
    Major change: gene.lower() is no longer. Idk why i was using it but it is not doing what it supposed to do
    genes_in_dataset: 6311
    genes_in_biogrid_pathways: 78397
    genes_in_common: 5940
    Du coup on a environ: 371 features (genes) a ne pas considérer car appartenant à inexistant group. Pour l'instant je vais
    construire le dico avec eux et aller ensuite voir dans reactome s'ils y sont. 
    Aussi je peux trouver les miRNA quel genes ils influencent avec msigDB.
    Maintenant la question est: est-ce qu'on mélange les BD? Clairement va falloir utliser msigDB pour les miRNA... 
    maintenant est-ce que réactome et biogrid speak the same language? 
    Args:
        data_path: str, data path
        return_views: str, correct view for the group
        output_file_name: str, output file name
    Return:
        output_file_name.pck
    """
    graph, _ = load_biogrid_network()
    _, _, features_names, _ = load_data(data=data_path, return_views=return_views, drop_inexistant_features=True, mad_selection=True)
    features_names = list(features_names)
    adjacency_matrix = np.asarray(list(graph.adjacency()))
    # nodes = np.asarray(list(graph.nodes))
    dico_results = {feature: [] for feature in features_names}
    # noeud + et toutes leurs interactions; 
    # el[0] is always the node and list(el[1].keys()) the genes whose are interacting with the node
    biogrid_pathways = [list(el[1].keys()) + [el[0]] for el in adjacency_matrix] # Obtention des pathways de biogrids
    biogrid_pathways.append('miRNA') # biogrid_pathways[20170]
    biogrid_pathways.append('clinical View') # biogrid_pathways[20171]
    biogrid_pathways.append('unknown') # biogrid_pathways[20172]
    # Biogrids rank from 1 to 20170 + 3: pathways
    for feature in features_names:
        if feature.find('_') != -1:
            # I went step by step for the comprehension but remember the gene is always at the end of the feature so use the [-1] access
            split_results = feature.split('_')
            gene_cible = split_results[-1]
            if gene_cible.find(';'): # Cas ou on a un feature lié à 2 ou plus genes (surtout pour les cpg)
                gene_cibles = gene_cible.split(';')
                for gene in gene_cibles: # Recupere chaque gene et on remplit le dico
                    for idx, pathway_in_biogrid in enumerate(biogrid_pathways):
                        # if gene.lower() in pathway_in_biogrid:
                        if gene in pathway_in_biogrid: # idk why i was using lower but it is clearly not working the way it supposed to do
                            dico_results[feature].append('G_{}'.format(idx))
            else: # Different du premier if du coup le feature est link à un seul gene et on remplit le dictionnaire de facon adéquate
                for idx, pathway_in_biogrid in enumerate(biogrid_pathways):
                    if gene_cible in pathway_in_biogrid:
                        dico_results[feature].append('G_{}'.format(idx))
        elif feature.find('|') != -1: # Here the gene is the 1st element always since it's directly the RNA view only
            split_results = feature.split('|')
            gene_cible = split_results[0].lower()
            for idx, pathway_in_biogrid in enumerate(biogrid_pathways):
                if gene_cible in pathway_in_biogrid:
                    dico_results[feature].append('G_{}'.format(idx))
        elif feature.startswith('hsa'):  # MiRNA View: faire le traitement directement
            dico_results[feature].append('G_20170')
        else:
            dico_results[feature].append('G_20171')
    for cle, valeur in dico_results.items():
        if valeur == []:
            dico_results[cle].append('G_20172')

    with open(f'{data_repository}/groups2genes_biogrid.pck', 'wb') as f: # Dict: {'G_number': []} # 
        dict_results = {f'G_{idx}': el for idx, el in enumerate(biogrid_pathways)}
        pickle.dump(dict_results, f)
    with open(f'{data_repository}/{output_file_name}', 'wb') as f:
        pickle.dump(dico_results, f)

def build_dictionnary_groups_prad(data_path=data_prad, return_views='all', output_file_name=''):
    graph, _ = load_biogrid_network()
    _, _, features_names, _ = load_prad_data(data=data_path, return_views=return_views)
    features_names = list(features_names)
    adjacency_matrix = np.asarray(list(graph.adjacency()))
    dico_results = {feature: [] for feature in features_names}
    biogrid_pathways = [list(el[1].keys()) + [el[0]] for el in adjacency_matrix] # Obtention des pathways de biogrids
    for feature in features_names:
        split_results = feature.split('_')
        gene_cible = split_results[-1].lower()
        for idx, pathway_in_biogrid in enumerate(biogrid_pathways):
            if gene_cible in pathway_in_biogrid:
                dico_results[feature].append('G_{}'.format(idx))
    for cle, valeur in dico_results.items():
        if valeur == []:
            dico_results[cle].append('G_20172') # unknown
    with open(f'{data_repository}/{output_file_name}', 'wb') as f:
        pickle.dump(dico_results, f)   
                
if __name__ == "__main__":
    build_dictionnary_groups(data_path=data_to_extract_features_from, return_views='all', output_file_name='groups2pathwaysTN_biogrid.pck')
    build_dictionnary_groups_prad(data_path=data_prad, return_views='all', output_file_name='groups2pathwaysPRAD_biogrid.pck')
    