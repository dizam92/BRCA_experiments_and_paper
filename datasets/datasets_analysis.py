# -*- coding: utf-8 -*-
__author__ = 'maoss2'
""" Fichier d'analyses de la dispersion des données. """
from datasets.datasets_utilitaires import *

methyl_450_file = "./methylome_450.tsv"
methyl_27_file = "./methylome_27.tsv"
rnaseq_genes_file = "./rnaseq_genes.tsv"
rnaseq_isoforms_file = "./rnaseq_genes.tsv"
snp_file = "./snp.tsv"
mirna_file = "./mirna.tsv"
label_file_triple_neg = "./labels_for_triple_negatives.tsv"


def load_data(fichier, label_file):
    """ A little utils function to made the code cleaner and reusable"""
    labels = pd.read_csv("%s" % label_file, index_col="example_id", sep="\t")
    data = pd.read_table(fichier)
    indexes = np.array(map(str, np.array(data["Unnamed: 0"])))
    data.set_index(indexes, inplace=True)
    data = data.T.loc[labels.index.values]
    data = data.loc[:, data.count() > 0]
    index_pos = labels.loc[labels['labels'] == 1].index.values
    index_neg = labels.loc[labels['labels'] == -1].index.values
    return data, index_pos, index_neg


def figure_1():
    """ Affiche la moyenne sur chaque attribut pour l'ensemble des données (attributs).
     Pourquoi je n'affiche plus les attribus au complets?
     Reponse: La figure devient ilisible. J'essaye d'afficher + de 25mille points.
     C'est quoi l'objectif: Ici je voulais voir la dispersion des données pour pouvoir trouver une facon plus
     intelligente de préprocesser les données. car pour l'instant on fait juste fillna(0)"""
    methyl_27, index_pos, index_neg = load_data(fichier=methyl_27_file, label_file=label_file_triple_neg)
    np.random.seed(42)
    attributs_to_be_plotted = np.random.choice(methyl_27.columns.values, 200)
    new_array = []
    for attribut_name in attributs_to_be_plotted:
        temp = methyl_27[attribut_name]
        new_array.append(temp[temp.notnull()].values)
    mean_values = [np.round(np.mean(el), 3) for el in new_array]
    sns.plt.figure(figsize=(20, 10))
    sns.plt.xlabel('attributs')
    sns.plt.ylabel('mean_values')
    sns.plt.plot(np.arange(0, len(mean_values)), mean_values, 'ro', label='mean_by_attributes')
    sns.plt.legend(loc='upper center', shadow=True)
    sns.plt.title('Moyenne des attributs (sans les nan)')
    sns.plt.savefig('mean_by_global_attributes.pdf')
    sns.plt.close()


def figure_2():
    """ Faire la meme chose que figure_1 sauf fait une distinction sur les labels et affiche les moyennes par labels.
     On remarque que la distribution est un peu différente. Je vais y aller avec le test de remplacer par la moyenne
     par phenotype lorsqu'il y a un nan """
    methyl_27, index_pos, index_neg = load_data(fichier=methyl_27_file, label_file=label_file_triple_neg)
    np.random.seed(42)
    attributs_to_be_plotted = np.random.choice(methyl_27.columns.values, 200)
    new_array_pos = []
    new_array_neg = []
    for attribut_name in attributs_to_be_plotted:
        temp_neg = methyl_27.loc[index_neg, attribut_name]
        temp_pos = methyl_27.loc[index_pos, attribut_name]
        new_array_pos.append(temp_pos[temp_pos.notnull()].values)
        new_array_neg.append(temp_neg[temp_neg.notnull()].values)
    mean_values_pos = [np.round(np.mean(el), 3) for el in new_array_pos]
    mean_values_neg = [np.round(np.mean(el), 3) for el in new_array_neg]
    sns.plt.figure(figsize=(20, 10))
    sns.plt.xlabel('attributs')
    sns.plt.ylabel('mean_values')
    sns.plt.plot(np.arange(0, len(mean_values_pos)), mean_values_pos, 'r-', label='mean_by_attributes_pos')
    sns.plt.plot(np.arange(0, len(mean_values_neg)), mean_values_neg, 'b-', label='mean_by_attributes_neg')
    sns.plt.legend(loc='upper center', shadow=True)
    sns.plt.title('Moyenne des attributs (sans les nan)')
    sns.plt.savefig('mean_by_phenotypes_attributes.pdf')
    sns.plt.close()


def statistics_on_missing_values():
    methyl_27_data = pd.read_table('methylome_27.tsv')
    indexes = np.array(map(str, np.array(methyl_27_data["Unnamed: 0"])))
    methyl_27_data.set_index(indexes, inplace=True)
    methyl_27_data.drop(["Unnamed: 0"], axis=1, inplace=True)
    valeurs_manquantes_dans_la_matrice = np.sum(methyl_27_data.T.isnull().sum().values)
    ratio = (valeurs_manquantes_dans_la_matrice / float(methyl_27_data.size)) * 100
    print('Methyl 27 & {} & {}({}) & {}'.format(valeurs_manquantes_dans_la_matrice, methyl_27_data.T.shape,
                                                methyl_27_data.size, ratio))

    methyl_450_data = pd.read_table('methylome_450.tsv')
    indexes = np.array(map(str, np.array(methyl_450_data["Unnamed: 0"])))
    methyl_450_data.set_index(indexes, inplace=True)
    methyl_450_data.drop(["Unnamed: 0"], axis=1, inplace=True)
    valeurs_manquantes_dans_la_matrice = np.sum(methyl_450_data.T.isnull().sum().values)
    ratio = (valeurs_manquantes_dans_la_matrice / float(methyl_450_data.size)) * 100
    print('Methyl 450 & {} & {}({}) & {}'.format(valeurs_manquantes_dans_la_matrice, methyl_450_data.T.shape,
                                                 methyl_450_data.size, ratio))

    methyl_fusion_data = pd.read_table('methylation_fusion.tsv')
    indexes = np.array(map(str, np.array(methyl_fusion_data["Unnamed: 0"])))
    methyl_fusion_data.set_index(indexes, inplace=True)
    methyl_fusion_data.drop(["Unnamed: 0"], axis=1, inplace=True)
    valeurs_manquantes_dans_la_matrice = np.sum(methyl_fusion_data.T.isnull().sum().values)
    ratio = (valeurs_manquantes_dans_la_matrice / float(methyl_fusion_data.size)) * 100
    print('Methyl fusion & {} & {}({}) & {}'.format(valeurs_manquantes_dans_la_matrice, methyl_fusion_data.shape,
                                                    methyl_fusion_data.size, ratio))

    rna_isoforms_data = pd.read_table('rnaseq_isoforms.tsv')
    indexes = np.array(map(str, np.array(rna_isoforms_data["Unnamed: 0"])))
    rna_isoforms_data.set_index(indexes, inplace=True)
    rna_isoforms_data.drop(["Unnamed: 0"], axis=1, inplace=True)
    valeurs_manquantes_dans_la_matrice = np.sum(rna_isoforms_data.T.isnull().sum().values)
    ratio = (valeurs_manquantes_dans_la_matrice / float(rna_isoforms_data.size)) * 100
    print('RNAs isoforms & {} & {}({}) & {}'.format(valeurs_manquantes_dans_la_matrice, rna_isoforms_data.T.shape,
                                                    rna_isoforms_data.size, ratio))

    mirna_data = pd.read_table('mirna.tsv')
    indexes = np.array(map(str, np.array(mirna_data["Unnamed: 0"])))
    mirna_data.set_index(indexes, inplace=True)
    mirna_data.drop(["Unnamed: 0"], axis=1, inplace=True)
    valeurs_manquantes_dans_la_matrice = np.sum(mirna_data.T.isnull().sum().values)
    ratio = (valeurs_manquantes_dans_la_matrice / float(mirna_data.size)) * 100
    print('miRNAs & {} & {}({}) & {}'.format(valeurs_manquantes_dans_la_matrice, mirna_data.T.shape,
                                             mirna_data.size, ratio))

    snps_data = pd.read_table('snp.tsv')
    indexes = np.array(map(str, np.array(snps_data["Unnamed: 0"])))
    snps_data.set_index(indexes, inplace=True)
    snps_data.drop(["Unnamed: 0"], axis=1, inplace=True)
    valeurs_manquantes_dans_la_matrice = np.sum(snps_data.T.isnull().sum().values)
    ratio = (valeurs_manquantes_dans_la_matrice / float(snps_data.size)) * 100
    print('miRNAs & {} & {}({}) & {}'.format(valeurs_manquantes_dans_la_matrice, snps_data.T.shape,
                                             snps_data.size, ratio))


if __name__ == '__main__':
    figure_2()
