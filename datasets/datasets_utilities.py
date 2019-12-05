# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import random
import h5py
import re
import os
import pandas as pd
import numpy as np

from glob import glob
from copy import deepcopy

# ******************************************** Global Values Section ***************************************************
project_path_on_is2 = '/is2/projects/JC_Cancers/TCGA_raw/'
ids_pattern = re.compile(r'(TCGA-\w+-\w+)', re.U | re.M | re.I)

label_file_triple_all = "/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/labels_for_triple_negatives_all.tsv"
new_label_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/clinical_views_labels_and_proba_copie.csv'

methyl_450_file = "/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/methylome_450.tsv"
methyl_27_file = "/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/methylome_27.tsv"
rnaseq_genes_file = "/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/rnaseq_genes.tsv"
rnaseq_isoforms_file = "/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/rnaseq_isoforms.tsv"
snp_file = "/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/snp.tsv"
mirna_file = "/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/mirna.tsv"
new_clinical_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/new_clinical_view_copie.tsv'
old_clinical_file = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/clinical_view.tsv'
# **********************************************************************************************************************


class BuildOmicsDatasets(object):
    def __init__(self, cancer_name, methyl_path_450, methyl_path_27, rnaseq_path, snps_path, mirna_file,
                 saving_repertory='./', label_file=None):
        self.saving_repertory = saving_repertory
        self.methyl_path_450 = methyl_path_450
        self.methyl_path_27 = methyl_path_27
        self.rnaseq_path = rnaseq_path
        self.snps_path = snps_path
        self.mirna_file = mirna_file
        self.cancer_name = cancer_name
        self.labels_file = label_file

    @staticmethod
    def method_to_fill_nan(data, idx_pos, idx_neg, filling_type='zero'):
        """ Replace nan values per zero, mean or median
         Args:  data, pandas dataset
                idx_pos, index positive, list
                idx_neg, index negative, list
                filling_type, str, mean, zero, median
        Return: data, a pandas dataframe
        """
        if filling_type == 'zero':
            data.fillna(0, inplace=True)
            return data
        if filling_type == 'mean':
            data_pos = np.asarray(data.loc[idx_pos].values, dtype=np.float64)
            data_neg = np.asarray(data.loc[idx_neg].values, dtype=np.float64)
            col_mean_data_pos = np.nanmean(data_pos, axis=0)
            col_mean_data_neg = np.nanmean(data_neg, axis=0)
            # Find index that you need to replace
            inds_data_pos = np.where(np.isnan(data_pos))
            inds_data_neg = np.where(np.isnan(data_neg))
            # Place column means in the indices. Align the arrays using take
            data_pos[inds_data_pos] = np.take(col_mean_data_pos, inds_data_pos[1])
            data_neg[inds_data_neg] = np.take(col_mean_data_neg, inds_data_neg[1])
            # Concatenate the numpy array
            matrix = np.vstack((data_pos, data_neg))
            # Build the new pandas dataFrame
            new_dataFrame = pd.DataFrame(data=matrix, index=data.index.values, columns=data.columns.values)
            return new_dataFrame
        if filling_type == 'median':
            data_pos = np.asarray(data.loc[idx_pos].values, dtype=np.float64)
            data_neg = np.asarray(data.loc[idx_neg].values, dtype=np.float64)
            col_mean_data_pos = np.nanmedian(data_pos, axis=0)
            col_mean_data_neg = np.nanmedian(data_neg, axis=0)
            # Find indicies that you need to replace
            inds_data_pos = np.where(np.isnan(data_pos))
            inds_data_neg = np.where(np.isnan(data_neg))
            # Place column means in the indices. Align the arrays using take
            data_pos[inds_data_pos] = np.take(col_mean_data_pos, inds_data_pos[1])
            data_neg[inds_data_neg] = np.take(col_mean_data_neg, inds_data_neg[1])
            # Concatenate the numpy array
            matrix = np.vstack((data_pos, data_neg))
            # Build the new pandas dataFrame
            new_dataFrame = pd.DataFrame(data=matrix, index=data.index.values, columns=data.columns.values)
            return new_dataFrame

    def build_methylome_450_tsv(self):
        """
        Build the methylome 450 view
        :return: methylome_450.tsv
        """
        os.chdir(self.methyl_path_450)
        new_list_dataframe = []
        omics_column_ids = []
        temp_list = []
        for fichier in glob("*-01A-*.txt"):
            temp_list.append(ids_pattern.search(fichier).group(0))
            temp_data_frame = pd.read_table('{}'.format(fichier), header='infer', skiprows=[0])
            temp_data_frame_methylation_value = temp_data_frame["Beta_value"].values
            omics_column_ids.append(temp_data_frame["Composite Element REF"].values)
            new_list_dataframe.append(temp_data_frame_methylation_value)
        new_list_dataframe = np.asarray(new_list_dataframe)
        data = pd.DataFrame(new_list_dataframe.T, index=omics_column_ids[0], columns=temp_list)
        saving_file = self.saving_repertory.format('methylome_450.tsv')
        data.to_csv(saving_file, sep="\t", encoding='utf-8')

    def build_methylome_27_tsv(self):
        """
        Build the methylome 27 view
        :return: methylome_27.tsv
        """
        os.chdir(self.methyl_path_27)
        new_list_dataframe = []
        omics_column_ids = []
        temp_list = []
        for fichier in glob("*-01A-*.txt"):  # TODO: Change for 03A for LAML case uniquement
            temp_list.append(ids_pattern.search(fichier).group(0))
            temp_data_frame = pd.read_table('{}'.format(fichier), header='infer', skiprows=[0])
            temp_data_frame_methylation_value = temp_data_frame["Beta_value"].values
            omics_column_ids.append(temp_data_frame["Composite Element REF"].values)
            new_list_dataframe.append(temp_data_frame_methylation_value)
        new_list_dataframe = np.asarray(new_list_dataframe)
        data = pd.DataFrame(new_list_dataframe.T, index=omics_column_ids[0], columns=temp_list)
        saving_file = self.saving_repertory.format('methylome_27.tsv')
        data.to_csv(saving_file, sep="\t", encoding='utf-8')

    def build_rnaseq_tsv(self):
        """
        Build the rnaseq view
        :return: rnaseq_isoforms.tsv, rnaseq_genes.tsv
        """
        os.chdir(self.rnaseq_path)
        new_list_dataframe = []
        omics_column_ids = []
        temp_list = []
        # La premiere partie est pour construire la correspondance entre les fichiers et les IDs
        if len(self.rnaseq_path) == 93:
            manifest_file = pd.read_table('{}/file_manifest.txt'.format(self.rnaseq_path[0:49]), header='infer')
        elif len(self.rnaseq_path) == 91:
            manifest_file = pd.read_table('{}/file_manifest.txt'.format(self.rnaseq_path[0:47]), header='infer')
        else:
            manifest_file = pd.read_table('{}/file_manifest.txt'.format(self.rnaseq_path[0:51]), header='infer')
        manifest_file = manifest_file.loc[manifest_file['Platform Type'] == 'RNASeqV2']
        sample_list = manifest_file["Sample"].values
        file_name = manifest_file["File Name"].values
        barcode_name = manifest_file["Barcode"].values
        matching_name_isoforms = {}
        matching_name_genes = {}
        for i in range(len(file_name)):
            if ".rsem.isoforms.normalized_results" in file_name[i] and "-01A-" in barcode_name[i]:
                if sample_list[i][0:12] in matching_name_isoforms.keys():
                    continue
                else:
                    matching_name_isoforms[sample_list[i][0:12]] = file_name[i]

            if ".rsem.genes.normalized_results" in file_name[i] and "-01A-" in barcode_name[i]:
                if sample_list[i][0:12] in matching_name_genes.keys():
                    continue
                else:
                    matching_name_genes[sample_list[i][0:12]] = file_name[i]

        # Section Isoforms normalized
        for ID in matching_name_isoforms.keys():
            temp_list.append(ID)
            path_fichier = '{}/{}'.format(self.rnaseq_path, matching_name_isoforms[ID])
            # if it exists in the right directory read it. Else, go watch for it in the other directory
            if os.path.exists('{}'.format(path_fichier)):
                temp_data_frame = pd.read_table('{}'.format(matching_name_isoforms[ID]), header='infer')
            else:
                path_fichier = '{}UNC__IlluminaGA_RNASeqV2/Level_3/{}'.format(self.rnaseq_path[:-35],
                                                                              matching_name_isoforms[ID])
                temp_data_frame = pd.read_table('{}'.format(path_fichier), header='infer')

            temp_data_frame_rnaseq_value = temp_data_frame["normalized_count"].values
            omics_column_ids.append(temp_data_frame["isoform_id"].values)
            # take the rnaseq value for each patient
            new_list_dataframe.append(temp_data_frame_rnaseq_value)
        new_list_dataframe = np.array(new_list_dataframe)
        data = pd.DataFrame(new_list_dataframe.T, index=omics_column_ids[0], columns=temp_list)
        saving_file = self.saving_repertory.format('rnaseq_isoforms.tsv')
        data.to_csv(saving_file, sep="\t", encoding='utf-8')

        # Section genes normalized
        new_list_dataframe = []
        omics_column_ids = []
        temp_list = []
        for ID in matching_name_genes.keys():
            temp_list.append(ID)
            path_fichier = '{}/{}'.format(self.rnaseq_path, matching_name_genes[ID])
            # if it exists in the right directory read it. Else, go watch for it in the other directory
            if os.path.exists('{}'.format(path_fichier)):
                temp_data_frame = pd.read_table('{}'.format(matching_name_genes[ID]), header='infer')
            else:
                path_fichier = '{}UNC__IlluminaGA_RNASeqV2/Level_3/{}'.format(self.rnaseq_path[:-35],
                                                                              matching_name_genes[ID])
                temp_data_frame = pd.read_table('{}'.format(path_fichier), header='infer')

            temp_data_frame_rnaseq_value = temp_data_frame["normalized_count"].values
            omics_column_ids.append(temp_data_frame["gene_id"].values)
            # take the rnaseq value for each patient
            new_list_dataframe.append(temp_data_frame_rnaseq_value)
        new_list_dataframe = np.array(new_list_dataframe)
        data = pd.DataFrame(new_list_dataframe.T, index=omics_column_ids[0], columns=temp_list)
        saving_file = self.saving_repertory.format('rnaseq_genes.tsv')
        data.to_csv(saving_file, sep="\t", encoding='utf-8')

    def build_mirna_tsv(self):
        """
        Build the mirna view
        :return: mirna.tsv
        """
        os.chdir(self.mirna_file)
        new_list_dataframe = []
        omics_column_ids = []
        temp_list = []
        matching_name = {}
        if len(self.mirna_file) == 96:
            manifest_file = pd.read_table('{}/file_manifest.txt'.format(self.mirna_file[0:50]), header='infer')
        elif len(self.mirna_file) == 94:
            manifest_file = pd.read_table('{}/file_manifest.txt'.format(self.mirna_file[0:48]), header='infer')
        else:
            manifest_file = pd.read_table('{}/file_manifest.txt'.format(self.mirna_file[0:46]), header='infer')
        manifest_file = manifest_file.loc[manifest_file['Platform Type'] == 'miRNASeq']
        sample_list = manifest_file["Sample"].values
        file_name = manifest_file["File Name"].values
        barcode_name = manifest_file["Barcode"].values
        for i in range(len(file_name)):
            if ".mirna.quantification" in file_name[i] and "-01A-" in barcode_name[i]:
                # verify if the key already exist
                if sample_list[i][0:12] in matching_name.keys():
                    continue
                else:
                    matching_name[sample_list[i][0:12]] = file_name[i]
        for ID in matching_name.keys():
            temp_list.append(ID)
            path_fichier = '{}/{}'.format(self.mirna_file, matching_name[ID])
            if os.path.exists('{}'.format(path_fichier)):
                temp_data_frame = pd.read_table('{}'.format(matching_name[ID]), header='infer')
            else:
                path_fichier = '{}BCGSC__IlluminaGA_miRNASeq/Level_3/{}'.format(self.mirna_file[:-37],
                                                                                matching_name[ID])
                temp_data_frame = pd.read_table('{}'.format(path_fichier), header='infer')

            # We can use the read_count attribute here too
            temp_data_frame_mirnaseq_value = temp_data_frame["reads_per_million_miRNA_mapped"].values
            omics_column_ids.append(temp_data_frame["miRNA_ID"].values)
            # take the mirnaseq value for each patient
            new_list_dataframe.append(temp_data_frame_mirnaseq_value)
        new_list_dataframe = np.array(new_list_dataframe)
        data = pd.DataFrame(new_list_dataframe.T, index=omics_column_ids[0], columns=temp_list)
        # go back to our directory to write the results
        saving_file = self.saving_repertory.format('mirna.tsv')
        data.to_csv(saving_file, sep="\t", encoding='utf-8')

    def build_snp_tsv(self):
        """
        Build the snp view
        :return: snp.tsv
        """

        def trim_example_id(id):
            return id[: 12]

        snp_data = pd.read_csv(self.snps_path, sep="\t", index_col="Tumor_Sample_Barcode")
        drop_columns = ["Center", "Ncbi_Build", "Entrez_Gene_Id", "Archive_Name", "Strand", "Dbsnp_Rs",
                        "Dbsnp_Val_Status",
                        "Verification_Status", "Sequencer", "Validation_Status", "Validation_Method", "Score",
                        "File_Name",
                        "Bam_File", "Mutation_Status", "Sequence_Source", "Sequencing_Phase", "Line_Number",
                        "Tumor_Validation_Allele1", "Tumor_Validation_Allele2", "Match_Norm_Validation_Allele1",
                        "Match_Norm_Validation_Allele2"]
        snp_data.drop(drop_columns, axis=1, inplace=True)
        index_patterns = re.compile(r'(TCGA-\w+-\w+-01A-\w+-\w+-\w+)', re.U | re.M | re.I)
        index_to_drop = [idx for idx in snp_data.index if index_patterns.match(idx) is None]
        snp_data.drop(index_to_drop, axis=0, inplace=True)

        # Identifiant unique des SNP
        snp_data["snp_id"] = snp_data.Chrom.map(str) + "_" + snp_data.Start_Position.map(str) + "_" \
                             + snp_data.End_Position.map(str) + "_" + snp_data.Reference_Allele.map(str) + "_" \
                             + snp_data.Tumor_Seq_Allele2

        # Compte le nombre d'occurences de chaque SNP
        snp_counts = snp_data["snp_id"].value_counts()

        # Converser seulement les SNP qui sont dans plus d'un exemple (filtre la majorité)
        # 1st assumption here, we're not sure if it's the right thing to do. We went from 38985 to 243 snps
        # in the BLCA dataset for example !!!!
        # Dans le cadre de BRCA on passe de 90490 à 4378 features.
        snps_to_keep = snp_counts.loc[snp_counts > 1].index.values

        snp_matrix = pd.DataFrame(index=snp_data.index, columns=snps_to_keep)

        for snp_id in snps_to_keep:
            temp_index = snp_data.loc[snp_data.snp_id == snp_id].index.tolist()
            snp_matrix.loc[temp_index, snp_id] = 1
        snp_matrix = snp_matrix[~snp_matrix.index.duplicated(keep='first')]
        # garder les index unique (j'ai verifier que la matrice se répétait donc mes résultats sont consistants.
        # On a par contre un peu moins de features. Mais si on veut trop parler j'enleve juste les lignes de index_to_drop
        snp_matrix.index = snp_matrix.index.map(trim_example_id)
        snp_matrix = snp_matrix.fillna(0)
        snp_matrix = snp_matrix.T
        saving_file = self.saving_repertory.format('snp.tsv')
        snp_matrix.to_csv(saving_file, sep="\t", encoding='utf-8')

    def main_build_tsv(self):
        if self.methyl_path_27 != '':
            self.build_methylome_27_tsv()
        if self.methyl_path_450 != '':
            self.build_methylome_450_tsv()
        if self.rnaseq_path != '':
            self.build_rnaseq_tsv()
        if self.snps_path != '':
            self.build_snp_tsv()
        if self.mirna_file != '':
            self.build_mirna_tsv()

    def combine_dataset(self, methyl_450_file=None, methyl_27_file=None, rnaseq_genes_file=None,
                        rnaseq_isoforms_file=None, snp_file=None, mirna_file=None, clinical_file=None,
                        filling_type='mean'):
        """
        Combine all the .tsv files to build the dataset
        Args: methyl_450_file, str, path to methyl 450 file,
              methyl_27_file=str, path to methyl 27 file,
              rnaseq_genes_file=str, path to rnaseq genes file,
              rnaseq_isoforms_file=str, path to rnaseq isoform file,
              snp_file=str, path to snp file,
              mirna_file=str, path to mirna file,
              clinical_file=str, path to clinical file
              filling_type, str, 'mean', 'zero', 'median' or None
        :return:
        """
        methylation_27 = None
        methylation_450 = None
        rnaseq_genes = None
        rnaseq_isoforms = None
        snps = None
        mirnas = None
        clinical_data = None
        methylation_fusion = None
        methylation_fusion_27 = None
        methylation_fusion_450 = None
        labels = pd.read_csv('{}'.format(self.labels_file), index_col="example_id", sep="\t")
        index_pos = labels.loc[labels['labels'] == 1].index.values
        index_neg = labels.loc[labels['labels'] == -1].index.values
        if methyl_27_file is not None:
            methylation_27 = pd.read_table(methyl_27_file)
            indexes = np.array(map(str, np.array(methylation_27["Unnamed: 0"])))
            methylation_27.set_index(indexes, inplace=True)
            methylation_27 = methylation_27.T.loc[labels.index.values]
            methylation_27 = methylation_27.loc[:, methylation_27.count() > 0]
        if methyl_450_file is not None:
            methylation_450 = pd.read_table(methyl_450_file)
            indexes = np.array(map(str, np.array(methylation_450["Unnamed: 0"])))
            methylation_450.set_index(indexes, inplace=True)
            methylation_450 = methylation_450.T.loc[labels.index.values]
            methylation_450 = methylation_450.loc[:, methylation_450.count() > 0]
        if methyl_27_file is not None and methyl_450_file is not None:
            indexes_fusion = [el for el in methylation_27.columns.values if el in methylation_450.columns.values]
            methylation_fusion_450 = methylation_450[indexes_fusion]
            methylation_fusion_27 = methylation_27[indexes_fusion]
            methylation_fusion = deepcopy(methylation_fusion_450)
            # Saving file (temp)
            methylation_fusion_450.to_csv('methylation_fusion_450.tsv', sep="\t", encoding='utf-8')
            methylation_fusion_27.to_csv('methylation_fusion_27.tsv', sep="\t", encoding='utf-8')
            # recuperer les endroits ou c'est nan dans methylation_fusion_450
            informations_on_the_nan_places = methylation_fusion_450.isnull().all(axis=1)
            for i in range(methylation_450.shape[0]):  # iterer sur le nombre des exemples, c'est lent mais bon
                if informations_on_the_nan_places[i] is True:
                    methylation_fusion.set_value(i, methylation_fusion_27[i])
            methylation_fusion.to_csv('methylation_fusion.tsv', sep="\t", encoding='utf-8')

        if filling_type is not None:
            if methyl_27_file is not None:
                methylation_27 = self.method_to_fill_nan(data=methylation_27, idx_pos=index_pos, idx_neg=index_neg,
                                                         filling_type=filling_type)
            if methyl_450_file is not None:
                methylation_450 = self.method_to_fill_nan(data=methylation_450, idx_pos=index_pos, idx_neg=index_neg,
                                                          filling_type=filling_type)
            if methyl_27_file is not None and methyl_450_file is not None:
                methylation_fusion = self.method_to_fill_nan(data=methylation_fusion, idx_pos=index_pos,
                                                             idx_neg=index_neg,
                                                             filling_type=filling_type)
                methylation_fusion_27 = self.method_to_fill_nan(data=methylation_fusion_27, idx_pos=index_pos,
                                                                idx_neg=index_neg,
                                                                filling_type=filling_type)
                methylation_fusion_450 = self.method_to_fill_nan(data=methylation_fusion_450, idx_pos=index_pos,
                                                                 idx_neg=index_neg,
                                                                 filling_type=filling_type)
        if rnaseq_genes_file is not None:
            rnaseq_genes = pd.read_table(rnaseq_genes_file)
            indexes = np.array(map(str, np.array(rnaseq_genes["Unnamed: 0"])))
            rnaseq_genes.set_index(indexes, inplace=True)
            rnaseq_genes = rnaseq_genes.T.loc[labels.index.values]
            rnaseq_genes = rnaseq_genes.loc[:, rnaseq_genes.count() > 0]
            if filling_type is not None:
                rnaseq_genes = self.method_to_fill_nan(data=rnaseq_genes, idx_pos=index_pos, idx_neg=index_neg,
                                                  filling_type=filling_type)
        if rnaseq_isoforms_file is not None:
            rnaseq_isoforms = pd.read_table(rnaseq_isoforms_file)
            indexes = np.array(map(str, np.array(rnaseq_isoforms["Unnamed: 0"])))
            rnaseq_isoforms.set_index(indexes, inplace=True)
            rnaseq_isoforms = rnaseq_isoforms.T.loc[labels.index.values]
            rnaseq_isoforms = rnaseq_isoforms.loc[:, rnaseq_isoforms.count() > 0]
            if filling_type is not None:
                rnaseq_isoforms = self.method_to_fill_nan(data=rnaseq_isoforms, idx_pos=index_pos, idx_neg=index_neg,
                                                     filling_type=filling_type)
        if snp_file is not None:
            snps = pd.read_table(snp_file)
            indexes = np.array(map(str, np.array(snps["Unnamed: 0"])))
            snps.set_index(indexes, inplace=True)
            snps = snps.T.loc[labels.index.values]
            snps = snps.loc[:, snps.count() > 0]
            if filling_type is not None:
                snps = self.method_to_fill_nan(data=snps, idx_pos=index_pos, idx_neg=index_neg,
                                               filling_type=filling_type)

        if mirna_file is not None:
            mirnas = pd.read_table(mirna_file)
            indexes = np.array(map(str, np.array(mirnas["Unnamed: 0"])))
            mirnas.set_index(indexes, inplace=True)
            mirnas = mirnas.T.loc[labels.index.values]
            mirnas = mirnas.loc[:, mirnas.count() > 0]
            if filling_type is not None:
                mirnas = self.method_to_fill_nan(data=mirnas, idx_pos=index_pos, idx_neg=index_neg,
                                            filling_type=filling_type)

        if clinical_file is not None:
            clinical_data = pd.read_table(clinical_file)
            indexes = np.array(map(str, clinical_data['bcr_patient_barcode'].values))
            clinical_data.set_index(indexes, inplace=True)
            clinical_data.drop(['Unnamed: 0', 'bcr_patient_barcode'], axis=1, inplace=True)
            clinical_data = clinical_data.loc[labels.index.values]

        # Check the examples
        if methylation_450 is not None:
            assert np.all(labels.index.values == methylation_450.index.values)
        if methylation_27 is not None:
            assert np.all(labels.index.values == methylation_27.index.values)
        if rnaseq_isoforms is not None:
            assert np.all(labels.index.values == rnaseq_isoforms.index.values)
        if rnaseq_genes is not None:
            assert np.all(labels.index.values == rnaseq_genes.index.values)
        if snps is not None:
            assert np.all(labels.index.values == snps.index.values)
        if mirnas is not None:
            assert np.all(labels.index.values == mirnas.index.values)
        if clinical_data is not None:
            assert np.all(labels.index.values == clinical_data.index.values)

        # Write the data to hdf5
        name = '{}_{}.h5'.format(self.cancer_name, filling_type)
        labels.to_hdf(name, "labels")
        if methylation_450 is not None:
            methylation_450.to_hdf(name, "methylation_450")
        if methylation_27 is not None:
            methylation_27.to_hdf(name, "methylation_27")
        if methylation_fusion_27 is not None:
            methylation_fusion_27.to_hdf(name, "methylation_fusion_27")
        if methylation_fusion_450 is not None:
            methylation_fusion_450.to_hdf(name, "methylation_fusion_450")
        if methylation_fusion is not None:
            methylation_fusion.to_hdf(name, "methylation_fusion")
        if rnaseq_genes is not None:
            rnaseq_genes.to_hdf(name, "rnaseq_genes")
        if rnaseq_isoforms is not None:
            rnaseq_isoforms.to_hdf(name, "rnaseq_isoforms")
        if snps is not None:
            snps.to_hdf(name, "snp")
        if mirnas is not None:
            mirnas.to_hdf(name, "mirna")
        if clinical_data is not None:
            clinical_data.to_hdf(name, 'clinical_view')


def build_cancer_dataset_for_graalpy(dataset='',
                                     name='',
                                     label_file='',
                                     output_path='./',
                                     methyl_example_file='',
                                     genes_example_file='',
                                     snp_data_file=''):
    """ Build a hdf5 dataset for to be used in graalpy """
    print('load the dataset')
    data = h5py.File(dataset, 'r')
    print("Methylation block")
    if 'methylation_fusion' in data.keys():
        x_methyl = data['methylation_fusion/block0_values'].value
        features_names_methyl = data['methylation_fusion/block0_items'].value

        # linked the methyl_name to the genes_name
        d = pd.read_table(methyl_example_file, skiprows=[0], header='infer')
        d.fillna('INEXISTANT', inplace=True)
        features_names_methyl_linked = ['{}_{}'.format(d['Composite Element REF'].values[i], d['Gene_Symbol'].values[i])
                                        for i in range(d.shape[0]) if
                                        d['Composite Element REF'].values[i] in features_names_methyl]

        features_names_methyl_linked = np.asarray(features_names_methyl_linked)
    else:
        x_methyl = data['methylation_450/block0_values'].value
        features_names_methyl = data['methylation_450/block0_items'].value

        # linked the methyl_name to the genes_name
        d = pd.read_table(methyl_example_file, skiprows=[0], header='infer')
        d.fillna('INEXISTANT', inplace=True)
        features_names_methyl_linked = ['{}_{}'.format(d['Composite Element REF'].values[i], d['Gene_Symbol'].values[i])
                                        for i in range(d.shape[0]) if
                                        d['Composite Element REF'].values[i] in features_names_methyl]
        features_names_methyl_linked = np.asarray(features_names_methyl_linked)

    print("MiRNA block")
    if 'mirna' in data.keys():
        x_mirna = data['mirna/block0_values'].value
        features_names_mirna = data['mirna/block0_items'].value
    else:
        pass

    print("Isoforms block")
    x_rna_isoforms = data['rnaseq_isoforms/block0_values'].value
    features_names_rna_isoforms = data['rnaseq_isoforms/block0_items'].value
    # linked the rna isoforms name to the genes names
    d = pd.read_table(genes_example_file)
    temp_genes_ids_names = d['gene_id'].values
    temp_genes_ids_names = np.asarray([temp_genes_ids_names[i].split('|')[1] if i in range(29) else
                                       temp_genes_ids_names[i].split('|')[0]
                                       for i in range(temp_genes_ids_names.shape[0])])
    temp_isoforms_ids_names = d['transcript_id'].values
    temp_features_names_rna_isoforms_linked = []
    temp_isoforms_ids_names_flatten_list = []
    for i, el in enumerate(temp_isoforms_ids_names):
        temp_splits = el.split(',')
        for names in temp_splits:
            temp_isoforms_ids_names_flatten_list.append(names)
            temp_features_names_rna_isoforms_linked.append('{}_{}'.format(names, temp_genes_ids_names[i]))
    temp_isoforms_ids_names_flatten_list = np.asarray(temp_isoforms_ids_names_flatten_list)
    temp_features_names_rna_isoforms_linked = np.asarray(temp_features_names_rna_isoforms_linked)
    features_names_rna_isoforms_linked = np.zeros((features_names_rna_isoforms.shape[0]), dtype='O')
    for i, el in enumerate(features_names_rna_isoforms):
        if el not in temp_isoforms_ids_names_flatten_list:  # si l'élément n'est pas dans la liste flatten c'est que
            # la correspondance au gene est inexistante. Alors il faut juste remplacer par inexistant le gene name
            features_names_rna_isoforms_linked[i] = '{}_INEXISTANT'.format(el)
        else:  # C'est que on est dans la liste flatten
            index_el = np.where(temp_isoforms_ids_names_flatten_list == el)[0][0]  # recuperer la position ou l'élément est
            features_names_rna_isoforms_linked[i] = temp_features_names_rna_isoforms_linked[index_el]  # remplacer par le nouveau nom

    print("Snp block")
    if 'snp' in data.keys():
        x_snps = data['snp/block0_values'].value
        features_names_snps = data['snp/block0_items'].value
        snp_data = pd.read_table(snp_data_file, sep="\t", index_col="Tumor_Sample_Barcode")
        drop_columns = ["Center", "Ncbi_Build", "Archive_Name", "Strand", "Dbsnp_Rs", "Dbsnp_Val_Status",
                        "Verification_Status", "Sequencer", "Validation_Status", "Validation_Method", "Score",
                        "File_Name",
                        "Bam_File", "Mutation_Status", "Sequence_Source", "Sequencing_Phase", "Line_Number",
                        "Tumor_Validation_Allele1", "Tumor_Validation_Allele2", "Match_Norm_Validation_Allele1",
                        "Match_Norm_Validation_Allele2"]
        snp_data.drop(drop_columns, axis=1, inplace=True)
        index_patterns = re.compile(r'(TCGA-\w+-\w+-01A-\w+-\w+-\w+)', re.U | re.M | re.I)
        index_to_drop = [idx for idx in snp_data.index if index_patterns.match(idx) is None]
        snp_data.drop(index_to_drop, axis=0, inplace=True)
        # Identifiant unique des SNP
        snp_data["snp_id"] = '{}_{}_{}_{}_{}_{}'.format(snp_data.Entrez_Gene_Id.map(str),
                                                        snp_data.Chrom.map(str),
                                                        snp_data.Start_Position.map(str),
                                                        snp_data.End_Position.map(str),
                                                        snp_data.Reference_Allele.map(str),
                                                        snp_data.Tumor_Seq_Allele2)
        snp_data = snp_data.loc[snp_data['snp_id'].isin(features_names_snps)]
        zipping_name = zip(snp_data['Hugo_Symbol'], snp_data['snp_id'])
        features_names_snps_linked = np.zeros((features_names_snps.shape[0]), dtype='O')
        for i, el in enumerate(features_names_snps):
            for zip_el in zipping_name:
                if zip_el[1] == el:
                    features_names_snps_linked[i] = '{}_{}'.format(el, zip_el[0])
    else:
        pass
    print('Labels block')
    y = data['labels/block0_values'].value
    y = y.reshape(-1)
    print('Concatenation block')
    x = np.hstack((x_methyl, x_rna_isoforms, x_mirna, x_snps))
    features_names_methyl_linked = np.asarray(features_names_methyl_linked, dtype='str')
    features_names_rna_isoforms_linked = np.asarray(features_names_rna_isoforms_linked, dtype='str')
    features_names_snps_linked = np.asarray(features_names_snps_linked, dtype='str')
    features_names = np.hstack((features_names_methyl_linked, features_names_rna_isoforms_linked,
                                features_names_mirna, features_names_snps_linked))
    print('Create the patient ID')
    label_data = pd.read_table(label_file)
    patients_ids = np.asarray(label_data['example_id'], dtype='str')
    print('Creation of the dataset')
    f = h5py.File(os.path.join(output_path, '{}.h5'.format(name)), 'w')
    f.create_dataset('data', data=x)
    f.create_dataset('target', data=y)
    f.create_dataset('features_names', data=features_names)
    f.create_dataset('patients_ids', data=patients_ids)


def main_build_omics_datasets_all_cancers(build_each_views=False, build_combine_dataset=False, build_graalpy_dataset=False):
    acc_builder = BuildOmicsDatasets(cancer_name='ACC',
                                     methyl_path_450=project_path_on_is2 + 'ACC/acc_methylation/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                     methyl_path_27='',
                                     rnaseq_path=project_path_on_is2 + 'ACC/acc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                     snps_path=project_path_on_is2 + 'ACC/acc_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                     mirna_file=project_path_on_is2 + 'ACC/acc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    blca_builder = BuildOmicsDatasets(cancer_name='BLCA',
                                      methyl_path_450=project_path_on_is2 + 'BLCA/blca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'BLCA/blca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'BLCA/blca_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'BLCA/blca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    cesc_builder = BuildOmicsDatasets(cancer_name='CESC',
                                      methyl_path_450=project_path_on_is2 + 'CESC/cesc_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'CESC/cesc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'CESC/cesc_exome/Somatic_Mutations/WUSM__IlluminaGA_DNASeq_curated/Level_2/genome.wustl.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'CESC/cesc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    chol_builder = BuildOmicsDatasets(cancer_name='CHOL',
                                      methyl_path_450=project_path_on_is2 + 'CHOL/chol_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'CHOL/chol_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'CHOL/chol_exome/Somatic_Mutations/BCM__Mixed_DNASeq_curated/Level_2/hgsc.bcm.edu__Mixed_curated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'CHOL/chol_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    coad_builder = BuildOmicsDatasets(cancer_name='COAD',
                                      methyl_path_450=project_path_on_is2 + 'COAD/coad_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'COAD/coad_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'COAD/coad_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'COAD/coad_exome/Somatic_Mutations/BCM__IlluminaGA_DNASeq/Level_2/hgsc.bcm.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'COAD/coad_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    dlbc_builder = BuildOmicsDatasets(cancer_name='DLBC',
                                      methyl_path_450=project_path_on_is2 + 'DLBC/dlbc_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'DLBC/dlbc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'DLBC/dlbc_exome/Somatic_Mutations/BCM__IlluminaGA_DNASeq_automated/Level_2/hgsc.bcm.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'DLBC/dlbc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    esca_builder = BuildOmicsDatasets(cancer_name='ESCA',
                                      methyl_path_450=project_path_on_is2 + 'ESCA/esca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'ESCA/esca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'ESCA/esca_exome/Somatic_Mutations/WUSM__IlluminaHiSeq_DNASeq_automated/Level_2/genome.wustl.edu__IlluminaHiSeq_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'ESCA/esca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    fppp_builder = BuildOmicsDatasets(cancer_name='FPPP',
                                     methyl_path_450='',
                                     methyl_path_27='',
                                     rnaseq_path='',
                                     snps_path=project_path_on_is2 + 'FPPP/fppp_exome/Somatic_Mutations/BCM__Mixed_DNASeq_automated/Level_2/hgsc.bcm.edu__Mixed_automated_DNA_sequencing_level2.maf',
                                     mirna_file=project_path_on_is2 + 'FPPP/fppp_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    gbm_builder = BuildOmicsDatasets(cancer_name='GBM',
                                     methyl_path_450=project_path_on_is2 + 'GBM/gbm_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                     methyl_path_27=project_path_on_is2 + 'GBM/gbm_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                     rnaseq_path=project_path_on_is2 + 'GBM/gbm_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                     snps_path=project_path_on_is2 + 'GBM/gbm_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq/Level_2/broad.mit.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                     mirna_file='')
    # mirna_file = project_path_on_is2 + 'GBM/gbm_mirna/Expression-miRNA/UNC__H-miRNA_8x15K/Level_3'
    hnsc_builder = BuildOmicsDatasets(cancer_name='HNSC',
                                      methyl_path_450=project_path_on_is2 + 'HNSC/hnsc_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'HNSC/hnsc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'HNSC/hnsc_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_automated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'HNSC/hnsc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    kich_builder = BuildOmicsDatasets(cancer_name='KICH',
                                      methyl_path_450=project_path_on_is2 + 'KICH/kich_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'KICH/kich_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'KICH/kich_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq/Level_2/broad.mit.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'KICH/kich_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    kirc_builder = BuildOmicsDatasets(cancer_name='KIRC',
                                      methyl_path_450=project_path_on_is2 + 'KIRC/kirc_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'KIRC/kirc_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'KIRC/kirc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'KIRC/kirc_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_automated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'KIRC/kirc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    kirp_builder = BuildOmicsDatasets(cancer_name='KIRP',
                                      methyl_path_450=project_path_on_is2 + 'KIRP/kirp_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'KIRP/kirp_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'KIRP/kirp_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'KIRP/kirp_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'KIRP/kirp_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    laml_builder = BuildOmicsDatasets(cancer_name='LAML',
                                      methyl_path_450=project_path_on_is2 + 'LAML/laml_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'LAML/laml_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'LAML/laml_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'LAML/laml_exome/Somatic_Mutations/WUSM__IlluminaGA_DNASeq/Level_2/genome.wustl.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'LAML/laml_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    lgg_builder = BuildOmicsDatasets(cancer_name='LGG',
                                     methyl_path_450=project_path_on_is2 + 'LGG/lgg_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                     methyl_path_27='',
                                     rnaseq_path=project_path_on_is2 + 'LGG/lgg_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                     snps_path=project_path_on_is2 + 'LGG/lgg_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                     mirna_file=project_path_on_is2 + 'LGG/lgg_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    lihc_builder = BuildOmicsDatasets(cancer_name='LIHC',
                                      methyl_path_450=project_path_on_is2 + 'LIHC/lihc_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'LIHC/lihc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'LIHC/lihc_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_automated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'LIHC/lihc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    luad_builder = BuildOmicsDatasets(cancer_name='LUAD',
                                     methyl_path_450=project_path_on_is2 + 'LUAD/luad_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                     methyl_path_27=project_path_on_is2 + 'LUAD/luad_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                     rnaseq_path=project_path_on_is2 + 'LUAD/luad_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                     snps_path=project_path_on_is2 + 'LUAD/luad_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                     mirna_file=project_path_on_is2 + 'LUAD/luad_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    lusc_builder = BuildOmicsDatasets(cancer_name='LUSC',
                                      methyl_path_450=project_path_on_is2 + 'LUSC/lusc_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'LUSC/lusc_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'LUSC/lusc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'LUSC/lusc_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq/Level_2/broad.mit.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'LUSC/lusc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    meso_builder = BuildOmicsDatasets(cancer_name='MESO',
                                      methyl_path_450=project_path_on_is2 + 'MESO/meso_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'MESO/meso_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path='',
                                      mirna_file=project_path_on_is2 + 'MESO/meso_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    ov_builder = BuildOmicsDatasets(cancer_name='OV',
                                    methyl_path_450=project_path_on_is2 + 'OV/ov_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                    methyl_path_27=project_path_on_is2 + 'OV/ov_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                    rnaseq_path=project_path_on_is2 + 'OV/ov_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                    snps_path=project_path_on_is2 + 'OV/ov_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq/Level_2/broad.mit.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                    mirna_file=project_path_on_is2 + 'OV/ov_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    paad_builder = BuildOmicsDatasets(cancer_name='PAAD',
                                      methyl_path_450=project_path_on_is2 + 'PAAD/paad_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'PAAD/paad_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'PAAD/paad_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'PAAD/paad_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    pcpg_builder = BuildOmicsDatasets(cancer_name='PCPG',
                                      methyl_path_450=project_path_on_is2 + 'PCPG/pcpg_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'PCPG/pcpg_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'PCPG/pcpg_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'PCPG/pcpg_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    prad_builder = BuildOmicsDatasets(cancer_name='PRAD',
                                      methyl_path_450=project_path_on_is2 + 'PRAD/prad_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'PRAD/prad_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'PRAD/prad_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'PRAD/prad_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    read_builder = BuildOmicsDatasets(cancer_name='READ',
                                      methyl_path_450=project_path_on_is2 + 'READ/read_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'READ/read_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'READ/read_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'READ/read_exome/Somatic_Mutations/BCM__IlluminaGA_DNASeq/Level_2/hgsc.bcm.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'READ/read_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    sarc_builder = BuildOmicsDatasets(cancer_name='SARC',
                                      methyl_path_450=project_path_on_is2 + 'SARC/sarc_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'SARC/sarc_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'SARC/sarc_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_automated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'SARC/sarc_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    skcm_builder = BuildOmicsDatasets(cancer_name='SKCM',
                                      methyl_path_450=project_path_on_is2 + 'SKCM/skcm_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'SKCM/skcm_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'SKCM/skcm_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_automated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'SKCM/skcm_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    stad_builder = BuildOmicsDatasets(cancer_name='STAD',
                                      methyl_path_450=project_path_on_is2 + 'STAD/stad_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'STAD/stad_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'STAD/stad_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'STAD/stad_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'STAD/stad_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    tgct_builder = BuildOmicsDatasets(cancer_name='TGCT',
                                      methyl_path_450=project_path_on_is2 + 'TGCT/tgct_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'TGCT/tgct_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'TGCT/tgct_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_automated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'TGCT/tgct_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    thca_builder = BuildOmicsDatasets(cancer_name='THCA',
                                      methyl_path_450=project_path_on_is2 + 'THCA/thca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'THCA/thca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'THCA/thca_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq/Level_2/broad.mit.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'THCA/thca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    thym_builder = BuildOmicsDatasets(cancer_name='THYM',
                                      methyl_path_450=project_path_on_is2 + 'THYM/thym_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27='',
                                      rnaseq_path=project_path_on_is2 + 'THYM/thym_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'THYM/thym_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_automated/Level_2/broad.mit.edu__IlluminaGA_automated_DNA_sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'THYM/thym_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    ucec_builder = BuildOmicsDatasets(cancer_name='UCEC',
                                      methyl_path_450=project_path_on_is2 + 'UCEC/ucec_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                      methyl_path_27=project_path_on_is2 + 'UCEC/ucec_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3',
                                      rnaseq_path=project_path_on_is2 + 'UCEC/ucec_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                      snps_path=project_path_on_is2 + 'UCEC/ucec_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq/Level_2/broad.mit.edu__Illumina_Genome_Analyzer_DNA_Sequencing_level2.maf',
                                      mirna_file=project_path_on_is2 + 'UCEC/ucec_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    ucs_builder = BuildOmicsDatasets(cancer_name='UCS',
                                     methyl_path_450=project_path_on_is2 + 'UCS/ucs_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                     methyl_path_27='',
                                     rnaseq_path=project_path_on_is2 + 'UCS/ucs_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                     snps_path=project_path_on_is2 + 'UCS/ucs_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                     mirna_file=project_path_on_is2 + 'UCS/ucs_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    uvm_builder = BuildOmicsDatasets(cancer_name='UVM',
                                     methyl_path_450=project_path_on_is2 + 'UVM/uvm_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3',
                                     methyl_path_27='',
                                     rnaseq_path=project_path_on_is2 + 'UVM/uvm_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3',
                                     snps_path=project_path_on_is2 + 'UVM/uvm_exome/Somatic_Mutations/BI__IlluminaGA_DNASeq_curated/Level_2/broad.mit.edu__IlluminaGA_curated_DNA_sequencing_level2.maf',
                                     mirna_file=project_path_on_is2 + 'UVM/uvm_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3')

    brca_builder = BuildOmicsDatasets(cancer_name='BRCA',
                                     methyl_path_450=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3",
                                     methyl_path_27=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3",
                                     rnaseq_path=project_path_on_is2 + "BRCA/brca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3",
                                     snps_path=project_path_on_is2 + "BRCA/brca_exome/Somatic_Mutations/WUSM__IlluminaGA_DNASeq_curated/Level_2/genome.wustl.edu__IlluminaGA_curated_DNA_sequencing_level2.maf",
                                     mirna_file=project_path_on_is2 + "BRCA/brca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3")

    listes_of_builders = [acc_builder, blca_builder, cesc_builder, chol_builder, coad_builder, dlbc_builder,
                          esca_builder, fppp_builder, gbm_builder, hnsc_builder, kich_builder, kirp_builder,
                          kirc_builder, laml_builder, lgg_builder, lihc_builder, luad_builder, lusc_builder,
                          meso_builder, ov_builder, paad_builder, pcpg_builder, prad_builder, read_builder,
                          sarc_builder, skcm_builder, stad_builder, tgct_builder, thca_builder, thym_builder,
                          ucec_builder, ucs_builder, uvm_builder, brca_builder]
    # listes_of_builders = [esca_builder]
    if build_each_views:  # just wanna build each dataset view: rnaseq, methyl, snp, mirna
        for builders in listes_of_builders:
            builders.main_build_tsv()
    if build_combine_dataset:  # combine all the dataset to build hdf5
        for builders in listes_of_builders:
            # TODO: LIHC, GBM, PCPG doesnt have mirna; MESO doesnt have SNP; OV there were no match up in the labels why?
            if builders.cancer_name in ['FPPP', 'GBM', 'LIHC', 'PCPG', 'MESO', 'OV', 'LAML', 'PAAD']:
                continue
            cancer_project_path = '/home/ossmaz01/EXPERIENCES/datasets/{}'.format(builders.cancer_name)
            methyl_27_path = cancer_project_path + "/methylome_27.tsv"
            if os.path.isfile(methyl_27_path):
                clinical_patient_file = cancer_project_path + '/nationwidechildrens.org_clinical_patient_{}.txt'.format(
                    builders.cancer_name.lower())
                clinical_file = pd.read_table(clinical_patient_file)
                labels = clinical_file[['bcr_patient_barcode', 'tumor_status']]
                labels_available_in_the_dataset = np.unique(labels['tumor_status'])
                for unique_labels in labels_available_in_the_dataset:
                    if unique_labels not in ['TUMOR FREE', 'WITH TUMOR', 'tumor free', 'with tumor']:
                        labels.drop(labels.loc[labels['tumor_status'] == unique_labels].index, inplace=True)
                labels.replace(to_replace=['TUMOR FREE', 'WITH TUMOR'], value=[-1, 1], inplace=True)
                labels = labels.values
                cancer_labels_file_name = '{}_tumor_status_labels.tsv'.format(builders.cancer_name.lower())
                with open(cancer_labels_file_name, 'w') as f:
                    f.write('{}\t{}\n'.format('example_id', 'labels'))
                    for el in labels:
                        f.write('{}\t{}\n'.format(el[0], el[1]))
                builders.labels_file = './{}'.format(cancer_labels_file_name)
                builders.combine_dataset(methyl_450_file=cancer_project_path + "/methylome_450.tsv",
                                         methyl_27_file=cancer_project_path + "/methylome_27.tsv",
                                         rnaseq_genes_file=cancer_project_path + "/rnaseq_genes.tsv",
                                         rnaseq_isoforms_file=cancer_project_path + "/rnaseq_isoforms.tsv",
                                         snp_file=cancer_project_path + "/snp.tsv",
                                         mirna_file=cancer_project_path + "/mirna.tsv",
                                         clinical_file=None,
                                         filling_type='mean')

            else:
                clinical_patient_file = cancer_project_path + '/nationwidechildrens.org_clinical_patient_{}.txt'.format(
                    builders.cancer_name.lower())
                clinical_file = pd.read_table(clinical_patient_file)
                labels = clinical_file[['bcr_patient_barcode', 'tumor_status']]
                labels_available_in_the_dataset = np.unique(labels['tumor_status'])
                for unique_labels in labels_available_in_the_dataset:
                    if unique_labels not in ['TUMOR FREE', 'WITH TUMOR', 'tumor free', 'with tumor']:
                        labels.drop(labels.loc[labels['tumor_status'] == unique_labels].index, inplace=True)
                labels.replace(to_replace=['TUMOR FREE', 'WITH TUMOR'], value=[-1, 1], inplace=True)
                labels = labels.values
                cancer_labels_file_name = '{}_tumor_status_labels.tsv'.format(builders.cancer_name.lower())
                with open(cancer_labels_file_name, 'w') as f:
                    f.write('{}\t{}\n'.format('example_id', 'labels'))
                    for el in labels:
                        f.write('{}\t{}\n'.format(el[0], el[1]))
                builders.labels_file = './{}'.format(cancer_labels_file_name)
                builders.combine_dataset(methyl_450_file=cancer_project_path + "/methylome_450.tsv",
                                         methyl_27_file=None,
                                         rnaseq_genes_file=cancer_project_path + "/rnaseq_genes.tsv",
                                         rnaseq_isoforms_file=cancer_project_path + "/rnaseq_isoforms.tsv",
                                         snp_file=cancer_project_path + "/snp.tsv",
                                         mirna_file=cancer_project_path + "/mirna.tsv",
                                         clinical_file=None,
                                         filling_type='mean')

    if build_graalpy_dataset:
        working_directory = '/home/ossmaz01/EXPERIENCES/breast_cancer/datasets'
        for builders in listes_of_builders:
            # Officially this is what i need to not be doing
            # if builders.cancer_name in ['FPPP', 'OV', 'GBM', 'LAML', 'MESO', 'PAAD', 'PCPG']: # the one i dont have any *_mean.h5
            #     continue
            # if builders.cancer_name in ['FPPP', 'OV', 'GBM', 'LAML', 'MESO', 'PAAD', 'PCPG', 'ACC', 'HNSC', 'BLCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'KICH', 'KIRP', 'KIRC',
            # 'LIHC', 'LGG']:
            #     continue
            cancer_project_path = '/home/ossmaz01/EXPERIENCES/datasets/{}'.format(builders.cancer_name)
            print(cancer_project_path)
            os.chdir(cancer_project_path)
            methyl_example_file_name = glob('jhu*')[0]
            genes_example_file_name = glob('unc*')[0]
            snp_data_file_name = glob('*.maf')[0]
            os.chdir(working_directory)
            build_cancer_dataset_for_graalpy(dataset='{}_mean.h5'.format(builders.cancer_name),
                                             name='{}_mean_tumor_status'.format(builders.cancer_name),
                                             label_file='{}_tumor_status_labels.tsv'.format(builders.cancer_name.lower()),
                                             output_path='./',
                                             methyl_example_file='{}/{}'.format(cancer_project_path,
                                                                                methyl_example_file_name),
                                             genes_example_file='{}/{}'.format(cancer_project_path,
                                                                               genes_example_file_name),
                                             snp_data_file='{}/{}'.format(cancer_project_path, snp_data_file_name))
            print('Dataset for {} is done'.format(builders.cancer_name))
