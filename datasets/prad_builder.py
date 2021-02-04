import numpy as np 
import pandas as pd 
import os
from glob import glob
# CNA Putative copy-number calls determined using CNVkit. Values: -2 = homozygous deletion; -1 = hemizygous deletion; 0 = neutral / no change; 1 = gain; 2 = high level amplification.

def read_cases_listes(path_fichier):
    cases_files = open(f'{path_fichier}', 'r')
    cases_files_lines = cases_files.readlines()
    cases_files_lines = [el.strip('\n') for el in cases_files_lines]
    ids_labels = cases_files_lines[-1].split(' ')[-1].split('\t')
    return ids_labels, len(ids_labels)


class BuildSu2c(object):
    def __init__(self, directory):
        self.directory = directory
        os.chdir(directory)
    
    def retrieve_patients_ids(self):
        self.cases_all_ids_labels, self.nb_cases_all_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_all.txt')
        self.cases_cna_ids_labels, self.nb_cases_cna_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cna.txt')
        self.cases_cnaseq_ids_labels, self.nb_cases_cnaseq_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cnaseq.txt')
        self.cases_fpkm_capture_ids_labels, self.nb_cases_fpkm_capture_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_fpkm_capture.txt')
        self.cases_fpkm_polya_ids_labels, self.nb_cases_fpkm_polya_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_fpkm_polya.txt')
        self.cases_sequenced_ids_labels, self.nb_cases_sequenced_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_sequenced.txt')
    
    def build_data_all_available(self):
        df_cna = pd.read_csv('data_CNA.txt', sep='\t') # Shape: (22058, 444)
        df_cna = df_cna.dropna()
        df_cna.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_cna = df_cna.set_index('Hugo_Symbol')
        df_fpkm_capture = pd.read_csv('data_mRNA_seq_fpkm_capture_Zscores.txt', sep='\t')
        df_fpkm_polyA = pd.read_csv('data_mRNA_seq_fpkm_polya_Zscores.txt', sep='\t')
        df_fpkm_capture = df_fpkm_capture.dropna() # (19158, 213)
        df_fpkm_polyA = df_fpkm_polyA.dropna() # (19165, 271)
        df_fpkm_capture.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) # (19154, 213)
        df_fpkm_polyA.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) # (9593, 271)
        common_hugo = [el for el in df_fpkm_polyA['Hugo_Symbol'].values[()] if el in df_fpkm_capture['Hugo_Symbol'].values[()]] # 9573
        df_fpkm_polyA = df_fpkm_polyA.loc[df_fpkm_polyA['Hugo_Symbol'].isin(common_hugo)]
        df_fpkm_capture = df_fpkm_capture.loc[df_fpkm_capture['Hugo_Symbol'].isin(common_hugo)]
        df_fpkm_polyA = df_fpkm_polyA.set_index('Hugo_Symbol')
        df_fpkm_capture = df_fpkm_capture.set_index('Hugo_Symbol')
        fpkm_ployA_columns_to_be_selected = [el for el in df_fpkm_polyA.columns if el not in df_fpkm_capture.columns]
        df_fpkm_fusion = pd.merge(df_fpkm_capture, df_fpkm_polyA[fpkm_ployA_columns_to_be_selected], right_index=True, left_index=True) # Shape: (9573, 332)
        df_cna = df_cna[df_fpkm_fusion.columns.values[()]] # (19062, 332)
        df_cna.T.to_csv('prad_su2c_metastaste_cna.csv')
        df_fpkm_fusion.T.to_csv('prad_su2c_metastaste_fpkm.csv') # same as mRNA
       
    def print_statistics(self):
        self.retrieve_patients_ids()
        print(f'The number of all cases available is {self.nb_cases_all_ids_labels}')
        print(f'The number of cna cases available is {self.nb_cases_cna_ids_labels}')
        print(f'The number of cnaseq cases available is {self.nb_cases_cnaseq_ids_labels}')
        print(f'The number of fpkm_capture cases available is {self.nb_cases_fpkm_capture_ids_labels}')
        print(f'The number of fpkm_polya cases available is {self.nb_cases_fpkm_polya_ids_labels}')
        print(f'The number of sequenced cases available is {self.nb_cases_sequenced_ids_labels}')

    
class BuildTCGA(object):
    def __init__(self, directory):
        self.directory = directory
        os.chdir(directory)
    
    def retrieve_patients_ids(self):
        self.cases_all_ids_labels, self.nb_cases_all_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_all.txt')
        self.cases_cna_ids_labels, self.nb_cases_cna_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cna.txt')
        self.cases_cnaseq_ids_labels, self.nb_cases_cnaseq_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cnaseq.txt')
        self.cases_mrna_ids_labels, self.nb_cases_mrna_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_RNA_Seq_v2_mRNA.txt')
        self.cases_methylation_ids_labels, self.nb_cases_methylation_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_methylation_all.txt')
        self.cases_rppa_ids_labels, self.nb_cases_rppa_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_rppa.txt')
        self.cases_complete_ids_labels, self.nb_cases_complete_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_complete.txt')
        
    def build_data_all_available(self):
        df_cna = pd.read_csv('data_CNA.txt', sep='\t') 
        df_cna = df_cna.dropna()
        df_cna.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_cna = df_cna.set_index('Hugo_Symbol') # shape (23286, 493)
        df_cna.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
        df_mrna = pd.read_csv('data_RNA_Seq_v2_mRNA_median_Zscores.txt', sep='\t')
        df_mrna = df_mrna.dropna()
        df_mrna.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_mrna = df_mrna.set_index('Hugo_Symbol')
        df_mrna.drop(['Entrez_Gene_Id'], axis=1, inplace=True) # shape (20160, 498)
        df_methylation = pd.read_csv('data_methylation_hm450.txt', sep='\t')
        df_methylation = df_methylation.dropna()
        df_methylation.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_methylation = df_methylation.set_index('Hugo_Symbol')
        df_methylation.drop(['Entrez_Gene_Id'], axis=1, inplace=True) # Shape (15297, 499)
        df_rppa = pd.read_csv('data_rppa_Zscores.txt', sep='\t')
        df_rppa = df_rppa.dropna()
        df_rppa.drop_duplicates(subset="Composite.Element.REF", keep = False, inplace=True) 
        df_rppa = df_rppa.set_index('Composite.Element.REF') # shape (186, 352)
        df_cna[self.cases_complete_ids_labels]
        df_mrna[self.cases_complete_ids_labels]
        df_methylation[self.cases_complete_ids_labels]
        df_cna.T.to_csv('prad_brca_non_metastaste_cna.csv')
        df_mrna.T.to_csv('prad_brca_non_metastaste_mrna.csv')
        df_methylation.T.to_csv('prad_brca_non_metastaste_methyl.csv')
        df_rppa.T.to_csv('prad_brca_non_metastaste_rppa.csv')
        
       
    def print_statistics(self):
        self.retrieve_patients_ids()
        print(f'The number of all cases available is {self.nb_cases_all_ids_labels}')
        print(f'The number of cna cases available is {self.nb_cases_cna_ids_labels}')
        print(f'The number of cnaseq cases available is {self.nb_cases_cnaseq_ids_labels}')
        print(f'The number of mrna cases available is {self.nb_cases_mrna_ids_labels}')
        print(f'The number of methylation cases available is {self.nb_cases_methylation_ids_labels}')
        print(f'The number of rppa cases available is {self.nb_cases_rppa_ids_labels}')


class BuildEURUROL(object):
    def __init__(self, directory):
        self.directory = directory
        os.chdir(directory)
    
    def retrieve_patients_ids(self):
        self.cases_all_ids_labels, self.nb_cases_all_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_all.txt')
        self.cases_cna_ids_labels, self.nb_cases_cna_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cna.txt')
        self.cases_cnaseq_ids_labels, self.nb_cases_cnaseq_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cnaseq.txt')
       
    def build_data_all_available(self):
        df_cna = pd.read_csv('data_CNA.txt', sep='\t') 
        df_cna = df_cna.dropna()
        df_cna.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_cna = df_cna.set_index('Hugo_Symbol') # shape (23286, 493)
        df_cna.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
        df_mrna = pd.read_csv('data_mRNA_seq_fpkm_capture_Zscores.txt', sep='\t')
        df_mrna = df_mrna.dropna()
        df_mrna.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_mrna = df_mrna.set_index('Hugo_Symbol')
        df_mrna.drop(['Entrez_Gene_Id'], axis=1, inplace=True) # shape (20160, 498)
        df_cna.T.to_csv('prad_eururol_non_metastaste_cna.csv')
        df_mrna.T.to_csv('prad_eururol_non_metastaste_mrna.csv')
       
    def print_statistics(self):
        self.retrieve_patients_ids()
        print(f'The number of all cases available is {self.nb_cases_all_ids_labels}')
        print(f'The number of cna cases available is {self.nb_cases_cna_ids_labels}')
        print(f'The number of cnaseq cases available is {self.nb_cases_cnaseq_ids_labels}')


class BuildFHCRC(object):
    def __init__(self, directory):
        self.directory = directory
        os.chdir(directory)
    
    def retrieve_patients_ids(self):
        self.cases_all_ids_labels, self.nb_cases_all_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_all.txt')
        self.cases_cna_ids_labels, self.nb_cases_cna_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cna.txt')
        self.cases_cnaseq_ids_labels, self.nb_cases_cnaseq_ids_labels = read_cases_listes(path_fichier=f'{self.directory}/case_lists/cases_cnaseq.txt')
        
    def build_data_all_available(self):
        df_cna = pd.read_csv('data_CNA.txt', sep='\t') 
        df_cna = df_cna.dropna()
        df_cna.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_cna = df_cna.set_index('Hugo_Symbol')
        df_cna.drop(['Entrez_Gene_Id'], axis=1, inplace=True)  
        df_mrna = pd.read_csv('data_mRNA_median_Zscores.txt', sep='\t')
        df_mrna = df_mrna.dropna()
        df_mrna.drop_duplicates(subset="Hugo_Symbol", keep = False, inplace=True) 
        df_mrna = df_mrna.set_index('Hugo_Symbol')
        df_mrna.drop(['Entrez_Gene_Id'], axis=1, inplace=True) 
        df_cna.T.to_csv('prad_fhrc_non_metastaste_cna.csv')
        df_mrna.T.to_csv('prad_fhrc_non_metastaste_mrna.csv')
        
       
    def print_statistics(self):
        self.retrieve_patients_ids()
        print(f'The number of all cases available is {self.nb_cases_all_ids_labels}')
        print(f'The number of cna cases available is {self.nb_cases_cna_ids_labels}')
        print(f'The number of cnaseq cases available is {self.nb_cases_cnaseq_ids_labels}')
      
      
if __name__ == "__main__":
    su2c_obj = BuildSu2c(directory='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository_prad/prad_su2c_2019')
    su2c_obj.print_statistics()
    su2c_obj.build_data_all_available()
    
    tcga_obj = BuildTCGA(directory='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository_prad/prad_tcga')
    tcga_obj.print_statistics()
    tcga_obj.build_data_all_available()
    
    eururol_obj = BuildEURUROL(directory='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository_prad/prad_eururol_2017')
    eururol_obj.print_statistics()
    eururol_obj.build_data_all_available()
    
    fhcrc_obj = BuildFHCRC(directory='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository_prad/prad_fhcrc')
    fhcrc_obj.print_statistics()
    fhcrc_obj.build_data_all_available()
