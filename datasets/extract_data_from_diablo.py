import pandas as pd
import numpy as np
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

os.chdir('/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository')
# Load the data
ro.r.load('TCGA.normalised.mixDIABLO_2.RData')
# ['data.train', 'data.test']
data_train = ro.r['data.train']  # len(data_train) == 6
data_test = ro.r['data.test']  # len(data_test) == 5

# Convert data_train_0 to pandasDataframe
with localconverter(ro.default_converter + pandas2ri.converter):
	pd_from_r_df = ro.conversion.rpy2py(data_train[0])
# Save to file
pd_from_r_df.to_csv('data_train_clinical.tsv', sep='\t')

data_train_methylation = pd.DataFrame(data=np.array(data_train[1]), index=pd_from_r_df.index)
data_train_methylation.to_csv('data_train_methylation.tsv', sep='\t')

data_train_mirna = pd.DataFrame(data=np.array(data_train[2]), index=pd_from_r_df.index)
data_train_mirna.to_csv('data_train_mirna.tsv', sep='\t')

data_train_rna = pd.DataFrame(data=np.array(data_train[3]), index=pd_from_r_df.index)
data_train_rna.to_csv('data_train_rna.tsv', sep='\t')

data_train_proteomics = pd.DataFrame(data=np.array(data_train[4]), index=pd_from_r_df.index)
data_train_proteomics.to_csv('data_train_proteomics.tsv', sep='\t')

data_train_labels = pd.DataFrame(data=np.array(data_train[5]), index=pd_from_r_df.index)
data_train_labels.to_csv('data_train_labels.tsv', sep='\t')


with localconverter(ro.default_converter + pandas2ri.converter):
	pd_from_r_df = ro.conversion.rpy2py(data_test[0])
# Save to file
pd_from_r_df.to_csv('data_test_clinical.tsv', sep='\t')

data_test_methylation = pd.DataFrame(data=np.array(data_test[1]), index=pd_from_r_df.index)
data_test_methylation.to_csv('data_test_methylation.tsv', sep='\t')

data_test_mirna = pd.DataFrame(data=np.array(data_test[2]), index=pd_from_r_df.index)
data_test_mirna.to_csv('data_test_mirna.tsv', sep='\t')

data_test_rna = pd.DataFrame(data=np.array(data_test[3]), index=pd_from_r_df.index)
data_test_rna.to_csv('data_test_rna.tsv', sep='\t')

data_test_labels = pd.DataFrame(data=np.array(data_test[4]), index=pd_from_r_df.index)
data_test_labels.to_csv('data_test_labels.tsv', sep='\t')


