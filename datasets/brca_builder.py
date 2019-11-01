# -*- coding: utf-8 -*-
__author__ = 'maoss2'
from datasets.datasets_utilitaires import *


class BuildBrcaDatasets(BuildOmicsDatasets):
    def __init__(self,
                 cancer_name='BRCA',
                 label_file=label_file_triple_all,
                 methyl_path_450=project_path_on_is2 +
                                 "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3",
                 methyl_path_27=project_path_on_is2 +
                                "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3",
                 rnaseq_path=project_path_on_is2 +
                             "BRCA/brca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3",
                 snps_path=project_path_on_is2 +
                           "BRCA/brca_exome/Somatic_Mutations/WUSM__IlluminaGA_DNASeq_curated/Level_2/genome.wustl.edu__IlluminaGA_curated_DNA_sequencing_level2.maf",
                 mirna_file=project_path_on_is2 +
                            "BRCA/brca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3"):
        super(BuildBrcaDatasets, self).__init__(cancer_name, methyl_path_450, methyl_path_27, rnaseq_path, snps_path,
                                                mirna_file)
        self.labels_file = label_file

    def combine_dataset(self,
                        file_name='',
                        methyl_450_file=methyl_450_file,
                        methyl_27_file=methyl_27_file,
                        rnaseq_genes_file=rnaseq_genes_file,
                        rnaseq_isoforms_file=rnaseq_isoforms_file,
                        snp_file=snp_file,
                        mirna_file=mirna_file,
                        clinical_file=new_clinical_file,
                        balanced_dataset=False,
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
              balanced_dataset, bool, False to build unbalanced dataset, True to build balanced one
              filling_type, str, 'mean', 'zero', 'median' or None
        :return:
        """
        if balanced_dataset:
            name = '{}_balanced_{}.h5'.format(file_name, filling_type)
        else:
            name = '{}_unbalanced_{}.h5'.format(file_name, filling_type)
        try:
            labels = pd.read_csv('{}'.format(self.labels_file), index_col="example_id", sep="\t")
            index_pos = labels.loc[labels['labels'] == 1].index.values
            index_neg = labels.loc[labels['labels'] == -1].index.values
            if balanced_dataset:
                index_neg_selected = np.asarray(random.sample(list(index_neg), len(index_pos)))
                index_neg_non_selected = [el for el in index_neg if el not in index_neg_selected]
                labels.drop(index_neg_non_selected, axis=0, inplace=True)
                index_neg = index_neg_selected
            # Save to file
            labels.to_hdf(name, "labels")
        except ValueError:
            labels = pd.read_csv('{}'.format(self.labels_file), sep=';', index_col="bcr_patient_barcode")
            y_labels = labels['phenotype_TN']
            index_pos = labels.loc[labels['phenotype_TN'] == 1].index.values
            index_neg = labels.loc[labels['phenotype_TN'] == -1].index.values
            if balanced_dataset:
                index_neg_selected = np.asarray(random.sample(list(index_neg), len(index_pos)))
                index_neg_non_selected = [el for el in index_neg if el not in index_neg_selected]
                labels.drop(index_neg_non_selected, axis=0, inplace=True)
                index_neg = index_neg_selected
            labels.replace(['LA', 'LB', 'HER2++', 'TN'], [1, 2, 3, 4],
                           inplace=True)  # IMPORTANT POUR DECODER LEUR SIGNIFICATION
            labels.replace(['NON TN', 'BASAL NON TN'], [1, 2],
                           inplace=True)  # IMPORTANT POUR DECODER LEUR SIGNIFICATION
            labels_phenotype_normal = labels['phenotype']
            labels_phenotype_diablo = labels['phenotype_DIABLO']
            labels_phenotype_diablo_TN_and_basal = labels['phenotype_DIABLO_TN_AND_BASAL']
            er_positive_probability = labels['er_status_ihc_Percent_Positive']
            pr_positive_probability = labels['pr_status_ihc_percent_Positive']
            her_positive_probability = labels['her2_ihc_percent_Positive']
            # Save to file
            y_labels.to_hdf(name, "labels")
            labels_phenotype_normal.to_hdf(name, 'label_normal_phenotype')
            labels_phenotype_diablo.to_hdf(name, 'label_diablo_phenotype')
            labels_phenotype_diablo_TN_and_basal.to_hdf(name, 'label_diablo_TN_and_basal_phenotype')
            er_positive_probability.to_hdf(name, 'er_positive_probability')
            pr_positive_probability.to_hdf(name, 'pr_positive_probability')
            her_positive_probability.to_hdf(name, 'her_positive_probability')
        methylation_27 = pd.read_csv(methyl_27_file, sep="\t")
        indexes = np.array(list(map(str, np.array(methylation_27["Unnamed: 0"]))))
        methylation_27.set_index(indexes, inplace=True)
        methylation_27 = methylation_27.T.loc[labels.index.values]
        methylation_27 = methylation_27.loc[:, methylation_27.count() > 0]

        methylation_450 = pd.read_csv(methyl_450_file, sep="\t")
        indexes = np.array(list(map(str, np.array(methylation_450["Unnamed: 0"]))))
        methylation_450.set_index(indexes, inplace=True)
        methylation_450 = methylation_450.T.loc[labels.index.values]
        methylation_450 = methylation_450.loc[:, methylation_450.count() > 0]

        indexes_fusion = [el for el in methylation_27.columns.values if el in methylation_450.columns.values]
        methylation_fusion_450 = methylation_450[indexes_fusion]
        methylation_fusion_27 = methylation_27[indexes_fusion]
        methylation_fusion = deepcopy(methylation_fusion_450)
        # Saving file (temp)
        # methylation_fusion_450.to_csv('methylation_fusion_450.tsv', sep="\t", encoding='utf-8')
        # methylation_fusion_27.to_csv('methylation_fusion_27.tsv', sep="\t", encoding='utf-8')
        # recuperer les endroits ou c'est nan dans methylation_fusion_450
        informations_on_the_nan_places = methylation_fusion_450.isnull().all(axis=1)
        for i in range(methylation_450.shape[0]):  # iterer sur le nombre des exemples, c'est lent mais bon
            if informations_on_the_nan_places[i] is True:
                methylation_fusion.set_value(i, methylation_fusion_27[i])
        methylation_fusion = methylation_fusion.apply(pd.to_numeric, errors='coerce')
        methylation_fusion.to_csv('methylation_fusion.tsv', sep="\t", encoding='utf-8')

        if filling_type is not None:
            methylation_27 = self.method_to_fill_nan(data=methylation_27, idx_pos=index_pos, idx_neg=index_neg,
                                                     filling_type=filling_type)
            methylation_450 = self.method_to_fill_nan(data=methylation_450, idx_pos=index_pos, idx_neg=index_neg,
                                                      filling_type=filling_type)
            methylation_fusion = self.method_to_fill_nan(data=methylation_fusion, idx_pos=index_pos, idx_neg=index_neg,
                                                         filling_type=filling_type)
            methylation_fusion_27 = self.method_to_fill_nan(data=methylation_fusion_27, idx_pos=index_pos,
                                                            idx_neg=index_neg, filling_type=filling_type)
            methylation_fusion_450 = self.method_to_fill_nan(data=methylation_fusion_450, idx_pos=index_pos,
                                                             idx_neg=index_neg, filling_type=filling_type)

        rnaseq_genes = pd.read_csv(rnaseq_genes_file, sep="\t")
        indexes = np.array(list(map(str, np.array(rnaseq_genes["Unnamed: 0"]))))
        rnaseq_genes.set_index(indexes, inplace=True)
        rnaseq_genes = rnaseq_genes.T.loc[labels.index.values]
        rnaseq_genes = rnaseq_genes.loc[:, rnaseq_genes.count() > 0]
        if filling_type is not None:
            rnaseq_genes = self.method_to_fill_nan(data=rnaseq_genes, idx_pos=index_pos, idx_neg=index_neg,
                                                   filling_type=filling_type)

        rnaseq_isoforms = pd.read_csv(rnaseq_isoforms_file, sep="\t")
        indexes = np.array(list(map(str, np.array(rnaseq_isoforms["Unnamed: 0"]))))
        rnaseq_isoforms.set_index(indexes, inplace=True)
        rnaseq_isoforms = rnaseq_isoforms.T.loc[labels.index.values]
        rnaseq_isoforms = rnaseq_isoforms.loc[:, rnaseq_isoforms.count() > 0]
        if filling_type is not None:
            rnaseq_isoforms = self.method_to_fill_nan(data=rnaseq_isoforms, idx_pos=index_pos, idx_neg=index_neg,
                                                      filling_type=filling_type)

        snps = pd.read_csv(snp_file, sep="\t")
        indexes = np.array(list(map(str, np.array(snps["Unnamed: 0"]))))
        snps.set_index(indexes, inplace=True)
        snps = snps.T.loc[labels.index.values]
        snps = snps.loc[:, snps.count() > 0]
        snps = snps.apply(pd.to_numeric, errors='coerce')
        if filling_type is not None:
            snps = self.method_to_fill_nan(data=snps, idx_pos=index_pos, idx_neg=index_neg, filling_type='zero')
        mirnas = pd.read_csv(mirna_file, sep="\t")
        indexes = np.array(list(map(str, np.array(mirnas["Unnamed: 0"]))))
        mirnas.set_index(indexes, inplace=True)
        mirnas = mirnas.T.loc[labels.index.values]
        mirnas = mirnas.loc[:, mirnas.count() > 0]
        if filling_type is not None:
            mirnas = self.method_to_fill_nan(data=mirnas, idx_pos=index_pos, idx_neg=index_neg,
                                             filling_type=filling_type)

        clinical_data = pd.read_csv(clinical_file, sep='\t')
        indexes = np.array(list(map(str, clinical_data['bcr_patient_barcode'].values)))
        clinical_data.set_index(indexes, inplace=True)
        clinical_data.drop(['Unnamed: 0', 'bcr_patient_barcode'], axis=1, inplace=True)
        clinical_data = clinical_data.loc[labels.index.values]
        clinical_data = clinical_data.apply(pd.to_numeric, errors='coerce')

        # Check the examples
        assert np.all(labels.index.values == methylation_450.index.values)
        assert np.all(labels.index.values == methylation_27.index.values)
        assert np.all(labels.index.values == rnaseq_isoforms.index.values)
        assert np.all(labels.index.values == rnaseq_genes.index.values)
        assert np.all(labels.index.values == snps.index.values)
        assert np.all(labels.index.values == mirnas.index.values)
        assert np.all(labels.index.values == clinical_data.index.values)

        # Write the data to hdf5
        # methylation_450.to_hdf(name, "methylation_450")
        # methylation_27.to_hdf(name, "methylation_27")
        # methylation_fusion_27.to_hdf(name, "methylation_fusion_27")
        # methylation_fusion_450.to_hdf(name, "methylation_fusion_450")
        methylation_fusion.to_hdf(name, "methylation_fusion")
        rnaseq_genes.to_hdf(name, "rnaseq_genes")
        rnaseq_isoforms.to_hdf(name, "rnaseq_isoforms")
        snps.to_hdf(name, "snp")
        mirnas.to_hdf(name, "mirna")
        clinical_data.to_hdf(name, 'clinical_view')

    def main_build_tsv(self):
        # self.build_methylome_450_tsv()
        # self.build_methylome_27_tsv()
        # self.build_rnaseq_tsv()
        # self.build_mirna_tsv()
        # self.build_clinical_view()
        self.combine_dataset(file_name='',
                             methyl_450_file=methyl_450_file,
                             methyl_27_file=methyl_27_file,
                             rnaseq_genes_file=rnaseq_genes_file,
                             rnaseq_isoforms_file=rnaseq_isoforms_file,
                             snp_file=snp_file,
                             mirna_file=mirna_file,
                             clinical_file=new_clinical_file,
                             filling_type='mean')


def build_brca_dataset_for_graalpy(dataset='',
                                   name='',
                                   output_path='./',
                                   methyl_example_file='/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/files_to_build_datasets/jhu-usc.edu_BRCA.HumanMethylation27.4.lvl-3.TCGA-E2-A15M-11A-22D-A12E-05.txt',
                                   genes_example_file='/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/files_to_build_datasets/unc.edu.ffd6c7c5-d4c4-4ead-9e55-de8f6aa62182.2248604.rsem.genes.results',
                                   snp_data_file='/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/files_to_build_datasets/genome.wustl.edu__IlluminaGA_curated_DNA_sequencing_level2.maf'):
    """ Transform the raw dataset to the view we explicitly want to be loaded. Get the dataset loader ready
    Args:
        dataset, str, path to raw dataset, .h5
        name, str, name of the final dataset
        output_path, str, place to save the file,
        methyl_example_file, a sample example file to build the right feature names for methyl view
        genes_example_file, a sample example file to build the right feature names for genes view
        snp_data_file, a sample example file to build the right feature names for snp view
    Returns:
            New dataset loader ready  with keys, data, target, features_names, patients_ids
    """
    data = h5py.File(dataset, 'r')
    x_methyl_fusion = data['methylation_fusion/block0_values'][()]
    features_names_methyl_fusion = data['methylation_fusion/block0_items'][()]
    features_names_methyl_fusion = np.asarray([el.decode('utf8') for el in features_names_methyl_fusion])
    # linked the methyl_name to the genes_name
    d = pd.read_table(methyl_example_file, skiprows=[0], header='infer')
    d.fillna('INEXISTANT', inplace=True)
    features_names_methyl_fusion_linked = ['{}_{}'.format(d['Composite Element REF'].values[i],
                                                          d['Gene_Symbol'].values[i])
                                           for i in range(d.shape[0]) if d['Composite Element REF'].values[i] in
                                           features_names_methyl_fusion]
    features_names_methyl_fusion_linked = np.asarray(features_names_methyl_fusion_linked)
    x_mirna = data['mirna/block0_values'][()]
    features_names_mirna = data['mirna/block0_items'][()]
    features_names_mirna = np.asarray([el.decode('utf8') for el in features_names_mirna])
    x_rna_isoforms = data['rnaseq_isoforms/block0_values'][()]
    features_names_rna_isoforms = data['rnaseq_isoforms/block0_items'][()]
    features_names_rna_isoforms = np.asarray([el.decode('utf8') for el in features_names_rna_isoforms])

    x_rna = data['rnaseq_genes/block0_values'][()]
    features_names_rna = data['rnaseq_genes/block0_items'][()]
    features_names_rna = np.asarray([el.decode('utf8') for el in features_names_rna])

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
            index_el = np.where(temp_isoforms_ids_names_flatten_list == el)[0][
                0]  # recuperer la position ou l'élément est
            features_names_rna_isoforms_linked[i] = temp_features_names_rna_isoforms_linked[
                index_el]  # remplacer par le nouveau nom

    x_snps = data['snp/block0_values'][()]
    features_names_snps = data['snp/block0_items'][()]
    features_names_snps = np.asarray([el.decode('utf8') for el in features_names_snps])
    snp_data = pd.read_table(snp_data_file, sep="\t", index_col="Tumor_Sample_Barcode")
    drop_columns = ["Center", "Ncbi_Build", "Archive_Name", "Strand", "Dbsnp_Rs", "Dbsnp_Val_Status",
                    "Verification_Status", "Sequencer", "Validation_Status", "Validation_Method", "Score", "File_Name",
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
    zipping_name = list(zip(snp_data['Hugo_Symbol'], snp_data['snp_id']))
    features_names_snps_linked = np.zeros((features_names_snps.shape[0]), dtype='O')
    for i, el in enumerate(features_names_snps):
        for zip_el in zipping_name:
            if zip_el[1] == el:
                features_names_snps_linked[i] = '{}_{}'.format(el, zip_el[0])

    x_clinical = np.hstack((data['clinical_view/block0_values'][()], data['clinical_view/block1_values'][()]))
    features_names_clinical = np.hstack((data['clinical_view/block0_items'][()],
                                         data['clinical_view/block1_items'][()]))
    features_names_clinical = np.asarray([el.decode('utf8') for el in features_names_clinical])
    try:
        y = data['labels/block0_values'][()]
    except KeyError:
        y = data['labels/values'][()]
    y = y.reshape(-1)
    x = np.hstack((x_methyl_fusion, x_rna_isoforms, x_mirna, x_snps, x_clinical, x_rna))
    features_names_methyl_fusion_linked = np.asarray(features_names_methyl_fusion_linked, dtype='str')
    features_names_rna_isoforms_linked = np.asarray(features_names_rna_isoforms_linked, dtype='str')
    features_names_snps_linked = np.asarray(features_names_snps_linked, dtype='str')
    features_names = np.hstack((features_names_methyl_fusion_linked, features_names_rna_isoforms_linked,
                                features_names_mirna, features_names_snps_linked, features_names_clinical,
                                features_names_rna))
    features_names = [str(x).encode('utf-8') for x in features_names]
    data.close()
    # New add section: Lire la matrice avec pandas :) et extraire les patients ids car je veux tester pour
    # voir sur qui on se trompe
    df_temp = pd.read_hdf(dataset, key='methylation_fusion')
    patients_ids = np.asarray(df_temp.index.values, dtype='str')
    patients_ids = [str(x).encode('utf-8') for x in patients_ids]
    f = h5py.File(os.path.join(output_path, '{}.h5'.format(name)), 'w')
    f.create_dataset('data', data=x)
    f.create_dataset('target', data=y)
    f.create_dataset('features_names', data=features_names)
    f.create_dataset('patients_ids', data=patients_ids)


def main_brca_dataset_builder():
    for filling in ['mean', 'median', 'zero']:
        for label_file in [new_label_file, label_file_triple_all]:
            for balanced in [False, True]:
                brca_builder = BuildBrcaDatasets(cancer_name='BRCA',
                                                 label_file=label_file,
                                                 methyl_path_450=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3",
                                                 methyl_path_27=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3",
                                                 rnaseq_path=project_path_on_is2 + "BRCA/brca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3",
                                                 snps_path=project_path_on_is2 + "BRCA/brca_exome/Somatic_Mutations/WUSM__IlluminaGA_DNASeq_curated/Level_2/genome.wustl.edu__IlluminaGA_curated_DNA_sequencing_level2.maf",
                                                 mirna_file=project_path_on_is2 + "BRCA/brca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3")
                if label_file == new_label_file:
                    brca_builder.combine_dataset(file_name='BRCA_triple_neg_new_labels',
                                                 methyl_450_file=methyl_450_file,
                                                 methyl_27_file=methyl_27_file,
                                                 rnaseq_genes_file=rnaseq_genes_file,
                                                 rnaseq_isoforms_file=rnaseq_isoforms_file,
                                                 snp_file=snp_file,
                                                 mirna_file=mirna_file,
                                                 clinical_file=new_clinical_file,
                                                 balanced_dataset=balanced,
                                                 filling_type=filling)
                    if balanced:
                        dataset_name = 'BRCA_triple_neg_new_labels_balanced_{}.h5'.format(filling)
                        final_dataset_name = 'triple_neg_new_labels_balanced_{}.h5'.format(filling)
                    else:
                        dataset_name = 'BRCA_triple_neg_new_labels_unbalanced_{}.h5'.format(filling)
                        final_dataset_name = 'triple_neg_new_labels_unbalanced_{}.h5'.format(filling)
                    build_brca_dataset_for_graalpy(dataset=dataset_name, name=final_dataset_name)
                else:
                    brca_builder.combine_dataset(file_name='BRCA_triple_neg_old_labels',
                                                 methyl_450_file=methyl_450_file,
                                                 methyl_27_file=methyl_27_file,
                                                 rnaseq_genes_file=rnaseq_genes_file,
                                                 rnaseq_isoforms_file=rnaseq_isoforms_file,
                                                 snp_file=snp_file,
                                                 mirna_file=mirna_file,
                                                 clinical_file=new_clinical_file,
                                                 balanced_dataset=balanced,
                                                 filling_type=filling)
                    if balanced:
                        dataset_name = 'BRCA_triple_neg_old_labels_balanced_{}.h5'.format(filling)
                        final_dataset_name = 'triple_neg_old_labels_balanced_mean_{}.h5'.format(filling)
                    else:
                        dataset_name = 'BRCA_triple_neg_old_labels_unbalanced_{}.h5'.format(filling)
                        final_dataset_name = 'triple_neg_old_labels_unbalanced_mean_{}.h5'.format(filling)
                    build_brca_dataset_for_graalpy(dataset=dataset_name, name=final_dataset_name)
# def main_brca_dataset_builder():
#     brca_builder = BuildBrcaDatasets(cancer_name='BRCA',
#                                      label_file=new_label_file,
#                                      methyl_path_450=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3",
#                                      methyl_path_27=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3",
#                                      rnaseq_path=project_path_on_is2 + "BRCA/brca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3",
#                                      snps_path=project_path_on_is2 + "BRCA/brca_exome/Somatic_Mutations/WUSM__IlluminaGA_DNASeq_curated/Level_2/genome.wustl.edu__IlluminaGA_curated_DNA_sequencing_level2.maf",
#                                      mirna_file=project_path_on_is2 + "BRCA/brca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3")
#     brca_builder.combine_dataset(file_name='BRCA_triple_neg_new_labels',
#                                  methyl_450_file=methyl_450_file,
#                                  methyl_27_file=methyl_27_file,
#                                  rnaseq_genes_file=rnaseq_genes_file,
#                                  rnaseq_isoforms_file=rnaseq_isoforms_file,
#                                  snp_file=snp_file,
#                                  mirna_file=mirna_file,
#                                  clinical_file=new_clinical_file,
#                                  filling_type='mean')
#     brca_builder.combine_dataset(file_name='BRCA_triple_neg_new_labels',
#                                  methyl_450_file=methyl_450_file,
#                                  methyl_27_file=methyl_27_file,
#                                  rnaseq_genes_file=rnaseq_genes_file,
#                                  rnaseq_isoforms_file=rnaseq_isoforms_file,
#                                  snp_file=snp_file,
#                                  mirna_file=mirna_file,
#                                  clinical_file=new_clinical_file,
#                                  filling_type='median')
#     brca_builder.combine_dataset(file_name='BRCA_triple_neg_new_labels',
#                                  methyl_450_file=methyl_450_file,
#                                  methyl_27_file=methyl_27_file,
#                                  rnaseq_genes_file=rnaseq_genes_file,
#                                  rnaseq_isoforms_file=rnaseq_isoforms_file,
#                                  snp_file=snp_file,
#                                  mirna_file=mirna_file,
#                                  clinical_file=new_clinical_file,
#                                  filling_type='zero')
#     build_brca_dataset_for_graalpy(dataset='BRCA_triple_neg_new_labels_unbalanced_mean.h5',
#                                    name='triple_neg_new_labels_unbalanced_mean.h5')
#     build_brca_dataset_for_graalpy(dataset='BRCA_triple_neg_new_labels_unbalanced_median.h5',
#                                    name='triple_neg_new_labels_unbalanced_median.h5')
#     build_brca_dataset_for_graalpy(dataset='BRCA_triple_neg_new_labels_unbalanced_zero.h5',
#                                    name='triple_neg_new_labels_unbalanced_zero.h5')
#
#     brca_builder = BuildBrcaDatasets(cancer_name='BRCA',
#                                      label_file=label_file_triple_all,
#                                      methyl_path_450=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3",
#                                      methyl_path_27=project_path_on_is2 + "BRCA/brca_methylome/DNA_Methylation/JHU_USC__HumanMethylation27/Level_3",
#                                      rnaseq_path=project_path_on_is2 + "BRCA/brca_rnaseq/RNASeqV2/UNC__IlluminaHiSeq_RNASeqV2/Level_3",
#                                      snps_path=project_path_on_is2 + "BRCA/brca_exome/Somatic_Mutations/WUSM__IlluminaGA_DNASeq_curated/Level_2/genome.wustl.edu__IlluminaGA_curated_DNA_sequencing_level2.maf",
#                                      mirna_file=project_path_on_is2 + "BRCA/brca_mirna/miRNASeq/BCGSC__IlluminaHiSeq_miRNASeq/Level_3")
#     brca_builder.combine_dataset(file_name='BRCA_triple_neg_old_labels',
#                                  methyl_450_file=methyl_450_file,
#                                  methyl_27_file=methyl_27_file,
#                                  rnaseq_genes_file=rnaseq_genes_file,
#                                  rnaseq_isoforms_file=rnaseq_isoforms_file,
#                                  snp_file=snp_file,
#                                  mirna_file=mirna_file,
#                                  clinical_file=new_clinical_file,
#                                  filling_type='mean')
#     brca_builder.combine_dataset(file_name='BRCA_triple_neg_old_labels',
#                                  methyl_450_file=methyl_450_file,
#                                  methyl_27_file=methyl_27_file,
#                                  rnaseq_genes_file=rnaseq_genes_file,
#                                  rnaseq_isoforms_file=rnaseq_isoforms_file,
#                                  snp_file=snp_file,
#                                  mirna_file=mirna_file,
#                                  clinical_file=new_clinical_file,
#                                  filling_type='median')
#     brca_builder.combine_dataset(file_name='BRCA_triple_neg_old_labels',
#                                  methyl_450_file=methyl_450_file,
#                                  methyl_27_file=methyl_27_file,
#                                  rnaseq_genes_file=rnaseq_genes_file,
#                                  rnaseq_isoforms_file=rnaseq_isoforms_file,
#                                  snp_file=snp_file,
#                                  mirna_file=mirna_file,
#                                  clinical_file=new_clinical_file,
#                                  filling_type='zero')
#     build_brca_dataset_for_graalpy(dataset='BRCA_triple_neg_old_labels_unbalanced_mean.h5',
#                                    name='triple_neg_nold_labels_unbalanced_mean.h5')
#     build_brca_dataset_for_graalpy(dataset='BRCA_triple_neg_old_labels_unbalanced_median.h5',
#                                    name='triple_neg_old_labels_unbalanced_median.h5')
#     build_brca_dataset_for_graalpy(dataset='BRCA_triple_neg_old_labels_unbalanced_zero.h5',
#                                    name='triple_neg_old_labels_unbalanced_zero.h5')


if __name__ == '__main__':
    main_brca_dataset_builder()
