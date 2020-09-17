#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24000M
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=00:20:00
#SBATCH --output=/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/dispatch_folder/dispatch_analysis.out

# remove all the files so far
# On the project repo
rm /home/maoss2/project/maoss2/saving_repository_article/group_scm_*
rm /home/maoss2/project/maoss2/saving_repository_article/normal_brca_results*
rm /home/maoss2/project/maoss2/saving_repository_article/normal_prad_results*
rm -rf /home/maoss2/project/maoss2/saving_repository_article/histograms_repo

# In my home repo
rm /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/group_scm_*
rm /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/normal_brca_results*
rm /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/normal_prad_results*
rm -rf /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/histograms_repo

# Compute the analysis
# Global Analysis
#python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/normal_experiments_brca --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid.pck --output-text-file normal_brca_results_analysis --type-experiment normal --sous-experiment-types 'dt scm rf' --plot-hist
#python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/normal_experiments_prad --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysPRAD_biogrid.pck --output-text-file normal_prad_results_analysis --type-experiment normal --sous-experiment-types 'dt scm rf' --plot-hist

python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/normal_experiments_brca --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid_msigDB.pck --output-text-file normal_brca_results_analysis --type-experiment normal --sous-experiment-types 'dt scm rf' --plot-hist
python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/normal_experiments_brca_old --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid.pck --output-text-file normal_brca_results_analysis_old --type-experiment normal --sous-experiment-types 'dt scm rf' --plot-hist

python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid.pck --output-text-file group_scm_brca_results_analysis --type-experiment group_scm --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' 
python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_prad_experiments --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysPRAD_biogrid.pck --output-text-file group_scm_prad_results_analysis --type-experiment group_scm --sous-experiment-types 'all' 

python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_false/inner_group --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid_msigDB.pck --output-text-file group_scm_brca_results_analysis_eliminate_false_inner_group --type-experiment group_scm --sous-experiment-types 'methyl_rna_iso_mirna' 
python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_false/inner_group --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid_msigDB.pck --output-text-file group_scm_brca_results_analysis_eliminate_false_outer_group --type-experiment group_scm --sous-experiment-types 'methyl_rna_iso_mirna' 

python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_true/outer_group --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid_msigDB.pck --output-text-file group_scm_brca_results_analysis_eliminate_true_inner_group --type-experiment group_scm --sous-experiment-types 'methyl_rna_iso_mirna' 
python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_true/outer_group --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid_msigDB.pck --output-text-file group_scm_brca_results_analysis_eliminate_true_outer_group --type-experiment group_scm --sous-experiment-types 'methyl_rna_iso_mirna' 

python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments_old/inner_group --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid.pck --output-text-file group_scm_brca_results_analysis_inner_group --type-experiment group_scm --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' 
python experiments/experiments_utilities.py run-analysis --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments_old/outer_group --dict-for-prior-rules /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/groups2pathwaysTN_biogrid.pck --output-text-file group_scm_brca_results_analysis_outer_group --type-experiment group_scm --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' 


# Plot Group Figures
python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_prad_experiments --cancer-name 'prad' --f exp --sous-experiment-types 'all' --type-of-update 'inner' --plot-mean --plot-best
python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_prad_experiments --cancer-name 'prad' --f exp --sous-experiment-types 'all' --type-of-update 'outer' --plot-mean --plot-best

python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments --cancer-name 'brca' --f exp --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --type-of-update 'inner' --plot-mean --plot-best
python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments --cancer-name 'brca' --f exp --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --type-of-update 'outer' --plot-mean --plot-best

python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_false/inner_group --fig-name 'inner_eliminate_false' --cancer-name 'brca' --f exp --sous-experiment-types 'methyl_rna_iso_mirna' --type-of-update 'inner' --plot-mean --plot-best
python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_false/outer_group --fig-name 'outer_eliminate_false' --cancer-name 'brca' --f exp --sous-experiment-types 'methyl_rna_iso_mirna' --type-of-update 'outer' --plot-mean --plot-best

python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_true/inner_group --fig-name 'inner_eliminate_true' --cancer-name 'brca' --f exp --sous-experiment-types 'methyl_rna_iso_mirna' --type-of-update 'inner' --plot-mean --plot-best
python experiments/experiments_utilities.py run-plot-groups --directory /home/maoss2/project/maoss2/saving_repository_article/groups_brca_experiments/eliminate_true/outer_group --fig-name 'outer_eliminate_true' --cancer-name 'brca' --f exp --sous-experiment-types 'methyl_rna_iso_mirna' --type-of-update 'outer' --plot-mean --plot-best
# Plot BoxPlots
# BoxPlots Groups SCM Features
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used 'group_scm' --target-features 'cg00347904_SCUBE3 cg20261915_GLP2R uc002acg.3_KIAA1370 cg20556988_CCL1 cg14620221_OR8B8 uc001xqa.2_LTBP2'
#python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used 'group_scm' --target-features 'mrna_LRIT1 mrna_SUN3 mrna_AGMO mrna_TMEM71'
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used 'group_scm' --target-features 'hsa-mir-190b hsa-mir-135b hsa-mir-505 hsa-mir-106b hsa-mir-412'

# BoxPlots Groups SCM Features: ADD features 2 and more
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used group_scm --target-features 'cg00347904_SCUBE3 cg20261915_GLP2R'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used group_scm --target-features 'cg00347904_SCUBE3 cg20261915_GLP2R uc002acg.3_KIAA1370'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used group_scm --target-features 'cg00347904_SCUBE3 cg20261915_GLP2R uc002acg.3_KIAA1370 cg20556988_CCL1'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used group_scm --target-features 'cg00347904_SCUBE3 cg20261915_GLP2R uc002acg.3_KIAA1370 cg20556988_CCL1 cg14620221_OR8B8'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used group_scm --target-features 'cg00347904_SCUBE3 cg20261915_GLP2R uc002acg.3_KIAA1370 cg20556988_CCL1 cg14620221_OR8B8 uc001xqa.2_LTBP2'

python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used group_scm --target-features 'mrna_LRIT1 mrna_SUN3'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used group_scm --target-features 'mrna_LRIT1 mrna_SUN3 mrna_AGMO'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used group_scm --target-features 'mrna_LRIT1 mrna_SUN3 mrna_AGMO mrna_TMEM71'

python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used group_scm --target-features 'hsa-mir-190b hsa-mir-135b'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used group_scm --target-features 'hsa-mir-190b hsa-mir-135b hsa-mir-505'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used group_scm --target-features 'hsa-mir-190b hsa-mir-135b hsa-mir-505 hsa-mir-106b'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used group_scm --target-features 'hsa-mir-190b hsa-mir-135b hsa-mir-505 hsa-mir-106b hsa-mir-412'

# BoxPlots SCM Features
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used 'scm' --target-features 'uc002vwt.2_MLPH uc002hul.3_RARA'
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used 'scm' --target-features 'uc002vwt.2_MLPH uc002hul.3_RARA uc009wsd.2_HDGF uc001jpo.1_TSPAN15'
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used 'scm' --target-features 'mrna_PSKH2 mrna_GABRB1 mrna_ZP2 mrna_HOXB9'

# BoxPlots SCM Features: ADD features 2 and more
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used scm --target-features 'uc002vwt.2_MLPH uc002hul.3_RARA'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used scm --target-features 'uc002vwt.2_MLPH uc002hul.3_RARA'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used scm --target-features 'uc002vwt.2_MLPH uc002hul.3_RARA uc009wsd.2_HDGF'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna_complet.h5 --sous-experiment-types 'methyl_rna_iso_mirna' --cancer-name 'brca' --algo-used scm --target-features 'uc002vwt.2_MLPH uc002hul.3_RARA uc009wsd.2_HDGF uc001jpo.1_TSPAN15'

python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used scm --target-features 'mrna_PSKH2 mrna_GABRB1'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used scm --target-features 'mrna_PSKH2 mrna_GABRB1 mrna_ZP2'
python experiments/figures.py run-add-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used scm --target-features 'mrna_PSKH2 mrna_GABRB1 mrna_ZP2 mrna_HOXB9'


# BoxPlots DT Features
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used 'dt' --target-features 'uc002vwt.2_MLPH cg26377677_FURIN uc003aed.2_XBP1'
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used 'dt' --target-features 'mrna_PSKH2 mrna_GABRB1 mrna_ZP2'

# BoxPlots RF Features
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used 'rf' --target-features 'uc003str.2_AGR2 uc003sts.2_AGR3 uc002vwt.2_MLPH'
python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used 'rf' --target-features 'mrna_GPR78 mrna_CCL1 mrna_SUN3'

# Copy to my home
cp /home/maoss2/project/maoss2/saving_repository_article/group_scm_* /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/
cp /home/maoss2/project/maoss2/saving_repository_article/normal_brca_results* /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/
cp /home/maoss2/project/maoss2/saving_repository_article/normal_prad_results* /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/
cp -R /home/maoss2/project/maoss2/saving_repository_article/histograms_repo /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/saving_repository_article/