import datetime
import os
import random
from os import makedirs
from os.path import abspath, dirname, exists, join
from subprocess import call

import numpy as np
from sklearn.model_selection import ParameterGrid
from experiments.experiments_utilities import data_prad, data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna, data_repository, saving_repository

print(__file__)
# RESULTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "saving_repository"))
# EXPERIMENTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "experiments"))
# DATAREPOSITORY_PATH = os.environ.get('', join(dirname(abspath(__file__)), "datasets/datasets_repository"))
RESULTS_PATH = os.environ.get('', join(dirname(abspath(__file__))))
EXPERIMENTS_PATH = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/experiments'
DATAREPOSITORY_PATH = '/home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository'
PROJECT_ROOT = dirname(abspath(__file__))
SAVING_REPO = '/home/maoss2/project/maoss2/saving_repository_article/'
saving_repository_tn = f'{saving_repository}groups_brca_experiments'
saving_repository_prad = f'{saving_repository}groups_prad_experiments'

def launch_slurm_experiment_group(return_views, dataset, cancer_expe, nb_repetitions, pathway_file, update_method, c, random_weights, 
                                  prior_dict_groups, prior_dict_rules, experiment_file, experiment_name, time, dispatch_path, saving_repo):
    exp_file = join(dispatch_path, experiment_name)
    submission_script = ""
    submission_script += f"#!/bin/bash\n"
    submission_script += f"#SBATCH --nodes=1\n" 
    submission_script += f"#SBATCH --ntasks-per-node=8\n" 
    submission_script += f"#SBATCH --mem=128000M\n" 
    submission_script += f"#SBATCH --account=rrg-corbeilj-ac\n"
    submission_script += f"#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca\n"
    submission_script += f"#SBATCH --mail-type=BEGIN\n"
    submission_script += f"#SBATCH --mail-type=END\n"
    submission_script += f"#SBATCH --mail-type=FAIL\n"
    submission_script += f"#SBATCH --time={time}:00:00\n" 
    submission_script += f"#SBATCH --output={exp_file + '.out'}\n\n" 
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -data {dataset} -cancer_expe {cancer_expe} -rt {return_views} -nb_r {nb_repetitions} -g_dict {pathway_file} -u_m {update_method} -c {c} -random_weights {random_weights} -prior_dict_groups {prior_dict_groups} -prior_dict_rules {prior_dict_rules} -exp_name {experiment_name} -o {saving_repo}" 
    
    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])
   
    
def main_group_TN():
    # return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
    #             'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']
    return_views = ['methyl_rna_iso_mirna_snp_clinical']
    dictionaries_paths = [f"{DATAREPOSITORY_PATH}/groups2pathwaysTN_biogrid.pck"]
    update_method = ['inner_group', 'outer_group']
    random.seed(42)
    c_list = np.round(np.linspace(0.1, 1, 10), 3)
    random_weights_list = [False]
    param_grid = {'view': return_views, 'update': update_method, 'c': c_list, 'random_weights': random_weights_list}
    dispatch_path = join(RESULTS_PATH, "dispatch_f_exp_group_TN_biogrid")
    if not exists(dispatch_path): makedirs(dispatch_path)
    if not exists(saving_repository_tn): makedirs(saving_repository_tn)
    for pathway_dict in dictionaries_paths:
        name_pathway_file = pathway_dict.split('/')[-1].split('.')[0]
        print(f"Launching {pathway_dict}")
        for params in ParameterGrid(param_grid):
            print(f"Launching {params}")
            nb_repetitions = 15
            exp_name = f"{params['view']}__group_scm__" + f"{name_pathway_file}__" + f"{params['update']}__" + f"c{params['c']}__" + f"random_weights{params['random_weights']}__" + f"{nb_repetitions}"
            launch_slurm_experiment_group(return_views=params['view'], 
                                          dataset=data_tn_new_label_unbalanced_cpg_rna_rna_iso_mirna,
                                          cancer_expe='brca',
                                            nb_repetitions=nb_repetitions, 
                                            pathway_file=pathway_dict, 
                                            update_method=params['update'], 
                                            c=params['c'],
                                            random_weights=params['random_weights'],
                                            prior_dict_groups=f"{data_repository}/groups2genes_biogrid.pck",
                                            prior_dict_rules=f"{data_repository}/groups2pathwaysTN_biogrid.pck",
                                            experiment_file='run_new_group_experiments.py', 
                                            experiment_name=exp_name, 
                                            time='15', 
                                            dispatch_path=dispatch_path,
                                            saving_repo=saving_repository_tn)
    print("### DONE ###") 


def main_group_PRAD():
    return_views = ['all']
    dictionaries_paths = [f"{DATAREPOSITORY_PATH}/groups2pathwaysPRAD_biogrid.pck"]
    update_method = ['inner_group', 'outer_group']
    random.seed(42)
    c_list = np.round(np.linspace(0.1, 1, 10), 3)
    random_weights_list = [False]
    param_grid = {'view': return_views, 'update': update_method, 'c': c_list, 'random_weights': random_weights_list}
    dispatch_path = join(RESULTS_PATH, "dispatch_f_exp_group_PRAD_biogrid")
    if not exists(dispatch_path): makedirs(dispatch_path)
    if not exists(saving_repository_prad): makedirs(saving_repository_prad)
    for pathway_dict in dictionaries_paths:
        name_pathway_file = pathway_dict.split('/')[-1].split('.')[0]
        print(f"Launching {pathway_dict}")
        for params in ParameterGrid(param_grid):
            print(f"Launching {params}")
            nb_repetitions = 15
            exp_name = f"{params['view']}__group_scm__" + f"{name_pathway_file}__" + f"{params['update']}__" + f"c{params['c']}__" + f"random_weights{params['random_weights']}__" + f"{nb_repetitions}"
            launch_slurm_experiment_group(return_views=params['view'], 
                                            nb_repetitions=nb_repetitions, 
                                            dataset=data_prad,
                                            cancer_expe='prad',
                                            pathway_file=pathway_dict, 
                                            update_method=params['update'], 
                                            c=params['c'],
                                            random_weights=params['random_weights'],
                                            prior_dict_groups=f"{data_repository}/groups2genes_biogrid.pck",
                                            prior_dict_rules=f"{data_repository}/groups2pathwaysPRAD_biogrid.pck",
                                            experiment_file='run_new_group_experiments.py', 
                                            experiment_name=exp_name, 
                                            time='15', 
                                            dispatch_path=dispatch_path,
                                            saving_repo=saving_repository_prad)
    print("### DONE ###") 

if __name__ == '__main__':
    main_group_TN()
    main_group_PRAD()
