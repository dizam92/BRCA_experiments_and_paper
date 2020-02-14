import os
import random
import numpy as np
from sklearn.model_selection import ParameterGrid
from os.path import join, abspath, dirname, exists
from os import makedirs
from subprocess import call
import datetime
print(__file__)
RESULTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "saving_repository"))
EXPERIMENTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "experiments"))
DATAREPOSITORY_PATH = os.environ.get('', join(dirname(abspath(__file__)), "datasets/datasets_repository"))
PROJECT_ROOT = dirname(abspath(__file__))

def launch_slurm_experiment(return_views, nb_repetitions, experiment_file, experiment_name, time, dispatch_path):
    exp_file = join(dispatch_path, f"{return_views}__" + f"{experiment_name}__" + f"{nb_repetitions}")
                        
    submission_script = ""
    submission_script += f"#!/bin/bash\n"
    submission_script += f"#SBATCH --nodes=1\n" 
    submission_script += f"#SBATCH --ntasks-per-node=8\n" 
    submission_script += f"#SBATCH --mem=128000M\n" 
    submission_script += f"#SBATCH --account=rpp-corbeilj\n"
    submission_script += f"#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca\n"
    submission_script += f"#SBATCH --mail-type=BEGIN\n"
    submission_script += f"#SBATCH --mail-type=END\n"
    submission_script += f"#SBATCH --mail-type=FAIL\n"
    submission_script += f"#SBATCH --time={time}:00:00\n" 
    submission_script += f"#SBATCH --output={exp_file + '.out'}\n\n" 
    submission_script += f"{datetime.date}\n" 
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -rt {return_views} -nb_r {nb_repetitions}"

    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])

def main():
    return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']
    dispatch_path = join(RESULTS_PATH, "dispatch")
    if not exists(dispatch_path): makedirs(dispatch_path)
    for view in return_views:
        print(f"Launching {view}")
        launch_slurm_experiment(return_views=view, nb_repetitions=15, experiment_file='run_tn_experiments.py', 
                                experiment_name='normal_experiments', time='1', dispatch_path=dispatch_path)
    print("### DONE ###")   
    

def launch_slurm_experiment_old_group(return_views, nb_repetitions, pathway_file, update_method, 
                                  experiment_file, prior_initialization_type, experiment_name, time, dispatch_path):
    exp_file = join(dispatch_path, experiment_name)
    submission_script = ""
    submission_script += f"#!/bin/bash\n"
    submission_script += f"#SBATCH --nodes=1\n" 
    submission_script += f"#SBATCH --ntasks-per-node=8\n" 
    submission_script += f"#SBATCH --mem=128000M\n" 
    submission_script += f"#SBATCH --account=rpp-corbeilj\n"
    submission_script += f"#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca\n"
    submission_script += f"#SBATCH --mail-type=BEGIN\n"
    submission_script += f"#SBATCH --mail-type=END\n"
    submission_script += f"#SBATCH --mail-type=FAIL\n"
    submission_script += f"#SBATCH --time={time}:00:00\n" 
    submission_script += f"#SBATCH --output={exp_file + '.out'}\n\n" 
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -rt {return_views} -nb_r {nb_repetitions} -g_dict {pathway_file} -u_m {update_method} -init {prior_initialization_type} -exp_name {experiment_name}" 
    
    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])
    
def main_old_group():
    return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']
    dictionaries_paths = [f"{DATAREPOSITORY_PATH}/pathways_biogrid_groups.pck", f"{DATAREPOSITORY_PATH}/pathways_biogrid_groups_unknown_features_are_groups.pck"]
    prior_init_and_update_method = [('default', 'pos_exp_group'), ('default', 'neg_exp_group'), 
                                    ('exp', 'pos_exp'), ('exp', 'neg_exp'), 
                                    ('normal', 'pos_exp'), ('normal', 'neg_exp')]
    dispatch_path = join(RESULTS_PATH, "dispatch")
    if not exists(dispatch_path): makedirs(dispatch_path)
    for pathway_dict in dictionaries_paths:
        name_pathway_file = pathway_dict.split('/')[-1].split('.')[0]
        print(f"Launching {pathway_dict}")
        for init_and_update_method in prior_init_and_update_method:
            print(f"Launching {init_and_update_method}")
            for view in return_views:
                print(f"Launching {view}")
                nb_repetitions = 5
                exp_name = f"{view}__group_scm__" + f"{name_pathway_file}__" + f"{init_and_update_method[0]}__" + f"{init_and_update_method[1]}__" + f"{nb_repetitions}"
                launch_slurm_experiment_old_group(return_views=view, 
                                            nb_repetitions=nb_repetitions, 
                                            pathway_file=pathway_dict, 
                                            update_method=init_and_update_method[1], 
                                            prior_initialization_type=init_and_update_method[0],
                                            experiment_file='run_new_group_experiments.py', 
                                            experiment_name=exp_name, 
                                            time='3', 
                                            dispatch_path=dispatch_path)
    print("### DONE ###") 


def launch_slurm_experiment_group(return_views, nb_repetitions, pathway_file, update_method, c, inverse_prior_group,
                                  experiment_file, experiment_name, time, dispatch_path):
    exp_file = join(dispatch_path, experiment_name)
    submission_script = ""
    submission_script += f"#!/bin/bash\n"
    submission_script += f"#SBATCH --nodes=1\n" 
    submission_script += f"#SBATCH --ntasks-per-node=8\n" 
    submission_script += f"#SBATCH --mem=128000M\n" 
    submission_script += f"#SBATCH --account=rpp-corbeilj\n"
    submission_script += f"#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca\n"
    submission_script += f"#SBATCH --mail-type=BEGIN\n"
    submission_script += f"#SBATCH --mail-type=END\n"
    submission_script += f"#SBATCH --mail-type=FAIL\n"
    submission_script += f"#SBATCH --time={time}:00:00\n" 
    submission_script += f"#SBATCH --output={exp_file + '.out'}\n\n" 
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -rt {return_views} -nb_r {nb_repetitions} -g_dict {pathway_file} -u_m {update_method} -c {c} -inverse_prior_group {inverse_prior_group} -exp_name {experiment_name}" 
    
    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])
    
def main_group():
    # return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
    #             'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']
    return_views = ['methyl_rna_iso_mirna']
    dictionaries_paths = [f"{DATAREPOSITORY_PATH}/pathways_biogrid_groups.pck"]
    update_method = ['inner_group', 'outer_group']
    random.seed(42)
    c_list = np.linspace(0.1, 1, 10)
    inverse_prior_group_list = [False, True]
    param_grid = {'view': return_views, 'update': update_method, 'c': c_list, 'inverse_prior_group': inverse_prior_group_list}
    dispatch_path = join(RESULTS_PATH, "dispatch_f_sqrt")
    if not exists(dispatch_path): makedirs(dispatch_path)
    for pathway_dict in dictionaries_paths:
        name_pathway_file = pathway_dict.split('/')[-1].split('.')[0]
        print(f"Launching {pathway_dict}")
        for params in ParameterGrid(param_grid):
            print(f"Launching {params}")
            nb_repetitions = 5
            exp_name = f"{params['view']}__group_scm__" + f"{name_pathway_file}__" + f"{params['update']}__" + f"c{params['c']}__" + f"inverse_prior_group{params['inverse_prior_group']}__" + f"{nb_repetitions}"
            launch_slurm_experiment_group(return_views=params['view'], 
                                            nb_repetitions=nb_repetitions, 
                                            pathway_file=pathway_dict, 
                                            update_method=params['update'], 
                                            c=params['c'],
                                            inverse_prior_group=params['inverse_prior_group'],
                                            experiment_file='run_new_group_experiments.py', 
                                            experiment_name=exp_name, 
                                            time='4', 
                                            dispatch_path=dispatch_path)
    print("### DONE ###") 

       
if __name__ == '__main__':
    main_group()