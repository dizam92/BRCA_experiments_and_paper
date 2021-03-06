import datetime
import os
import random
from os import makedirs
from os.path import abspath, dirname, exists, join
from subprocess import call

import numpy as np
from sklearn.model_selection import ParameterGrid

print(__file__)
RESULTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "saving_repository"))
EXPERIMENTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "experiments"))
DATAREPOSITORY_PATH = os.environ.get('', join(dirname(abspath(__file__)), "datasets/datasets_repository"))
PROJECT_ROOT = dirname(abspath(__file__))

def launch_slurm_experiment_group(subsampling, nb_repetitions, pathway_file, update_method, c, inverse_prior_group,
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
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -subs {subsampling} -nb_r {nb_repetitions} -g_dict {pathway_file} -u_m {update_method} -c {c} -inverse_prior_group {inverse_prior_group} -exp_name {experiment_name}" 
    
    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])
    
def main_group():
    dictionaries_paths = [f"{DATAREPOSITORY_PATH}/pathways_multiview_groups.pck"]
    update_method = ['inner_group', 'outer_group']
    subsampling_state = [False, True]
    random.seed(42)
    c_list = np.round(np.linspace(0.1, 1, 10), 3)
    inverse_prior_group_list = [False, True]
    param_grid = {'subsampling': subsampling_state, 'update': update_method, 'c': c_list, 'inverse_prior_group': inverse_prior_group_list}
    dispatch_path = join(RESULTS_PATH, "dispatch_baptiste_group_exp")
    if not exists(dispatch_path): makedirs(dispatch_path)
    for pathway_dict in dictionaries_paths:
        name_pathway_file = pathway_dict.split('/')[-1].split('.')[0]
        print(f"Launching {pathway_dict}")
        for params in ParameterGrid(param_grid):
            print(f"Launching {params}")
            nb_repetitions = 5
            exp_name = f"{params['subsampling']}__group_scm__" + f"{name_pathway_file}__" + f"{params['update']}__" + f"c{params['c']}__" + f"inverse_prior_group{params['inverse_prior_group']}__" + f"{nb_repetitions}"
            launch_slurm_experiment_group(subsampling=params['subsampling'], 
                                            nb_repetitions=nb_repetitions, 
                                            pathway_file=pathway_dict, 
                                            update_method=params['update'], 
                                            c=params['c'],
                                            inverse_prior_group=params['inverse_prior_group'],
                                            experiment_file='run_groups_scm_multiview_data_experiments.py', 
                                            experiment_name=exp_name, 
                                            time='5', 
                                            dispatch_path=dispatch_path)
    print("### DONE ###") 

       
if __name__ == '__main__':
    main_group()
