import datetime
import os
import random
from os import makedirs
from os.path import abspath, dirname, exists, join
from subprocess import call

import numpy as np
from sklearn.model_selection import ParameterGrid
from experiments.utilities import *

RESULTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "saving_repository"))
EXPERIMENTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "experiments"))
DATAREPOSITORY_PATH = os.environ.get('', join(dirname(abspath(__file__)), "datasets/datasets_repository"))
PROJECT_ROOT = dirname(abspath(__file__))
SAVING_REPO = '/home/maoss2/project/maoss2/saving_repository_article/normal_experiments'

def launch_slurm_experiment(return_views, nb_repetitions, experiment_file, experiment_name, time, dispatch_path):
    exp_file = join(dispatch_path, f"{return_views}__" + f"{experiment_name}__" + f"{nb_repetitions}")
                        
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
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -rt {return_views} -nb_r {nb_repetitions} -o {SAVING_REPO}"

    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])

def main():
    # return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
    #             'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']
    return_views = ['methyl_rna_iso_mirna_snp_clinical']
    dispatch_path = join(RESULTS_PATH, "dispatch")
    if not exists(dispatch_path): makedirs(dispatch_path)
    for view in return_views:
        print(f"Launching {view}")
        launch_slurm_experiment(return_views=view, nb_repetitions=15, experiment_file='run_tn_experiments.py', 
                                experiment_name='normal_experiments', time='1', dispatch_path=dispatch_path)
    print("### DONE ###")   
    
    
if __name__ == '__main__':
    main()
