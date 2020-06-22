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

def launch_slurm_experiment(experiment_file, saving_file, subsampling, nb_repetitions, time, dispatch_path):
    exp_file = join(dispatch_path, f"{saving_file}__" + f"{subsampling}__" + f"{nb_repetitions}")         
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
    submission_script += f"python  {EXPERIMENTS_PATH}/{experiment_file} -subs {subsampling} -nb_r {nb_repetitions}"

    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])


def main():
    subsampling_state = [False, True]
    dispatch_path = join(RESULTS_PATH, "dispatch_baptiste_exp")
    if not exists(dispatch_path): makedirs(dispatch_path)
    for subs in subsampling_state:
        print(f"Launching Subsampling {subs}")
        launch_slurm_experiment(experiment_file='run_multiview_data_experiments.py', 
                                saving_file='experiments_scm_removed', 
                                subsampling=subs, 
                                nb_repetitions=15, 
                                time='5',
                                dispatch_path=dispatch_path)
    print("### DONE ###")   
      
if __name__ == '__main__':
    main()
