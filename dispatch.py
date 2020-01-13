import os
from os.path import join, abspath, dirname, exists
from os import makedirs
from subprocess import call

print(__file__)
RESULTS_PATH = os.environ.get('saving_repository', join(dirname(abspath(__file__)), "saving_repository"))
EXPERIMENTS_PATH = os.environ.get('experiments', join(dirname(abspath(__file__)), "experiments"))
PROJECT_ROOT = dirname(abspath(__file__))

def launch_slurm_experiment(return_views, nb_repetitions, experiment_file, experiment_name, time, dispatch_path):
    exp_file = join(dispatch_path, f"{return_views}" + f"{experiment_name}" + f"{nb_repetitions}")
                        
    submission_script = ""
    submission_script += f"#!/bin/bash\n"
    submission_script += f"#SBATCH --nodes=1\n" 
    submission_script += f"#SBATCH --ntasks-per-node=32\n" 
    submission_script += f"#SBATCH --mem=128000M\n" 
    submission_script += f"#SBATCH --account=rpp-corbeilj\n"
    submission_script += f"#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca\n"
    submission_script += f"#SBATCH --mail-type=BEGIN\n"
    submission_script += f"#SBATCH --mail-type=END\n"
    submission_script += f"#SBATCH --mail-type=FAIL\n"
    submission_script += f"#SBATCH --time={time}:00:00\n" 
    submission_script += f"#SBATCH --output={exp_file + '.out'}\n\n" 
    submission_script += f"date\n" 
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -rt {return_views} -nb_r {nb_repetitions}"

    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    # call(["sbatch", submission_path])

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
    
# def main():
#     datasets = ["breast", "ads", "adult", "mnist17", "mnist49", "mnist56"]
#     experiments = ["greedy_kernel"]
#     landmarks_method = ["random"]
#     n_cpu = 40
#     time = 6
    
#     dispatch_path = join(RESULTS_PATH, "dispatch")
#     if not exists(dispatch_path): makedirs(dispatch_path)
    
#     for dataset in datasets:
#         print(f"Launching {dataset}")
#         launch_slurm_experiment(dataset, experiments, landmarks_method, n_cpu, time, dispatch_path)
    
#     print("### DONE ###")

if __name__ == '__main__':
    main()