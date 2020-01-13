import os
from os.path import join, abspath, dirname, exists
from os import makedirs
from subprocess import call
import datetime
print(__file__)
RESULTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "saving_repository"))
EXPERIMENTS_PATH = os.environ.get('', join(dirname(abspath(__file__)), "experiments"))
DATAREPOSITORY_PATH = os.environ.get('', join(dirname(abspath(__file__)), "datasets/data_repository"))
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
    

def launch_slurm_experiment_group(return_views, nb_repetitions, pathway_file, update_method, 
                                  experiment_file, prior_initialization_type, experiment_name, time, dispatch_path):
    exp_file = join(dispatch_path, f"{return_views}__" + f"{experiment_name}__" + f"{pathway_file}__" + 
                    f"{update_method}__" + f"{prior_initialization_type}__" + f"{nb_repetitions}")
                        
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
    submission_script += f"python {EXPERIMENTS_PATH}/{experiment_file} -rt {return_views} -nb_r {nb_repetitions} 
                -g_dict {pathway_file} -u_m {update_method} -init {prior_initialization_type}"
    
    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    # call(["sbatch", submission_path])
    
def main_group():
    return_views = ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all']
    dictionaries_paths = [f"{DATAREPOSITORY_PATH}/pathways_biogrid_groups.pck", f"{DATAREPOSITORY_PATH}/pathways_biogrid_groups_unknown_features_are_groups.pck"]
    prior_init_and_update_method = [('default', 'pos_exp_group'), ('default', 'neg_exp_group'), 
                                    ('exp', 'pos_exp'), ('exp', 'neg_exp'), 
                                    ('normal', 'pos_exp'), ('normal', 'neg_exp')]
    dispatch_path = join(RESULTS_PATH, "dispatch")
    if not exists(dispatch_path): makedirs(dispatch_path)
    for pathway_dict in dictionaries_paths:
        for init_and_update_method in prior_init_and_update_method:
            for view in return_views:
                print(f"Launching {view}")
                launch_slurm_experiment_group(return_views=view, nb_repetitions=1, pathway_file=pathway_dict, 
                                            update_method=init_and_update_method[1], prior_initialization_type=init_and_update_method[0],
                                            experiment_file='run_new_group_experiments.py', experiment_name='group', time='2', 
                                            dispatch_path=dispatch_path)
    print("### DONE ###") 
    
if __name__ == '__main__':
    main_group()