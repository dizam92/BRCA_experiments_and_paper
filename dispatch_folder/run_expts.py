#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import uuid
import time
import json
import hashlib
import subprocess
from pathlib import Path

import click
from sklearn.model_selection import ParameterGrid

from copy import deepcopy
print(__file__)
EXP_FOLDER = 'experiments'

def get_hash(task):
    task = deepcopy(task)
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def run_cmd(cmd):
    print(">>> Running the following command:")
    print(cmd)
    print(">>> Output")
    process = subprocess.Popen(cmd, bufsize=1, stdout=sys.stdout,
                               stderr=sys.stderr, shell=True)
    process.communicate()

sbatch_template = r"""#!/bin/bash
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={duration}
#SBATCH --account={account_id}
#SBATCH --array={start_idx}-{end_idx}

date
SECONDS=0
which python
python3 {dispatcher_file} run-sbatch  --exp_name {exp_name} --hpid $SLURM_ARRAY_TASK_ID -e {exec_file}
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
"""

@click.group()
def cli():
    pass


@cli.command(help="Run a dispatched experiment on sbatch cluster. The id of the experiment must be provided")
@click.option('-n', '--exp_name', type=str, default='test', help="Unique name for the experiment.")
@click.option('-p', '--hpid', type=int, default=0, help="""Position of the config file to run""")
@click.option('-e', '--exec_file', type=str, default='train', help=""" path to script that will be run. It is only used if instance_type is 'local'
                        and imagename is None. """)

def run_sbatch(exp_name, hpid, exec_file):
    exp_dir = os.path.join(Path.home(), EXP_FOLDER, exp_name)
    all_filenames_location = os.path.join(exp_dir, 'configs', 'configs.txt')
    with open(all_filenames_location, 'r') as fd:
        config_file = fd.readlines()[hpid].rstrip()

    print(f"{exec_file} -p {config_file} -o {exp_dir}")
    process = subprocess.Popen(f"{exec_file} -p {config_file} -o {exp_dir}", shell=True)
    process.communicate()
    if process.returncode != 0:
        exit()
    os.rename(config_file, config_file.replace('.json', '.done'))


def dispatch_sbatch(exp_name, config_file, exec_file, memory, duration, cpus, account_id):
    exp_dir = os.path.join(Path.home(), EXP_FOLDER, exp_name)

    config_dir = os.path.join(exp_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    with open(config_file, 'r') as fd:
        task_config = json.load(fd)

    task_grid = list(ParameterGrid(task_config))

    task_grid = {get_hash(task): task for task in task_grid}
    print(f"- Experiment has {len(task_grid)} different tasks:")

    existing_exp_files = [os.path.splitext(f) for f in os.listdir(os.path.join(exp_dir, 'configs'))]

    done_task_ids = [
        task_id for task_id, ext in existing_exp_files
        if (task_id in task_grid.keys() and ext == '.done')
    ]
    planned_task_ids = [
        task_id for task_id, ext in existing_exp_files if
        (task_id in task_grid.keys() and ext == '.json')
    ]
    new_task_ids = [
        task_id for task_id in task_grid
        if task_id not in done_task_ids + planned_task_ids
    ]

    if new_task_ids:
        print(f'\nNew:', *new_task_ids, sep='\n')
    if planned_task_ids:
        print('\nPlanned:', *planned_task_ids, sep='\n')
    if done_task_ids:
        print('\nCompleted:', *done_task_ids, sep='\n')

    print(f"\n\t*New: {len(new_task_ids)}\n"
          f"\t*Planned: {len(planned_task_ids)}\n"
          f"\t*Completed: {len(done_task_ids)}\n")

    planned_as_well = len(planned_task_ids) == 0 \
                      or input('>> Relaunch already planned tasks ? [N/y]').lower() in {'y', 'yes'}

    tasks = new_task_ids + planned_task_ids if planned_as_well else new_task_ids

    # Uploading on the exp folder
    all_filenames = []
    for task_id in tasks:
        fname = os.path.join(config_dir, f"{task_id}.json")
        with open(fname, 'w') as f:
            json.dump(task_grid[task_id], f)
        all_filenames.append(fname)
    all_filenames_location = os.path.join(config_dir, 'configs.txt')
    if os.path.exists(all_filenames_location):
        with open(all_filenames_location, 'r') as fd:
            start_idx = len(fd.readlines())
    else:
        start_idx = 0
    with open(all_filenames_location, 'a') as fd:
        fd.writelines([el + '\n' for el in all_filenames])
    end_idx = start_idx + len(all_filenames)

    dispatcher_file = os.path.abspath(__file__)
    template_args = dict(start_idx=start_idx, end_idx=end_idx - 1,
                         exp_name=exp_name, exec_file=exec_file,
                         duration=duration, cpus=cpus, memory=memory,
                         dispatcher_file=dispatcher_file, account_id=account_id)
    sbatch_script = sbatch_template.format(**template_args)

    sbatch_script_location = os.path.join(exp_dir, 'submit.sh')
    with open(sbatch_script_location, 'w') as fd:
        fd.write(sbatch_script)

    print(sbatch_script)
    os.chdir(exp_dir)
    process = subprocess.Popen(f"sbatch {sbatch_script_location} -D {exp_dir}", shell=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)


def dispatch_local(config_file, exec_file):

    with open(config_file, 'r') as fd:
        config = json.load(fd)

    hp_grid = list(ParameterGrid(config))

    print("Quick summary")
    print(f"Number of tasks: {len(hp_grid)}")

    for _, hp in enumerate(hp_grid):
        config_dir = os.path.dirname(os.path.abspath(
            os.path.expandvars(config_file)))
        local_cfg_file = os.path.join(config_dir, f"temp_{str(uuid.uuid4())}.json")
        try:
            with open(local_cfg_file, 'w') as fp:
                json.dump(hp, fp)
            print(f"Executing ... \n>>> python {exec_file} -p {local_cfg_file}")
            process = subprocess.Popen(f"python {exec_file} -p {local_cfg_file}", shell=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                exit(44)
        except:
            pass
        finally:
            os.remove(local_cfg_file)


@cli.command(help="Run a dispatched experiment on sbatch cluster. The id of the experiment must be provided")
@click.option('--exp-name', type=str, default='test', help="Unique name for the experiment.")
@click.option('--server', type=str, default='local',
              help=""" server where the experiments is running: local | graham. """)
@click.option('--config-file', type=str, default=None,
              help="""The name/path of the config file (in json format) that contains all the parameters for 
                    the experiment. This config file should be at the same location as the train file""")
@click.option('--duration', type=str, default='12:00:00',
              help="""Duration of each task in the experiment. Will only be used on clusters with schedulers""")
@click.option('--cpus', type=int, default=16,
              help="""Number of cpus per task""")
@click.option('--memory', type=str, default="32G",
              help="""Number of cpus per task""")
@click.option('--account_id', type=str, default="rrg-corbeilj-ac",
              help="""Number of cpus per task""")

def dispatch(exp_name, server, config_file, duration, cpus, memory, account_id):
    expt_folder = os.path.dirname(os.path.abspath(__file__))

    if config_file is None:
        config_file = os.path.join(expt_folder, "config.json")

    if not os.path.exists(config_file):
        raise Exception("We were expecting an existing config file or a config.json in "
                        "the experiment folder but none were given")

    if server == "local":
        dispatch_local(
            config_file=config_file,
            exec_file=f"{expt_folder}/trainer.py") # change that
    elif server == "graham":
        dispatch_sbatch(
            exp_name=exp_name,
            config_file=config_file,
            exec_file=f"{expt_folder}/run_tn_experiments.py", 
            memory=memory, duration=duration, cpus=cpus, account_id=account_id)
    else:
        raise Exception("unexpected server type")


if __name__ == '__main__':
    cli()

