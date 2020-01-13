#!/usr/bin/env python3
import os
import sys
import json
import torch
import click
import traceback
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import sparse
from copy import deepcopy

def run_expt(model_name, dataset_name, output_path=None):
    model_funcs = dict(
        scvi=run_scvi,
        scgen=run_scgen,
        saucie=run_saucie
    )
    assert model_name in list(model_funcs.keys())
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
    subdir = os.path.join(output_path, f"{dataset_name}_{model_name}")
    os.makedirs(subdir, exist_ok=True)

    pre_scdata = load_datasets(dataset_name)
    post_adata = model_funcs[model_name](pre_scdata, n_epochs=100, batch_size=128,
                                         model_dir=subdir, lambda_b=0.1)
    pre_adata = scvi2ann_data(pre_scdata)

    evaluator = BatchcorrectionEvaluator()
    results = evaluator.pre_post_eval(pre_adata, post_adata, y_field='batch')
    with open(os.path.join(subdir, 'metrics.json'), 'w') as fd:
        json.dump(results, fd, indent=2)

    generate_plots(pre_adata, post_adata, subdir)


@click.command(help="Train models")
@click.option('--config_file', '-p', 
              help="Path to the config file (json) that contains the parameters for the experiment.")
@click.option('--output_path', '-o', default=None,
              help="Location for saving the training results (model artifacts and output files).")
def main(config_file, output_path):
    try:
        # Read config_file
        with open(config_file, 'r') as tc:
            train_params = json.load(tc)
            if output_path is not None:
                os.makedirs(output_path, exist_ok=True)

        print('Params for this experiement')
        print(train_params)
        print()

        print('Starting the training.')
        run_expt(**train_params, output_path=output_path)

    except Exception as e:
        trc = traceback.format_exc()
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' +
              str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    main()



