#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128000M
#SBATCH --time=1:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#--mail-type=REQUEUE
# --mail-type=ALL
#date
#SECONDS=0
python PycharmProjects/BRCA_experiments_and_paper/datasets/brca_builder.py
