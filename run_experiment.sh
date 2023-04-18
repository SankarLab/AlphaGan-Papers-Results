#!/bin/bash

#SBATCH -N 1
#SBATCH -p htc
#SBATCH -q public
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 00-4:00
#SBATCH -o results/outputs/%j.out
#SBATCH -e results/errors/%j.err

module load mamba/latest
source activate deep_learning
/home/kotstot/.conda/envs/deep_learning/bin/python3 experiments/$1
source deactivate
