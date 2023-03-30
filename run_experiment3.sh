#!/bin/bash

#SBATCH -N 1
#SBATCH -p htc
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 00-4:00
#SBATCH -o experiment/outputs/%j.out
#SBATCH -e experiment/outputs/%j.err

module load mamba/latest
source activate deep_learning
/home/kotstot/.conda/envs/deep_learning/bin/python3 experiment3.py $1
source deactivate
