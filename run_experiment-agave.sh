#!/bin/bash

#SBATCH -N 1
#SBATCH -p htcgpu
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -q normal
#SBATCH -t 00-4:00
#SBATCH -o results/outputs/%j.out
#SBATCH -e results/error/%j.err

module load anaconda/py3
source activate lsankar_env
python3 experiments/$1
conda deactivate
