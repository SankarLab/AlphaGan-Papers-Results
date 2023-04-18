#!/bin/bash
module load mamba/latest
mamba create --name lsankar_env
source activate lsankar_env
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
mamba install matplotlib scipy scikit-learn tqdm python-lmdb -y
