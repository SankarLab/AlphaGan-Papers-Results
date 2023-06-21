
from itertools import product
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Run (aD, aG)-GAN Experiments')
parser.add_argument('--experiment', type=str, default=None,
    choices={'2D', 'stacked_mnist', 'cfiar10', 'celeba', 'lsun_classroom'}, help='experiment to run')
parser.add_argument('--sbatch', action='store_true', help='run parameter grid on sbatch')
parser.add_argument('--no_reset', action='store_true', help='don\'t clear the results directory')
parser.set_defaults(sbatch=False, no_reset=False)
args = parser.parse_args()

if not args.no_reset:

    # Prepare results directory
    if os.path.isdir('results'):
        os.system('rm -rf results')

    os.mkdir('results')
    os.mkdir('results/outputs')
    os.mkdir('results/errors')
    os.mkdir('results/data')

# String of args for single use
single_params = '--seed 1 --lr 1e-4 --d_alpha 1 --g_alpha 1 --non_saturating --save_images --n_epochs 5 --epoch_step 5'

# Grid of hyperparameters for sbatch
"""
grid = {
    'seed' : list(range(1, 51)),
    'non_saturating' : [True],
    'n_epochs' : [150],
    'epoch_step' : [10],
    'lr' : [1e-4, 2e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3],
    'd_alpha' : [1],
    'g_alpha' : [1],
    'ls_gan' : [True],
    'amp' : [True]
}
"""

grid = {
    'seed' : list(range(1, 41)),
    'non_saturating' : [True],
    'n_epochs' : [150],
    'epoch_step' : [10],
    'lr' : [1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
    'd_alpha' : [0.6],
    'g_alpha' : [1.05, 1.1, 1.2],
}

# Utility function
def make_sbatch_params(grid):

    trials = [ { p : t for p, t in zip(grid.keys(), trial) }
                    for trial in list(product(*grid.values())) ]

    def trial_to_args(trial):
        arg_list = ['--' + param + ' ' + str(val) if type(val) != type(True)
                else '--' + param if val else '' for param, val in trial.items()]
        return ' '.join(arg_list)

    sbatch_params = [trial_to_args(trial) for trial in trials]

    return sbatch_params

sbatch_params = make_sbatch_params(grid)

if args.sbatch:

    print(len(sbatch_params), 'jobs will be submitted.')

    for params in sbatch_params:
        os.system('sbatch run_experiment.sh \'' + args.experiment + '.py ' + params + '\'')

else:

    print('Interactive mode.')
    os.system('python3 experiments/' + args.experiment + '.py ' + single_params)
