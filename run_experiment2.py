
from itertools import product
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Run (aD, aG)-GAN Experiment #2: Stacked MNIST')
parser.add_argument('--sbatch', action='store_true', help='run parameter grid on sbatch')
parser.add_argument('--no_reset', action='store_true', help='don\'t clear the results directory')
parser.set_defaults(sbatch=False, no_reset=False)
args = parser.parse_args()

if not args.no_reset:

    # Prepare results directory
    if os.path.isdir('experiment'):
        os.system('rm -rf experiment')

    os.mkdir('experiment')
    os.mkdir('experiment/outputs')
    os.mkdir('experiment/data')

    row = ['Setting', 'Seed', 'FID Score']
    for conf in [i/10 for i in range(10)]:
        for mode in [1,3,5,10]:
            row.append(str(conf) + '-' + str(mode))
    row += ['Epochs', 'Modes', 'FIDs']

    with open('experiment/metrics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# String of args for single use
single_params = '--seed 1 --non_saturating --save_bursts --n_epochs 3 --epoch_step 1'

# Grid of hyperparameters for sbatch
grid = {
    'seed' : list(range(1,101)),
    'non_saturating' : [True],
    'n_epochs' : [200],
    'epoch_step' : [10],
    'g_lr' : [5e-4],
    'g_alpha' : [1],
    'ls_gan' : [True]
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
        os.system('sbatch run_experiment2.sh \'' + params + '\'')

else:

    print('Interactive mode.')
    os.system('python3 experiment2.py ' + single_params)
