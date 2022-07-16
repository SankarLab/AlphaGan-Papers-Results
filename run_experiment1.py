
from itertools import product
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Run alpha-GAN experiment')
parser.add_argument('--sbatch', action='store_true', help='run parameter grid on sbatch')
parser.set_defaults(sbatch=False)
args = parser.parse_args()

# Prepare results directory
if os.path.isdir('experiment1'):
    os.system('rm -rf experiment1')

os.mkdir('experiment1')
os.mkdir('experiment1/bursts')
os.mkdir('experiment1/figures')
os.mkdir('experiment1/outputs')

with open('experiment1/metrics.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Setting', 'Seed', 'Modes', 'High-Quality Samples', 'Reverse KL'])

# String of args for single use
single_params = '--save_bursts --alpha 1.0 --epoch_step 2'

# Grid of hyperparameters for sbatch
grid = {
    'seed' : [1, 2, 3, 4, 5],
    'alpha' : [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0],
    'd_layers' : [1, 2, 3, 4]
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
        os.system('sbatch run_experiment1.sh \'' + params + '\'')

else:

    print('Interactive mode.')
    os.system('python3 experiment1.py ' + single_params)
