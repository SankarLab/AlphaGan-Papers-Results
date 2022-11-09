
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
os.mkdir('experiment1/outputs')
os.mkdir('experiment1/data')

with open('experiment1/metrics.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Setting', 'Seed',
        'Modes', 'Major-Modes', 'Minor-Modes', 'HQS',
        'Real-KL', 'Real-RKL', 'Real-JSD', 'Real-TVD',
        'Uniform-KL', 'Uniform-RKL', 'Uniform-JSD', 'Uniform-TVD'])

# String of args for single use
single_params = '--seed 1 --n_epochs 400 --dataset polar --g_layers 2 --d_layers 2 --save_bursts --save_grads --epoch_step 1'

# Grid of hyperparameters for sbatch
grid = {
    'seed' : list(range(151,201)),
    'dataset' : ['polar'],
    'non_saturating' : [True],
    'd_layers' : [4],
    'g_layers' : [4],
    'd_alpha' : [0.5,0.6,0.7,0.8,0.9,1],
    'g_alpha' : [1, 1.2],
    'epoch_step' : [400],
    #'save_bursts' : [True],
    #'save_grads' : [True],
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
