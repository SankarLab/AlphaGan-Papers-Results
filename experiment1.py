
print()
print('(aD, aG)-GAN Experiment #1: 2D Ring')
print('Author: Kyle Otstot')
print('-------------------------------')
print()

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import os
import time
import csv
from gan import *

# PARAMETERS

parser = argparse.ArgumentParser(description='(aD, aG)-GAN Experiment #1')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Dataset
parser.add_argument('--dataset', type=str, default='2Dring', choices={'2Dgrid', '2Dring'}, help='dataset type')
parser.add_argument('--train_size', type=int, default=50000, help='number of train examples')
parser.add_argument('--test_size', type=int, default=25000, help='number of test examples')
parser.add_argument('--batch_size', type=int, default=128, help='batch size used during training/testing')
parser.add_argument('--save_bursts', action='store_true', help='saves the plotted output bursts for each checkpoint')
parser.set_defaults(save_bursts=False)

# Network settings
parser.add_argument('--d_layers', type=int, default=2, help='number of hidden layers in discriminator')
parser.add_argument('--g_layers', type=int, default=2, help='number of hidden layers in generator')
parser.add_argument('--d_width', type=int, default=200, help='hidden layer width in discriminator')
parser.add_argument('--g_width', type=int, default=400, help='hidden layer width in generator')

# Training
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--epoch_step', type=int, default=401, help='number of epochs between validation checkpoints')
parser.add_argument('--d_lr', type=float, default=1e-4, help='learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for generator')

# Loss function
parser.add_argument('--d_alpha', type=float, default=1.0, help='alpha parameter for discriminator')
parser.add_argument('--g_alpha', type=float, default=1.0, help='alpha parameter for generator')
parser.add_argument('--non_saturating', action='store_true', help='uses non saturating loss function')
parser.add_argument('--ls_gan', action='store_true', help='uses LS GAN loss functions')
parser.set_defaults(non_saturating=False, ls_gan=False)

args = parser.parse_args()

# PATHS

setting = '-'.join(sys.argv[1:]).replace('---', '--').replace('--', '-')

if 'seed-' not in setting:
    print('Please specify the seed number as an argument in the command line.')
    exit()

setting_split = setting.split('seed-')
setting = setting_split[0] + '-'.join(setting_split[1].split('-')[1:])

unique_setting = setting + '-seed-' + str(args.seed) + '_time-' + str(time.time())

paths = { 'base' : 'experiment/data/' + unique_setting + '/' }

if args.save_bursts:
    paths['bursts'] = paths['base'] + 'bursts/'

for path in paths.values():
    os.mkdir(path)

print('Setting:')
print(unique_setting)

# DATASET

# Fixed seed for dataset generation
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

def make_noise(data_size):

    data_tensor = torch.randn(data_size, 2)
    data = [data_tensor[i,:] for i in range(data_size)]
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return data_loader

def make_grid():

    means = [torch.Tensor([-4 + 2 * i, -4 + 2 * j]) for j in range(5) for i in range(5)]
    stds = [np.sqrt(0.0025)] * 25
    modes = list(zip(means, stds))
    probs = [1 / len(modes)] * len(modes)

    return modes, probs

def make_ring():

    means = [torch.Tensor([np.cos(2 * np.pi * i / 8), np.sin(2 * np.pi * i / 8)]) for i in range(1,9)]
    stds = [np.sqrt(1e-4)] * 8
    modes = list(zip(means, stds))
    probs = [1 / len(modes)] * len(modes)

    return modes, probs

def get_data(data_size, modes, probs):

    indices = np.random.choice(len(modes), data_size, p=probs)
    data = [modes[i][0] + modes[i][1] * torch.randn(2) for i in indices]
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    return data_loader

make_dataset = make_grid if args.dataset == '2Dgrid' else make_ring
modes, probs = make_dataset()

train_real_loader = get_data(args.train_size, modes, probs)
test_real_loader = get_data(args.test_size, modes, probs)

train_noise_loader = make_noise(args.train_size)
test_noise_loader = make_noise(args.test_size)

# MODELS

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        layers = [nn.Linear(2,args.g_width), nn.ReLU()]

        # Hidden layers
        for i in range(args.g_layers - 1):
            layers += [nn.Linear(args.g_width,args.g_width), nn.ReLU()]

        layers += [nn.Linear(args.g_width, 2)]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [nn.Linear(2, args.d_width), nn.LeakyReLU(0.2)]

        # Hidden layers
        for i in range(args.d_layers - 1):
            layers += [nn.Linear(args.d_width, args.d_width), nn.LeakyReLU(0.2)]

        layers += [nn.Linear(args.d_width, 1)]

        if not args.ls_gan:
            layers += [nn.Sigmoid()]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out

# Define instances
generator = Generator()
discriminator = Discriminator()

# OPTIMIZERS

g_optimizer = optim.Adam(generator.parameters(), lr=args.g_lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr)

# LOSS FUNCTION

class AlphaLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, label, ep=1e-7):
        output = torch.clamp(output, min=ep, max=1 - ep)
        A = (self.alpha / (self.alpha - 1))
        real_term = A * (1 - label * (output ** (1/A)))
        fake_term = -A * ((1 - label) * (1 - output) ** (1/A))
        loss = torch.mean(real_term + fake_term)
        return loss

class LSLoss(nn.Module):

    def __init__(self):
        super(LSLoss, self).__init__()

    def forward(self, output, labels):
        return 0.5 * torch.mean((output - labels) ** 2)

if args.ls_gan:
    d_criterion = LSLoss()
    g_criterion = LSLoss()
else:
    d_criterion = nn.BCELoss() if args.d_alpha == 1 else AlphaLoss(args.d_alpha)
    g_criterion = nn.BCELoss() if args.g_alpha == 1 else AlphaLoss(args.g_alpha)

# MAKE DATA

def make_burst(gan, epoch):

    if epoch < args.n_epochs:
        fig = plt.figure(figsize=(8,6))
        x, y, z = gan.get_decisions()
        plt.contourf(x, y, z, cmap=plt.cm.Spectral, levels=20)
        plt.colorbar()
    else:
        fig = plt.figure(figsize=(6,6))

    fake_output = gan.get_fake_output()
    plt.scatter(fake_output[:,0], fake_output[:,1], c=[(0,0,0)], s=0.2)

    interval = [-5, 5] if args.dataset == '2Dgrid' else [-2, 2]
    plt.xlim(interval)
    plt.ylim(interval)

    plt.title('Epoch ' + str(epoch))

    plt.savefig(paths['bursts'] + 'epoch-' + str(epoch) + '.png')
    plt.clf()
    plt.close(fig)

# METRICS

def compute_metrics(gan, epoch):

    output = gan.get_fake_output()
    means, stds = [m.numpy() for m, _ in modes], [s for _, s in modes]
    means_dict = { str(mean) : [] for mean in means }

    # Assign output samples to closest mode
    for i in range(output.shape[0]):

        sample = output[i,:]
        dists = [(mean, np.linalg.norm(sample - mean)) for mean in means]
        means_dict[str(min(dists, key=lambda x: x[1])[0])].append(sample)

    # KL & Reverse KL & JSD

    P_fake = (np.array([len(means_dict[str(mean)]) for mean in means]) + 1) / (output.shape[0] + len(means))
    P_real = np.array(probs)

    def kl_div(P, Q):
        return np.sum(P * np.log(P / Q))

    kl = kl_div(P_real, P_fake)
    rkl = kl_div(P_fake, P_real)

    P_mix = (P_real + P_fake) / 2
    jsd = (kl_div(P_real, P_mix) + kl_div(P_fake, P_mix)) / 2

    tvd = np.sum(np.abs(P_real - P_fake)) / 2

    metrics = {'kl' : kl, 'rkl' : rkl, 'jsd' : jsd, 'tvd' : tvd}

    lost_modes = 0
    hq_samples = 0

    def high_quality(s, m, std):
        return np.linalg.norm(s - m) <= 3*std

    # Compute lost modes and high-quality samples
    for mean, std in zip(means, stds):
        samples = means_dict[str(mean)]
        hqs_mode = np.sum([high_quality(sample, mean, std) for sample in samples]) if samples else 0
        lost_modes += int(hqs_mode < 1)
        hq_samples += hqs_mode

    metrics['modes'] = len(means) - lost_modes
    metrics['hqs'] = hq_samples / output.shape[0]

    if 'epochs' not in gan.metrics:
        gan.metrics['epochs'] = []

    if 'modes' not in gan.metrics:
        gan.metrics['modes'] = []

    gan.metrics['epochs'].append(epoch)
    gan.metrics['modes'].append(metrics['modes'])
    gan.metrics['last'] = metrics

def make_data(gan, epoch):

    compute_metrics(gan, epoch)

    if args.save_bursts:
        make_burst(gan, epoch)

# GAN TRAINING

gan_model = GAN(
                data_loaders = ((train_noise_loader, train_real_loader), (test_noise_loader, test_real_loader)),
                models = (discriminator, generator),
                optimizers = (d_optimizer, g_optimizer),
                criteria = (d_criterion, g_criterion),
                dataset_name = args.dataset
                )

gan_model.train(n_epochs=args.n_epochs, epoch_step=args.epoch_step,
                                            seed=args.seed,
                                            flip=args.non_saturating,
                                            make_data=make_data
                                            )

M = gan_model.metrics['last']

row = [setting, args.seed, M['modes'], M['hqs'], M['kl'], M['rkl'], M['jsd'], M['tvd']]

with open('experiment/metrics.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(row)

with open(paths['base'] + 'metrics.csv', 'w') as f:
    f.write(str(gan_model.metrics))

print()
print('--------------------')
print('Metrics:', M)
