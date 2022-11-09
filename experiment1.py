
print()
print('alpha-GAN Experiment 1: 2D Grid')
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

parser = argparse.ArgumentParser(description='alpha-GAN Experiment 1')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Dataset
parser.add_argument('--dataset', type=str, default='lattice', choices={'lattice', 'polar'}, help='dataset type')
parser.add_argument('--minor_prop', type=float, default=0, help='proportion of minority modes in dataset')
parser.add_argument('--major_weight', type=float, default=1, help='weighing scale for majority modes')
parser.add_argument('--train_size', type=int, default=50000, help='number of train examples')
parser.add_argument('--test_size', type=int, default=25000, help='number of test examples')
parser.add_argument('--batch_size', type=int, default=128, help='batch size used during training/testing')
parser.add_argument('--save_bursts', action='store_true', help='saves the plotted output bursts for each checkpoint')
parser.add_argument('--save_grads', action='store_true', help='saves the plotted grad distributions for each checkpoint')
parser.set_defaults(save_bursts=False, save_grads=False)

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
parser.set_defaults(non_saturating=False)

args = parser.parse_args()

# PATHS

setting = '-'.join(sys.argv[1:]).replace('---', '--').replace('--', '-')
setting_split = setting.split('seed-')
setting = setting_split[0] + '-'.join(setting_split[1].split('-')[1:])

unique_setting = setting + '-seed-' + str(args.seed) + '_time-' + str(time.time())

paths = { 'base' : 'experiment1/data/' + unique_setting + '/' }

if args.save_bursts:
    paths['bursts'] = paths['base'] + 'bursts/'

if args.save_grads:
    paths['2d_grads'] = paths['base'] + '2d_grads/'
    paths['mag_grads'] = paths['base'] + 'mag_grads/'
    paths['log_mag_grads'] = paths['base'] + 'log_mag_grads/'
    paths['dir_grads'] = paths['base'] + 'dir_grads/'

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

def make_lattice():

    means = [torch.Tensor([-4 + 2 * i + rand_shift(), -4 + 2 * j + rand_shift()]) for j in range(5) for i in range(5)]
    stds = [np.sqrt(0.0025)] * 25
    modes = list(zip(means, stds))

    mode_types = list(np.random.choice([0,1], 25, # 1 = majority, 0 = minority
                        p=[args.minor_prop, 1 - args.minor_prop]))
    probs = np.array([[1, args.major_weight][m] for m in mode_types])
    probs = list(probs / probs.sum())

    return modes, mode_types, probs

def make_polar():

    means = [torch.Tensor([np.cos(2 * np.pi * i / 8), np.sin(2 * np.pi * i / 8)]) for i in range(1,9)]
    stds = [np.sqrt(1e-4)] * 8
    modes = list(zip(means, stds))

    mode_types = list(np.random.choice([0,1], 8, # 1 = majority, 0 = minority
                        p=[args.minor_prop, 1 - args.minor_prop]))
    probs = np.array([[1, args.major_weight][m] for m in mode_types])
    probs = list(probs / probs.sum())

    return modes, mode_types, probs

def get_data(data_size, modes, probs):

    indices = np.random.choice(len(modes), data_size, p=probs)

    data = [modes[i][0] + modes[i][1] * torch.randn(2) for i in indices]

    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    return data_loader

make_dataset = make_lattice if args.dataset == 'lattice' else make_polar
modes, mode_types, probs = make_dataset()

train_real_loader = get_data(args.train_size, modes, probs)
test_real_loader = get_data(args.test_size, modes, probs)

train_noise_loader = make_noise(args.train_size)
test_noise_loader = make_noise(args.test_size)

# MODELS

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        layers = [nn.Linear(2,args.g_width), nn.ReLU()]

        # Hidden layers
        for i in range(args.g_layers - 1):
            layers += [nn.Linear(args.g_width,args.g_width), nn.ReLU()]

        layers += [nn.Linear(args.g_width, 2)]

        self.main = nn.Sequential(*layers)

        self.states = None

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

        layers += [nn.Linear(args.d_width, 1), nn.Sigmoid()]

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

d_criterion = nn.BCELoss() if args.d_alpha == 1 else AlphaLoss(args.d_alpha)
g_criterion = nn.BCELoss() if args.g_alpha == 1 else AlphaLoss(args.g_alpha)

# MAKE DATA

def make_burst(gan, epoch):

    x, y, z = gan.get_decisions()
    fake_output = gan.get_fake_output()
    real_output = gan.get_real_output()

    fig = plt.figure(figsize=(8,6))
    plt.contourf(x, y, z, cmap=plt.cm.Spectral, levels=20)
    plt.colorbar()
    plt.scatter(fake_output[:,0], fake_output[:,1], c=[(0,0,0)], s=0.2)

    interval = [-5, 5] if args.dataset == 'lattice' else [-2, 2]
    plt.xlim(interval)
    plt.ylim(interval)

    plt.title('Epoch ' + str(epoch))

    plt.savefig(paths['bursts'] + 'epoch-' + str(epoch) + '.png')
    plt.clf()
    plt.close(fig)

def make_grad(gan, epoch):

    grads = gan.all_grads.cpu().numpy()
    sample_grads = grads
    plt.scatter(sample_grads[:,0], sample_grads[:,1], s=1, alpha=0.7)
    plt.title('Epoch ' + str(epoch))
    plt.savefig(paths['2d_grads'] + 'epoch-' + str(epoch) + '.png')
    plt.clf()
    plt.close()

    mag_grads = np.sqrt(np.sum(sample_grads ** 2, axis=1))
    sns.displot(np.log(mag_grads))
    plt.title('Epoch ' + str(epoch))
    plt.savefig(paths['log_mag_grads'] + 'epoch-' + str(epoch) + '.png')
    plt.clf()
    plt.close()

    dir_grads = np.arctan(sample_grads[:,1] / sample_grads[:,0])
    sns.displot(dir_grads)
    plt.title('Epoch ' + str(epoch))
    plt.savefig(paths['dir_grads'] + 'epoch-' + str(epoch) + '.png')
    plt.clf()
    plt.close()

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

    def get_metrics(P_real):

        def kl_div(P, Q):
            return np.sum(P * np.log(P / Q))

        kl = kl_div(P_real, P_fake)
        rkl = kl_div(P_fake, P_real)

        P_mix = (P_real + P_fake) / 2
        jsd = (kl_div(P_real, P_mix) + kl_div(P_fake, P_mix)) / 2

        tvd = np.sum(np.abs(P_real - P_fake)) / 2

        return {'kl' : kl, 'rkl' : rkl, 'jsd' : jsd, 'tvd' : tvd}

    P_real = np.array(probs)
    P_uniform = np.ones(len(probs)) / len(probs)

    metrics = {}

    metrics['real'] = get_metrics(P_real)
    metrics['uniform'] = get_metrics(P_uniform)

    lost_modes = 0
    lost_major_modes = 0
    lost_minor_modes = 0
    hq_samples = 0

    def high_quality(s, m, std):
        return np.linalg.norm(s - m) <= 3*std

    # Compute lost modes and high-quality samples
    for mode_type, mean, std in zip(mode_types, means, stds):
        samples = means_dict[str(mean)]
        hqs_mode = np.sum([high_quality(sample, mean, std) for sample in samples]) if samples else 0
        lost_modes += int(hqs_mode < 1)
        lost_major_modes += mode_type * int(hqs_mode < 1)
        lost_minor_modes += (1 - mode_type) * int(hqs_mode < 1)
        hq_samples += hqs_mode

    metrics['modes'] = len(means) - lost_modes
    metrics['major_modes'] = sum(mode_types) - lost_major_modes
    metrics['minor_modes'] = (len(means) - sum(mode_types)) - lost_minor_modes
    metrics['hqs'] = hq_samples / output.shape[0]

    gan.metrics['epochs'].append(epoch)
    gan.metrics['modes'].append(metrics['modes'])
    gan.metrics['hqs'].append(metrics['hqs'])
    gan.metrics['last'] = metrics

def make_data(gan, epoch):

    compute_metrics(gan, epoch)

    if args.save_bursts:
        make_burst(gan, epoch)

    if args.save_grads:
        make_grad(gan, epoch)

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

R, U = M['real'], M['uniform']

row = ([setting, args.seed]
   + [M['modes'], M['major_modes'], M['minor_modes'], M['hqs']]
   + [R['kl'], R['rkl'], R['jsd'], R['tvd']]
   + [U['kl'], U['rkl'], U['jsd'], U['tvd']])

with open('experiment1/metrics.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(row)

with open(paths['base'] + 'metrics.csv', 'w') as f:
    f.write(str(gan_model.metrics))

print()
print('--------------------')
print('Metrics:', M)
