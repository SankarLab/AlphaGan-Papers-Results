
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
parser.add_argument('--train_size', type=int, default=10000, help='number of train examples')
parser.add_argument('--test_size', type=int, default=2000, help='number of test examples')
parser.add_argument('--batch_size', type=int, default=100, help='batch size used during training/testing')
parser.add_argument('--save_bursts', action='store_true', help='saves the plotted output bursts for each checkpoint')
parser.set_defaults(save_bursts=False)

# Network settings
parser.add_argument('--d_layers', type=int, default=3, help='number of hidden layers in discriminator')
parser.add_argument('--g_layers', type=int, default=3, help='number of hidden layers in generator')
parser.add_argument('--d_width', type=int, default=200, help='hidden layer width in discriminator')
parser.add_argument('--g_width', type=int, default=400, help='hidden layer width in generator')

# Training
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--epoch_step', type=int, default=5, help='number of epochs between validation checkpoints')
parser.add_argument('--d_lr', type=float, default=1e-4, help='learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for generator')

# Loss function
parser.add_argument('--alpha', type=float, default=1.0, help='alpha parameter for alpha loss')

args = parser.parse_args()

# PATHS

setting = 'alpha-' + str(args.alpha) + '_'
setting += 'dlayers-' + str(args.d_layers) + '_'
setting += 'glayers-' + str(args.g_layers) + '_'
setting += 'dwidth-' + str(args.d_width) + '_'
setting += 'gwidth-' + str(args.g_width) + '_'
setting += 'dlr-' + str(args.d_lr) + '_'
setting += 'glr-' + str(args.g_lr) + '_'

unique_setting = setting + 'seed-' + str(args.seed) + '_time-' + str(time.time())

burst_path = 'experiment1/bursts/' + unique_setting + '/'

os.mkdir(burst_path)

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

def make_2Dgrid(data_size):

    data = []

    gauss_size = data_size // 25

    std = np.sqrt(0.0025)

    for i in range(5):
        for j in range(5):
            mean = torch.Tensor([-4 + 2 * i, -4 + 2 * j]).reshape(1,2)
            new_data = mean + std * torch.randn(gauss_size, 2)
            data += [new_data[k,:] for k in range(gauss_size)]

    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    return data_loader

train_real_loader = make_2Dgrid(args.train_size)
test_real_loader = make_2Dgrid(args.test_size)

train_noise_loader = make_noise(args.train_size)
test_noise_loader = make_noise(args.test_size)

# MODELS

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        layers = [nn.Linear(2,args.g_width), nn.BatchNorm1d(args.g_width), nn.ReLU()]

        # Hidden layers
        for i in range(args.g_layers):
            layers += [nn.Linear(args.g_width, args.g_width), nn.BatchNorm1d(args.g_width), nn.ReLU()]

        layers += [nn.Linear(args.g_width, 2)]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out

# Combination of linear layer and maxout activation
class LinearMaxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [LinearMaxout(2, args.d_width, 5)]

        # Hidden layers
        for i in range(args.d_layers):
            layers += [LinearMaxout(args.d_width, args.d_width, 5)]

        layers += [nn.Linear(args.d_width, 1), nn.Sigmoid()]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out

# Define instances
generator = Generator()
discriminator = Discriminator()

# OPTIMIZERS

g_optimizer = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.8,0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.8,0.999))

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

criterion = nn.BCELoss() if args.alpha == 1 else AlphaLoss(args.alpha)

# MAKE FIGURE

def make_burst(gan, epoch):

    fig_path = burst_path + 'epoch-' + str(epoch) + '.png'
    save_figure(gan, fig_path, epoch=epoch)

def save_figure(gan, fig_path, epoch=None):

    output = gan.get_output()

    fig = plt.figure(figsize=(6,6))
    plt.scatter(output[:,0], output[:,1], c=[(0,0,0)], s=0.8)

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    if epoch is not None:
        plt.title('Epoch ' + str(epoch))

    plt.savefig(fig_path)
    plt.close(fig)

# GAN TRAINING

gan_model = GAN(
                data_loaders = ((train_noise_loader, train_real_loader), (test_noise_loader, test_real_loader)),
                models = (discriminator, generator),
                optimizers = (d_optimizer, g_optimizer),
                criterion = criterion
                )

gan_model.train(n_epochs=args.n_epochs, epoch_step=args.epoch_step,
                                            make_burst=make_burst if args.save_bursts else None)

save_figure(gan_model, 'experiment1/figures/' + unique_setting + '.png')

# METRICS

def high_quality(s, m, std=np.sqrt(0.0025)):
    is_hq = (m[0] - 3*std <= s[0] <= m[0] + 3*std) and (m[1] - 3*std <= s[1] <= m[1] + 3*std)
    return is_hq

def compute_metrics(output):

    # Create 25 grid modes
    modes = []
    for x in [-4, -2, 0, 2, 4]:
        for y in [-4, -2, 0, 2, 4]:
            modes.append(np.array([x,y]))
    modes_dict = { str(mode) : [] for mode in modes }

    # Assign output samples to closest mode
    for i in range(output.shape[0]):

        sample = output[i,:]
        dists = [(mode, np.linalg.norm(sample - mode)) for mode in modes]
        modes_dict[str(min(dists, key=lambda x: x[1])[0])].append(sample)

    # Compute reverse KL
    P_real = np.ones(len(modes)) / len(modes) + 1e-10
    P_fake = np.array([len(modes_dict[str(mode)]) for mode in modes]) / output.shape[0] + 1e-10
    reverse_kl = np.sum(P_fake * np.log(P_fake / P_real))

    lost_modes = 0
    hq_samples = 0

    # Compute lost modes and high-quality samples
    for mode in modes:
        samples = modes_dict[str(mode)]
        hqs_mode = np.sum([high_quality(sample, mode) for sample in samples]) if samples else 0
        lost_modes += int(hqs_mode == 0)
        hq_samples += hqs_mode

    recovered_modes = len(modes) - lost_modes
    hq_samples /= output.shape[0]

    return recovered_modes, hq_samples, reverse_kl

modes, hqs, rkl = compute_metrics(gan_model.get_output())

with open('experiment1/metrics.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([setting, str(args.seed), str(modes), str(hqs), str(rkl)])

print()
print('--------------------')
print('Modes:', modes)
print('High-Quality Samples:', hqs)
print('Reverse KL:', rkl)
