
print()
print('(aD, aG)-GAN Experiment: 2D Datasets')
print('Author: Kyle Otstot')
print('-------------------------------')
print()

import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import random
import os

sys.path.append('..')
from alphaGAN.gan import *

# PARAMETERS

parser = argparse.ArgumentParser(description='(aD, aG)-GAN Experiment: 2D Datasets')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Dataset
parser.add_argument('--dataset', type=str, default='2Dring', choices={'2Dgrid', '2Dring'}, help='dataset type')
parser.add_argument('--train_size', type=int, default=50000, help='number of train examples')
parser.add_argument('--test_size', type=int, default=25000, help='number of test examples')
parser.add_argument('--batch_size', type=int, default=128, help='batch size used during training/testing')
parser.add_argument('--save_images', action='store_true', help='saves the plotted output bursts for each checkpoint')
parser.set_defaults(save_images=False)

# Network settings
parser.add_argument('--d_layers', type=int, default=4, help='number of hidden layers in discriminator')
parser.add_argument('--g_layers', type=int, default=4, help='number of hidden layers in generator')
parser.add_argument('--d_width', type=int, default=200, help='hidden layer width in discriminator')
parser.add_argument('--g_width', type=int, default=400, help='hidden layer width in generator')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter for adam optimization')
parser.add_argument('--amp', action='store_true', help='uses automatic mixed precision training')
parser.set_defaults(amp=False)

# Training
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--epoch_step', type=int, default=401, help='number of epochs between validation checkpoints')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for discriminator & generator')

# Loss function
parser.add_argument('--d_alpha', type=float, default=1.0, help='alpha parameter for discriminator')
parser.add_argument('--g_alpha', type=float, default=1.0, help='alpha parameter for generator')
parser.add_argument('--non_saturating', action='store_true', help='uses non saturating loss function')
parser.add_argument('--ls_gan', action='store_true', help='uses LS GAN loss functions')
parser.set_defaults(non_saturating=False, ls_gan=False)

args = parser.parse_args()

# PATHS

setting, paths = create_setting(args, sys.argv[1:])

# DATASET

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

def make_noise(data_size):

    data_tensor = torch.randn(data_size, 2)
    data = [data_tensor[i,:] for i in range(data_size)]
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
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
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

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

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out

# Define instances
generator = Generator()
discriminator = Discriminator()

# EVALUATION FUNCTIONS

def visualize_data(gan, epoch):

    if epoch < args.n_epochs:
        fig = plt.figure(figsize=(8,6))
        x, y, z = gan.get_decisions()
        plt.contourf(x, y, z, cmap=plt.cm.Spectral, levels=20)
        plt.colorbar()
    else:
        fig = plt.figure(figsize=(6,6))

    fake_output = gan.get_fake_output().numpy()
    plt.scatter(fake_output[:,0], fake_output[:,1], c=[(0,0,0)], s=0.2)

    interval = [-5, 5] if args.dataset == '2Dgrid' else [-2, 2]
    plt.xlim(interval)
    plt.ylim(interval)

    plt.title('Epoch ' + str(epoch))

    plt.savefig(paths['images'] + 'epoch-' + str(epoch) + '.png')
    plt.clf()
    plt.close(fig)

def compute_metrics(gan, epoch):

    output = gan.get_fake_output().numpy()
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

    gan.metrics['epochs'].append(epoch)

    for metric, val in metrics.items():

        if metric not in gan.metrics:
            gan.metrics[metric] = []

        gan.metrics[metric].append(val)

    gan.metrics['last'] = metrics

    print('Modes:', metrics['modes'])
    print('HQS:', metrics['hqs'])
    print('KL:', metrics['kl'])

def eval_fn(gan, epoch):

    if args.save_images:
        visualize_data(gan, epoch)

    compute_metrics(gan, epoch)

# GAN TRAINING

gan_model = GAN(
                data_loaders = ((train_noise_loader, train_real_loader), (test_noise_loader, test_real_loader)),
                gan_models = (discriminator, generator),
                lr = args.lr, beta1 = args.beta1,
                d_alpha = args.d_alpha if not args.ls_gan else None,
                g_alpha = args.g_alpha if not args.ls_gan else None,
                dataset_name = args.dataset,
                amp = args.amp
                )

gan_model.train(n_epochs=args.n_epochs, epoch_step=args.epoch_step,
                                            flip=args.non_saturating,
                                            eval_fn=eval_fn
                                            )

gan_model.store_results(setting=setting, args=args, paths=paths)
