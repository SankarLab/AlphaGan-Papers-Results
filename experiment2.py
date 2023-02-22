
print()
print('(aD, aG)-GAN Experiment #2: Stacked MNIST')
print('Author: Kyle Otstot')
print('-------------------------------')
print()

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from math import ceil
import random
import os
import time
import csv
from gan import *

# PARAMETERS

parser = argparse.ArgumentParser(description='(aD, aG)-GAN Experiment #2')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Dataset
parser.add_argument('--train_size', type=int, default=100000, help='number of train examples')
parser.add_argument('--test_size', type=int, default=25000, help='number of test examples')
parser.add_argument('--noise_dim', type=int, default=100, help='dimensionality of latent noise vectors')
parser.add_argument('--batch_size', type=int, default=64, help='batch size used during training/testing')
parser.add_argument('--save_bursts', action='store_true', help='saves the plotted output bursts for each checkpoint')
parser.set_defaults(save_bursts=False)

# Training
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--epoch_step', type=int, default=51, help='number of epochs between validation checkpoints')
parser.add_argument('--d_lr', type=float, default=1e-3, help='learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=1e-3, help='learning rate for generator')
parser.add_argument('--d_width', type=int, default=1, help='channel multiplier for discriminator')
parser.add_argument('--g_width', type=int, default=1, help='channel multiplier for generator')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter for adam optimization')

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

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

def make_noise(data_size):

    data_tensor = torch.randn(data_size, args.noise_dim)
    data = [data_tensor[i,:] for i in range(data_size)]
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return data_loader

def make_stacks(data, data_size):
    data = torch.concat(data, dim=0)
    stacked_data = data[torch.randint(data.shape[0], (data_size * 3,))]
    stacked_data = stacked_data.reshape(data_size, 3, data.shape[1], data.shape[2])
    return [stacked_data[i, :, :, :] for i in range(data_size)]

transform_data = transforms.Compose([transforms.ToTensor()])

train_data = [x for x, _ in list(datasets.MNIST('data', download=True, train=True, transform=transform_data))]
train_data = make_stacks(train_data, args.train_size)
test_data = [x for x, _ in list(datasets.MNIST('data', download=True, train=False, transform=transform_data))]
test_data = make_stacks(test_data, args.test_size)

train_real_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_real_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

train_noise_loader = make_noise(len(train_data))
test_noise_loader = make_noise(len(test_data))

# MODELS

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.main(x)

classifier = Classifier()
classifier.load_state_dict(torch.load('pretrained_models/-save_model.pt'))
classifier.eval()

inception_model = models.inception_v3(weights=models.inception.Inception_V3_Weights.IMAGENET1K_V1)
inception_model.eval()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.noise_dim, args.g_width * 32, 3, stride=2, bias=False),
            nn.BatchNorm2d(args.g_width * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.g_width * 32, args.g_width * 16, 3, stride=2, bias=False),
            nn.BatchNorm2d(args.g_width * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.g_width * 16, args.g_width * 8, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(args.g_width * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.g_width * 8, 3, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        out = self.main(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, args.d_width * 8, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.d_width * 8, args.d_width * 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(args.d_width * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.d_width * 16, args.d_width * 32, 3, stride=2, bias=False),
            nn.BatchNorm2d(args.d_width * 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.d_width * 32, 1, 3, stride=2, bias=False),
        )

        self.sig = (lambda x: x) if args.ls_gan else nn.Sigmoid()

    def forward(self, x):
        out = self.sig(self.main(x))
        return out.reshape(-1,1)

# Define instances
generator = Generator()
discriminator = Discriminator()

# OPTIMIZERS

g_optimizer = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.beta1, 0.99))
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.beta1, 0.99))

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

    images = gan.get_fake_output()

    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111, nrows_ncols=(5,5), axes_pad=0.1)
    for i in range(25):
        grid[i].imshow(np.clip(images[i,:, :, :],0,1).transpose(1,2,0))

    plt.title('Epoch ' + str(epoch))
    plt.savefig(paths['bursts'] + 'epoch-' + str(epoch) + '.png')
    plt.clf()
    plt.close(fig)

# METRICS

def compute_FID(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_metrics(gan, epoch):

    # Get real & generated output
    real_images = gan.get_real_output()
    fake_images = torch.Tensor(gan.get_fake_output()).to(gan.device)

    # FID Score
    global inception_model
    inception_model = inception_model.to(gan.device)
    preprocess = models.inception.Inception_V3_Weights.IMAGENET1K_V1.transforms()

    real_features, fake_features = [], []
    batch_size = 32

    for k in tqdm(range(ceil(len(real_images) / batch_size))):

        real_batch = preprocess(real_images[batch_size*k:batch_size*(k+1),:,:,:])
        fake_batch = preprocess(fake_images[batch_size*k:batch_size*(k+1),:,:,:])
        features = inception_model(torch.concat((real_batch, fake_batch), dim=0))
        real_features.append(features[:batch_size,:].detach().cpu().numpy())
        fake_features.append(features[batch_size:,:].detach().cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    fid_score = compute_FID(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6)

    print("FID score: ", fid_score)

    metrics = [fid_score]

    # Mode coverage w/ pretrained classifier
    global classifier
    classifier = classifier.to(gan.device)

    all_images = torch.concat([fake_images[:, i, :, :] for i in range(3)], dim=0)
    output = F.softmax(classifier(all_images.reshape(-1, 1, all_images.shape[1], all_images.shape[2])), dim=1)

    confs = np.array(output.amax(dim=1).cpu().tolist()).reshape(3,-1).prod(axis=0)
    labels = np.array(output.argmax(dim=1).cpu().tolist()).reshape(3,-1)
    labels = np.array([''.join(map(str, labels[:, i])) for i in range(labels.shape[1])])

    for limit in [n/10 for n in range(10)]:
        print('Limit:', limit)

        lim_labels = labels[confs >= limit]

        label_dict = {}
        for l in lim_labels:
            if l not in label_dict:
                label_dict[l] = 0
            label_dict[l] += 1

        counts = np.array(list(label_dict.values()))
        for modes in [1,3,5,10]:
            metrics.append((counts >= modes).sum())
            print('Modes', modes, '=', metrics[-1])

    gan.metrics['last'] = metrics

    if 'epochs' not in gan.metrics:
        gan.metrics['epochs'] = []

    gan.metrics['epochs'].append(epoch)

    if 'modes' not in gan.metrics:
        gan.metrics['modes'] = []

    gan.metrics['modes'].append(metrics[1])

    if 'FID' not in gan.metrics:
        gan.metrics['FID'] = []

    gan.metrics['FID'].append(metrics[0])

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
                dataset_name = 'stacked-mnist'
                )

gan_model.train(n_epochs=args.n_epochs, epoch_step=args.epoch_step,
                                            seed=args.seed,
                                            flip=args.non_saturating,
                                            make_data=make_data
                                            )

row = ([setting, args.seed] + gan_model.metrics['last'] + [str(gan_model.metrics['epochs'])]
        + [str(gan_model.metrics['modes'])] + [str(gan_model.metrics['FID'])])

with open('experiment/metrics.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(row)

with open(paths['base'] + 'metrics.csv', 'w') as f:
    f.write(str(gan_model.metrics))

print()
print('--------------------')
print('Metrics:', gan_model.metrics['last'])
