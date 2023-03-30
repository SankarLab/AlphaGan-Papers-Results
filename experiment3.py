
print()
print('(aD, aG)-GAN Experiment #3: Celeb-A')
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
import torchvision.utils as vutils
from tqdm import tqdm
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

parser = argparse.ArgumentParser(description='(aD, aG)-GAN Experiment #3')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Dataset
parser.add_argument('--train_test_split', type=float, default=0.8, help='percentage of train samples')
parser.add_argument('--noise_dim', type=int, default=100, help='dimensionality of latent noise vectors')
parser.add_argument('--batch_size', type=int, default=128, help='batch size used during training/testing')
parser.add_argument('--save_bursts', action='store_true', help='saves the plotted output bursts for each checkpoint')
parser.set_defaults(save_bursts=False)

# Training
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--epoch_step', type=int, default=51, help='number of epochs between validation checkpoints')
parser.add_argument('--d_lr', type=float, default=1e-4, help='learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for generator')
parser.add_argument('--d_width', type=int, default=64, help='channel multiplier for discriminator')
parser.add_argument('--g_width', type=int, default=64, help='channel multiplier for generator')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter for adam optimization')

# Loss function
parser.add_argument('--d_alpha', type=float, default=1.0, help='alpha parameter for discriminator')
parser.add_argument('--g_alpha', type=float, default=1.0, help='alpha parameter for generator')
parser.add_argument('--non_saturating', action='store_true', help='uses non saturating loss function')
parser.add_argument('--ls_gan', action='store_true', help='uses LS GAN loss functions')
parser.set_defaults(non_saturating=False, ls_gan=False)

args = parser.parse_args()

# DELETE
args.d_lr = args.g_lr
#args.d_alpha = args.g_alpha

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

print('Loading Celeb-A dataset...')
data_iter = iter(tqdm(range(len(os.listdir('data/CelebA/img_align_celeba')))))
def transform_data(x):
    global data_iter
    next(data_iter)
    return (transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])(x))

full_data = [x for x, _ in list(datasets.ImageFolder(root='data/CelebA', transform=transform_data))]
train_size = int(args.train_test_split * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = torch.utils.data.random_split(full_data, [train_size, test_size])

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

inception_model = models.inception_v3(weights=models.inception.Inception_V3_Weights.IMAGENET1K_V1)
inception_model.eval()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.noise_dim, args.g_width * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.g_width * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.g_width * 8, args.g_width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.g_width * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.g_width * 4, args.g_width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.g_width * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.g_width * 2, args.g_width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.g_width),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.g_width, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, args.d_width, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.d_width, args.d_width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.d_width * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.d_width * 2, args.d_width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.d_width * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.d_width * 4, args.d_width * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.d_width * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.d_width * 8, 1, 4, 1, 0, bias=False),
        )

        self.sig = (lambda x: x) if args.ls_gan else nn.Sigmoid()

    def forward(self, x):
        out = self.sig(self.main(x))
        return out.reshape(-1,1)

# Define instances
generator = Generator()
generator.apply(weights_init)
discriminator = Discriminator()
discriminator.apply(weights_init)

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

    images = gan.get_fake_output(25)

    fig = plt.figure(figsize=(10,10))

    plt.imshow(np.transpose(vutils.make_grid(torch.Tensor(images), nrow=5, padding=2, normalize=True),(1,2,0)))

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
    fake_images = torch.Tensor(gan.get_fake_output())

    real_loader = DataLoader([real_images[i,:] for i in range(real_images.shape[0])],
                        batch_size=args.batch_size, shuffle=False)
    fake_loader = DataLoader([fake_images[i,:] for i in range(fake_images.shape[0])],
                        batch_size=args.batch_size, shuffle=False)

    # FID Score
    global inception_model

    inception_model = inception_model.to(gan.device)
    preprocess = transforms.Compose([
                        transforms.Normalize([0,0,0],[2,2,2]),
                        transforms.Normalize([-0.5,-0.5,-0.5], [1,1,1]),
                        models.inception.Inception_V3_Weights.IMAGENET1K_V1.transforms()
                        ])

    real_features, fake_features = [], []

    for reals, fakes in tqdm(zip(real_loader, fake_loader), total=len(real_loader)):

        reals = preprocess(reals).to(gan.device)
        fakes = preprocess(fakes).to(gan.device)

        real_features.append(inception_model(reals).detach().cpu().numpy())
        fake_features.append(inception_model(fakes).detach().cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    fid_score = compute_FID(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6)

    print("FID score: ", fid_score)

    gan.metrics['last'] = [fid_score]

    if 'epochs' not in gan.metrics:
        gan.metrics['epochs'] = []

    gan.metrics['epochs'].append(epoch)

    if 'FID' not in gan.metrics:
        gan.metrics['FID'] = []

    gan.metrics['FID'].append(fid_score)

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
                dataset_name = 'celeb-a'
                )

gan_model.train(n_epochs=args.n_epochs, epoch_step=args.epoch_step,
                                            seed=args.seed,
                                            flip=args.non_saturating,
                                            make_data=make_data
                                            )

row = ([setting, args.seed] + gan_model.metrics['last'] + [str(gan_model.metrics['epochs'])]
        + [str(gan_model.metrics['FID'])])

with open('experiment/metrics.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(row)

with open(paths['base'] + 'metrics.csv', 'w') as f:
    f.write(str(gan_model.metrics))

print()
print('--------------------')
print('Metrics:', gan_model.metrics['last'])
