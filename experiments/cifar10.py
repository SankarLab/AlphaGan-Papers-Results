print()
print('(aD, aG)-GAN Experiment: CIFAR-10')
print('Author: Kyle Otstot')
print('-------------------------------')
print()

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from tqdm import tqdm

sys.path.append('..')
from alphaGAN.gan import *

# PARAMETERS

parser = argparse.ArgumentParser(description='(aD, aG)-GAN Experiment: CIFAR-10')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Dataset
parser.add_argument('--noise_dim', type=int, default=100, help='dimensionality of latent noise vectors')
parser.add_argument('--batch_size', type=int, default=128, help='batch size used during training/testing')
parser.add_argument('--save_images', action='store_true', help='saves the plotted output bursts for each checkpoint')
parser.set_defaults(save_images=False)

# Training
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--epoch_step', type=int, default=51, help='number of epochs between validation checkpoints')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for discriminator & generator')
parser.add_argument('--model_width', type=int, default=64, help='channel multiplier for discriminator & generator')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter for adam optimization')
parser.add_argument('--amp', action='store_true', help='uses automatic mixed precision training')
parser.set_defaults(amp=False)

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

    data_tensor = torch.randn(data_size, args.noise_dim)
    data = [data_tensor[i,:] for i in range(data_size)]
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return data_loader

transform_data = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

print('Loading training data...')
train_data = [x for x, _ in list(datasets.CIFAR10('data', download=True, train=True, transform=transform_data))]
train_real_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
train_noise_loader = make_noise(len(train_data))

print('Loading testing data...')
test_data = [x for x, _ in list(datasets.CIFAR10('data', download=True, train=False, transform=transform_data))]
test_real_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_noise_loader = make_noise(len(test_data))

# MODELS

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

classifier = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True)
classifier.eval()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.noise_dim, args.model_width * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(args.model_width * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.model_width * 8, args.model_width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.model_width * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.model_width * 4, args.model_width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.model_width * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.model_width * 2, args.model_width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.model_width),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.model_width, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        out = self.main(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, args.model_width, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.model_width, args.model_width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.model_width * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.model_width * 2, args.model_width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.model_width * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.model_width * 4, args.model_width * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.model_width * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.model_width * 8, 1, 2, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).reshape(-1,1)

# Define instances
generator = Generator()
discriminator = Discriminator()

# EVALUATION FUNCTIONS

def visualize_data(gan, epoch):
    gan.make_images(epoch, paths)

# METRICS

def compute_metrics(gan, epoch):

    # FID SCORE

    fid_score = gan.get_fid_score(batch_size=args.batch_size)
    print('FID score: ', fid_score)

    gan.metrics['last'] = {'fid_score' : fid_score}

    if 'epochs' not in gan.metrics:
        gan.metrics['epochs'] = []

    gan.metrics['epochs'].append(epoch)

    if 'fid_scores' not in gan.metrics:
        gan.metrics['fid_scores'] = []

    gan.metrics['fid_scores'].append(fid_score)

    # MODE COVERAGE
    global classifier
    classifier = classifier.to(gan.device)
    preprocess = transforms.Compose([
                        transforms.Normalize([0,0,0],[2,2,2]),
                        transforms.Normalize([-0.5,-0.5,-0.5], [1,1,1])
                        ])

    fake_images = preprocess(gan.get_fake_output())
    fake_loader = DataLoader(fake_images, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    confs, labels = [], []

    for images in fake_loader:

        output = F.softmax(classifier(images.to(gan.device)), dim=1)
        confs += output.amax(dim=1).cpu().tolist()
        labels += output.argmax(dim=1).cpu().tolist()

    confs, labels = np.array(confs), np.array(labels)

    for limit in [0, 0.2, 0.5, 0.9]:

        lim_labels = labels[confs >= limit]

        label_dict = {}
        for l in lim_labels:
            if l not in label_dict:
                label_dict[l] = 0
            label_dict[l] += 1

        counts = np.array(list(label_dict.values()))
        for modes in [1,10]:

            mode_type = 'modes-' + str(limit) + ',' + str(modes)

            if mode_type not in gan.metrics:
                gan.metrics[mode_type] = []

            n_modes = (counts >= modes).sum()
            gan.metrics[mode_type].append(n_modes)
            gan.metrics['last'][mode_type] = n_modes

            print(mode_type, ':', n_modes)

def make_data(gan, epoch):

    compute_metrics(gan, epoch)

    if args.save_bursts:
        make_burst(gan, epoch)

# GAN TRAINING

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
                dataset_name = 'cifar10',
                amp = args.amp
                )

gan_model.train(n_epochs=args.n_epochs, epoch_step=args.epoch_step,
                                            flip=args.non_saturating,
                                            eval_fn=eval_fn
                                            )

gan_model.store_results(setting=setting, args=args, paths=paths)
