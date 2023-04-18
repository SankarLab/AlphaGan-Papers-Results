
"""
gan.py
-------------------
General procedure for training with dataset, discriminator, generator, and loss function.
-------------------
Kyle Otstot
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchvision.utils as vutils
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import csv
import time

# UTILITY FUNCTIONS

def create_setting(args, argv):

    setting = '-'.join(argv)

    while '--' in setting:
        setting = '-' + setting.replace('--', '-') + '-'
        setting = setting.strip('--')

    if 'seed' in setting:
        s_split = setting.split('-seed-')
        setting = s_split[0] + '-'.join(s_split[1].split('-')[1:])

    setting = setting.strip('-')
    unique_setting = setting + '-seed-' + str(args.seed) + '-time-' + str(time.time())

    paths = { 'base' : 'results/data/' + unique_setting + '/' }

    if args.save_images:
        paths['images'] = paths['base'] + 'images/'

    for path in paths.values():
        os.mkdir(path)

    print('Setting:')
    print(unique_setting)

    return setting, paths

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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

# LOSS FUNCTIONS

class AlphaLoss(nn.Module):

    def __init__(self, alpha, ep):
        super().__init__()
        self.alpha = alpha
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss()
        self.ep = ep

    def forward(self, logits, labels):

        if self.alpha == 1:
            return self.bce(logits, labels)

        output = torch.clamp(self.sigmoid(logits), min=self.ep, max=1 - self.ep)
        A = (self.alpha / (self.alpha - 1))
        real_term = A * (1 - labels * (output ** (1/A)))
        fake_term = -A * ((1 - labels) * (1 - output) ** (1/A))
        loss = torch.mean(real_term + fake_term)
        return loss

class LSLoss(nn.Module):

    def __init__(self):
        super(LSLoss, self).__init__()

    def forward(self, output, labels):
        return 0.5 * torch.mean((output - labels) ** 2)

# BASE GAN CLASS

class GAN:

    def __init__(self, data_loaders, gan_models, lr, beta1, d_alpha, g_alpha, dataset_name, amp=False):

        # Prep datasets
        train_loaders, test_loaders = data_loaders
        self.train_noise_loader, self.train_real_loader = train_loaders
        self.test_noise_loader, self.test_real_loader = test_loaders
        self.dataset_name = dataset_name

        # For GPU use
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device:', self.device)

        # Models
        self.discriminator, self.generator = gan_models
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)

        # Optimizers
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.99))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.99))

        # Mixed precision grad scaler
        self.amp = amp
        print('AMP?', 'Yes' if self.amp else 'No')
        if self.amp:
            self.scaler = GradScaler()

        # Loss functions
        self.d_criterion = AlphaLoss(d_alpha, ep=1e-3 if self.amp else 1e-7) if d_alpha is not None else LSLoss()
        self.d_criterion = self.d_criterion.to(self.device)
        self.g_criterion = AlphaLoss(g_alpha, ep=1e-3 if self.amp else 1e-7) if g_alpha is not None else LSLoss()
        self.g_criterion = self.g_criterion.to(self.device)

        # Non-2D datasets use DCGANs and evaluate with FID
        if self.dataset_name not in ['2Dring', '2Dgrid']:

            self.discriminator.apply(weights_init)
            self.generator.apply(weights_init)

            self.inception_model = models.inception_v3(
                                weights=models.inception.Inception_V3_Weights.IMAGENET1K_V1
                                ).to(self.device)
            self.inception_model.eval()

        self.metrics = {'errors' : []}


    def train_discriminator(self, real_data, fake_data):

        b_size = real_data.shape[0]
        self.d_optimizer.zero_grad()

        def forward_pass():

            # Real data is labeled 1. Compute loss
            output_real = self.discriminator(real_data)
            loss_real = self.d_criterion(output_real, torch.ones(b_size,1).to(self.device))

            # Fake data is labeled 0. Compute loss
            output_fake = self.discriminator(fake_data)
            loss_fake = self.d_criterion(output_fake, torch.zeros(b_size,1).to(self.device))

            return loss_real + loss_fake

        if self.amp:

            with autocast():
                loss = forward_pass()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()

        else:
            loss = forward_pass()
            loss.backward()
            self.d_optimizer.step()

        return float(loss)

    def train_generator(self, fake_data, flip=False):

        b_size = fake_data.shape[0]
        self.g_optimizer.zero_grad()

        def forward_pass():

            output = self.discriminator(fake_data)

            if flip:
                loss = self.g_criterion(output, torch.ones(b_size,1).to(self.device))
            else:
                loss = -self.g_criterion(output, torch.zeros(b_size,1).to(self.device))

            return loss

        if self.amp:

            with autocast():
                loss = forward_pass()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.g_optimizer)
            self.scaler.update()

        else:

            loss = forward_pass()
            loss.backward()
            self.g_optimizer.step()

        return float(loss)

    def train_loop(self, flip=False, epoch=1):

        d_losses, g_losses = [], []

        for noise_data, real_data in tqdm(zip(self.train_noise_loader, self.train_real_loader),
                                                                    total=len(self.train_noise_loader)):

            real_data = real_data[0] if isinstance(real_data, list) else real_data

            # Send to GPU
            noise_data, real_data = noise_data.to(self.device), real_data.to(self.device)

            fake_data = self.generator(noise_data)

            # Compute losses and backpropogate
            d_losses.append(self.train_discriminator(real_data, fake_data.detach()))
            g_losses.append(self.train_generator(fake_data, flip=flip))

        d_loss, g_loss = np.mean(d_losses), np.mean(g_losses)

        return d_loss, g_loss

    def train(self, n_epochs, epoch_step, eval_fn, flip=False):

        self.discriminator.train()
        self.generator.train()

        start_time = time.time()

        for epoch in range(1, n_epochs+1):

            print('Epoch', epoch)

            d_loss, g_loss = self.train_loop(flip=flip, epoch=epoch)

            print()
            print('Epoch', epoch, 'of', n_epochs)
            print('-----------------------')
            print('Discriminator loss:', d_loss)
            print('Generator loss:', g_loss)

            if epoch % epoch_step == 0:

                # Output checkpoint metrics
                self.evaluate()
                eval_fn(self, epoch)

        end_time = time.time()
        self.metrics['time'] = int(end_time - start_time)

    def evaluate(self):

        self.discriminator.eval()
        self.generator.eval()

        labels, preds = [], []

        with torch.no_grad():

            for noise_data, real_data in zip(self.test_noise_loader, self.test_real_loader):

                real_data = real_data[0] if isinstance(real_data, list) else real_data

                # Send to GPU
                noise_data, real_data = noise_data.to(self.device), real_data.to(self.device)

                # Set up classification task
                def classify():
                    b_size = len(noise_data)
                    features = torch.cat((real_data, self.generator(noise_data)), dim=0)
                    label = [1] * b_size + [0] * b_size
                    output = self.discriminator(features)
                    output = nn.Sigmoid()(output)
                    pred = torch.round(output).reshape(-1).detach().cpu().tolist()
                    return label, pred

                if self.amp:
                    with autocast():
                        label, pred = classify()
                else:
                    label, pred = classify()

                labels += label
                preds += pred

        # Compute metrics
        labels, preds = np.array(labels), np.array(preds)
        error = np.mean(labels != preds)
        self.metrics['errors'].append(error)
        CM = confusion_matrix(labels, preds)

        print('Classification error:', error)
        print('Confusion Matrix:')
        print(CM / np.sum(CM))

        self.discriminator.train()
        self.generator.train()

    def get_fake_output(self, num_samples=None):

        self.discriminator.eval()
        self.generator.eval()

        with torch.no_grad():

            count = 0
            output = []
            for noise in self.test_noise_loader:

                if num_samples is not None and count >= num_samples:
                    break

                if self.amp:
                    with autocast():
                        output.append(self.generator(noise.to(self.device)))
                else:
                    output.append(self.generator(noise.to(self.device)))

                count += noise.shape[0]

            output = torch.concat(output, dim=0).detach().cpu()

            if num_samples is not None:
                output = output[:num_samples,:,:,:]

        self.discriminator.train()
        self.generator.train()

        return output

    def get_real_output(self):

        first = self.test_real_loader.dataset[0]
        if isinstance(first, list) or isinstance(first, tuple):
            return torch.concat([t.reshape(1, *t.shape) for t, _ in self.test_real_loader.dataset], dim=0)
        else:
            return torch.concat([t.reshape(1, *t.shape) for t in self.test_real_loader.dataset], dim=0)

    def get_decisions(self):

        assert self.dataset_name in ['2Dgrid', '2Dring'] # For 2D datasets only

        limit = 5 if self.dataset_name == '2Dgrid' else 2

        xs = torch.linspace(-limit, limit, steps=100)
        ys = torch.linspace(-limit, limit, steps=100)

        x, y = torch.meshgrid(xs, ys, indexing='xy')
        xy = torch.concat((x.reshape(-1,1), y.reshape(-1,1)), dim=1)

        with autocast():
            z = self.discriminator(xy.to(self.device)).detach().reshape(x.shape).cpu()

        return x, y, z

    def get_fid_score(self, batch_size):

        print('Computing FID Score:')

        # Get real & generated output
        real_images = self.get_real_output()
        fake_images = self.get_fake_output()

        real_loader = DataLoader([real_images[i,:] for i in range(real_images.shape[0])],
                            batch_size=batch_size, shuffle=False, pin_memory=True)
        fake_loader = DataLoader([fake_images[i,:] for i in range(fake_images.shape[0])],
                            batch_size=batch_size, shuffle=False, pin_memory=True)

        preprocess = transforms.Compose([
                            transforms.Normalize([0,0,0],[2,2,2]),
                            transforms.Normalize([-0.5,-0.5,-0.5], [1,1,1]),
                            models.inception.Inception_V3_Weights.IMAGENET1K_V1.transforms()
                            ])

        real_features, fake_features = [], []

        for reals, fakes in tqdm(zip(real_loader, fake_loader), total=len(real_loader)):

            reals = preprocess(reals).to(self.device)
            fakes = preprocess(fakes).to(self.device)

            def forward_pass():
                real_feature = self.inception_model(reals).detach().cpu().numpy()
                fake_feature = self.inception_model(fakes).detach().cpu().numpy()
                return real_feature, fake_feature

            if self.amp:
                with autocast():
                    real_feature, fake_feature = forward_pass()
            else:
                real_feature, fake_feature = forward_pass()

            real_features.append(real_feature)
            fake_features.append(fake_feature)

        real_features = np.concatenate(real_features, axis=0)
        fake_features = np.concatenate(fake_features, axis=0)

        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)

        fid_score = compute_FID(mu_real, sigma_real, mu_fake, sigma_fake)

        return fid_score

    def make_images(self, epoch, paths):

        images = self.get_fake_output(25).to(torch.float32)

        fig = plt.figure(figsize=(10,10))

        plt.imshow(np.transpose(vutils.make_grid(images, nrow=5, padding=2, normalize=True),(1,2,0)))

        plt.title('Epoch ' + str(epoch))
        plt.savefig(paths['images'] + 'epoch-' + str(epoch) + '.png')
        plt.clf()
        plt.close(fig)

    def store_results(self, setting, args, paths):

        row_data = {'setting' : setting}

        for arg in vars(args):
            row_data[arg] = getattr(args, arg)

        for metric in [key for key in self.metrics if key != 'last']:
            row_data[metric] = str(self.metrics[metric])

        if not os.path.exists('results/metrics.csv'):

            with open('results/metrics.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(list(row_data.keys()))

        with open('results/metrics.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list(row_data.values()))

        with open(paths['base'] + 'metrics.csv', 'w') as f:
            f.write(str(self.metrics))

        print()
        print('--------------------')
        print('Metrics:', self.metrics['last'])
