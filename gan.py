
"""
gan.py
-------------------
General procedure for training with dataset, discriminator, generator, and loss function.
-------------------
Kyle Otstot
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os

from matplotlib import pyplot as plt

class GAN:

    def __init__(self, data_loaders, models, optimizers, criteria, dataset_name):

        # Prep datasets
        train_loaders, test_loaders = data_loaders
        self.train_noise_loader, self.train_real_loader = train_loaders
        self.test_noise_loader, self.test_real_loader = test_loaders
        self.dataset_name = dataset_name

        # For GPU use
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()

        # Models
        self.discriminator, self.generator = models

        print('Num GPUs:', self.n_gpu)

        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)

        # Optimizers
        self.d_optimizer, self.g_optimizer = optimizers

        # Loss functions
        self.d_criterion = criteria[0].to(self.device)
        self.g_criterion = criteria[1].to(self.device)

        self.metrics = {}


    def train_discriminator(self, real_data, fake_data):

        b_size = real_data.shape[0]
        self.d_optimizer.zero_grad()

        # Real data is labeled 1. Compute loss
        output_real = self.discriminator(real_data)
        #loss_real = self.d_criterion(disc=(output_real, 1, 1/9))#output_real, torch.ones(b_size,1).to(self.device))
        loss_real = self.d_criterion(output_real, torch.ones(b_size,1).to(self.device))

        # Fake data is labeled 0. Compute loss
        output_fake = self.discriminator(fake_data)
        #loss_fake = self.d_criterion(disc=(output_fake, -1, 1/9))#output_fake, torch.zeros(b_size,1).to(self.device))
        loss_fake = self.d_criterion(output_fake, torch.zeros(b_size,1).to(self.device))

        # Gradient descent
        (1 * (loss_real + loss_fake)).backward()
        #loss_fake.backward()
        self.d_optimizer.step()

        return 1 * (float(loss_real) + float(loss_fake))


    def train_generator(self, real_data, fake_data, flip=False):

        b_size = fake_data.shape[0]
        self.g_optimizer.zero_grad()

        # Generator tries to maximize the value function
        #real_output = self.discriminator(real_data)
        #fake_output = self.discriminator(fake_data)

        output = self.discriminator(fake_data)

        #loss = self.g_criterion(gen=(real_output.detach(), fake_output))

        if flip:
            loss = self.g_criterion(output, torch.ones(b_size,1).to(self.device))
        else:
            loss = -self.g_criterion(output, torch.zeros(b_size,1).to(self.device))

        loss.backward()

        self.g_optimizer.step()

        return float(loss)

    def train_loop(self, train_d=True, train_g=True, flip=False, epoch=1):

        d_losses, g_losses = [], []

        for i, (noise_data, real_data) in enumerate(tqdm(zip(self.train_noise_loader, self.train_real_loader),
                                                                    total=len(self.train_noise_loader))):

            # Send to GPU
            noise_data, real_data = noise_data.to(self.device), real_data.to(self.device)

            fake_data = self.generator(noise_data)

            # Compute losses and backpropogate
            d_losses.append(self.train_discriminator(real_data, fake_data.detach()))

            """
            for j in range(4):
                self.train_generator(real_data, fake_data, flip=flip)
                fake_data = self.generator(noise_data)
            """

            g_losses.append(self.train_generator(real_data, fake_data, flip=flip))

        d_loss, g_loss = np.mean(d_losses), np.mean(g_losses)

        return d_loss, g_loss

    def train(self, n_epochs, epoch_step, make_data, flip=False, seed=1):

        self.discriminator.train()
        self.generator.train()

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

                make_data(self, epoch)

    def evaluate(self):

        self.discriminator.eval()
        self.generator.eval()

        labels, preds = [], []

        with torch.no_grad():

            for noise_data, real_data in zip(self.test_noise_loader, self.test_real_loader):

                # Send to GPU
                noise_data, real_data = noise_data.to(self.device), real_data.to(self.device)

                # Set up classification task
                b_size = len(noise_data)
                features = torch.cat((real_data, self.generator(noise_data)), dim=0)
                labels += [1] * b_size + [0] * b_size
                output = self.discriminator(features)
                #output = nn.Sigmoid()(output)
                pred = torch.round(output).reshape(-1).detach().cpu().tolist()
                preds += pred

        # Compute metrics
        labels, preds = np.array(labels), np.array(preds)
        error = np.mean(labels != preds)
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

            dataset = (list(self.test_noise_loader.dataset)[:num_samples]
                                if num_samples is not None else self.test_noise_loader.dataset)

            # Generate output for fixed validation noise
            noise = torch.Tensor([list(d.numpy()) for d in dataset]).to(self.device)
            output = self.generator(noise).detach().cpu().numpy()

        self.discriminator.train()
        self.generator.train()

        return output

    def get_real_output(self):
        return torch.concat([t.reshape(1, *t.shape) for t in self.test_real_loader.dataset], dim=0)

    def get_decisions(self):

        assert self.dataset_name in ['2Dgrid', '2Dring'] # For 2D datasets only

        limit = 5 if self.dataset_name == '2Dgrid' else 2

        xs = torch.linspace(-limit, limit, steps=100)
        ys = torch.linspace(-limit, limit, steps=100)

        x, y = torch.meshgrid(xs, ys, indexing='xy')
        xy = torch.concat((x.reshape(-1,1), y.reshape(-1,1)), dim=1)

        z = self.discriminator(xy.to(self.device)).detach().reshape(x.shape).cpu()

        return x, y, z
