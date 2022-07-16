
"""
gan.py
-------------------
General procedure for training with dataset, discriminator, generator, and loss function.
-------------------
Kyle Otstot
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class GAN:

    def __init__(self, data_loaders, models, optimizers, criterion):

        # Prep datasets
        train_loaders, test_loaders = data_loaders
        self.train_noise_loader, self.train_real_loader = train_loaders
        self.test_noise_loader, self.test_real_loader = test_loaders

        # For GPU use
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Models
        self.discriminator, self.generator = models
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)

        # Optimizers
        self.d_optimizer, self.g_optimizer = optimizers

        # Loss function
        self.criterion = criterion.to(self.device)


    def train_discriminator(self, real_data, fake_data):

        b_size = real_data.shape[0]
        self.d_optimizer.zero_grad()

        # Real data is labeled 1. Compute loss
        output_real = self.discriminator(real_data)
        loss_real = self.criterion(output_real, torch.ones(b_size,1).to(self.device))

        # Fake data is labeled 0. Compute loss
        output_fake = self.discriminator(fake_data)
        loss_fake = self.criterion(output_fake, torch.zeros(b_size,1).to(self.device))

        # Gradient descent
        loss_real.backward()
        loss_fake.backward()
        self.d_optimizer.step()

        return float(loss_real) + float(loss_fake)


    def train_generator(self, fake_data):

        b_size = fake_data.shape[0]
        self.g_optimizer.zero_grad()

        # Generator tries to maximize the value function
        output = self.discriminator(fake_data)
        loss = -self.criterion(output, torch.zeros(b_size,1).to(self.device))
        loss.backward()
        self.g_optimizer.step()

        return float(loss)


    def train(self, n_epochs, epoch_step, make_burst=None):

        self.discriminator.train()
        self.generator.train()

        for epoch in range(1, n_epochs+1):

            d_losses, g_losses = [], []

            for noise_data, real_data in tqdm(zip(self.train_noise_loader, self.train_real_loader),
                                                                        total=len(self.train_noise_loader)):

                # Send to GPU
                noise_data, real_data = noise_data.to(self.device), real_data.to(self.device)

                fake_data = self.generator(noise_data)

                # Compute losses and backpropogate
                d_losses.append(self.train_discriminator(real_data, fake_data.detach()))
                g_losses.append(self.train_generator(fake_data))

            d_loss, g_loss = np.mean(d_losses), np.mean(g_losses)

            if epoch % epoch_step == 0:

                # Output checkpoint metrics
                print()
                print('Epoch', epoch, 'of', n_epochs)
                print('-----------------------')
                print('Discriminator loss:', d_loss)
                print('Generator loss:', g_loss)
                self.evaluate()

                # Draw snapshot of generated output
                if make_burst is not None:
                    make_burst(self, epoch)


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
                pred = torch.round(self.discriminator(features)).reshape(-1).detach().tolist()
                #print(pred)
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

    def get_output(self):

        self.discriminator.eval()
        self.generator.eval()

        with torch.no_grad():

            # Generate output for fixed validation noise
            noise = torch.Tensor([list(t.numpy()) for t in self.test_noise_loader.dataset]).to(self.device)
            output = self.generator(noise).detach().numpy()

        self.discriminator.train()
        self.generator.train()

        return output
