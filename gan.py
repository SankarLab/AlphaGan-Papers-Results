
"""
gan.py
-------------------
General procedure for training with dataset, discriminator, generator, and loss function.
-------------------
Kyle Otstot
"""

import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

class AlphaLoss(torch.nn.Module):

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

class GAN:

    def __init__(self, data_loaders, models, optimizers, criteria, dataset_name):

        # Prep datasets
        train_loaders, test_loaders = data_loaders
        self.train_noise_loader, self.train_real_loader = train_loaders
        self.test_noise_loader, self.test_real_loader = test_loaders
        self.dataset_name = dataset_name

        # For GPU use
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Models
        self.discriminator, self.generator = models
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)

        # Optimizers
        self.d_optimizer, self.g_optimizer = optimizers

        # Loss functions
        self.d_criterion = criteria[0].to(self.device)
        self.g_criterion = criteria[1].to(self.device)

        self.metrics = {'epochs' : [], 'modes' : [], 'hqs' : []}


    def train_discriminator(self, real_data, fake_data):

        b_size = real_data.shape[0]
        self.d_optimizer.zero_grad()

        # Real data is labeled 1. Compute loss
        output_real = self.discriminator(real_data)
        loss_real = self.d_criterion(output_real, torch.ones(b_size,1).to(self.device))

        # Fake data is labeled 0. Compute loss
        output_fake = self.discriminator(fake_data)
        loss_fake = self.d_criterion(output_fake, torch.zeros(b_size,1).to(self.device))

        # Gradient descent
        loss_real.backward()
        loss_fake.backward()
        self.d_optimizer.step()

        return float(loss_real) + float(loss_fake)


    def train_generator(self, fake_data, flip=False):

        #fake_data = Variable(fake_data, requires_grad=True)

        b_size = fake_data.shape[0]
        self.g_optimizer.zero_grad()

        # Generator tries to maximize the value function
        output = self.discriminator(fake_data)

        if flip:
            loss = self.g_criterion(output, torch.ones(b_size,1).to(self.device))
        else:
            loss = -self.g_criterion(output, torch.zeros(b_size,1).to(self.device))

        loss.backward()

        self.g_optimizer.step()

        return float(loss)

    def generator_grads(self, fake_data, flip=False):

        fake_data = Variable(fake_data, requires_grad=True)

        b_size = fake_data.shape[0]
        self.g_optimizer.zero_grad()

        # Generator tries to maximize the value function
        output = self.discriminator(fake_data)

        if flip:
            loss = self.g_criterion(output, torch.ones(b_size,1).to(self.device))
        else:
            loss = -self.g_criterion(output, torch.zeros(b_size,1).to(self.device))

        loss.backward()

        return fake_data.grad.data


    def train_loop(self, train_d=True, train_g=True, flip=False, count=-1, epoch=1):

        d_losses, g_losses = [], []

        self.all_grads = None

        for i, (noise_data, real_data) in enumerate(tqdm(zip(self.train_noise_loader, self.train_real_loader),
                                                                    total=len(self.train_noise_loader))):

            if 0 <= count <= i:
                break

            # Send to GPU
            noise_data, real_data = noise_data.to(self.device), real_data.to(self.device)

            fake_data = self.generator(noise_data)

            # Compute losses and backpropogate
            d_losses.append(
                    self.train_discriminator(real_data, fake_data.detach()) if train_d else -1)

            grads = self.generator_grads(fake_data, flip=flip)

            g_losses.append(
                    self.train_generator(fake_data, flip=flip) if train_d else -1)


            self.all_grads = grads if self.all_grads is None else torch.concat((self.all_grads, grads), dim=0)


        #plt.scatter(all_grads[:,0], all_grads[:,1], s=0.2, alpha=0.5)
        #plt.savefig('experiment1/grads/epoch-' + str(epoch) + '.png')
        #plt.close()


        d_loss, g_loss = np.mean(d_losses), np.mean(g_losses)

        return d_loss, g_loss

    def train(self, n_epochs, epoch_step, make_data, flip=False, seed=1):

        self.discriminator.train()
        self.generator.train()

        limit_dict = {'1' : 48, '5' : 20, '6' : 17, '8' : 81, '9' : 19}
        limit = 0#limit_dict[str(seed)]

        for epoch in range(1, n_epochs+1):

            #if epoch > limit:
                #self.criterion = AlphaLoss(0.2)

            print('Epoch', epoch)

            d_loss, g_loss = self.train_loop(train_d=True, train_g=True, flip=flip, epoch=epoch)
            #_, g_loss = self.train_loop(train_d=False, train_g=True, count=5)

            if epoch % epoch_step == 0:

                # Output checkpoint metrics
                print()
                print('Epoch', epoch, 'of', n_epochs)
                print('-----------------------')
                print('Discriminator loss:', d_loss)
                print('Generator loss:', g_loss)
                self.evaluate()

                make_data(self, epoch)

                # Draw snapshot of generated output
                #if epoch >= limit and make_burst is not None:
                    #make_burst(self, epoch)




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
                pred = torch.round(self.discriminator(features)).reshape(-1).detach().cpu().tolist()
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

    def get_fake_output(self):

        self.discriminator.eval()
        self.generator.eval()

        with torch.no_grad():

            # Generate output for fixed validation noise
            noise = torch.Tensor([list(t.numpy()) for t in self.test_noise_loader.dataset]).to(self.device)
            output = self.generator(noise).detach().cpu().numpy()

        self.discriminator.train()
        self.generator.train()

        return output

    def get_real_output(self):
        return torch.Tensor([list(t.numpy()) for t in self.test_real_loader.dataset])

    def get_decisions(self):

        limit = 5 if self.dataset_name == 'lattice' else 2

        xs = torch.linspace(-limit, limit, steps=100)
        ys = torch.linspace(-limit, limit, steps=100)

        x, y = torch.meshgrid(xs, ys, indexing='xy')
        xy = torch.concat((x.reshape(-1,1), y.reshape(-1,1)), dim=1)

        z = self.discriminator(xy.to(self.device)).detach().reshape(x.shape).cpu()

        return x, y, z
