import torch
import torch.nn as nn
import torch.optim as optim

import IPython
from utils import mnist, mnist_batched
from utils import interrupted, enumerate_cycle
import numpy as np


## PCA-like Autoencoder (based on Ladjal et al., 2019)

class PCA_AE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.LeakyReLU(),
            # Batch normalization layer (with B = 0) just before latent space
            nn.BatchNorm1d(1, affine = False)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1728),
            nn.LeakyReLU(),
            Reshaper((-1, 12, 12, 12)),
            nn.Conv2d(12, 36, 3, 1),
            nn.LeakyReLU(),
            Reshaper((-1, 4, 30, 30)),
            nn.Conv2d(4, 4, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 1, 1, 1),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim

    def forward(self, x, prev_z):
        z = self.encoder(x)

        # concatenate with previous latent space if k > 1
        if self.latent_dim > 1:
            new_z = torch.cat((prev_z, z), 1)
        else:
            new_z = z

        return new_z, self.decoder(new_z)


# All the standard functions you might like to apply come in two versions:
# as a pure function (e.g. torch.nn.functional.sigmoid) and as a Module (class torch.nn.Sigmoid).
# All except for torch.reshape... so here is a Module version of it.

class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.reshape(x, self.dim)


def reconstruct(imgs, models):
    z = torch.empty((np.shape(imgs)[0], 0))
    rx = torch.empty((0,1,28,28))
    for i in range(np.shape(models)[0]):
        z = torch.cat((z, models[i].encoder(imgs)), 1)
        rx = torch.cat((rx, models[i].decoder(z)), 0)
    return z, rx

if __name__ == '__main__':

    # max latent space size
    d_max = 8
    prev_z = []
    models = []

    for k in range(1, d_max + 1):
        iter_training_data = enumerate_cycle(mnist_batched)

        fit = PCA_AE(latent_dim=k)
        optimizer = optim.Adadelta(fit.parameters(), lr=1)

        print("Press Ctrl+C to end training and save parameters")
        epoch = 0
        max_epochs = 5

        while epoch < max_epochs:
            (epoch, batch_num), (imgs, lbls) = next(iter_training_data)
            optimizer.zero_grad()

            # train for new dimension (passing in previous latent space)
            prev_z = torch.empty(np.shape(imgs)[0],0)
            for i in range(k-1):
                if k == 2:
                    prev_z = models[i].encoder(imgs)
                else:
                    prev_z = torch.cat((prev_z, models[i].encoder(imgs)), 1)
            z, rx = fit(imgs, prev_z)

            # add covariance term
            lam = 0.05
            cov = 0.0
            for i in range(k - 1):
                cov += np.dot(z[:, i].detach(), z[:, k - 1].detach())
            cov = cov / np.shape(z)[0]

            e = nn.functional.mse_loss(imgs, rx) + lam * cov
            e.backward(retain_graph = True)
            optimizer.step()

            if batch_num % 25 == 0:
                IPython.display.clear_output(wait=True)
                print(f'k={k} epoch={epoch} batch={batch_num}/{len(mnist_batched)} loss={e.item()}')

        # store current model
        models.append(fit)

        # save all models
        torch.save(fit, 'pca_ae_8d_'+str(k)+'.pt')