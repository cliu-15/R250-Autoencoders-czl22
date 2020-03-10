import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional

import IPython
from utils import mnist, mnist_batched
from utils import interrupted, enumerate_cycle

## Beta-variational autoencoder (based on Higgins et al., 2016)
# Adapted from code written by Andrei Margeloiu

class B_VAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        # variance is always positive, thus this layer encodes log(variance).
        # we compute the std in reparameterize()
        self.fc_logvar = nn.Linear(128, latent_dim)

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

    def encode(self, x):
        latent = functional.relu(self.shared_encoder(x))
        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

    @staticmethod
    def loss_function(reconstructed_x, x, mu, logvar, beta=3):
        BCE = functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')
        KLD = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


# All the standard functions you might like to apply come in two versions:
# as a pure function (e.g. torch.nn.functional.sigmoid) and as a Module (class torch.nn.Sigmoid).
# All except for torch.reshape... so here is a Module version of it.

class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.reshape(x, self.dim)


if __name__ == '__main__':

    # Create a new model object, and initialize its parameters
    fit = B_VAE(latent_dim=6)

    # Prepare to iterate through all the training data.
    # See the note at the top, under Utilities.
    iter_training_data = enumerate_cycle(mnist_batched)

    optimizer = optim.Adadelta(fit.parameters(), lr=0.05)

    print("Press Ctrl+C to end training and save parameters")

    beta = 3

    while not interrupted():
        (epoch, batch_num), (imgs, lbls) = next(iter_training_data)
        optimizer.zero_grad()
        rx, mu, logvar, z = fit(imgs)
        e = B_VAE.loss_function(rx, imgs, mu, logvar, beta=beta)
        e.backward()
        optimizer.step()

        if batch_num % 25 == 0:
            IPython.display.clear_output(wait=True)
            print(f'epoch={epoch} batch={batch_num}/{len(mnist_batched)} loss={e.item()}')

    # Optionally, save all the parameters
    torch.save(fit.state_dict(), 'models/b_vae_6d.pt')
