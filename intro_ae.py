import torch
import torch.nn as nn
import torch.optim as optim

## Basic autoencoder
# Adapted from introductory code by Dr. Damon Wischik

import IPython
from utils import mnist, mnist_batched
from utils import interrupted, enumerate_cycle


## Simple autoencoder
class SimpleAE(nn.Module):
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
            nn.Linear(128, latent_dim),
            nn.LeakyReLU()
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

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.reshape(x, self.dim)


if __name__ == '__main__':

    # Create a new model object, and initialize its parameters
    fit = SimpleAE(latent_dim=6)

    # Prepare to iterate through all the training data.
    # See the note at the top, under Utilities.
    iter_training_data = enumerate_cycle(mnist_batched)

    optimizer = optim.Adadelta(fit.parameters(), lr=1)

    print("Press Ctrl+C to end training and save parameters")

    while not interrupted():
        (epoch, batch_num), (imgs,lbls) = next(iter_training_data)
        optimizer.zero_grad()
        _,rx = fit(imgs)
        e = nn.functional.mse_loss(imgs, rx)
        e.backward()
        optimizer.step()

        if batch_num % 25 == 0:
            IPython.display.clear_output(wait=True)
            print(f'epoch={epoch} batch={batch_num}/{len(mnist_batched)} loss={e.item()}')

    # Optionally, save all the parameters
    torch.save(fit.state_dict(), 'intro_ae_6d.pt')
