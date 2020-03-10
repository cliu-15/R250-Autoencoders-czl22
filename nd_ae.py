import torch
import torch.nn as nn
import torch.optim as optim

import IPython
from utils import mnist, mnist_batched
from utils import interrupted, enumerate_cycle
import numpy as np

## Nested dropout autoencoder (based on Rippel et al., 2014)

class ND_AE(nn.Module):
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
        self.latent_dim = latent_dim

    def truncate(self, z):
        #sample over geometric distribution -- ****determine p???
        b = np.random.geometric(p = (1.0 - 0.1), size = self.latent_dim)[0]
        if b < np.shape(z)[0]:
            new_z = z.clone()
            new_z[:, b:] = 0
        else:
            new_z = z
        return new_z

    def forward(self, x):
        z = self.encoder(x)
        z = self.truncate(z)
        return z, self.decoder(z)


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
    fit = ND_AE(latent_dim=8)

    # Prepare to iterate through all the training data.
    # See the note at the top, under Utilities.
    iter_training_data = enumerate_cycle(mnist_batched)

    optimizer = optim.Adadelta(fit.parameters(), lr=1)

    print("Press Ctrl+C to end training and save parameters")

    epoch = 0
    while not interrupted() and epoch < 10:
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
    torch.save(fit.state_dict(), 'nd_ae_8d.pt')
