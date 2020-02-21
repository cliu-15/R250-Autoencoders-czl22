import torch
import torch.nn as nn
import torch.optim as optim

import IPython
from utils import mnist, mnist_batched
from utils import interrupted, enumerate_cycle


## Simple autoencoder
# 
# PyTorch has a [good tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html).
# 
# Briefly, PyTorch encourages you to write your neural networks as Modules. 
# 
# * A Module is basically a function, and you have to specify an entry point
# `forward(self, x)` which evaluates the function.
# 
# * For neural network training, we want to work with parameterized functions
# (parameters for edge weights). If you define tensors (or modules containing tensors)
# in `__init__`, then it treats them as parameters, meaning it will take care of all
# the gradient calculations for you. 


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


# All the standard functions you might like to apply come in two versions:
# as a pure function (e.g. torch.nn.functional.sigmoid) and as a Module (class torch.nn.Sigmoid).
# All except for torch.reshape... so here is a Module version of it.

class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.reshape(x, self.dim)



# As a sanity check, and to get the syntax right, I first tested out my neural network layers using simple imperative commands. It makes it easier to figure out the shapes involved at each stage.
# 
# The convention, in all of the standard `torch.nn` layers, is that the input data consists of batches, and the index within the batch is stored as the first index.
# 
# ```
# # Sanity check that the encoder works
# 
# imgs,lbls = next(iter(mnist_batched))
# print(imgs.shape) # 5 images * 1 channel * 28 width * 28 height
# 
# l1 = nn.Conv2d(1, 32, 3, 1)(imgs) # (in_channels, out_channels, kernel_size, stride)
# print(l1.shape) # 5 images * 32 filters * 26 width * 26 height
# 
# l2 = nn.Conv2d(32, 64, 3, 1)(l1)
# print(l2.shape) # 5 images * 64 filters * 24 width * 24 height
# 
# l3 = nn.functional.max_pool2d(l2, 2) # kernel_size
# print(l3.shape) # 5 images * 64 filters * 12 width * 12 height
# 
# l3f = torch.flatten(l3, 1)
# print(l3f.shape) # 5 images * 9216
# 
# l4 = nn.Linear(9216, 128)(l3f)
# print(l4.shape) # 5 images * 128
# 
# z = nn.Linear(128, 4)(l4)
# print(z.shape) # 5 images * 4
# 
# # Sanity check that the decoder works
# 
# l1 = nn.Linear(4, 128)(z)
# print(l1.shape) # 5 images * 128
# 
# l2 = nn.Linear(128, 1728)(l1)
# print(l2.shape) # 5 images * 1728
# 
# l2w = torch.reshape(l2, (-1, 12, 12, 12))
# print(l2w.shape) # 5 images * 12 filters * 12 width * 12 height
# 
# l3 = nn.Conv2d(12, 36, 3, 1)(l2w)
# print(l3.shape) # 5 images * 36 filters * 10 width * 10 height
# 
# l3w = torch.reshape(l3, (-1, 4, 30, 30))
# print(l3w.shape) # 5 images * 4 filters * 30 width * 30 height
# 
# l4 = nn.Conv2d(4, 4, 3, 1)(l3w)
# print(l4.shape) # 5 images * 4 filters * 28 width * 28 height
# 
# x = nn.Conv2d(4, 1, 1, 1)(l4)
# print(x.shape) # 5 images * 1 channel * 28 width * 28 height
# 
# # Sanity check that the whole model runs (has the right dimensions etc.)
# 
# img,lbl = mnist[0]
# f = SimpleAE(latent_dim=4)
# x = img.unsqueeze(0)
# z,rx = f(x)
# ```


if __name__ == '__main__':

    # Create a new model object, and initialize its parameters
    fit = SimpleAE(latent_dim=4)

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
    torch.save(fit.state_dict(), 'intro_ae_4d.pt')
