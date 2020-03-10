import numpy as np
import torch
import torchvision
import sklearn.decomposition
import pandas as pd
from sklearn.metrics import mutual_info_score
from typing import Iterable

from utils import interrupted, enumerate_cycle
from utils import mnist, mnist_batched
from intro_ae import SimpleAE
from pca_ae import PCA_AE, Reshaper, reconstruct
from nd_ae import ND_AE
from b_vae import B_VAE

## Code for computing mutual information gap (MIG) disentanglement metric
# Parts are adapted from Morpho-MNIST (Castro et al., 2019) GitHub code

def _discrete_mutual_info(x, y):
    return np.array([[mutual_info_score(yj, xi) for yj in y] for xi in x])


def _discrete_entropy(x):
    return np.array([mutual_info_score(xi, xi) for xi in x])


def _is_float(x):
    return np.issubsctype(x, np.floating)


def _discretize(x, discretize, bins):
    x = np.asarray(x)
    if discretize is None:
        discretize = _is_float(x)
    if not isinstance(discretize, Iterable):
        discretize = [discretize] * x.shape[1]
    if x.shape[1] != len(discretize):
        raise ValueError(f"Expected 1 or {x.shape[1]} discretization flags, got {len(discretize)}")
    return [np.digitize(xi, np.histogram_bin_edges(xi, bins)[:-1]) if disc else xi
            for xi, disc in zip(x.T, discretize)]


def compute_mig(codes: np.ndarray, factors: np.ndarray, discretize_codes=None, discretize_factors=None,
        bins=20):
    """Mutual information gap score (MIG) [1]_.
    Parameters
    ----------
    codes : (N, C) array_like
        Latent representations inferred by a model.
    factors : (N, F) array_like
        Generative factor annotations.
    discretize_codes : bool or sequence of bool, optional
        Whether the given `codes` should be discretized. If None (default), float inputs will be
        discretized.
    discretize_factors : bool or sequence of bool, optional
        Whether the given `factors` should be discretized. If None (default), float inputs will be
        discretized.
    bins : int or sequence of scalars or str, optional
        Argument to the discretization function (`np.histogram_bin_edges`): number of bins,
        sequence of bin edges, or name of the method to compute optimal bin width. Is ignored if
        neither `codes` nor `factors` need discretizing.
    Returns
    -------
    mig_score : float
        The computed MIG score.
    mi : (C, F) np.ndarray
        The mutual information matrix.
    entropy : (F,) np.ndarray
        The entropies for each generative factor.
    See Also
    --------
    np.histogram_bin_edges
    References
    ----------
    .. [1] Chen, T. Q., Li, X., Grosse, T. B.. & Duvenaud, D. K. (2018). Isolating Sources of
       Disentanglement in Variational Autoencoders. In Advances in Neural Information Processing
       Systems 31 (NeurIPS 2018), pp. 2610-2620.
    """
    codes = _discretize(codes, discretize_codes, bins)
    factors = _discretize(factors, discretize_factors, bins)

    mi = _discrete_mutual_info(codes, factors)
    entropy = _discrete_entropy(factors)
    sorted_mi = np.sort(mi, axis=0)[::-1]
    return np.mean((sorted_mi[0] - sorted_mi[1]) / entropy), mi, entropy


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
mnist_data = torchvision.datasets.MNIST(root = 'data/', download = True, train = True, transform = transform)
dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=len(mnist_data))
data = next(iter(dataloader))[0].numpy()
data = np.reshape(data, (data.shape[0], data.shape[2] * data.shape[3]))

#Choose model here
pca = sklearn.decomposition.PCA(n_components = 8)
pca_result = pca.fit_transform(data)

# fit = SimpleAE(latent_dim=8)
# fit.load_state_dict(torch.load('models/intro_ae_8d.pt'))

# models = []
# for i in range(1, 9):
#     fit = PCA_AE(latent_dim = i)
#     fit = torch.load('models/pca_ae_8d_' + str(i) + '.pt')
#     models.append(fit)

# fit = ND_AE(latent_dim=8)
# fit.load_state_dict(torch.load('models/nd_ae_8d.pt'))
#
# fit = B_VAE(latent_dim=8)
# fit.load_state_dict(torch.load('models/b_vae_8d.pt'))

# morphometric table taken from https://github.com/dccastro/Morpho-MNIST
metrics = pd.read_csv("train-morpho.csv")
cols = ['length', 'thickness', 'slant', 'width', 'height']
factors = metrics[cols].values

codes = torch.empty(0,8)
iter_training_data = enumerate_cycle(mnist_batched)
batch_num = 0
while batch_num < 11999:
    (epoch, batch_num), (imgs,lbls) = next(iter_training_data)
    #z,rx = fit(imgs)
    #z, rx = reconstruct(imgs, models)
    #rx, mu, logvar, z = fit(imgs)
    imgs = imgs.detach().numpy()
    imgs = np.reshape(imgs, (5, 784))
    z = torch.from_numpy(pca.transform(imgs)).float()
    codes = torch.cat([codes, z], 0)
    if batch_num %100 == 0:
      print("batch", batch_num)
codes = codes.detach().numpy()

mig = compute_mig(codes, factors)
print(mig)