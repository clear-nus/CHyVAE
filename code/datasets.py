from __future__ import print_function
from __future__ import division
import glob
import os
import numpy as np
import synthetic

DATA_ROOT = '../data'


def load_dataset(name, size=64):
    if name == 'celeba':
        raise NotImplementedError('To be implemented')
    elif name == '3dfaces':
        raise NotImplementedError('To be implemented')
    elif name == '3dchairs':
        raise NotImplementedError('To be implemented')
    elif name == '2dshapes':
        return d2DShapes()
    elif name == 'correlated_ellipses':
        return CorrelatedEllipses()
    else:
        raise ValueError('Error: Dataset not supported')


class d2DShapes:
    def __init__(self, include_discrete=True):
        self.name = '2dshapes'
        self.n_factors = 5 if include_discrete else 4
        self.diff = 1 if include_discrete else 2
        train_data_file = os.path.join(DATA_ROOT, '2dshapes', 'dsprites.npz')
        dataset_zip = np.load(train_data_file, encoding='latin1')
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate(
            (self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def sample_fixed_factor(self, size=20):
        if self.n_factors == 5:
            factor = np.random.randint(5) + 1
        else:
            factor = np.random.randint(4) + 2
        latents_sampled = self.sample_latent(size=size)
        latents_sampled[:, factor] = latents_sampled[0, factor]
        idx = self.latent_to_index(latents_sampled)
        xi = self.imgs[idx].reshape(size, 64, 64, 1)
        return xi, factor


class CorrelatedEllipses:
    def __init__(self):
        self.name = 'correlated_ellipses'
        self.n_factors = 4
        self.diff = 0
        train_data_file = os.path.join(
            DATA_ROOT, 'corr-ell', 'correlated_ellipses.npz')
        dataset_zip = np.load(train_data_file)
        self.imgs = dataset_zip['dataset'] / 255.

    def sample_fixed_factor(self, size=20):
        factor = np.random.randint(4)
        imgs = synthetic.conditional_sample(factor, size=size, normalize=True)
        xi = imgs.reshape(size, 64, 64, 1)
        return xi, factor
