from __future__ import print_function
from __future__ import division
import glob
import os
import numpy as np
import synthetic

DATA_ROOT = '../data'


def load_dataset(name, size=64):
    if name == 'celeba':
        return load_celeba(size)
    elif name == '3dfaces':
        return load_3dfaces(size)
    elif name == '3dchairs':
        return load_3dchairs(size)
    elif name == 'mnist':
        return load_mnist(size)
    elif name == 'churches':
        return load_churches(size)
    elif name == '2dshapes':
        return d2DShapes()
    elif name == 'correlated_ellipses':
        return CorrelatedEllipses()
    else:
        print('Error: Dataset not supported')


def get_fixed_test_idx(name='celeba'):
    if name == 'celeba':
        return [8207, 5426, 1850, 7598, 19606, 6180, 17274, 7413, 10771,
                2585, 6070, 16872, 12040, 7537, 15483, 10771, 9744, 13340,
                2385, 13803, 13634, 14049, 8215, 4662, 15689, 4404, 9913,
                6008, 15367, 5324, 10226, 19551, 17235, 16796, 19063, 15948,
                5251, 3360, 19648, 7864, 9942, 3008, 10487, 8448, 12925,
                9099, 6647, 11790, 13131, 12177]
    elif name == '3dfaces':
        return [3603, 11442, 20232, 36429, 94724, 81726, 91719, 9493, 61921,
                20820, 14391, 45172, 56290, 14156, 61776, 23067, 42869, 47156,
                52274, 64100, 6454, 89297, 49043, 81803, 58686, 51309, 2654,
                32220, 66605, 34983, 47579, 24478, 30877, 88138, 77061, 72960,
                25245, 90074, 48003, 6535, 27132, 54468, 48693, 33808, 4976,
                44336, 64571, 61399, 96972, 77233]
    elif name == '3dchairs':
        return [6781, 629, 6113, 5376, 3996, 6966, 7086, 3277, 4564, 5800, 1628,
                1857, 5757, 534, 7393, 6883, 198, 369, 2490, 3832, 6868, 5887,
                7429, 607, 2117, 6229, 7011, 6349, 3084, 5209, 4743, 7732, 4890,
                1558, 5880, 5996, 3958, 7426, 4256, 1499, 4570, 4405, 630, 5374,
                3002, 5512, 558, 3720, 7567, 2133]
    else:
        return []


def load_celeba(size):
    train_data_file = os.path.join(DATA_ROOT, 'CelebA_train_X.npy')
    test_data_file = os.path.join(DATA_ROOT, 'CelebA_test_X.npy')
    train_X = np.load(train_data_file)
    scale = np.max(train_X)
    train_X = train_X / scale
    test_X = np.load(test_data_file)
    test_X = test_X / scale
    imsize = train_X.shape[1]
    if imsize != size:
        print('Scaling')
        train_X = _resize_images(train_X, size)
        test_X = _resize_images(test_X, size)
    return train_X, test_X


def load_3dfaces(size):
    train_files = glob.glob(os.path.join(DATA_ROOT, '3dfaces_train*.npy'))
    train_X = [np.load(trf) for trf in train_files]
    train_X = np.vstack(train_X)[:, :, :, None]
    test_files = glob.glob(os.path.join(DATA_ROOT, '3dfaces_test*.npy'))
    test_X = [np.load(tef) for tef in test_files]
    test_X = np.vstack(test_X)[:, :, :, None]
    imsize = train_X.shape[1]
    if imsize != size:
        print('Scaling')
        train_X = _resize_images(train_X, size)
        test_X = _resize_images(test_X, size)
    return train_X, test_X


def load_3dchairs(size):
    train_data_file = os.path.join(DATA_ROOT, '3dchairs_X.npy')
    train_X = np.load(train_data_file)
    train_X = train_X / np.max(train_X)
    upto = int(0.1 * train_X.shape[0])
    test_X = train_X[:upto]
    train_X = train_X[upto:]
    imsize = train_X.shape[1]
    if imsize != size:
        print('Scaling')
        train_X = _resize_images(train_X, size)
        test_X = _resize_images(test_X, size)
    return train_X, test_X


def load_churches(size):
    train_data_file = os.path.join(DATA_ROOT, 'churches_X.npy')
    train_X = np.load(train_data_file)
    train_X = train_X / np.max(train_X)
    upto = int(0.1 * train_X.shape[0])
    test_X = train_X[:upto]
    train_X = train_X[upto:]
    imsize = train_X.shape[1]
    if imsize != size:
        print('Scaling')
        train_X = _resize_images(train_X, size)
        test_X = _resize_images(test_X, size)
    return train_X, test_X


def load_mnist(size):
    train_data_file = os.path.join(DATA_ROOT, 'mnist_train_X.npy')
    test_data_file = os.path.join(DATA_ROOT, 'mnist_test_X.npy')
    train_X = np.load(train_data_file)
    test_X = np.load(test_data_file)
    imsize = train_X.shape[1]
    if imsize != size:
        print('Scaling')
        train_X = _resize_images(train_X, size)
        test_X = _resize_images(test_X, size)
    return train_X, test_X


def load_2dshapes(size):
    train_data_file = os.path.join(DATA_ROOT, 'dsprites_X.npz')
    dataset_zip = np.load(train_data_file, encoding='bytes')
    imgs = dataset_zip['imgs']
    train_X = imgs[:, :, :, None]
    upto = int(0.1 * train_X.shape[0])
    test_X = train_X[:upto]
    train_X = train_X[upto:]
    imsize = train_X.shape[1]
    if imsize != size:
        print('Scaling')
        train_X = _resize_images(train_X, size)
        test_X = _resize_images(test_X, size)
    return train_X, test_X


def _resize_images(X, imsize):
    from skimage.transform import resize
    X_resized = np.zeros([X.shape[0], imsize, imsize, X.shape[-1]])
    for i, x in enumerate(X):
        X_resized[i] = resize(x, (imsize, imsize))
    return X_resized


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
        dataset_zip = np.load(train_data_file) / 255.
        self.imgs = dataset_zip['dataset']

    def sample_fixed_factor(self, size=20):
        factor = np.random.randint(4)
        imgs = synthetic.conditional_sample(factor, size=size, normalize=True)
        xi = imgs.reshape(size, 64, 64, 1)
        return xi, factor
