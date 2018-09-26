from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import numpy.random as npr
import tensorflow as tf
from numpy.linalg import inv, cholesky
from scipy.stats import chi2

# Training Utils


def batch_generator(X, batch_size=100):
    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    while True:
        permutation = np.random.permutation(X.shape[0])
        for b in range(n_batches):
            batch_idx = permutation[b * batch_size:(b + 1) * batch_size]
            batch = X[batch_idx]
            if batch.shape[0] is not batch_size:
                continue
            if len(batch.shape) == 3:
                batch = batch[:, :, :, None]
            yield batch

# ImageIO Utils


def render_reconstructions(x, x_hat, path):
    imsize, channels = x.shape[1], x.shape[3]
    x = np.reshape(x[:50], [5, 10, imsize, imsize, channels])
    x_hat = np.reshape(x_hat[:50], [5, 10, imsize, imsize, channels])
    canvas = np.zeros([imsize * 10, imsize * 10, channels])
    for i in range(5):
        canvas[2 * i * imsize:(2 * i + 1) * imsize, :, :] = np.transpose(np.transpose(
            x[i], [0, 2, 1, 3]).reshape([imsize * 10, imsize, channels]), [1, 0, 2])
        canvas[(2 * i + 1) * imsize:(2 * i + 2) * imsize, :, :] = np.transpose(np.transpose(
            x_hat[i], [0, 2, 1, 3]).reshape([imsize * 10, imsize, channels]), [1, 0, 2])
    _save_image(canvas, path)


def render_images(np_x, path, n_rows=10, n_cols=10):
    imsize, channels = np_x.shape[1], np_x.shape[3]
    np_x = np_x.reshape((n_rows, n_cols, imsize, imsize, channels))
    np_x = np.concatenate(np.split(np_x, n_rows, axis=0), axis=3)
    np_x = np.concatenate(np.split(np_x, n_cols, axis=1), axis=2)
    x_img = np.squeeze(np_x)
    _save_image(x_img, path)


def _save_image(img, name):
    img = (img * 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)


# Sampling Utils

def invwishartrand(nu, phi):
    return inv(_wishartrand(nu, inv(phi)))


def _wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    foo = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                foo[i, j] = np.sqrt(chi2.rvs(nu - (i + 1) + 1))
            else:
                foo[i, j] = npr.normal(0, 1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


def sample_noise(Psi, nu, size):
    p = Psi.shape[0]
    covs = [invwishartrand(nu, Psi) for i in range(10)]
    samples = [npr.multivariate_normal(
        np.zeros(p), covs[i], size // 10) for i in range(10)]
    return np.concatenate(samples, axis=0)


# TF Utils

def multivariate_digamma(a, p):
    a_reshaped = tf.expand_dims(a, -1)
    diff_np = (1 - np.arange(1, p + 1)) / 2
    diff = tf.constant(diff_np[None, None, :].astype(np.float32))
    return tf.reduce_sum(tf.digamma(a_reshaped + diff), -1)


def multivariate_lgamma(a, p):
    a_reshaped = tf.expand_dims(a, -1)
    diff_np = (1 - np.arange(1, p + 1)) / 2
    diff = tf.constant(diff_np[None, None, :].astype(np.float32))
    return 0.5 * p * tf.log(np.pi) + tf.reduce_sum(tf.lgamma(a_reshaped + diff), -1)

# Metric Utils


def _sample_var_batch(model, scale, B=20):
    bX, by = [], []
    for b in range(B):
        xi, factor = model.dataset.sample_fixed_factor(size=200)
        zi = model.sess.run(model.mu, feed_dict={model.x: xi})
        zi /= scale
        D = np.argmin(np.var(zi, axis=0, ddof=1))
        f = factor - model.dataset.diff
        bX.append(D)
        by.append(f)
    return np.asarray(bX), np.asarray(by)


def _compute_std(model, iters=100):
    std_batch = []
    for i in range(1, iters + 1):
        bX = next(model.test_batches).reshape(50, 64, 64, 1)
        mu_np = model.sess.run(model.mu, feed_dict={model.x: bX})
        std_batch.append(mu_np)
    scale = np.std(np.vstack(std_batch), 0)
    return scale


def compute_metric(model):
    scale = _compute_std(model)
    bX, by = _sample_var_batch(model, scale, B=800)
    V = np.zeros([model.z_dim, model.dataset.n_factors])
    for bx, by in zip(bX, by):
        V[bx, by] += 1
    tX, ty = _sample_var_batch(model, scale, B=800)
    preds = [np.argmax(V[tx]) for tx in tX]
    return V, np.mean(preds == ty)
