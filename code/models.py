from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import time
import os
import datasets
import utils
import pickle
ds = tf.contrib.distributions


class CHyVAE:
    def __init__(self, dataset, z_dim, imsize, channels, batch_size, n_steps, nu, prior_cov, run_no=None):
        self.dataset = datasets.load_dataset(dataset)
        self.z_dim = z_dim
        self.imsize = imsize
        self.channels = channels
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.nu_np = nu
        self.n_fc_units = 128 if dataset in {'2dshapes', 'correlated_ellipses'} else 256
        scale = max(self.nu_np - self.z_dim - 1, 1)
        self.Psi_np = scale * prior_cov
        print((dataset, z_dim, nu, run_no))
        self.results_path = '../results/{:s}_z_{:d}_nu_{:d}_run{:d}/'.format(dataset, z_dim, nu, run_no)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        self._build_model()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _encoder(self, x, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            z = tf.layers.conv2d(x, 32, 4, 2, padding='same', activation=tf.nn.relu)
            z = tf.layers.conv2d(z, 32, 4, 2, padding='same', activation=tf.nn.relu)
            z = tf.layers.conv2d(z, 64, 4, 2, padding='same', activation=tf.nn.relu)
            z = tf.layers.conv2d(z, 64, 4, 2, padding='same', activation=tf.nn.relu)
            z = tf.layers.flatten(z)
            z = tf.layers.dense(z, self.n_fc_units, activation=tf.nn.relu)
            mu = tf.layers.dense(z, self.z_dim, activation=None)
            A = tf.reshape(tf.layers.dense(z, self.z_dim * self.z_dim, activation=None), [-1, self.z_dim, self.z_dim])
            L = tf.matrix_band_part(A, -1, 0)
            diag = tf.nn.softplus(tf.matrix_diag_part(A)) + 1e-4
            L = tf.matrix_set_diag(L, diag)
            L_LT = tf.matmul(L, L, transpose_b=True)
            Sigma = L_LT + 1e-4 * tf.eye(self.z_dim)
            return mu, Sigma

    def _decoder(self, z, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            z = tf.layers.dense(z, self.n_fc_units, activation=tf.nn.relu)
            z = tf.layers.dense(z, 4 * 4 * 64, activation=tf.nn.relu)
            z = tf.reshape(z, [-1, 4, 4, 64])
            z = tf.layers.conv2d_transpose(z, 64, 4, 2, padding='same', activation=tf.nn.relu)
            z = tf.layers.conv2d_transpose(z, 32, 4, 2, padding='same', activation=tf.nn.relu)
            z = tf.layers.conv2d_transpose(z, 32, 4, 2, padding='same', activation=tf.nn.relu)
            x_logits = tf.layers.conv2d_transpose(z, self.channels, 4, 2, padding='same', activation=None)
            return x_logits

    def _regularizer(self, z, mu, Sigma, Psi, nu, B):
        psi_zzT = Psi + tf.matmul(z, z, transpose_b=True)
        mu = tf.expand_dims(mu, -1)
        sigma_mumuT_psi = Sigma + tf.matmul(mu, mu, transpose_b=True) + Psi
        return -\
            0.5 * (nu + 1) * (tf.linalg.logdet(psi_zzT)) +\
            0.5 * tf.linalg.logdet(Sigma) -\
            0.5 * (nu + B) * tf.trace(tf.matmul(sigma_mumuT_psi, tf.matrix_inverse(psi_zzT)))

    def _build_model(self):
        # Model
        self.x = tf.placeholder(tf.float32, [None, self.imsize, self.imsize, self.channels])
        self.Psi = tf.placeholder(tf.float32, [self.z_dim, self.z_dim])
        self.nu = tf.placeholder(tf.float32, ())
        self.mu, Sigma = self._encoder(self.x)
        mvn = ds.MultivariateNormalFullCovariance(loc=self.mu, covariance_matrix=Sigma)
        z = mvn.sample()
        z2 = tf.transpose(mvn.sample(1), perm=[1, 2, 0])
        x_hat_logits = self._decoder(z)
        self.loglikelihood = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=x_hat_logits), [1, 2, 3]))
        self.regularizer = -tf.reduce_mean(self._regularizer(
            z2, self.mu, Sigma, self.Psi, self.nu, 1))
        self.loss = self.loglikelihood + self.regularizer
        self.optim_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # Reconstruction
        self.x_test = tf.placeholder(tf.float32, [None, self.imsize, self.imsize, self.channels])
        z_test = self._encoder(self.x_test, reuse=True)[0]
        self.x_recons = tf.nn.sigmoid(self._decoder(z_test, reuse=True))

        # Generation
        self.noise = tf.placeholder(tf.float32, [None, self.z_dim])
        self.fake_images = tf.nn.sigmoid(self._decoder(self.noise, reuse=True))

    def train(self):
        # self.fixed_idx =  datasets.get_fixed_test_idx(name=self.dataset)
        self.train_batches = utils.batch_generator(self.dataset.imgs, self.batch_size)
        self.test_batches = utils.batch_generator(self.dataset.imgs, batch_size=50)
        train_path = os.path.join(self.results_path, 'train')
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        image_base_name = os.path.join(train_path, '{:s}_step_{:d}.png')
        metrics_history = {'iter': [], 'recons': [], 'disent': []}
        start_time = time.time()
        for stp in range(1, self.n_steps + 1):
            x_np = next(self.train_batches)
            _, loss_np, rec_np, reg_np = self.sess.run([self.optim_op, self.loss, self.loglikelihood, self.regularizer],
                                                       feed_dict={self.x: x_np, self.Psi: self.Psi_np, self.nu: self.nu_np})
            if stp % 1000 == 0 or stp == 1:
                end_time = time.time()
                print('Step: {:d} in {:.2f}s:: Loss: {:.3f} => Recons.: {:.3f}, Reg: {:.3f}'.format(
                    stp, end_time - start_time, loss_np, -rec_np, -reg_np))
                start_time = end_time
                x_test_np = next(self.test_batches)
                x_recons_np = self.sess.run(self.x_recons, feed_dict={self.x_test: x_test_np})
                utils.render_reconstructions(x_test_np, x_recons_np, image_base_name.format('rec', stp))

                z_np = utils.sample_noise(self.Psi_np, self.nu_np, 100)
                x_hat_np = self.sess.run(self.fake_images, feed_dict={self.noise: z_np})
                utils.render_images(x_hat_np, image_base_name.format('iw', stp))
            if stp % 10000 == 0:
                disent_metric = utils.compute_metric(self)[1]
                metrics_history['iter'].append(stp)
                metrics_history['recons'].append(-rec_np)
                metrics_history['disent'].append(disent_metric)
                print('Metric: {:.4f}'.format(disent_metric))

        with open(os.path.join(train_path, 'metrics.pkl'), 'wb') as pkl:
            pickle.dump(metrics_history, pkl)

    def generate(self):
        gen_path = os.path.join(self.results_path, 'gen')
        if not os.path.exists(gen_path):
            os.mkdir(gen_path)
        nu = self.z_dim + 1
        while True:
            z_np = utils.sample_noise(self.Psi_np, self.nu_np, 100)
            x_hat_np = self.sess.run(self.fake_images, feed_dict={self.noise: z_np})
            utils.render_images(x_hat_np, os.path.join(gen_path, 'nu_{:d}.png'.format(nu)))
            if nu >= 2 * self.nu_np:
                break
            nu = nu * 4

        x_test_np = next(self.test_batches)
        x_np = np.vstack([x_test_np, next(self.train_batches), next(self.test_batches)])
        means, = self.sess.run([self.mu], feed_dict={self.x: x_np})
        for num, base_point in enumerate(means):
            n_images_per_latent = 20
            z_np = []
            for i in range(self.z_dim):
                dim_i = np.repeat(base_point[None, :], n_images_per_latent, 0)
                dim_i[np.arange(n_images_per_latent), i] = np.linspace(-3, 3, n_images_per_latent)
                z_np.append(dim_i)
            z_np = np.vstack(z_np)
            x_hat_np = self.sess.run(self.fake_images, feed_dict={self.noise: z_np})
            img_path = os.path.join(gen_path, 'inter_{:d}_{:d}.png'.format(1, num))
            utils.render_images(x_hat_np, img_path, n_rows=self.z_dim, n_cols=n_images_per_latent)
