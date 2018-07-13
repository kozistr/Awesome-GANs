import tensorflow as tf
import numpy as np

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)


class FGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1,
                 sample_num=8 * 8, sample_size=8,
                 z_dim=128, dfc_unit=256, gfc_unit=1024, lr=2e-4,
                 divergence_method='KL'):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1

        # Output Settings
        :param sample_num: the number of sample images, default 64
        :param sample_size: sample image size, default 8

        # Model Settings
        :param z_dim: z noise dimension, default 128
        :param dfc_unit: the number of fully connected units used at disc, default 256
        :param gfc_unit: the number of fully connected units used at gen, default 1024

        # Training Settings
        :param lr: learning rate, default 2e-4
        :param divergence_method: the method of f-divergences, default 'KL'
        """

        self.s = s
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.image_shape = [self.height, self.width, self.channel]
        self.n_input = self.height * self.width * self.channel

        self.z_dim = z_dim

        self.dfc_unit = dfc_unit
        self.gfc_unit = gfc_unit

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')

        # Training Options
        self.beta1 = 0.5
        self.lr = lr

        self.divergence = divergence_method

        self.bulid_fgan()  # build f-GAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = t.dense(x, self.dfc_unit, name='disc-fc-1')
            x = tf.nn.elu(x)

            x = t.dense(x, self.dfc_unit, name='disc-fc-2')
            x = tf.nn.elu(x)

            x = tf.layers.flatten(x)

            x = t.dense(x, 1, name='disc-fc-3')
            return x

    def generator(self, z, reuse=None, is_train=True):
        with tf.variable_scope('generator', reuse=reuse):
            x = t.dense(z, self.gfc_unit, name='gen-fc-1')
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-1')
            x = tf.nn.relu(x)

            x = t.dense(x, self.gfc_unit, name='gen-fc-2')
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-2')
            x = tf.nn.relu(x)

            x = t.dense(x, self.n_input, name='gen-fc-3')
            x = tf.nn.sigmoid(x)
            return x

    def bulid_fgan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Losses
        if self.divergence == 'GAN':
            def activation(x): return -tf.reduce_mean(-t.safe_log(1. + tf.exp(-x)))

            def conjugate(x): return -tf.reduce_mean(-t.safe_log(1. - tf.exp(x)))
        elif self.divergence == 'KL':  # tf.distribution.kl_divergence
            def activation(x): return -tf.reduce_mean(x)

            def conjugate(x): return -tf.reduce_mean(tf.exp(x - 1.))
        elif self.divergence == 'Reverse-KL':
            def activation(x): return -tf.reduce_mean(-tf.exp(x))

            def conjugate(x): return -tf.reduce_mean(-1. - -x)  # remove log
        elif self.divergence == 'JS':
            def activation(x): return -tf.reduce_mean(tf.log(2.) - t.safe_log(1. + tf.exp(-x)))

            def conjugate(x): return -tf.reduce_mean(-t.safe_log(2. - tf.exp(x)))
        elif self.divergence == 'JS-Weighted':
            def activation(x): return -tf.reduce_mean(-np.pi * np.log(np.pi) - t.safe_log(1. + tf.exp(-x)))

            def conjugate(x): return -tf.reduce_mean((1. - np.pi) *
                                                     t.safe_log((1. - np.pi) / (1. - np.pi * tf.exp(x / np.pi))))
        elif self.divergence == 'Squared-Hellinger':
            def activation(x): return -tf.reduce_mean(1. - tf.exp(x))

            def conjugate(x): return -tf.reduce_mean(x / (1. - x))
        elif self.divergence == 'Pearson':
            def activation(x): return -tf.reduce_mean(x)

            def conjugate(x): return -tf.reduce_mean(tf.square(x) / 4. + x)
        elif self.divergence == 'Neyman':
            def activation(x): return -tf.reduce_mean(1. - tf.exp(x))

            def conjugate(x): return -tf.reduce_mean(2. - 2. * tf.sqrt(1. - x))
        elif self.divergence == 'Jeffrey':
            from scipy.special import lambertw

            def activation(x): return -tf.reduce_mean(x)

            def conjugate(x):
                lambert_w = lambertw(self.s.run(tf.exp(1. - x)))  # need to be replaced with another tensor func
                return -tf.reduce_mean(lambert_w + 1. / lambert_w + x - 2.)
        elif self.divergence == 'Total-Variation':
            def activation(x): return -tf.reduce_mean(tf.nn.tanh(x) / 2.)

            def conjugate(x): return -tf.reduce_mean(x)
        else:
            raise NotImplementedError("[-] Not Implemented f-divergence %s" % self.divergence)

        d_real_loss = activation(d_real)
        d_fake_loss = conjugate(d_fake)
        self.d_loss = d_real_loss - d_fake_loss
        self.g_loss = activation(d_fake)

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Collect trainer values
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        # Optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/%s/' % self.divergence, self.s.graph)
