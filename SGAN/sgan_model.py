from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


tf.set_random_seed(777)


def conv2d(x, f=64, k=5, d=2, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param d: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: covn2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=d,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad, name=name)


def deconv2d(x, f=64, k=5, d=2, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param d: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: decovn2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=d,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad, name=name)


def batch_norm(x, momentum=0.9, eps=1e-9):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         training=True)


def gaussian_noise(x, std=5e-2):
    noise = tf.random_normal(x.get_shape(), mean=0., stddev=std, dtype=tf.float32)
    return x + noise


class SGAN:

    def __init__(self, s, batch_size=64, input_height=32, input_width=32, input_channel=3,
                 sample_size=8, sample_num=64,
                 z_dim=(16, 32), gf_dim=256, df_dim=96,
                 eps=1e-12):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 32
        :param input_width: input image width, default 32
        :param input_channel: input image channel, default 3 (RGB)
        - in case of CIFAR, image size is 32x32x3(HWC).

        # Output Settings
        :param sample_size: sample image size, default 8
        :param sample_num: the number of sample images, default 64

        # Model Settings
        :param z_dim: z noise dimension, default 128
        :param gf_dim: the number of generator filters, default 256
        :param df_dim: the number of discriminator filters, default 96

        # Training Settings
        :param eps: epsilon, default 1e-12
        """

        self.s = s
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.n_classes = 10

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.image_shape = [self.input_height, self.input_width, self.input_channel]

        self.z_dim = z_dim

        self.eps = eps

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[-1] + self.image_shape, name='x-images')
        self.z_0 = tf.placeholder(tf.float32, shape=[-1, self.z_dim[0]], name='z-noise-1')
        self.z_1 = tf.placeholder(tf.float32, shape=[-1, self.z_dim[1]], name='z-noise-2')

        # Training Options
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 2e-4

        self.d_loss = 0.
        self.g_loss = 0.

        self.bulid_sgan()  # build SGAN model

    def encoder(self, x, reuse=None):
        with tf.variable_scope('encoder', reuse=reuse):
            for i in range(1, 4):
                x = conv2d(x, self.df_dim * i, name='enc-conv2d-%d' % i)
                x = tf.nn.leaky_relu(x)
                # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='enc-max_pool2d-%d' % i) # for speed

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, units=256, name='enc-fc-1')
            x = tf.nn.leaky_relu(x)

            logits = tf.layers.dense(x, units=self.n_classes, name='enc-fc-2')
            prob = tf.nn.softmax(logits)

            return prob

    def discriminator_0(self, x, reuse=None):
        with tf.variable_scope('discriminator_0', reuse=reuse):
            x = gaussian_noise(x)

            x = conv2d(x, self.df_dim, k=3, d=1, name='d_0-conv2d-1')
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim, k=3, d=2, name='d_0-conv2d-2')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim * 2, k=3, d=2, name='d_0-conv2d-3')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim * 2, k=3, d=2, name='d_0-conv2d-4')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, self.df_dim * 2, name='d-fc-1')

            prob = tf.nn.sigmoid(x)

            return prob

    def generator_0(self, z, h, reuse=None):
        with tf.variable_scope('generator_0', reuse=reuse):
            for i in range(1, 3):
                z = tf.layers.dense(z, self.gf_dim // 2, name='g_0-fc-%d' % i)
                z = tf.nn.leaky_relu(z)

                z = batch_norm(z)

            # z : (batch_size, 128)
            # h : (batch_size, 128)
            # x : (batch_size, 256)
            x = tf.concat([h, z], axis=1)

            x = tf.layers.dense(x, self.gf_dim * 4 * 4, name='g_0-fc-3')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 4, 4, self.gf_dim])

            for i in range(1, 3):
                f = self.gf_dim // i
                x = deconv2d(x, f, name='g_0-deconv2d-%d' % i)
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x)

            logits = deconv2d(x, self.input_channel, name='g_0-deconv2d-3')  # 32x32x3
            prob = tf.nn.sigmoid(logits)

            return prob

    def bulid_sgan(self):
        def log(x):
            return tf.log(x + self.eps)

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = -tf.reduce_mean(log(d_real))
        d_fake_loss = -tf.reduce_mean(log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = -tf.reduce_mean(log(d_fake))

        # Summary
        tf.summary.histogram("z", self.z)
        tf.summary.image("g", self.g)  # generated image from G model
        tf.summary.histogram("d_real", d_real)
        tf.summary.histogram("d_fake", d_fake)

        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Collect trainer values
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]

        # Optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
