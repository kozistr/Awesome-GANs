from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


tf.set_random_seed(777)
np.random.seed(777)


class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.eps = epsilon
            self.momentum = momentum
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)

            self.name = name

    def __call__(self, x, train=True):
        with tf.variable_scope(self.name) as scope:
            return tf.contrib.layers.batch_norm(x,
                                                decay=self.momentum,
                                                updates_collections=None,
                                                epsilon=self.eps,
                                                scale=True,
                                                is_training=train,
                                                scope=scope)


def lrelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)


class DiscoGAN:

    def __init__(self, s, input_height=64, input_width=64, batch_size=64,
                 sample_size=32, sample_num=64, z_dim=100, gf_dim=64, df_dim=64, c_dim=3,
                 learning_rate=2e-4, beta1=0.5, beta2=0.999, eps=1e-12):

        self.s = s
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = c_dim
        self.image_shape = [self.input_height, self.input_width, self.input_channel]

        self.z_dim = z_dim

        self.eps = eps
        self.mm1 = beta1
        self.mm2 = beta2

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # batch normalization
        self.d_bn1 = batch_norm(self.df_dim * 2, name='d_bn1')
        self.d_bn2 = batch_norm(self.df_dim * 4, name='d_bn2')
        self.d_bn3 = batch_norm(self.df_dim * 8, name='d_bn3')

        self.g_bn1 = batch_norm(self.batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(self.batch_size, name='g_bn2')
        self.g_bn3 = batch_norm(self.batch_size, name='g_bn3')
        self.g_bn4 = batch_norm(self.batch_size, name='g_bn4')
        self.g_bn5 = batch_norm(self.batch_size, name='g_bn5')
        self.g_bn6 = batch_norm(self.batch_size, name='g_bn6')

        self.d_lr, g_lr = learning_rate, learning_rate

        self.build_discogan()

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding="SAME",
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                weights_regularizer=slim.l2_regularizer(2e-4)):
                net = slim.conv2d(x, self.df_dim, 4, 2)
                net = lrelu(net)

                net = slim.conv2d(net, self.df_dim * 2, 4, 2)
                net = self.d_bn1(net)
                net = lrelu(net)

                net = slim.conv2d(net, self.df_dim * 4, 4, 2)
                net = self.d_bn2(net)
                net = lrelu(net)

                net = slim.conv2d(net, self.df_dim * 8, 4, 2)
                net = self.d_bn3(net)
                net = lrelu(net)

                net = slim.conv2d(net, 1, 4, 1)
                net = tf.squeeze(net, squeeze_dims=[1])  # logits

        return tf.nn.sigmoid(net)  # return prob

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding="SAME",
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                weights_regularizer=slim.l2_regularizer(2e-4)):

                pass

    def build_discogan(self):
        # x, z placeholder
        x = tf.placeholder(tf.float32, [-1, self.input_height, self.input_width, self.input_channel], name='x-image')
        z = tf.placeholder(tf.float32, [-1, self.z_dim], name='z-noise')


