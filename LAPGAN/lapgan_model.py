from __future__ import print_function

import tensorflow as tf
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


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes, y_shapes = x.get_shape(), y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=2e-2, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.variance_scaling_initializer())

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        return conv


def deconv2d(input_, output_shape, k_h=3, k_w=3, d_h=2, d_w=2, stddev=2e-2, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.variance_scaling_initializer())

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        if with_w:
            return deconv, w
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)


def linear(input_, output_size, scope=None, stddev=2e-2, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", shape=[shape[1], output_size],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())

        bias_term = tf.get_variable("Bias", shape=[output_size],
                                    initializer=tf.constant_initializer(bias_start))

        layer = tf.nn.bias_add(tf.matmul(input_, matrix), bias_term)

        if with_w:
            return layer, matrix
        else:
            return layer


class LAPGAN:

    def __init__(self, s, input_height=32, input_width=32,
                 batch_size=64, sample_size=32, sample_num=64,
                 z_dim=100, gf_dim=64, df_dim=64, c_dim=3, eps=1e-12):
        self.s = s
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.input_size = self.input_height * self.input_width
        self.input_channel = c_dim
        self.image_shape = [self.input_height, self.input_width, self.input_channel]

        self.z_dim = z_dim

        self.eps = eps

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # batch normalization
        self.d_bn1 = batch_norm(self.batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(self.batch_size, name='d_bn2')
        self.d_bn3 = batch_norm(self.batch_size, name='d_bn3')

        # self.g_bn0 = batch_norm(self.batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(self.batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(self.batch_size, name='g_bn2')
        self.g_bn3 = batch_norm(self.batch_size, name='g_bn3')

        self.bulid_lapgan()

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):



    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):



    def bulid_lapgan(self):
        # x, z placeholder
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size] + self.image_shape, name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='z-noise')

        # generator
        self.G = self.generator(self.z)

        # discriminator
        self.D = self.discriminator(self.x)

        # discriminate
        self.D_ = self.discriminator(self.G, reuse=True)

        # maximize log(D(G(z)))
        self.g_loss = -tf.reduce_mean(tf.log(self.D_ + self.eps))

        # maximize log(D(x)) + log(1 - D(G(z)))
        self.d_real_loss = -tf.reduce_mean(tf.log(self.D + self.eps))
        self.d_fake_loss = -tf.reduce_mean(tf.log((1. - self.D_) + self.eps))
        self.d_loss = self.d_real_loss + self.d_fake_loss

        # summary
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G_sum = tf.summary.image("G", self.G)  # generated image from G model
        self.D_sum = tf.summary.histogram("D", self.D)
        self.D__sum = tf.summary.histogram("D_", self.D_)

        self.d_real_loss_sum = tf.summary.scalar("d_real_loss", self.d_real_loss)
        self.d_fake_loss_sum = tf.summary.scalar("d_fake_loss", self.d_fake_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # collect trainer values
        vars = tf.trainable_variables()
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

        self.saver = tf.train.Saver()

        # training # optimizer
        beta1 = 0.5
        self.learning_rate = 1e-4

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=beta1).\
            minimize(self.d_loss, var_list=self.d_vars)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=beta1).\
            minimize(self.g_loss, var_list=self.g_vars)

        # merge summary
        self.g_sum = tf.summary.merge([self.z_sum, self.D__sum, self.G_sum, self.d_fake_loss_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.D_sum, self.d_real_loss_sum, self.d_loss_sum])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
