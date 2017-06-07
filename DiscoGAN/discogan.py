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
                 sample_size=32, sample_num=64, z_dim=100, gf_dim=32, df_dim=32, c_dim=3,
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
        self.d_bn4 = batch_norm(self.df_dim * 16, name='d_bn4')
        self.d_bn = [self.d_bn1, self.d_bn2, self.d_bn3, self.d_bn4]

        self.g_bn1 = batch_norm(self.gf_dim * 2, name='g_bn1')
        self.g_bn2 = batch_norm(self.gf_dim * 4, name='g_bn2')
        self.g_bn3 = batch_norm(self.gf_dim * 8, name='g_bn3')
        self.g_bn4 = batch_norm(self.gf_dim * 16, name='g_bn4')
        self.g_bn5 = batch_norm(self.gf_dim * 8, name='g_bn5')
        self.g_bn6 = batch_norm(self.gf_dim * 4, name='g_bn6')
        self.g_bn7 = batch_norm(self.gf_dim * 2, name='g_bn7')
        self.g_bn_1 = [self.g_bn1, self.g_bn2, self.g_bn3]
        self.g_bn_2 = [self.g_bn4, self.g_bn5, self.g_bn6, self.g_bn7]

        self.lr = learning_rate

        self.build_discogan()

    def discriminator(self, x, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                weights_regularizer=slim.l2_regularizer(2e-4)):
                with slim.arg_scope([slim.conv2d], padding="SAME", stride=2, kernel_size=4):
                    net = slim.conv2d(x, self.df_dim)
                    net = lrelu(net)

                    mul = 2
                    for bn in self.d_bn:
                        net = slim.conv2d(net, self.df_dim * mul)
                        net = bn(net)
                        net = lrelu(net)
                        mul *= 2

                net = tf.reshape(net, shape=[-1, 2*2*512])
                net = slim.fully_connected(net, 512, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
                net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
            return net  # return prob

    def generator(self, x, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], padding="SAME", kernel_size=4,
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                weights_regularizer=slim.l2_regularizer(2e-4)):
                with slim.arg_scope([slim.conv2d], stride=2):
                    net = slim.conv2d(x, self.gf_dim)
                    net = lrelu(net)

                    mul = 2
                    for bn in self.g_bn_1:
                        net = slim.conv2d(net, self.gf_dim * mul)
                        net = bn(net)
                        net = lrelu(net)
                        mul *= 2

                with slim.arg_scope([slim.conv2d], stride=1, activation_fn=tf.nn.relu):
                    hw, hidden = 8, 128
                    for bn in self.g_bn_2:
                        net = slim.conv2d(net, self.gf_dim * mul)
                        net = bn(net)

                        net = tf.reshape(net, shape=[-1, hw, hw, hidden])

                        mul = int(mul / 2)
                        hw *= 2
                        hidden = int(hidden / 2)

                    net = slim.conv2d(net, 3)
        return net  # logits

    def build_discogan(self):
        # x, z placeholder
        self.shoes_x = tf.placeholder(tf.float32, [None,
                                                   self.input_height,
                                                   self.input_width,
                                                   self.input_channel], name='x-shoes_image')  # -1x64x64x3
        self.bags_x = tf.placeholder(tf.float32, [None,
                                                  self.input_height,
                                                  self.input_width,
                                                  self.input_channel], name='x-bags_image')  # -1x64x64x3

        # generator
        # s : shoes, b : bags, 2 : to
        self.G_s2b = self.generator(self.shoes_x, "generator_s2b")
        self.G_b2s = self.generator(self.bags_x, "generator_b2s")

        self.G_s2b2s = self.generator(self.G_s2b, "generator_s2b", reuse=True)
        self.G_b2s2b = self.generator(self.G_b2s, "generator_b2s", reuse=True)

        # discriminator
        self.D_s_real = self.discriminator(self.shoes_x, "discriminator_real")
        self.D_b_real = self.discriminator(self.bags_x, "discriminator_fake")

        self.D_s_fake = self.discriminator(self.G_s2b2s, "discriminator_real", reuse=True)
        self.D_b_fake = self.discriminator(self.G_b2s2b, "discriminator_fake", reuse=True)

        # loss
        self.s_loss = tf.reduce_sum(tf.losses.mean_squared_error(self.shoes_x, self.G_s2b2s))
        self.b_loss = tf.reduce_sum(tf.losses.mean_squared_error(self.bags_x, self.G_b2s2b))

        # self.g_shoes_loss = tf.reduce_sum(tf.square(self.D_s_fake - 1)) / 2
        # self.g_bags_loss = tf.reduce_sum(tf.square(self.D_b_fake - 1)) / 2

        # self.d_shoes_real_loss = tf.reduce_sum(tf.square(self.D_s_real - 1)) / 2
        # self.d_shoes_fake_loss = tf.reduce_sum(tf.square(self.D_s_fake)) / 2
        # self.d_bags_real_loss = tf.reduce_sum(tf.square(self.D_b_real - 1)) / 2
        # self.d_bags_fake_loss = tf.reduce_sum(tf.square(self.D_b_fake)) / 2

        # self.d_shoes_loss = self.d_shoes_real_loss + self.d_shoes_fake_loss
        # self.d_bags_loss = self.d_bags_real_loss + self.d_bags_fake_loss

        # self.g_loss = 10 * (self.s_loss + self.b_loss) + self.g_shoes_loss + self.g_bags_loss
        # self.d_loss = self.d_shoes_loss + self.d_bags_loss
        # sigmoid cross entropy loss
        self.g_s_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_s_fake,
                                                                               labels=tf.ones_like(self.D_s_fake)))
        self.g_b_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_b_fake,
                                                                               labels=tf.ones_like(self.D_b_fake)))

        self.d_s_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_s_real,
                                                                                    labels=tf.ones_like(self.D_s_real)))
        self.d_b_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_b_real,
                                                                                    labels=tf.ones_like(self.D_b_real)))
        self.d_s_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_s_fake,
                                                                                    labels=tf.zeros_like(self.D_s_fake)))
        self.d_b_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_b_fake,
                                                                                    labels=tf.zeros_like(self.D_b_fake)))

        self.g_loss = (self.s_loss + self.g_s_loss) + (self.b_loss + self.g_b_loss)

        self.d_s_loss = self.d_s_real_loss + self.d_s_fake_loss
        self.d_b_loss = self.d_b_real_loss + self.d_b_fake_loss
        self.d_loss = self.d_s_loss + self.d_b_loss

        # collect trainer values
        vars = tf.trainable_variables()
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_*")
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator_*")

        # summary
        self.g_shoes_sum = tf.summary.histogram("G_shoes", self.g_s_loss)
        self.g_bags_sum = tf.summary.histogram("G_bags", self.g_b_loss)

        self.d_shoes_sum = tf.summary.histogram("D_shoes", self.d_s_loss)
        self.d_bags_sum = tf.summary.histogram("D_bags", self.d_b_loss)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.saver = tf.train.Saver()

        # train op
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.mm1, beta2=self.mm2).\
            minimize(self.g_loss, var_list=self.g_vars)
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.mm1, beta2=self.mm2).\
            minimize(self.d_loss, var_list=self.d_vars)

        # merge summary
        self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_shoes_sum, self.g_bags_sum])
        self.d_sum = tf.summary.merge([self.d_loss_sum, self.d_shoes_sum, self.d_bags_sum])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
