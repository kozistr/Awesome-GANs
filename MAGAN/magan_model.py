import tensorflow as tf

import sys

from adamax import AdamaxOptimizer

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class MAGAN:

    def __init__(self, s, batch_size=64, height=64, width=64, channel=3, n_classes=41,
                 sample_num=10 * 10, sample_size=10,
                 df_dim=64, gf_dim=64, fc_unit=512, z_dim=350, lr=5e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 64
        :param width: input image width, default 64
        :param channel: input image channel, default 3
        :param n_classes: input DataSet's classes, default 41

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # For CNN model
        :param df_dim: discriminator conv2d filter, default 64
        :param gf_dim: generator conv2d filter, default 64
        :param fc_unit: the number of fully connected filters, default 512

        # Training Option
        :param z_dim: z dimension (kinda noise), default 350
        :param lr: generator learning rate, default 5e-4
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.lr = lr
        self.pt_lambda = 0.1

        # pre-defined
        self.g_loss = 0.
        self.d_loss = 0.
        self.d_real_loss = 0.
        self.d_fake_loss = 0.

        self.g = None
        self.g_test = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name="x-image")
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')
        self.m = tf.placeholder(tf.float32, name='margin')

        self.build_magan()  # build MAGAN model

    def encoder(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope('encoder', reuse=reuse):
            for i in range(1, 5):
                x = t.conv2d(x, self.df_dim * (2 ** (i - 1)), 4, 2, name='enc-conv2d-%d' % i)
                if i > 1:
                    x = t.batch_norm(x, name='enc-bn-%d' % (i - 1))
                x = tf.nn.leaky_relu(x)

            return x

    def decoder(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope('decoder', reuse=reuse):
            x = z
            for i in range(1, 4):
                x = t.deconv2d(x, self.df_dim * 8 // (2 ** i), 4, 2, name='dec-deconv2d-%d' % i)
                x = t.batch_norm(x, name='dec-bn-%d' % i)
                x = tf.nn.leaky_relu(x)

            x = t.deconv2d(x, self.channel, 4, 2, name='enc-deconv2d-4')
            x = tf.nn.tanh(x)
            return x

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: prob, embeddings, gen-ed_image
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            embeddings = self.encoder(x, reuse=reuse)
            decoded = self.decoder(embeddings, reuse=reuse)

            return embeddings, decoded

    def generator(self, z, reuse=None, is_train=True):
        """
        :param z: embeddings
        :param reuse: re-usable
        :param is_train: trainable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.reshape(z, (-1, 1, 1, self.z_dim))

            x = t.deconv2d(x, self.df_dim * 8, 4, 1, name='gen-deconv2d-1')
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-1')
            x = tf.nn.relu(x)

            for i in range(1, 4):
                x = t.deconv2d(x, self.df_dim * 8 // (2 ** i), 4, 2, name='gen-deconv2d-%d' % (i + 1))
                x = t.batch_norm(x, is_train=is_train, name='gen-bn-%d' % (i + 1))
                x = tf.nn.relu(x)

            x = t.deconv2d(x, self.channel, 4, 2, name='gen-deconv2d-5')
            x = tf.nn.tanh(x)
            return x

    def build_magan(self):
        # Generator
        self.g = self.generator(self.z)
        self.g_test = self.generator(self.z, reuse=True, is_train=False)

        # Discriminator
        _, d_real = self.discriminator(self.x)
        _, d_fake = self.discriminator(self.g, reuse=True)

        self.d_real_loss = t.mse_loss(self.x, d_real, self.batch_size)
        self.d_fake_loss = t.mse_loss(self.g, d_fake, self.batch_size)
        self.d_loss = self.d_real_loss + tf.maximum(0., self.m - self.d_fake_loss)
        self.g_loss = self.d_fake_loss

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/d_real_loss", self.d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", self.d_fake_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = AdamaxOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.d_loss,
                                                                                      var_list=d_params)
        self.g_op = AdamaxOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.g_loss,
                                                                                      var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
