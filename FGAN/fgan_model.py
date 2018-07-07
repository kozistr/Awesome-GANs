import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)


class FGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1,
                 sample_num=8 * 8, sample_size=8,
                 z_dim=128, gf_dim=64, df_dim=64, lr=2e-4):

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
        :param gf_dim: the number of generator filters, default 64
        :param df_dim: the number of discriminator filters, default 64

        # Training Settings
        :param lr: learning rate, default 2e-4
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

        self.gf_dim = gf_dim
        self.df_dim = df_dim

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

        self.bulid_fgan()  # build FGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            for i in range(1, 4):
                x = t.dense(x, self.gf_dim * (2 ** (i - 1)), name='disc-fc-%d' % i)
                x = tf.nn.elu(x)

            x = tf.layers.flatten(x)

            x = t.dense(x, 1, name='disc-fc-1')
            return x

    def generator(self, z, reuse=None, is_train=True):
        with tf.variable_scope('generator', reuse=reuse):
            x = t.dense(z, self.gf_dim, name='gen-fc-1')
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-1')
            x = tf.nn.relu(x)

            x = t.dense(x, self.gf_dim, name='gen-fc-2')
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
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

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
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
