import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class InfoGAN:

    def __init__(self, s, batch_size=64, height=32, width=32, channel=3,
                 sample_num=10 * 10, sample_size=10,
                 df_dim=64, gf_dim=64, fc_unit=128, n_categories=10, n_continous_factor=10,
                 z_dim=128, g_lr=1e-3, d_lr=2e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 32
        :param width: input image width, default 32
        :param channel: input image channel, default 3 (RGB)

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # Hyper-parameters
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_unit: fully connected unit, default 128

        # Training Option
        :param n_categories: the number of categories, default 10
        :param n_continous_factor: the number of cont factors, default 10
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 1e-3
        :param d_lr: discriminator learning rate, default 2e-4
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        """
        - MNIST
        n_cat : 10, n_cont : 2, z : 62 => embeddings : 10 + 2 + 62 = 74
        - SVHN
        n_cat : 10, n_cont : 4, z : 124 => embeddings : 40 + 124 = 168
        - Celeb-A
        n_cat : 10, n_cont : 10, z : 128 => embeddings : 100 + 128 = 228
        """
        self.n_cat = n_categories         # category dist, label
        self.n_cont = n_continous_factor  # gaussian dist, rotate, etc
        self.z_dim = z_dim
        self.lambda_ = 1.  # sufficient for discrete latent codes # less than 1

        self.beta1 = 0.5
        self.beta2 = 0.999
        self.d_lr = d_lr
        self.g_lr = g_lr

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.

        self.g_loss = 0.
        self.d_loss = 0.
        self.q_loss = 0.

        self.g = None
        self.g_test = None

        self.d_op = None
        self.g_op = None
        self.q_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name="x-image")                                                # (-1, 32, 32, 3)
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_cat], name='y-label')          # (-1, 10)
        self.z_cat = tf.placeholder(tf.float32, shape=[None, self.n_cat], name='z-label')      # (-1, 10)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')          # (-1, 128)
        self.z_con = tf.concat([self.z, self.z_cat], axis=1)

        self.build_infogan()  # build InfoGAN model

    def classifier(self, x, reuse=None):
        """
        :param x: ~ D
        :param reuse: re-usable
        :return: prob, logits
        """
        with tf.variable_scope("classifier", reuse=reuse):
            x = t.dense(x, 128, name='rec-fc-1')
            x = t.batch_norm(x, name='rec-bn-1')
            x = tf.nn.leaky_relu(x, alpha=0.1)

            logits = t.dense(x, self.n_cat * self.n_cont, name='rec-fc-2')
            prob = tf.nn.softmax(logits)

            return prob, logits

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = t.conv2d(x, self.df_dim * 1, 4, 2, name='disc-conv2d-1')
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = t.conv2d(x, self.df_dim * 2, 4, 2, name='disc-conv2d-2')
            x = t.batch_norm(x, name='disc-bn-1')
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = t.conv2d(x, self.df_dim * 4, 4, 2, name='disc-conv2d-3')
            x = t.batch_norm(x, name='disc-bn-2')
            x = tf.nn.leaky_relu(x, alpha=0.1)

            net = tf.layers.flatten(x)

            x = t.dense(net, 1, name='disc-fc-1')

            return x, net

    def generator(self, z, c, reuse=None, is_train=True):
        """
        :param z: 228 z-noise
        :param c: 10 categories * 10 dimensions
        :param reuse: re-usable
        :param is_train: trainable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([z, c], axis=1)  # (-1, 138)
            assert x.get_shape()[-1] == self.n_cat + self.z_dim

            x = t.dense(x, 2 * 2 * 448, name='gen-fc-1')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.relu(x)

            x = tf.reshape(x, (-1, 2, 2, 448))

            x = t.deconv2d(x, self.gf_dim * 4, 4, 2, name='gen-deconv2d-1')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.relu(x)

            x = t.deconv2d(x, self.gf_dim * 2, 4, 2, name='gen-deconv2d-2')
            x = tf.nn.relu(x)

            x = t.deconv2d(x, self.gf_dim * 1, 4, 2, name='gen-deconv2d-3')
            x = tf.nn.relu(x)

            x = t.deconv2d(x, 3, 4, 2, name='gen-deconv2d-2')
            x = tf.nn.tanh(x)
            return x

    def build_infogan(self):
        # Generator
        self.g = self.generator(self.z, self.y)
        self.g_test = self.generator(self.z, self.y, is_train=False)

        # Discriminator
        d_real, d_real_cat = self.discriminator(self.x)
        d_fake, d_fake_cat = self.discriminator(self.g, reuse=True)

        # Classifier
        _, c_real = self.classifier(d_real_cat)
        _, c_fake = self.classifier(d_fake_cat, reuse=True)

        # Losses
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        q_real_loss = t.softce_loss(c_real, self.y)
        q_fake_loss = t.softce_loss(c_fake, self.z_cat)

        self.d_loss += self.lambda_ * q_real_loss
        self.g_loss += self.lambda_ * q_fake_loss

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("loss/q_real_loss", q_real_loss)
        tf.summary.scalar("loss/q_fake_loss", q_fake_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
