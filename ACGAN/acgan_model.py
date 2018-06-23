import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class ACGAN:

    def __init__(self, s, batch_size=100, height=32, width=32, channel=3, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 df_dim=16, gf_dim=384, z_dim=128, lr=2e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 100
        :param height: image height, default 32
        :param width: image width, default 32
        :param channel: image channel, default 3
        :param n_classes: DataSet's classes

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # For CNN model
        :param df_dim: discriminator filter, default 16
        :param gf_dim: generator filter, default 384

        # Training Option
        :param z_dim: z dimension (kinda noise), default 100
        :param lr: learning rate, default 2e-4
        :param epsilon: epsilon, default 1e-9
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

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = lr
        self.eps = epsilon

        # pre-defined
        self.g_loss = 0.
        self.d_loss = 0.
        self.c_loss = 0.
        
        self.g = None
        self.g_test = None
        
        self.d_op = None
        self.g_op = None
        self.c_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name="x-image")  # (-1, 32, 32, 3)
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="y-label")  # (-1, 10)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="z-noise")      # (-1, 128)

        self.build_acgan()  # build ACGAN model

    def discriminator(self, x, reuse=None):
        """
        # Following a D Network, CiFar-like-hood, referred in the paper
        :param x: image
        :param reuse: re-usable
        :return: classification, probability (fake or real), network
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = t.conv2d(x, self.df_dim, 3, 2, name='disc-conv2d-1')
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, 0.5, name='disc-dropout2d-1')

            for i in range(5):
                x = t.conv2d(x, self.df_dim * (2 ** (i + 1)), k=3, s=(i % 2 + 1), name='disc-conv2d-%d' % (i + 2))
                x = t.batch_norm(x, reuse=reuse, name="disc-bn-%d" % (i + 1))
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.layers.dropout(x, 0.5, name='disc-dropout2d-%d' % (i + 1))

            x = tf.layers.flatten(x)

            x = t.dense(x, self.n_classes + 1, name='disc-fc-1')

            logit = x[:, self.n_classes:]
            cat = x[:, :self.n_classes]

            return cat, logit

    def generator(self, z, y, reuse=None, is_train=True):
        """
        # Following a G Network, CiFar-like-hood, referred in the paper
        :param z: noise
        :param y: image label
        :param reuse: re-usable
        :param is_train: trainable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([z, y], axis=1)  # 128 + 10

            # x = tf.reshape(x, (-1, 1, 1, x.get_shape()[-1]))  # (batch_size, 1, 1, fm_size)

            x = t.dense(x, self.gf_dim, name='gen-fc-1')
            x = tf.nn.relu(x)

            x = tf.reshape(x, (-1, 4, 4, 24))

            for i in range(1, 3):
                x = t.deconv2d(x, self.gf_dim // (2 ** i), 5, 2, name='gen-deconv2d-%d' % (i + 1))
                x = t.batch_norm(x, is_train=is_train, reuse=reuse, name="gen-bn-%d" % i)
                x = tf.nn.relu(x)

            x = t.deconv2d(x, self.channel, 5, 2, name='gen-deconv2d-4')
            x = tf.nn.tanh(x)  # scaling to [-1, 1]

            return x

    def build_acgan(self):
        # Generator
        self.g = self.generator(self.z, self.y)
        self.g_test = self.generator(self.z, self.y, reuse=True, is_train=False)

        # Discriminator
        c_real, d_real = self.discriminator(self.x)
        c_fake, d_fake = self.discriminator(self.g, reuse=True)

        # sigmoid ce loss
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = (d_real_loss + d_fake_loss) / 2.
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        # sparse softmax ce loss
        c_real_loss = t.softce_loss(c_real, self.y)
        c_fake_loss = t.softce_loss(c_fake, self.y)
        self.c_loss = (c_real_loss + c_fake_loss) / 2.

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/c_real_loss", c_real_loss)
        tf.summary.scalar("loss/c_fake_loss", c_fake_loss)
        tf.summary.scalar("loss/c_loss", self.c_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]
        c_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)
        self.c_op = tf.train.AdamOptimizer(self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.c_loss, var_list=c_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
