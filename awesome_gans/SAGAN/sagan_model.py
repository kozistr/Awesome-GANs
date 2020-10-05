import numpy as np
import tensorflow as tf

import awesome_gans.tfutil as t
from awesome_gans.config import get_config

cfg, _ = get_config()

np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)  # reproducibility


class SAGAN:
    def __init__(
        self,
        s,
        batch_size=64,
        height=64,
        width=64,
        channel=3,
        n_classes=10,
        sample_num=10 * 10,
        sample_size=10,
        df_dim=64,
        gf_dim=64,
        z_dim=128,
        lr=1e-4,
        use_gp=False,
        use_hinge_loss=True,
        graph_path="./model",
    ):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: image height, default 64
        :param width: image width, default 64
        :param channel: image channel, default 3
        :param n_classes: DataSet's classes, default 10

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # For Model
        :param df_dim: discriminator conv filter, default 64
        :param gf_dim: generator conv filter, default 64

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param lr: learning rate, default 1e-4
        :param use_gp: using gradient penalty, default False
        :param use_hinge_loss: using hinge loss, default True

        # Etc
        :param graph_path: path to save graph file, default "./model"
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]
        self.n_classes = n_classes

        self.n_layer = int(np.log2(self.height)) - 2  # 5
        assert self.height == self.width

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.up_sampling = True

        self.z_dim = z_dim
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lr = lr

        self.gp = 0.0
        self.lambda_ = 10.0  # for gradient penalty

        self.graph_path = graph_path

        # pre-defined
        self.g_loss = 0.0
        self.d_loss = 0.0
        self.c_loss = 0.0

        self.g = None
        self.g_test = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        self.use_gp = use_gp
        self.use_hinge_loss = use_hinge_loss

        # Placeholders
        self.x = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.height, self.width, self.channel], name="x-image"
        )  # (64, 64, 64, 3)
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name="z-noise")  # (-1, 128)
        self.z_test = tf.placeholder(tf.float32, shape=[self.sample_num, self.z_dim], name="z-test-noise")  # (-1, 128)

        self.build_sagan()  # build SAGAN model

    @staticmethod
    def attention(x, f_, reuse=None):
        with tf.variable_scope("attention", reuse=reuse):
            f = t.conv2d_alt(x, f_ // 8, 1, 1, sn=True, name='attention-conv2d-f')
            g = t.conv2d_alt(x, f_ // 8, 1, 1, sn=True, name='attention-conv2d-g')
            h = t.conv2d_alt(x, f_, 1, 1, sn=True, name='attention-conv2d-h')

            f, g, h = t.hw_flatten(f), t.hw_flatten(g), t.hw_flatten(h)

            s = tf.matmul(g, f, transpose_b=True)
            attention_map = tf.nn.softmax(s, axis=-1, name='attention_map')

            o = tf.reshape(tf.matmul(attention_map, h), shape=x.get_shape())
            gamma = tf.get_variable('gamma', shape=[1], initializer=tf.zeros_initializer())

            x = gamma * o + x
            return x

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param y: labels
        :param reuse: re-usable
        :return: classification, probability (fake or real), network
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            f = self.gf_dim

            x = t.conv2d_alt(x, f, 4, 2, pad=1, sn=True, name='disc-conv2d-1')
            x = tf.nn.leaky_relu(x, alpha=0.1)

            for i in range(self.n_layer // 2):
                x = t.conv2d_alt(x, f * 2, 4, 2, pad=1, sn=True, name='disc-conv2d-%d' % (i + 2))
                x = tf.nn.leaky_relu(x, alpha=0.1)

                f *= 2

            # Self-Attention Layer
            x = self.attention(x, f, reuse=reuse)

            for i in range(self.n_layer // 2, self.n_layer):
                x = t.conv2d_alt(x, f * 2, 4, 2, pad=1, sn=True, name='disc-conv2d-%d' % (i + 2))
                x = tf.nn.leaky_relu(x, alpha=0.1)

                f *= 2

            x = t.flatten(x)

            x = t.dense_alt(x, 1, sn=True, name='disc-fc-1')
            return x

    def generator(self, z, reuse=None, is_train=True):
        """
        :param z: noise
        :param y: image label
        :param reuse: re-usable
        :param is_train: trainable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            f = self.gf_dim * 8

            x = t.dense_alt(z, 4 * 4 * f, sn=True, name='gen-fc-1')

            x = tf.reshape(x, (-1, 4, 4, f))

            for i in range(self.n_layer // 2):
                if self.up_sampling:
                    x = t.up_sampling(x, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    x = t.conv2d_alt(x, f // 2, 5, 1, pad=2, sn=True, use_bias=False, name='gen-conv2d-%d' % (i + 1))
                else:
                    x = t.deconv2d_alt(x, f // 2, 4, 2, sn=True, use_bias=False, name='gen-deconv2d-%d' % (i + 1))

                x = t.batch_norm(x, is_train=is_train, name='gen-bn-%d' % i)
                x = tf.nn.relu(x)

                f //= 2

            # Self-Attention Layer
            x = self.attention(x, f, reuse=reuse)

            for i in range(self.n_layer // 2, self.n_layer):
                if self.up_sampling:
                    x = t.up_sampling(x, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    x = t.conv2d_alt(x, f // 2, 5, 1, pad=2, sn=True, use_bias=False, name='gen-conv2d-%d' % (i + 1))
                else:
                    x = t.deconv2d_alt(x, f // 2, 4, 2, sn=True, use_bias=False, name='gen-deconv2d-%d' % (i + 1))

                x = t.batch_norm(x, is_train=is_train, name='gen-bn-%d' % i)
                x = tf.nn.relu(x)

                f //= 2

            x = t.conv2d_alt(x, self.channel, 5, 1, pad=2, sn=True, name='gen-conv2d-%d' % (self.n_layer + 1))
            x = tf.nn.tanh(x)
            return x

    def build_sagan(self):
        # Generator
        self.g = self.generator(self.z)
        self.g_test = self.generator(self.z_test, reuse=True)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Losses
        if self.use_hinge_loss:
            d_real_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real))
            d_fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
            self.d_loss = d_real_loss + d_fake_loss
            self.g_loss = -tf.reduce_mean(d_fake)
        else:
            d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
            d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
            self.d_loss = d_real_loss + d_fake_loss
            self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        # gradient-penalty
        if self.use_gp:
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0, name='alpha')
            interp = alpha * self.x + (1.0 - alpha) * self.g
            d_interp = self.discriminator(interp, reuse=True)
            gradients = tf.gradients(d_interp, interp)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            self.gp = tf.reduce_mean(tf.square(slopes - 1.0))

            # Update D loss
            self.d_loss += self.lambda_ * self.gp

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        if self.use_gp:
            tf.summary.scalar("misc/gp", self.gp)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(self.lr * 4, beta1=self.beta1, beta2=self.beta2).minimize(
            self.d_loss, var_list=d_params
        )
        self.g_op = tf.train.AdamOptimizer(self.lr * 1, beta1=self.beta1, beta2=self.beta2).minimize(
            self.g_loss, var_list=g_params
        )

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.graph_path, self.s.graph)
