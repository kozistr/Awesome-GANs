import numpy as np
import tensorflow as tf

import awesome_gans.modules as t

np.random.seed(777)
tf.set_random_seed(777)  # reproducibility


class BigGAN:
    def __init__(
        self,
        s,
        batch_size=64,
        height=128,
        width=128,
        channel=3,
        n_classes=10,
        sample_num=10 * 10,
        sample_size=10,
        df_dim=64,
        gf_dim=64,
        fc_unit=512,
        z_dim=128,
        lr=1e-4,
    ):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: image height, default 128
        :param width: image width, default 128
        :param channel: image channel, default 3
        :param n_classes: number of classes, default 10

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # For Model
        :param df_dim: discriminator conv filter, default 64
        :param gf_dim: generator conv filter, default 64
        :param fc_unit: the number of fully connected layer units, default 512

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param lr: learning rate, default 1e-4
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]
        self.n_classes = n_classes

        assert self.height == self.width
        assert self.height in [128, 256, 512]

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.up_sampling = True

        self.gain = 2 ** 0.5

        self.z_dim = z_dim
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lr = lr

        self.res_block_disc = None
        self.res_block_gen = None
        if self.height == 128:
            self.res_block_disc = ([16, 8, 4, 2], [1])
            self.res_block_gen = ([1], [2, 4, 8, 16, 16])
        elif self.height == 256:
            self.res_block_disc = ([16, 8, 8, 4, 2], [1])
            self.res_block_gen = ([1, 2], [4, 8, 8, 16, 16])
        elif self.height == 512:
            self.res_block_disc = ([16, 8, 8, 4], [2, 1, 1])
            self.res_block_gen = ([1, 1, 2], [4, 8, 8, 16, 16])
        else:
            raise NotImplementedError

        # pre-defined
        self.g_loss = 0.0
        self.d_loss = 0.0
        self.c_loss = 0.0

        self.inception_score = None
        self.fid_score = None

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(
            tf.float32, shape=[None, self.height, self.width, self.channel], name="x-image"
        )  # (bs, 128, 128, 3)
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="y-label")  # (bs, n_classes)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="z-noise")  # (-1, 128)

        self.build_sagan()  # build BigGAN model

    @staticmethod
    def res_block(x, f, scale_type, use_bn=True, name=""):
        with tf.variable_scope("res_block-%s" % name):
            assert scale_type in ["up", "down"]
            scale_up = False if scale_type == "down" else True

            ssc = x

            x = t.batch_norm(x, name="bn-1") if use_bn else x
            x = tf.nn.relu(x)
            x = t.conv2d_alt(x, f, sn=True, name="conv2d-1")

            x = t.batch_norm(x, name="bn-2") if use_bn else x
            x = tf.nn.relu(x)

            if not scale_up:
                x = t.conv2d_alt(x, f, sn=True, name="conv2d-2")
                x = tf.layers.average_pooling2d(x, pool_size=(2, 2))
            else:
                x = t.deconv2d_alt(x, f, sn=True, name="up-sampling")

            return x + ssc

    @staticmethod
    def self_attention(x, f_, reuse=None):
        with tf.variable_scope("attention", reuse=reuse):
            f = t.conv2d_alt(x, f_ // 8, k=1, s=1, sn=True, name='attention-conv2d-f')
            g = t.conv2d_alt(x, f_ // 8, k=1, s=1, sn=True, name='attention-conv2d-g')
            h = t.conv2d_alt(x, f_, k=1, s=1, sn=True, name='attention-conv2d-h')

            f, g, h = t.hw_flatten(f), t.hw_flatten(g), t.hw_flatten(h)

            s = tf.matmul(g, f, transpose_b=True)
            attention_map = tf.nn.softmax(s, axis=-1, name='attention_map')

            o = tf.reshape(tf.matmul(attention_map, h), shape=x.get_shape())
            gamma = tf.get_variable('gamma', shape=[1], initializer=tf.zeros_initializer())

            x = gamma * o + x
            return x

    @staticmethod
    def non_local_block(x, f, sub_sampling=False, name="nonlocal"):
        """ non-local block, https://arxiv.org/pdf/1711.07971.pdf """
        with tf.variable_scope("non_local_block-%s" % name):
            with tf.name_scope("theta"):
                theta = t.conv2d(x, f=f, k=1, s=1, name="theta")
                if sub_sampling:
                    theta = tf.layers.max_pooling2d(theta, pool_size=(2, 2), name="max_pool-theta")
                theta = tf.reshape(theta, (-1, theta.get_shape().as_list()[-1]))

            with tf.name_scope("phi"):
                phi = t.conv2d(x, f=f, k=1, s=1, name="phi")
                if sub_sampling:
                    phi = tf.layers.max_pooling2d(theta, pool_size=(2, 2), name="max_pool-phi")
                phi = tf.reshape(phi, (-1, phi.get_shape().as_list()[-1]))
                phi = tf.transpose(phi, [1, 0])

            with tf.name_scope("g"):
                g = t.conv2d(x, f=f, k=1, s=1, name="g")
                if sub_sampling:
                    g = tf.layers.max_pooling2d(theta, pool_size=(2, 2), name="max_pool-g")
                g = tf.reshape(g, (-1, g.get_shape().as_list()[-1]))

            with tf.name_scope("self-attention"):
                theta_phi = tf.tensordot(theta, phi, axis=-1)
                theta_phi = tf.nn.softmax(theta_phi)

                theta_phi_g = tf.tensordot(theta_phi, g, axis=-1)

            theta_phi_g = t.conv2d(theta_phi_g, f=f, k=1, s=1, name="theta_phi_g")
            return x + theta_phi_g

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: classification, probability (fake or real), network
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            f = 1 * self.channel

            x = self.res_block(x, f=f, scale_type="down", name="disc-res1")

            x = self.self_attention(x, f_=f)

            for i in range(4):
                f *= 2
                x = self.res_block(x, f=f, scale_type="down", name="disc-res%d" % (i + 1))

            x = self.res_block(x, f=f, scale_type="down", use_bn=False, name="disc-res5")
            x = tf.nn.relu(x)

            with tf.name_scope("global_sum_pooling"):
                x_shape = x.get_shape().as_list()
                x = tf.reduce_mean(x, axis=-1) * (x_shape[1] * x_shape[2])

            x = t.dense_alt(x, 1, sn=True, name='disc-dense-last')
            return x

    def generator(self, z, reuse=None):
        """
        :param z: noise
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            # split
            z = tf.split(z, num_or_size_splits=4, axis=-1)  # expected [None, 32] * 4

            # linear projection
            x = t.dense_alt(z, f=4 * 4 * 16 * self.channel, sn=True, use_bias=False, name="gen-dense-1")
            x = tf.nn.relu(x)

            x = tf.reshape(x, (-1, 4, 4, 16 * self.channel))

            res = x

            f = 16 * self.channel
            for i in range(4):
                res = self.res_block(res, f=f, scale_type="up", name="gen-res%d" % (i + 1))
                f //= 2

            x = self.self_attention(res, f_=f * 2)

            x = self.res_block(x, f=1 * self.channel, scale_type="up", name="gen-res4")

            x = t.batch_norm(x, name="gen-bn-last")  # <- noise
            x = tf.nn.relu(x)
            x = t.conv2d_alt(x, f=self.channel, k=3, sn=True, name="gen-conv2d-last")

            x = tf.nn.tanh(x)
            return x

    def build_sagan(self):
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

        self.inception_score = t.inception_score(self.g)

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("metric/inception_score", self.inception_score)
        tf.summary.scalar("metric/fid_score", self.fid_score)

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
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
