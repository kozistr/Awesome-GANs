import tensorflow as tf
import numpy as np
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class BEGAN:

    def __init__(self, s, batch_size=16, height=64, width=64, channel=3,
                 sample_num=8 * 8, sample_size=8,
                 df_dim=64, gf_dim=64, gamma=0.5, lambda_k=1e-3, z_dim=256, g_lr=2e-4, d_lr=2e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 16
        :param height: image height, default 64
        :param width: image width, default 64
        :param channel  image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 32x32x3/64x64x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 64

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
        :param gamma: gamma value, default 0.4
        :param lambda_k: lr adjustment value lambda k, default 1e-3
        :param z_dim: z dimension (kinda noise), default 256
        :param g_lr: generator learning rate, default 1e-4
        :param d_lr: discriminator learning rate, default 1e-4
        :param epsilon: epsilon, default 1e-9
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

        self.gamma = gamma  # 0.3 ~ 0.7
        self.lambda_k = lambda_k
        self.z_dim = z_dim
        self.beta1 = .5
        self.beta2 = .9
        self.d_lr = tf.Variable(d_lr, name='d_lr')
        self.g_lr = tf.Variable(g_lr, name='g_lr')
        self.lr_decay_rate = .5
        self.lr_low_boundary = 1e-5
        self.eps = epsilon

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.g_loss = 0.
        self.d_loss = 0.
        self.m_global = 0.
        self.balance = 0.

        self.g = None
        self.g_test = None

        self.k_update = None
        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # LR/k update
        self.k = tf.Variable(0., trainable=False, name='k_t')  # 0 < k_t < 1, k_0 = 0

        self.lr_update_step = 100000
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * self.lr_decay_rate, self.lr_low_boundary))
        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * self.lr_decay_rate, self.lr_low_boundary))

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name="x-image")  # (-1, 32 or 64, 32 or 64, 3)
        self.z = tf.placeholder(tf.float32,
                                shape=[None, self.z_dim],
                                name='z-noise')  # (-1, 128)

        self.build_began()  # build BEGAN model

    def encoder(self, x, reuse=None):
        """
        :param x: Input images (32x32x3 or 64x64x3)
        :param reuse: re-usable
        :return: embeddings
        """
        with tf.variable_scope('encoder', reuse=reuse):
            repeat = int(np.log2(self.height)) - 2

            x = t.conv2d(x, f=self.df_dim, name="enc-conv-0")
            x = tf.nn.elu(x)

            for i in range(1, repeat + 1):
                f = self.df_dim * i

                x = t.conv2d(x, f=f, name="enc-conv-%d" % (i * 2 - 1))
                x = tf.nn.elu(x)
                x = t.conv2d(x, f=f, name="enc-conv-%d" % (i * 2))
                x = tf.nn.elu(x)

                if i < repeat:
                    """
                        You can choose one of them. max-pool or avg-pool or conv-pool.
                        Speed Order : conv-pool > avg-pool > max-pool. i guess :)
                    """
                    # x = tf.layers.max_pooling2d(x, 2, 2)
                    x = t.conv2d(x, f=f, s=2, name='enc-conv-pool-%d' % i)  # conv pooling
                    x = tf.nn.elu(x)
                    # x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME', name="enc-subsample-%d" % i)

            x = tf.layers.flatten(x)

            z = t.dense(x, self.z_dim, name='enc-fc-1')  # normally, (-1, 128)

            return z

    def decoder(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope('decoder', reuse=reuse):
            repeat = int(np.log2(self.height)) - 2

            x = t.dense(z, self.z_dim * 8 * 8, name='dec-fc-1')
            x = tf.reshape(x, [-1, 8, 8, self.z_dim])

            for i in range(1, repeat + 1):
                x = t.conv2d(x, f=self.gf_dim, name="dec-conv-%d" % (i * 2 - 1))
                x = tf.nn.elu(x)
                x = t.conv2d(x, f=self.gf_dim, name="dec-conv-%d" % (i * 2))
                x = tf.nn.elu(x)

                if i < repeat:
                    x = t.resize_nn(x, x.get_shape().as_list()[1] * 2)  # NN up-sampling

            x = t.conv2d(x, f=self.channel)

            return x

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            z = self.encoder(x, reuse=reuse)
            x = self.decoder(z, reuse=reuse)

            return x

    def generator(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("generator", reuse=reuse):
            repeat = int(np.log2(self.height)) - 2

            x = t.dense(z, self.z_dim * 8 * 8, name='g-fc-1')
            x = tf.nn.elu(x)

            x = tf.reshape(x, [-1, 8, 8, self.z_dim])

            for i in range(1, repeat + 1):
                x = t.conv2d(x, f=self.gf_dim, name="g-conv-%d" % (i * 2 - 1))
                x = tf.nn.elu(x)
                x = t.conv2d(x, f=self.gf_dim, name="g-conv-%d" % (i * 2))
                x = tf.nn.elu(x)

                if i < repeat:
                    x = t.resize_nn(x, x.get_shape().as_list()[1] * 2)  # NN up-sampling

            x = t.conv2d(x, f=self.channel)

            return x

    def build_began(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Loss
        d_real_loss = t.l1_loss(self.x, d_real)
        d_fake_loss = t.l1_loss(self.g, d_fake)
        self.d_loss = d_real_loss - self.k * d_fake_loss
        self.g_loss = d_fake_loss

        # Convergence Metric
        self.balance = self.gamma * d_real_loss - self.g_loss  # (=d_fake_loss)
        self.m_global = d_real_loss + tf.abs(self.balance)

        # k_t update
        self.k_update = tf.assign(self.k,  tf.clip_by_value(self.k + self.lambda_k * self.balance, 0, 1))

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("misc/balance", self.balance)
        tf.summary.scalar("misc/m_global", self.m_global)
        tf.summary.scalar("misc/k_t", self.k)
        tf.summary.scalar("misc/d_lr", self.d_lr)
        tf.summary.scalar("misc/g_lr", self.g_lr)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr_update,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr_update,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
