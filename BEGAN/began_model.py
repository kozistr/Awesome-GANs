import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # reproducibility


def conv2d(x, f=64, k=3, d=1, act=tf.nn.elu, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param d: strides, default 2
    :param act: activation function, default elu
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: covn2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=d,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.zeros_initializer(),
                            activation=act,
                            padding=pad, name=name)


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


class BEGAN:

    def __init__(self, s, batch_size=64, input_height=64, input_width=64, input_channel=3,
                 sample_num=16 * 16, sample_size=16, output_height=64, output_width=64,
                 df_dim=64, gf_dim=64,
                 gamma=0.5, lambda_k=1e-3, z_dim=128, g_lr=1e-4, d_lr=1e-4, epsilon=1e-12):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 64
        :param input_width: input image width, default 64
        :param input_channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 32x32x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 256
        :param sample_size: sample image size, default 16
        :param output_height: output images height, default 64
        :param output_width: output images width, default 64

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
        :param gamma: gamma value, default 0.4
        :param lambda_k: lr adjustment value lambda k, default 1e-3
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 1e-4
        :param d_lr: discriminator learning rate, default 1e-4
        :param epsilon: epsilon, default 1e-12
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.gamma = gamma  # 0.3 ~ 0.7
        self.lambda_k = lambda_k
        self.z_dim = z_dim
        self.beta1 = .5
        self.beta2 = .999
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

        self.k_update = None
        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # LR/k update
        self.k = tf.Variable(0., trainable=False, name='k_t')  # 0 < k_t < 1, k_0 = 0

        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * self.lr_decay_rate, self.lr_low_boundary))
        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * self.lr_decay_rate, self.lr_low_boundary))

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.input_height, self.input_width, self.input_channel],
                                name="x-image")                                        # (-1, 32, 32, 3)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')  # (-1, 128)

        self.build_began()  # build BEGAN model

    def encoder(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: embeddings
        """
        with tf.variable_scope('encoder', reuse=reuse):
            repeat = int(np.log2(self.input_height)) - 2

            x = conv2d(x, f=self.df_dim, name="enc-conv-0")

            for i in range(1, repeat + 1):
                f = self.df_dim * i

                x = conv2d(x, f=f, name="enc-conv-%d" % (i * 2 - 1))
                x = conv2d(x, f=f, name="enc-conv-%d" % (i * 2))

                if i < repeat:
                    # x = tf.layers.max_pooling2d(x, 2, 2)
                    x = conv2d(x, f=f, d=2, name='enc-conv-pool-%d' % i)  # conv pooling
                    # x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME', name="enc-subsample-%d" % i)

            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, units=self.z_dim * 8 * 8, name='enc-fc-1')

            return x

    def decoder(self, x, reuse=None):
        """
        :param x: embeddings
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope('decoder', reuse=reuse):
            repeat = int(np.log2(self.input_height)) - 2

            # x = tf.layers.dense(x, units=self.z_dim * 8 * 8, name='dec-fc-1')
            x = tf.reshape(x, [-1, 8, 8, self.z_dim])

            for i in range(1, repeat + 1):
                x = conv2d(x, f=self.gf_dim, name="dec-conv-%d" % (i * 2 - 1))
                x = conv2d(x, f=self.gf_dim, name="dec-conv-%d" % (i * 2))

                if i < repeat:
                    x = resize_nn(x, x.get_shape().as_list()[1] * 2)  # NN up-sampling

            x = conv2d(x, self.input_channel)

            return x

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = self.encoder(x, reuse=reuse)
            x = self.decoder(x, reuse=reuse)

            return x

    def generator(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("generator", reuse=reuse):
            repeat = int(np.log2(self.input_height)) - 2

            x = tf.layers.dense(z, units=self.z_dim * 8 * 8, activation=tf.nn.elu, name='g-fc-1')
            x = tf.reshape(x, [-1, 8, 8, self.z_dim])

            for i in range(1, repeat + 1):
                x = conv2d(x, f=self.gf_dim, name="g-conv-%d" % (i * 2 - 1))
                x = conv2d(x, f=self.gf_dim, name="g-conv-%d" % (i * 2))

                if i < repeat:
                    x = resize_nn(x, x.get_shape().as_list()[1] * 2)  # NN up-sampling

            x = conv2d(x, 3)

            return x

    def build_began(self):
        def l1_loss(x, y):
            return tf.reduce_mean(tf.abs(x - y))

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Loss
        d_real_loss = l1_loss(self.x, d_real)
        d_fake_loss = l1_loss(self.g, d_fake)
        self.d_loss = d_real_loss - self.k * d_fake_loss
        self.g_loss = d_fake_loss

        # Convergence Metric
        self.balance = self.gamma * d_real_loss - self.g_loss  # (=d_fake_loss)
        self.m_global = d_real_loss + tf.abs(self.balance)

        # k_t update
        self.k_update = tf.assign(self.k,  tf.clip_by_value(self.k + self.lambda_k * self.balance, 0, 1))

        # Summary
        tf.summary.histogram("z-noise", self.z)

        # tf.summary.image("g", self.g)  # generated images by Generative Model
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("balance", self.balance)
        tf.summary.scalar("m_global", self.m_global)
        tf.summary.scalar("k_t", self.k)

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
