import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # reproducibility


def instance_normalize(x, eps=1e-9):
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(x - mean, tf.sqrt(var + eps))


def conv2d(x, f=64, k=3, d=1, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param d: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: covn2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=d,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            use_bias=False,
                            padding=pad, name=name)


def residual_block(x, f):
    with tf.variable_scope("residual_block"):
        x = conv2d(x, f)
        x = instance_normalize(x)
        x = tf.nn.leaky_relu(x)

        x = conv2d(x, f)
        x = instance_normalize(x)

        return x


class StarGAN:

    def __init__(self, s, batch_size=64, input_height=32, input_width=32, input_channel=3,
                 sample_num=64, sample_size=8, output_height=32, output_width=32,
                 df_dim=64, gf_dim=64,
                 gamma=0.4, lambda_k=1e-3, z_dim=128, g_lr=0.0001, d_lr=0.0001, epsilon=1e-12):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 32
        :param input_width: input image width, default 32
        :param input_channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 32x32x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 9
        :param sample_size: sample image size, default 32
        :param output_height: output images height, default 32
        :param output_width: output images width, default 32

        # Hyper Parameters
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
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
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.d_lr = tf.Variable(d_lr, name='d_lr')
        self.g_lr = tf.Variable(g_lr, name='g_lr')
        self.lr_decay_rate = 0.5
        self.lr_low_boundary = 1e-5
        self.eps = epsilon

        self.d_real = 0
        self.d_fake = 0

        self.g_loss = 0.
        self.d_loss = 0.
        self.m_global = 0.
        self.balance = 0.

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=self.image_shape, name="x-image")               # (-1, 32, 32, 3)
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='z-noise')  # (-1, 128)

        self.build_stargan()  # build StarGAN model

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            # ...

            return x

    def generator(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = conv2d(z, k=7)
            x = instance_normalize(x)
            x = tf.nn.leaky_relu(x)

            # ...

            return x

    def build_stargan(self):
        def l1_loss(x, y):
            return tf.reduce_mean(tf.abs(x - y))

        self.k = tf.Variable(0., trainable=False, name='k_t')  # 0 < k_t < 1, k_0 = 0

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Loss
        d_real_loss = l1_loss(self.x, d_real)
        d_fake_loss = l1_loss(self.g, d_fake)
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = d_fake_loss

        # Summary
        tf.summary.histogram("z-noise", self.z)

        tf.summary.image("g", self.g)  # generated images by Generative Model
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Optimizer
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
