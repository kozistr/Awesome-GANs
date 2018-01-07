import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # reproducibility


def conv2d(x, f=64, k=3, s=1, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 1
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: conv2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad,
                            name=name)


def batch_norm(x, momentum=0.9, eps=1e-5):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         training=True)


class DualGAN:

    def __init__(self, s, batch_size=64, input_height=64, input_width=64, input_channel=3,
                 sample_num=16 * 16, sample_size=16, output_height=64, output_width=64,
                 df_dim=64, gf_dim=64,
                 lambda_a=20., lambda_b=20., z_dim=128, g_lr=1e-4, d_lr=1e-4, epsilon=1e-12):

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
        :param lambda_a: weight of A recovery loss, default 20
        :param lambda_a: weight of B recovery loss, default 20
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

        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.z_dim = z_dim
        self.decay = .9
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.eps = epsilon

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.g_loss = 0.
        self.d_loss = 0.

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x_A = tf.placeholder(tf.float32,
                                  shape=[None, self.input_height, self.input_width, self.input_channel],
                                  name="x-image-A")
        self.x_B = tf.placeholder(tf.float32,
                                  shape=[None, self.input_height, self.input_width, self.input_channel],
                                  name="x-image-B")
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')  # (-1, 128)

        self.build_dualgan()  # build DualGAN model

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv2d(x, self.df_dim * 1, name='d-conv2d-1')
            x = tf.nn.leaky_relu(x)

            for i in range(1, 3):
                x = conv2d(x, self.df_dim * (2 ** i), name='d-conv2d-%d' % (i + 1))
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x)

            x = conv2d(x, 1, s=1, name='d-conv2d-4')

            return x

    def generator(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("generator", reuse=reuse):


            return x

    def build_dualgan(self):
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
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = d_fake_loss

        # Summary
        tf.summary.histogram("z-noise", self.z)

        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.RMSPropOptimizer(learning_rate=self.d_lr,
                                              decay=self.decay).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.RMSPropOptimizer(learning_rate=self.g_lr,
                                              decay=self.decay).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
