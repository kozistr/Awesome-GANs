import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # reproducibility


def conv2d(x, f=64, k=4, d=2, pad='SAME', name='conv2d'):
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
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad, name=name)


def deconv2d(x, f=64, k=4, d=2, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param d: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: decovn2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=d,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad, name=name)


def batch_norm(x, momentum=0.9, eps=1e-9):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         training=True)


class InfoGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1,
                 sample_num=64, sample_size=8, output_height=28, output_width=28,
                 df_dim=64, gf_dim=64, fc_unit=1024,
                 n_categories=10, n_continous_factor=2, z_dim=62,
                 g_lr=1e-3, d_lr=2e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 28
        :param input_width: input image width, default 28
        :param input_channel: input image channel, default 1 (RGB)
        - in case of MNIST, image size is 28x28x1(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 9
        :param sample_size: sample image size, default 8
        :param output_height: output images height, default 28
        :param output_width: output images width, default 28

        # Hyper-parameters
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_unit: fully connected unit, default 1024

        # Training Option
        :param n_categories: the number of categories, default 10 (For MNIST)
        :param n_continous_factor: the number of cont factors, default 2 (For MNIST)
        :param z_dim: z dimension (kinda noise), default 62 (For MNIST)
        :param g_lr: generator learning rate, default 1e-3
        :param d_lr: discriminator learning rate, default 2e-4
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
        self.fc_unit = fc_unit

        """
        - MNIST
        n_cat : 10, n_cont : 2, z : 62 => embeddings : 10 + 2 + 62 = 74
        - SVHN
        n_cat : 40, n_cont : 4, z : 124 => embeddings : 40 + 4 + 124 = 168
        - Celeb-A
        n_cat : 100, n_cont : 0, z : 128 => embeddings : 100 + 0 + 128 = 228
        """
        self.n_cat = n_categories         # category dist, label
        self.n_cont = n_continous_factor  # gaussian dist, rotate, etc
        self.z_dim = z_dim
        self.lambda_ = 1  # sufficient for discrete latent codes # less than 1

        self.beta1 = 0.5
        self.beta2 = 0.999
        self.d_lr = d_lr
        self.g_lr = g_lr

        self.d_real = 0
        self.d_fake = 0

        self.g_loss = 0.
        self.d_loss = 0.
        self.q_loss = 0.

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=self.image_shape, name="x-image")               # (-1, 32, 32, 3)
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='z-noise')  # (-1, 128)
        self.c = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_cat + self.n_cont], name='c')  # (-1, 12)

        self.build_infogan()  # build InfoGAN model

    def classifier(self, x, reuse=None):
        """
        # This is a network architecture for MNIST DataSet referred in the paper
        :param x: ~ D
        :param reuse: re-usable
        :return: prob, logits
        """
        with tf.variable_scope("classifier", reuse=reuse):
            x = tf.layers.dense(x, units=128, name='d-fc-1')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            logits = tf.layers.dense(x, units=self.n_cat + self.n_cont, name='d-fc-2')

            prob = tf.nn.softmax(logits)

            return prob, logits

    def discriminator(self, x, reuse=None):
        """
        # This is a network architecture for MNIST DataSet referred in the paper
        :param x: 28x28x1 images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv2d(x, f=self.df_dim, name='d-conv2d-0')
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = conv2d(x, f=self.df_dim * 2, name='d-conv2d-1')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, units=self.fc_unit, name='d-fc-0')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            logits = tf.layers.dense(x, units=1, name='d-fc-1')
            prob = tf.nn.sigmoid(logits)

            return prob, logits, x

    def generator(self, z, c, reuse=None):
        """
        # This is a network architecture for MNIST DataSet referred in the paper
        :param z: 62 z-noise
        :param c: 10 categories + 2 dimensions
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([z, c], axis=1)  # (-1, 74)

            x = tf.layers.dense(x, units=self.fc_unit, name='g-fc-0')
            x = batch_norm(x)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, units=7 * 7 * self.gf_dim * 2, name='g-fc-1')
            x = batch_norm(x)
            x = tf.nn.relu(x)

            x = tf.reshape(x, shape=[self.batch_size, 7, 7, self.gf_dim * 2])

            x = deconv2d(x, f=self.gf_dim, name='g-conv2d-0')
            x = batch_norm(x)
            x = tf.nn.relu(x)

            x = deconv2d(x, f=1, name='g-conv2d-2')
            # x = tf.nn.tanh(x)
            x = tf.nn.sigmoid(x)  # tanh is used in the paper

            return x

    def build_infogan(self):
        # Generator
        self.g = self.generator(self.z, self.c)

        # Discriminator
        d_real, d_real_logits, _ = self.discriminator(self.x)
        d_fake, d_fake_logits, d_fake_d = self.discriminator(self.g, reuse=True)

        # Classifier
        c_fake, c_fake_logits = self.classifier(d_fake_d)  # Q net

        # Losses
        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits,
                                                                             labels=tf.ones_like(d_real)))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                             labels=tf.zeros_like(d_fake)))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                             labels=tf.ones_like(d_fake)))

        # categorical
        q_cat_logits = c_fake_logits[:, :self.n_cat]
        q_cat_labels = self.c[:, :self.n_cat]

        q_cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=q_cat_logits,
                                                                            labels=q_cat_labels))
        # gaussian
        q_cont_logits = c_fake[:, self.n_cat:]
        q_cont_labels = self.c[:, self.n_cat:]

        q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(q_cont_labels - q_cont_logits), axis=1))  # l2 loss
        self.q_loss = q_cat_loss + q_cont_loss

        # Summary
        tf.summary.histogram("z-noise", self.z)

        tf.summary.image("g", self.g)  # generated images by Generative Model
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("q_loss", self.q_loss)

        # Optimizer
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]
        q_params = [v for v in vars if v.name.startswith('d') or v.name.startswith('g') or v.name.startswith('c')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)
        self.q_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.q_loss, var_list=q_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
