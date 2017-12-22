from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


tf.set_random_seed(777)


class BatchNorm(object):

    def __init__(self, epsilon=1e-6, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.eps = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.layers.batch_normalization(inputs=x,
                                             momentum=self.momentum,
                                             epsilon=self.eps,
                                             scale=True,
                                             training=train)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, name="Conv2D"):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', shape=[k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable('biases', [output_dim],
                            initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv


def deconv2d(input_, output_shape, k_h=3, k_w=3, d_h=2, d_w=2, name="DeConv2D"):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', shape=[k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable('biases', [output_shape[-1]],
                            initializer=tf.constant_initializer(0.))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        return deconv


def linear(input_, output_size, scope=None, bias_start=0.):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("matrix", shape=[shape[1], output_size],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())

        bias_term = tf.get_variable("bias", shape=[output_size],
                                    initializer=tf.constant_initializer(bias_start))

        layer = tf.nn.bias_add(tf.matmul(input_, matrix), bias_term)

        return layer


class WGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1, n_classes=10,
                 sample_num=64, sample_size=8, n_input=784,
                 z_dim=128, gf_dim=64, df_dim=64, epsilon=1e-12,
                 enable_bn=False, enable_selu=False, enable_adam=False, enable_gp=False):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
        :param input_height: input image height, default 28
        :param input_width: input image width, default 28
        :param input_channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input dataset's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # For model
        :param n_input: input image size, default 784(28x28)

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param gf_dim: the number of generator filters, default 64
        :param df_dim: the number of discriminator filters, default 64
        :param epsilon: epsilon, default 1e-9
        :param enable_bn: enabling batch normalization, default False
        :param enable_selu: enabling selu activation func, default False
        :param enable_adam: enabling adam optimizer, default False
        :param enable_gp: enabling gradient penaly, default False
        """

        self.s = s
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.n_classes = n_classes

        self.sample_size = sample_size
        self.sample_num = sample_num
        self.n_input = n_input

        self.image_shape = [self.input_height, self.input_width, self.input_channel]

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.eps = epsilon

        # Custom Batch Normalization
        self.d_bn1 = BatchNorm(self.batch_size, name='d_bn1')
        self.d_bn2 = BatchNorm(self.batch_size, name='d_bn2')
        self.d_bn3 = BatchNorm(self.batch_size, name='d_bn3')

        self.g_bn1 = BatchNorm(self.batch_size, name='g_bn1')
        self.g_bn2 = BatchNorm(self.batch_size, name='g_bn2')
        self.g_bn3 = BatchNorm(self.batch_size, name='g_bn3')

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size] + self.image_shape, name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='z-noise')

        # Training Options - based on the WGAN paper
        self.learning_rate = 5e-5  # very slow
        self.critic = 5
        self.clip = 0.01
        self.d_clip = []  # (-0.01 ~ 0.01)
        self.d_lambda = 10
        self.decay = 0.90

        self.EnableBN = enable_bn
        self.EnableSELU = enable_selu
        self.EnableAdam = enable_adam
        self.EnableGP = enable_gp

        self.d_op = None
        self.g_op = None

        self.build_wgan()  # build WGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            h0 = conv2d(x, self.df_dim, name='d_h0_conv')
            if self.EnableBN:
                h0 = self.d_bn1(h0)
            if self.EnableSELU:
                h0 = tf.nn.selu(h0)
            else:
                h0 = tf.nn.leaky_relu(h0, alpha=0.2)

            h1 = conv2d(h0, self.df_dim * 2, name='d_h1_conv')
            if self.EnableBN:
                h1 = self.d_bn2(h1)
            if self.EnableSELU:
                h1 = tf.nn.selu(h1)
            else:
                h1 = tf.nn.leaky_relu(h1, alpha=0.2)

            h2 = conv2d(h1, self.df_dim * 4, name='d_h2_conv')
            if self.EnableBN:
                h2 = self.d_bn3(h2)
            if self.EnableSELU:
                h2 = tf.nn.selu(h2)
            else:
                h2 = tf.nn.leaky_relu(h2, alpha=0.2)

            h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_linear')

            return tf.nn.sigmoid(h3)

    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            h0 = tf.reshape(linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin'), [-1, 4, 4, self.gf_dim * 8])
            if self.EnableBN:
                h0 = self.d_bn1(h0)
            if self.EnableSELU:
                h0 = tf.nn.selu(h0)
            else:
                h0 = tf.nn.leaky_relu(h0, alpha=0.2)

            h1 = deconv2d(h0, [self.batch_size, 7, 7, self.gf_dim * 4], name='g_h1')
            if self.EnableBN:
                h1 = self.d_bn2(h1)
            if self.EnableSELU:
                h1 = tf.nn.selu(h1)
            else:
                h1 = tf.nn.leaky_relu(h1, alpha=0.2)

            h2 = deconv2d(h1, [self.batch_size, 14, 14, self.gf_dim * 2], name='g_h2')
            if self.EnableBN:
                h2 = self.d_bn3(h2)
            if self.EnableSELU:
                h2 = tf.nn.selu(h2)
            else:
                h2 = tf.nn.leaky_relu(h2, alpha=0.2)

            h3 = deconv2d(h2, [self.batch_size,  # output shape is same as input shape
                               self.input_height, self.input_width, self.input_channel], name='g_h3')

            return tf.nn.tanh(h3)

    def build_wgan(self):
        # Using DCGAN D/G Model

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # The WGAN losses
        # maximize log(D(G(z)))
        # maximize log(D(x)) + log(1 - D(G(z)))

        log = lambda x: tf.log(x + self.eps)

        self.g_loss = -tf.reduce_mean(log(d_fake))
        d_real_loss = -tf.reduce_mean(log(d_real))
        d_fake_loss = -tf.reduce_mean(log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss

        # The gradient penalty loss
        if self.EnableGP:
            alpha = tf.random_uniform(shape=self.x.get_shape(),
                                      minval=0., maxval=1., name='alpha')
            diff = self.g - self.x  # fake data - real data
            interpolates = self.x + alpha * diff
            d_interp = self.discriminator(interpolates, reuse=True)
            gradients = tf.gradients(d_interp, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            # Update D loss
            self.d_loss += self.d_lambda * gradient_penalty

        # Summary
        z_sum = tf.summary.histogram("z", self.z)
        g = tf.reshape(self.g, shape=[-1] + self.image_shape)
        g_sum = tf.summary.image("g", g)  # generated image from G model
        d_real_sum = tf.summary.histogram("d_real", d_real)
        d_fake_sum = tf.summary.histogram("d_fake", d_fake)

        d_real_loss_sum = tf.summary.scalar("d_real_loss", d_real_loss)
        d_fake_loss_sum = tf.summary.scalar("d_fake_loss", d_fake_loss)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # Collect trainer values
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('discriminator')]
        g_params = [v for v in vars if v.name.startswith('generator')]

        self.d_clip = [v.assign(tf.clip_by_value(v, -self.clip, self.clip)) for v in d_params]

        # Model Saver
        self.saver = tf.train.Saver()

        # Optimizer
        if self.EnableAdam:
            self.d_op = tf.train.AdamOptimizer(learning_rate=2e-4,
                                               beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_params)
            self.g_op = tf.train.AdamOptimizer(learning_rate=2e-4,
                                               beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_params)
        else:
            self.d_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                  decay=self.decay).minimize(self.d_loss, var_list=d_params)
            self.g_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                  decay=self.decay).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.g_sum = tf.summary.merge([z_sum, d_fake_sum, g_sum, d_fake_loss_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([z_sum, d_real_sum, d_real_loss_sum, d_loss_sum])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
