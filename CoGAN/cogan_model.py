import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


def conv2d(x, f=64, k=4, s=2, reuse=False, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param reuse: param re-usability, default False
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: covn2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad,
                            reuse=reuse,
                            name=name)


def deconv2d(x, f=64, k=4, s=2, reuse=False, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param reuse: param re-usability, default False
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: decovn2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad,
                                      reuse=reuse,
                                      name=name)


def batch_norm(x, momentum=0.9, eps=1e-5, reuse=False, training=True):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         reuse=reuse,
                                         training=training)


def prelu(x, stddev=1e-2, reuse=False, name='prelu'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        _alpha = tf.get_variable('_alpha',
                                 shape=x.get_shape(),
                                 initializer=tf.constant_initializer(stddev),
                                 dtype=x.dtype)

        return tf.maximum(_alpha * x, x)


class CoGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10, output_height=28, output_width=28,
                 n_input=784, fc_unit=1024, df_dim=64, gf_dim=64,
                 z_dim=128, g_lr=2e-4, d_lr=2e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 28
        :param input_width: input image width, default 28
        :param input_channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input dataset's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10
        :param output_height: output images height, default 28
        :param output_width: output images width, default 28

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param fc_unit: fully connected units, default 1024
        :param df_dim: the number of disc filters, default 64
        :param gf_dim: the number of gen filters, default 64

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 2e-4
        :param d_lr: discriminator learning rate, default 2e-4
        :param epsilon: epsilon, default 1e-9
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.n_input = n_input
        self.fc_unit = fc_unit
        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.z_dim = z_dim
        self.beta1 = .5
        self.beta2 = .999
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.g_1 = None
        self.g_2 = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")    # (-1, 784)
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="y-label")  # (-1, 10)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')      # (-1, 128)

        self.build_cogan()  # build CoGAN model

    def discriminator(self, x, share_params=False, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv2d(x, f=self.df_dim, k=5, s=1, reuse=False, name='disc-conv2d-0')
            x = prelu(x, reuse=False, name='disc-prelu-0')
            x = tf.nn.max_pool(x, ksize=2, strides=2, name='disc-max_pool2d-0')

            x = conv2d(x, f=self.df_dim * 2, k=5, s=1, reuse=share_params, name='disc-conv2d-1')
            x = prelu(x, reuse=share_params, name='disc-prelu-1')
            x = tf.nn.max_pool(x, ksize=2, strides=2, name='disc-max_pool2d-1')

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, self.fc_unit, reuse=share_params, name='disc-dense-0')
            x = prelu(x, reuse=share_params, name='disc-prelu-2')

            x = tf.layers.dense(x, 1, reuse=share_params, name='disc-dense-1')

            return x

    def generator(self, x, y, share_params=False, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([x, y], axis=0)

            return x

    def build_cogan(self):
        def log(x):
            return tf.log(x + self.eps)

        # Generator
        self.g_1 = self.generator(self.z, self.y)
        self.g_2 = self.generator(self.z, self.y, reuse=True)

        # Discriminator
        d_real, _ = self.discriminator(self.x)
        d_fake, _ = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = -tf.reduce_mean(log(d_real))
        d_fake_loss = -tf.reduce_mean(log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(tf.square(log(d_fake) + log(1. - d_fake))) / 2.

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

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
