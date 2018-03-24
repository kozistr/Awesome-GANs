import tensorflow as tf


tf.set_random_seed(777)


def conv2d(x, f=64, k=3, s=2, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: conv2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad,
                            name=name)


def deconv2d(x, f=64, k=3, s=2, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: deconv2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad,
                                      name=name)


def batch_norm(x, momentum=0.9, eps=1e-5, train=True):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         training=train)


class DRAGAN:

    def __init__(self, s, batch_size=64, input_height=32, input_width=32, input_channel=3,
                 sample_num=10 * 10, sample_size=10,
                 z_dim=128, gf_dim=64, df_dim=64,
                 eps=1e-12):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 32
        :param input_width: input image width, default 32
        :param input_channel: input image channel, default 3 (RGB)
        - in case of CIFAR, image size is 32x32x3(HWC).

        # Output Settings
        :param sample_num: the number of sample images, default 100
        :param sample_size: sample image size, default 10

        # Model Settings
        :param z_dim: z noise dimension, default 128
        :param gf_dim: the number of generator filters, default 64
        :param df_dim: the number of discriminator filters, default 64

        # Training Settings
        :param eps: epsilon, default 1e-12
        """

        self.s = s
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.image_shape = [self.input_height, self.input_width, self.input_channel]

        self.z_dim = z_dim

        self.eps = eps

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.input_height, self.input_width, self.input_channel],
                                name='x-images')
        self.z = tf.placeholder(tf.float32,
                                shape=[None, self.z_dim],
                                name='z-noise')

        # Training Options
        self.beta1 = .5
        self.beta2 = .9
        self.lr = 2e-4

        self.bulid_dragan()  # build DRAGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = conv2d(x, self.df_dim, name='d-conv-0')
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim * 2, name='d-conv-1')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim * 4, name='d-conv-2')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim * 8, name='d-conv-3')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            logits = tf.layers.dense(x, 1, name='d-fc-1')
            prob = tf.nn.sigmoid(logits)

            return prob, logits

    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            x = tf.layers.dense(z, self.gf_dim * 8 * 4 * 4)

            x = tf.reshape(x, [-1, 4, 4, self.gf_dim * 8])
            # x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = deconv2d(x, self.gf_dim * 4, name='g-deconv-1')
            # x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = deconv2d(x,  self.gf_dim * 2, name='g-deconv-2')
            # x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            logits = deconv2d(x, self.input_channel, name='g-deconv-3')
            prob = tf.nn.tanh(logits)

            return prob

    def bulid_dragan(self):
        def log(x, eps=self.eps):
            return tf.log(x + eps)

        def sce_loss(logits, labels):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        _, d_real = self.discriminator(self.x)
        _, d_fake = self.discriminator(self.g, reuse=True)

        # Losses
        """
        d_real_loss = -tf.reduce_mean(log(d_real))
        d_fake_loss = -tf.reduce_mean(log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = -tf.reduce_mean(log(d_fake))
        """
        d_real_loss = sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = sce_loss(d_fake, tf.ones_like(d_fake))

        # Summary
        # tf.summary.histogram("z", self.z)
        # tf.summary.histogram("d_real", d_real)
        # tf.summary.histogram("d_fake", d_fake)

        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Collect trainer values
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        # Optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
