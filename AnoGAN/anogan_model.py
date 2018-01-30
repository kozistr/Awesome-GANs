import tensorflow as tf


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
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad, name=name)


def deconv2d(x, f=64, k=3, s=1, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 1
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: decovn2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad, name=name)


def batch_norm(x, momentum=0.9, eps=1e-9, train=True):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         training=train)


class AnoGAN:

    def __init__(self, s, batch_size=64, input_height=64, input_width=64, input_channel=3,
                 sample_num=16 * 16, sample_size=16, output_height=64, output_width=64,
                 df_dim=64, gf_dim=64,
                 lambda_=1e-1, z_dim=128, g_lr=2e-4, d_lr=2e-4, epsilon=1e-12):

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
        :param lambda_: anomaly lambda value, default 1e-1
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

        self.lambda_ = lambda_
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
        self.d_loss = 0.
        self.g_loss = 0.
        self.anomaly_loss = 0.

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # learning rate update
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * self.lr_decay_rate, self.lr_low_boundary))
        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * self.lr_decay_rate, self.lr_low_boundary))

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.input_height, self.input_width, self.input_channel],
                                name="x-image")                                        # (-1, 32, 32, 3)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')  # (-1, 128)

        self.build_anogan()  # build AnoGAN model

    def sampler(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(z, units=self.z_dim * 4 * 4, name='s-fc-0')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 4, 4, self.z_dim])

            for i in range(1, 4):
                x = deconv2d(x, f=self.gf_dim * (2 ** i), k=3, s=2, name="s-conv2d-%d" % i)
                x = batch_norm(x, train=False)
                x = tf.nn.leaky_relu(x)

            x = deconv2d(x, f=3, k=3, s=1, name="s-conv2d-4")
            x = tf.nn.tanh(x)

            return x

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv2d(x, f=self.gf_dim * 1, name="d-conv2d-0")
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            for i in range(1, 4):
                x = conv2d(x, f=self.gf_dim * (2 ** i), k=3, s=2, name="d-conv2d-%d" % i)
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, 1, name='d-fc-0')
            # x = tf.nn.sigmoid(x)

            return x

    def generator(self, z, reuse=None):
        """
        :param z: embeddings
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(z, units=self.z_dim * 4 * 4, name='g-fc-0')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 4, 4, self.z_dim])

            for i in range(1, 4):
                x = deconv2d(x, f=self.gf_dim * (2 ** i), k=3, s=2, name="g-conv2d-%d" % i)
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x)

            x = deconv2d(x, f=3, k=3, s=1, name="g-conv2d-4")
            x = tf.nn.tanh(x)

            return x

    def build_anogan(self):
        def l1_loss(x, y):
            return tf.reduce_mean(tf.abs(x - y))

        def sce_loss(logits, label):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label))

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Loss
        d_real_loss = tf.reduce_mean(l1_loss(self.x, d_real))
        d_fake_loss = tf.reduce_mean(l1_loss(self.g, d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = d_fake_loss
        self.anomaly_loss = (1. - self.lambda_) * 1 + self.lambda_ * self.d_loss

        # Summary
        # tf.summary.histogram("z-noise", self.z)

        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("anomaly_loss", self.anomaly_loss)

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
