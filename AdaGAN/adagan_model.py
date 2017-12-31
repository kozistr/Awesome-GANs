import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


def conv2d(x, f=16, k=5, d=2, pad='SAME', name='conv2d'):
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


def deconv2d(x, f=16, k=5, d=2, pad='SAME', name='deconv2d'):
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


class AdaGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1, n_classes=10,
                 sample_num=64, sample_size=8, output_height=28, output_width=28,
                 n_input=784, df_dim=16, gf_dim=16, fc_unit=256,
                 z_dim=100, g_lr=5e-3, d_lr=1e-3, c_lr=1e-4, epsilon=1e-9):

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
        :param output_height: output images height, default 28
        :param output_width: output images width, default 28

        # Hyper Parameters
        :param n_input: input image size, default 784(28x28)
        :param df_dim: D net filter, default 16
        :param gf_dim: G net filter, default 16
        :param fc_unit: fully connected units, default 256

        # Training Option
        :param z_dim: z dimension (kinda noise), default 100
        :param g_lr: generator learning rate, default 5e-3
        :param d_lr: discriminator learning rate, default 1e-3
        :param c_lr: classifier learning rate, default 1e-4
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
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.c_lr = c_lr
        self.eps = epsilon

        self.d_loss = 0.
        self.g_loss = 0.
        self.c_loss = 0.

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")  # (-1, 784)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')    # (-1, 128)

        self.build_adagan()  # build AdaGAN model

    def classifier(self, x, reuse=None):
        with tf.variable_scope("classifier", reuse=reuse):
            pass

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            for i in range(1, 3):
                x = conv2d(x, self.df_dim * i, name='d-conv2d-%d' % i)
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x, alpha=0.3)

            logits = tf.layers.dense(x, units=1, name='d-fc-1')
            prob = tf.nn.sigmoid(logits)

        return prob, logits

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(z, self.gf_dim * 7 * 7, name='g-fc-1')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x, alpha=0.3)

            x = tf.reshape(x, [self.batch_size, 7, 7, self.gf_dim])

            for i in range(1, 3):
                x = deconv2d(x, self.gf_dim, name='g-fc-%d' % i)
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x)

            logits = deconv2d(x, f=1, d=1, name='g-deconv-3')
            prob = tf.nn.sigmoid(logits)

        return prob

    def build_adagan(self):
        def log(x):
            return tf.log(x + self.eps)

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real, _ = self.discriminator(self.x)
        d_fake, _ = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = -tf.reduce_mean(log(d_real))
        d_fake_loss = -tf.reduce_mean(log(1. - d_fake))  # proposed way : log(1. - d_fake) TO -log(d_fake)
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(log(d_fake))

        # Summary
        tf.summary.histogram("z-noise", self.z)

        g = tf.reshape(self.g, shape=self.image_shape)
        tf.summary.image("generated", g)  # generated images by Generative Model
        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("c_loss", self.c_loss)

        # Optimizer
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]
        c_params = [v for v in vars if v.name.startswith('c')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)
        self.c_op = tf.train.AdamOptimizer(learning_rate=self.c_lr,
                                           beta1=self.beta1).minimize(self.c_loss, var_list=c_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
