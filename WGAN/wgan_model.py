import tensorflow as tf


tf.set_random_seed(777)


def batch_normalize(x, eps=1e-5):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=eps,
                                         scale=True,
                                         training=True)


def conv2d(x, f=64, k=3, s=2, pad='SAME'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param pad: padding (valid or same), default same
    :return: conv2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            use_bias=False,
                            padding=pad)


def deconv2d(x, f=64, k=3, s=2, pad='SAME'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param pad: padding (valid or same), default same
    :return: deconv2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                      use_bias=False,
                                      padding=pad)


class WGAN:

    def __init__(self, s, batch_size=32, input_height=28, input_width=28, input_channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 z_dim=100, gf_dim=32, df_dim=32, epsilon=1e-12,
                 enable_bn=False, enable_adam=False, enable_gp=False):

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
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # Training Option
        :param z_dim: z dimension (kinda noise), default 100
        :param gf_dim: the number of generator filters, default 32
        :param df_dim: the number of discriminator filters, default 32
        :param epsilon: epsilon, default 1e-12
        :param enable_bn: enabling batch normalization, default False
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

        self.image_shape = [self.input_height, self.input_width, self.input_channel]

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.eps = epsilon

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.input_channel],
                                name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim],
                                name='z-noise')

        # Training Options - based on the WGAN paper
        self.beta1 = 0.5
        self.learning_rate = 1e-4  # very slow
        self.critic = 5
        self.clip = .01
        self.d_clip = []  # (-0.01 ~ 0.01)
        self.d_lambda = .25
        self.decay = .90

        self.EnableBN = enable_bn
        self.EnableAdam = enable_adam
        self.EnableGP = enable_gp

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        self.build_wgan()  # build WGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = conv2d(x, self.df_dim)
            if self.EnableBN:
                x = batch_normalize(x)
            else:
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = conv2d(x, self.df_dim * 2)
            if self.EnableBN:
                x = batch_normalize(x)
            else:
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = conv2d(x, self.df_dim * 4, s=1)
            if self.EnableBN:
                x = batch_normalize(x)
            else:
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, 1)

            return x

    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            x = tf.layers.dense(z, self.gf_dim * 4 * 7 * 7)
            x = tf.reshape(x, [-1, 7, 7, self.gf_dim * 4])

            if self.EnableBN:
                x = batch_normalize(x)
            else:
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = deconv2d(x, self.gf_dim * 2)
            if self.EnableBN:
                x = batch_normalize(x)
            else:
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = deconv2d(x, self.gf_dim * 1)
            if self.EnableBN:
                x = batch_normalize(x)
            else:
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = deconv2d(x, 1, s=1)
            x = tf.nn.sigmoid(x)  # tf.nn.tanh(x)

            return x

    def build_wgan(self):
        def log(x, eps=self.eps):
            return tf.log(x + eps)

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # The WGAN losses
        # d_real_loss = -tf.reduce_mean(log(d_real))
        # d_fake_loss = -tf.reduce_mean(log(1. - d_fake))
        # self.d_loss = d_real_loss + d_fake_loss
        # self.g_loss = -tf.reduce_mean(log(d_fake))
        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real,
                                                                             labels=tf.ones_like(d_real)))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                                             labels=tf.zeros_like(d_fake)))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                                             labels=tf.ones_like(d_fake)))

        # The gradient penalty loss
        if self.EnableGP:
            alpha = tf.random_uniform(shape=[self.batch_size] + self.image_shape,
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
        tf.summary.histogram("z", self.z)

        # g = tf.reshape(self.g, shape=[-1] + self.image_shape)
        # tf.summary.image("g", g)  # generated image from G model
        tf.summary.histogram("d_real", d_real)
        tf.summary.histogram("d_fake", d_fake)

        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Collect trainer values
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('discriminator')]
        g_params = [v for v in t_vars if v.name.startswith('generator')]

        self.d_clip = [v.assign(tf.clip_by_value(v, -self.clip, self.clip)) for v in d_params]

        # Optimizer
        if self.EnableAdam:
            self.d_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate * 2,
                                               beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
            self.g_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate * 2,
                                               beta1=self.beta1).minimize(self.g_loss, var_list=g_params)
        else:
            self.d_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                  decay=self.decay).minimize(self.d_loss, var_list=d_params)
            self.g_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                  decay=self.decay).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
