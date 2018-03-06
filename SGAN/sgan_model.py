import tensorflow as tf


tf.set_random_seed(777)


def conv2d(x, f=64, k=5, d=2, pad='SAME', name='conv2d'):
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


def deconv2d(x, f=64, k=5, d=2, pad='SAME', name='deconv2d'):
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


def gaussian_noise(x, std=5e-2):
    noise = tf.random_normal(x.get_shape(), mean=0., stddev=std, dtype=tf.float32)
    return x + noise


class SGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1,
                 n_classes=10, n_input=784, sample_size=8, sample_num=64,
                 z_dim=100, gf_dim=128, df_dim=96, fc_unit=256):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 28
        :param input_width: input image width, default 28
        :param input_channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: the classes, default 10
        - in case of MNIST, there're 10 classes
        :param n_input: flatten size of image, default 784

        # Output Settings
        :param sample_size: sample image size, default 8
        :param sample_num: the number of sample images, default 64

        # Model Settings
        :param z_dim: z noise dimension, default 128
        :param gf_dim: the number of generator filters, default 128
        :param df_dim: the number of discriminator filters, default 32
        :param fc_unit: fully connected units, default 256
        """

        self.s = s
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.n_input = n_input

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.image_shape = [self.input_height, self.input_width, self.input_channel]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.fc_unit = fc_unit

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name='x-images')
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-classes')
        self.z_0 = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise-1')
        self.z_1 = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise-2')

        # Training Options
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.lr = 2e-4

        self.d_1_loss = 0.
        self.d_0_loss = 0.
        self.g_1_loss = 0.
        self.g_0_loss = 0.

        self.d_w_loss = 1.   # weight for adversarial loss
        self.c_w_loss = 1.   # weight for conditional loss
        self.e_w_loss = 10.  # weight for entropy loss

        # pre-defined
        self.g_0 = None
        self.g_1 = None

        self.d_1_op = None
        self.d_0_op = None
        self.g_1_op = None
        self.g_0_op = None

        self.merged = None
        self.saver = None
        self.writer = None

        self.bulid_sgan()  # build SGAN model

    def encoder(self, x, reuse=None):
        """
        :param x: images, (-1, 28, 28, 1)
        :param reuse: re-usability
        :return: embeddings(256), y-labels' prob
        """
        with tf.variable_scope('encoder', reuse=reuse):
            x = tf.reshape(x, [-1] + self.image_shape)  # (-1, 28, 28, 1)

            for i in range(1, 3):
                x = conv2d(x, self.df_dim, name='enc-conv2d-%d' % i)
                x = tf.nn.leaky_relu(x)
                # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='enc-max_pool2d-%d' % i)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, self.fc_unit, name='enc-fc-1')
            x = tf.nn.leaky_relu(x)

            logits = tf.layers.dense(x, self.n_classes, name='enc-fc-2')
            prob = tf.nn.softmax(logits)

            return x, prob

    def discriminator_1(self, x, reuse=None):
        """
        :param x: features, (-1, 256)
        :param reuse: re-usability
        :return: z prob, disc prob
        """
        with tf.variable_scope('discriminator_1', reuse=reuse):
            for i in range(1, 3):
                x = tf.layers.dense(x, self.fc_unit, name='d_1-fc-%d' % i)

            z = tf.layers.dense(x, self.z_dim, activation=tf.nn.sigmoid, name='d_1-fc-3')
            prob = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='d_1-fc-4')

            return z, prob

    def discriminator_0(self, x, reuse=None):
        """
        :param x: MNIST image, (-1, 784)
        :param reuse: re-usability
        :return: z prob, disc prob
        """
        with tf.variable_scope('discriminator_0', reuse=reuse):
            x = tf.reshape(x, [-1] + self.image_shape)  # (-1, 28, 28, 1)
            x = gaussian_noise(x)

            x = conv2d(x, self.df_dim * 1, name='d_0-conv2d-1')
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim * 2, name='d_0-conv2d-2')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, self.df_dim * 4, name='d_0-conv2d-3')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, self.fc_unit, name='d_0-fc-1')

            z = tf.layers.dense(x, self.z_dim, activation=tf.nn.sigmoid, name='d_0-fc-2')
            prob = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='d_0-fc-3')

            return z, prob

    def generator_1(self, z, y, reuse=None):
        with tf.variable_scope('generator_1', reuse=reuse):
            # z : (batch_size, 100)
            # y : (batch_size, 10)
            # x : (batch_size, 110)
            x = tf.concat([z, y], axis=1)

            x = tf.layers.dense(x, self.fc_unit * 2, name='g_1-fc-1')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dense(x, self.fc_unit * 2, name='g_1-fc-2')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dense(x, self.fc_unit * 1, name='g_1-fc-3')
            x = tf.nn.leaky_relu(x)

            return x

    def generator_0(self, z, h, reuse=None):
        with tf.variable_scope('generator_0', reuse=reuse):
            z = tf.layers.dense(z, self.gf_dim, name='g_0-fc-1')
            z = tf.nn.leaky_relu(z)

            # z : (batch_size, 100)
            # h : (batch_size, 256)
            # x : (batch_size, 356)
            x = tf.concat([h, z], axis=1)

            x = tf.layers.dense(x, self.gf_dim * 7 * 7, name='g_0-fc-2')
            x = batch_norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 7, 7, self.gf_dim])

            for i in range(1, 3):
                f = self.gf_dim // i
                x = deconv2d(x, f, name='g_0-deconv2d-%d' % i)
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x)

            logits = deconv2d(x, self.input_channel, d=1, name='g_0-deconv2d-3')  # 28x28x1
            prob = tf.nn.sigmoid(logits)

            return prob

    def bulid_sgan(self):
        # Generator
        self.g_1 = self.generator_1(self.z_1, self.y)    # embeddings
        self.g_0 = self.generator_0(self.z_0, self.g_1)  # generated image

        # Encoder
        enc_real_f_prob, enc_real_c_prob = self.encoder(self.x)
        enc_fake_f_prob, enc_fake_c_prob = self.encoder(self.g_0, reuse=True)

        # Discriminator
        d_1_real_z_prob, d_1_real_prob = self.discriminator_1(enc_real_f_prob)
        d_1_fake_z_prob, d_1_fake_prob = self.discriminator_1(self.g_1, reuse=True)
        d_0_real_z_prob, d_0_real_prob = self.discriminator_0(self.x)
        d_0_fake_z_prob, d_0_fake_prob = self.discriminator_0(self.g_0, reuse=True)

        # Losses
        # Discriminator 1
        d_1_real_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_1_real_prob,
                                                                                labels=tf.ones_like(d_1_real_prob)))
        d_1_fake_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_1_fake_prob,
                                                                                labels=tf.zeros_like(d_1_fake_prob)))
        g_1_ent = tf.reduce_mean(tf.square(d_1_fake_z_prob - self.z_1))
        self.d_1_loss = 0.5 * self.d_w_loss * (d_1_real_loss + d_1_fake_loss) + self.e_w_loss * g_1_ent

        # Discriminator 0
        d_0_real_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_0_real_prob,
                                                                                labels=tf.ones_like(d_0_real_prob)))
        d_0_fake_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_0_fake_prob,
                                                                                labels=tf.zeros_like(d_0_fake_prob)))
        g_0_ent = tf.reduce_mean(tf.square(d_0_fake_z_prob - self.z_0))
        self.d_0_loss = 0.5 * self.d_w_loss * (d_0_real_loss + d_0_fake_loss) + self.e_w_loss * g_0_ent

        # Generator 1
        g_1_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_1_fake_prob,
                                                                              labels=tf.ones_like(d_1_fake_prob)))
        g_1_cond_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=enc_real_c_prob,
                                                                               labels=self.y))
        self.g_1_loss = self.d_w_loss * g_1_adv_loss + self.c_w_loss * g_1_cond_loss + self.e_w_loss * g_1_ent

        # Generator 0
        g_0_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_0_fake_prob,
                                                                              labels=tf.ones_like(d_0_fake_prob)))
        g_0_cond_loss = tf.reduce_mean(tf.square(enc_fake_f_prob - enc_real_f_prob))
        self.g_0_loss = self.d_w_loss * g_0_adv_loss + self.c_w_loss * g_0_cond_loss + self.e_w_loss * g_0_ent

        # Summary
        tf.summary.histogram("z_0", self.z_0)
        tf.summary.histogram("z_1", self.z_1)

        g_0 = tf.reshape(self.g_0, [-1] + self.image_shape)
        tf.summary.image("g_0", g_0)  # generated image from G model

        tf.summary.histogram("d_1_real_loss", d_1_real_loss)
        tf.summary.histogram("d_1_fake_loss", d_1_fake_loss)
        tf.summary.histogram("d_0_real_loss", d_0_real_loss)
        tf.summary.histogram("d_0_fake_loss", d_0_fake_loss)

        tf.summary.scalar("d_1_loss", self.d_1_loss)
        tf.summary.scalar("d_0_loss", self.d_0_loss)
        tf.summary.scalar("g_1_loss", self.g_1_loss)
        tf.summary.scalar("g_0_loss", self.g_0_loss)

        # Collect trainer values
        t_vars = tf.trainable_variables()
        d_1_params = [v for v in t_vars if v.name.startswith('discriminator_1')]
        d_0_params = [v for v in t_vars if v.name.startswith('discriminator_0')]
        g_1_params = [v for v in t_vars if v.name.startswith('generator_1')]
        g_0_params = [v for v in t_vars if v.name.startswith('generator_0')]

        # Optimizer
        self.d_1_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                             beta1=self.beta1).minimize(self.d_1_loss, var_list=d_1_params)
        self.d_0_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                             beta1=self.beta1).minimize(self.d_0_loss, var_list=d_0_params)
        self.g_1_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                             beta1=self.beta1).minimize(self.g_1_loss, var_list=g_1_params)
        self.g_0_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                             beta1=self.beta1).minimize(self.g_0_loss, var_list=g_0_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
