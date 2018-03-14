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
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad,
                            name=name)


def deconv2d(x, f=64, k=3, s=1, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 1
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: deconv2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad,
                                      name=name)


def instance_normalize(x, eps=1e-5, affine=True, name=""):
    with tf.variable_scope('instance_normalize-affine_%s' % name):
        mean, variance = tf.nn.moments(x, [1, 2], keep_dims=True)

        normalized = tf.div(x - mean, tf.sqrt(variance + eps))

        if not affine:
            return normalized
        else:
            depth = x.get_shape()[3]  # input channel

            scale = tf.get_variable('scale', [depth],
                                    initializer=tf.random_normal_initializer(mean=1., stddev=.02, dtype=tf.float32))
            offset = tf.get_variable('offset', [depth],
                                     initializer=tf.zeros_initializer())

        return scale * normalized + offset


def batch_normalize(x, eps=1e-5):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=eps,
                                         scale=False,
                                         center=False,
                                         training=True)


class CycleGAN:

    def __init__(self, s, batch_size=64, input_height=64, input_width=64, input_channel=3,
                 sample_num=1 * 1, sample_size=1, output_height=64, output_width=64,
                 df_dim=32, gf_dim=32, fc_unit=512,
                 g_lr=2e-4, c_lr=2e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 64
        :param input_width: input image width, default 64
        :param input_channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 64x64x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 4
        :param sample_size: sample image size, default 2
        :param output_height: output images height, default 64
        :param output_width: output images width, default 64

        # For CNN model
        :param df_dim: discriminator filter, default 32
        :param gf_dim: generator filter, default 32
        :param fc_unit: fully connected units, default 512

        # Training Option
        :param g_lr: generator learning rate, default 2e-4
        :param c_lr: classifier learning rate, default 2e-4
        :param epsilon: epsilon, default 1e-9
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

        self.beta1 = .5
        self.beta2 = .9
        self.c_lr = c_lr
        self.g_lr = g_lr
        self.lambda_ = 10.
        self.lambda_cycle = 10
        self.n_train_critic = 10
        self.eps = epsilon

        # pre-defined
        self.g_a2b = None
        self.g_b2a = None
        self.g_a2b2a = None
        self.g_b2a2b = None

        self.w_a = None
        self.w_b = None
        self.w = None

        self.gp_a = None
        self.gp_b = None
        self.gp = None

        self.g_loss = 0.
        self.g_a_loss = 0.
        self.g_b_loss = 0.
        self.c_loss = 0.
        self.cycle_loss = 0.

        self.c_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # placeholders
        self.a = tf.placeholder(tf.float32,
                                [None, self.input_height, self.input_width, self.input_channel], name='image-a')
        self.b = tf.placeholder(tf.float32,
                                [None, self.input_height, self.input_width, self.input_channel], name='image-b')
        self.lr_decay = tf.placeholder(tf.float32, None, name='learning_rate-decay')

        self.build_cyclegan()  # build CycleGAN

    def encoder(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: embeddings
        """

        with tf.variable_scope('encoder', reuse=reuse):
            def residual_block(x, f, name='residual_block'):
                with tf.variable_scope(name, reuse=reuse):
                    x = conv2d(x, f=f, k=3, s=1, name='encoder-residual_block-conv2d-1')
                    x = instance_normalize(x, name='encoder-residual_block-instance_norm-1')
                    x = tf.nn.leaky_relu(x, alpha=0.2)

                    x = conv2d(x, f=f, k=3, s=1, name='encoder-residual_block-conv2d-2')

                return x

            x = conv2d(x, f=self.df_dim, k=7, s=1, name='encoder-conv2d-1')
            x = tf.nn.leaky_relu(x)

            for i in range(1, 3):
                x = conv2d(x, f=self.df_dim * i, k=4, s=2, name='encoder-conv2d-%d' % (i + 1))
                x = instance_normalize(x, name='encoder-instance_norm-%d' % i)
                x = tf.nn.leaky_relu(x)

            for i in range(1, 10):
                x = residual_block(x, f=self.df_dim * 4, name='encoder-residual_block-%d' % i)

            # x = tf.layers.flatten(x)  # embeddings

            return x

    def decoder(self, x, reuse=None):
        """
        :param x: embeddings
        :param reuse: re-usable
        :return: logits
        """

        with tf.variable_scope('decoder', reuse=reuse):
            for i in range(1, 3):
                x = deconv2d(x, f=self.gf_dim * 4 // i, k=4, s=2, name='decoder-deconv2d-%d' % i)
                x = instance_normalize(x, name='decoder-instance_norm-%d' % i)
                x = tf.nn.leaky_relu(x)

            x = conv2d(x, f=self.input_channel, k=7, s=1, name='decoder-conv2d-1')
            # x = tf.nn.tanh(x)

            return x

    def enc_dec(self, x, reuse=None, name=""):
        """
        :param x: embeddings
        :param reuse: re-usable
        :param name: name

        :return: logits
        """

        with tf.variable_scope('encoder-decoder-%s' % name, reuse=reuse):
            x = self.encoder(x, reuse=reuse)
            x = self.decoder(x, reuse=reuse)

            return x

    def classifier(self, x, reuse=None):
        with tf.variable_scope('classifier', reuse=reuse):
            f = self.gf_dim * 2
            for i, f_ in enumerate([f, f * 2, f * 4, f * 4, f * 4]):
                x = conv2d(x, f=f_, k=4, s=2, name='classifier-conv2d-%d' % (i + 1))
                # x = instance_norm(x, name='classifier-instance_norm-%d' % (i + 1))
                x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, self.fc_unit, activation=tf.nn.leaky_relu, name='classifier-fc-1')
            x = tf.layers.dense(x, self.fc_unit, activation=tf.nn.leaky_relu, name='classifier-fc-2')
            x = tf.layers.dense(x, 1, name='classifier-fc-3')

            return x

    def build_cyclegan(self):
        # Generator
        with tf.variable_scope("generator-a2b"):
            self.g_a2b = self.enc_dec(self.a, name="a2b")  # a to b
        with tf.variable_scope("generator-b2a"):
            self.g_b2a = self.enc_dec(self.b, name="b2a")  # b to a

        with tf.variable_scope("generator-b2a", reuse=True):
            self.g_a2b2a = self.enc_dec(self.g_a2b, reuse=True, name="b2a")  # a to b to a
        with tf.variable_scope("generator-a2b", reuse=True):
            self.g_b2a2b = self.enc_dec(self.g_b2a, reuse=True, name="a2b")  # b to a to b

        # Classifier
        with tf.variable_scope("classifier-a"):
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            a_hat = alpha * self.a + (1. - alpha) * self.g_b2a

            c_a = self.classifier(self.a)
            c_b2a = self.classifier(self.g_b2a, reuse=True)
            c_a_hat = self.classifier(a_hat, reuse=True)
        with tf.variable_scope("classifier-b"):
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            b_hat = alpha * self.b + (1. - alpha) * self.g_a2b

            c_b = self.classifier(self.b)
            c_a2b = self.classifier(self.g_a2b, reuse=True)
            c_b_hat = self.classifier(b_hat, reuse=True)

        # Training Ops
        self.w_a = tf.reduce_mean(c_a) - tf.reduce_mean(c_b2a)
        self.w_b = tf.reduce_mean(c_b) - tf.reduce_mean(c_a2b)
        self.w = self.w_a + self.w_b

        self.gp_a = tf.reduce_mean(
            (tf.sqrt(tf.reduce_sum(tf.gradients(c_a_hat, a_hat)[0]**2, reduction_indices=[1, 2, 3])) - 1.) ** 2
        )
        self.gp_b = tf.reduce_mean(
            (tf.sqrt(tf.reduce_sum(tf.gradients(c_b_hat, b_hat)[0]**2, reduction_indices=[1, 2, 3])) - 1.) ** 2
        )
        self.gp = self.gp_a + self.gp_b

        # loss
        self.c_loss = self.lambda_ * self.gp - self.w

        cycle_a_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(self.a - self.g_a2b2a), reduction_indices=[1, 2, 3]))
        cycle_b_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(self.b - self.g_b2a2b), reduction_indices=[1, 2, 3]))
        self.cycle_loss = cycle_a_loss + cycle_b_loss

        self.g_a_loss = -1. * tf.reduce_mean(c_b2a)
        self.g_b_loss = -1. * tf.reduce_mean(c_a2b)
        self.g_loss = self.g_a_loss + self.g_b_loss + self.lambda_cycle * self.cycle_loss

        # Summary
        tf.summary.scalar("loss/c_loss", self.c_loss)
        tf.summary.scalar("loss/cycle_loss", self.cycle_loss)
        tf.summary.scalar("loss/cycle_a_loss", cycle_a_loss)
        tf.summary.scalar("loss/cycle_b_loss", cycle_b_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("loss/g_a_loss", self.g_a_loss)
        tf.summary.scalar("loss/g_b_loss", self.g_b_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        c_params = [v for v in t_vars if v.name.startswith('c')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.c_op = tf.train.AdamOptimizer(learning_rate=self.c_lr * self.lr_decay,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.c_loss, var_list=c_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr * self.lr_decay,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
