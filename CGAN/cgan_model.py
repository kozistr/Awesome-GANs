import tensorflow as tf
import tfutil as t


tf.set_random_seed(777)


class CGAN:

    def __init__(self, s, batch_size=32, height=28, width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 n_input=784, maxout_unit=8, fc_unit=128,
                 z_dim=128, g_lr=8e-4, d_lr=8e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input dataset's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param maxout_unit : max-out unit, default 8
        :param fc_unit: fully connected units, default 128

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 8e-4
        :param d_lr: discriminator learning rate, default 8e-4
        :param epsilon: epsilon, default 1e-9
        """

        self.s = s
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.n_input = n_input
        self.maxout_unit = maxout_unit
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta1 = 0.5
        self.eps = epsilon

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.g = None
        self.g_test = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")        # (-1, 784)
        self.c = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='c-condition')  # (-1, 10)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')          # (-1, 100)

        self.build_cgan()  # build CGAN model

    def discriminator(self, x, y, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.concat([x, y], axis=1)

            x = t.dense(x, self.maxout_unit * self.fc_unit, name='d-fc-1')

            x = tf.reshape(x, [-1, self.maxout_unit, self.fc_unit])

            x = tf.reduce_max(x, reduction_indices=[1], name='d-reduce_max-1')
            x = tf.nn.dropout(x, .5)

            x = t.dense(x, 1, name='d-fc-2')
            x = tf.nn.sigmoid(x)

            return x

    def generator(self, z, y, reuse=None, is_train=True):
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([z, y], axis=1)

            x = t.dense(x, self.fc_unit, name='g-fc-1')
            x = tf.nn.dropout(x, .5) if is_train else tf.nn.dropout(x, 1.)
            x = tf.nn.leaky_relu(x)

            x = t.dense(x, self.n_input, name='g-fc-2')
            x = tf.nn.sigmoid(x)

            return x

    def build_cgan(self):
        # Generator
        self.g = self.generator(self.z, self.c)
        self.g_test = self.generator(self.z, self.c, is_train=False)

        # Discriminator
        d_real = self.discriminator(self.x, self.c)
        d_fake = self.discriminator(self.g, self.c, reuse=True)

        # Losses
        d_real_loss = -tf.reduce_mean(t.safe_log(d_real))
        d_fake_loss = -tf.reduce_mean(t.safe_log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = -tf.reduce_mean(t.safe_log(d_fake))

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Collect trainer values
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        # Optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
