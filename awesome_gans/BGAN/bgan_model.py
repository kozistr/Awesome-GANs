import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class BGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 n_input=784, fc_unit=256, z_dim=128, g_lr=1e-4, d_lr=1e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input dataset's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param fc_unit: fully connected units, default 256

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 1e-4
        :param d_lr: discriminator learning rate, default 1e-4
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.n_input = n_input
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = .5
        self.beta2 = .9
        self.d_lr, self.g_lr = d_lr, g_lr

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")  # (-1, 784)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')    # (-1, 100)

        self.build_bgan()  # build BGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            for i in range(2):
                x = t.dense(x, self.fc_unit, name='disc-fc-%d' % (i + 1))
                x = tf.nn.leaky_relu(x)

            logits = t.dense(x, 1, name='disc-fc-3')
            prob = tf.nn.sigmoid(logits)

        return prob, logits

    def generator(self, z, reuse=None, is_train=True):
        with tf.variable_scope("generator", reuse=reuse):
            x = z
            for i in range(2):
                x = t.dense(x, self.fc_unit, name='gen-fc-%d' % (i + 1))
                x = t.batch_norm(x, is_train=is_train, name='gen-bn-%d' % (i + 1))
                x = tf.nn.leaky_relu(x)

            logits = t.dense(x, self.n_input, name='gen-fc-3')
            prob = tf.nn.sigmoid(logits)

        return prob

    def build_bgan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real, _ = self.discriminator(self.x)
        d_fake, _ = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = -tf.reduce_mean(t.safe_log(d_real))
        d_fake_loss = -tf.reduce_mean(t.safe_log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(tf.square(t.safe_log(d_fake) + t.safe_log(1. - d_fake))) / 2.

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
