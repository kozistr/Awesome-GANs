import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class AdaGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1, n_classes=10,
                 sample_num=64, sample_size=8,
                 n_input=784, df_dim=16, gf_dim=16, fc_unit=256, z_dim=100, d_lr=1e-3, g_lr=5e-3, c_lr=1e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (gray-scale)
        :param n_classes: input DataSet's classes

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # Hyper Parameters
        :param n_input: input image size, default 784(28x28)
        :param df_dim: D net filter, default 16
        :param gf_dim: G net filter, default 16
        :param fc_unit: fully connected units, default 256

        # Training Option
        :param z_dim: z dimension (kinda noise), default 100
        :param d_lr: discriminator learning rate, default 1e-3
        :param g_lr: generator learning rate, default 5e-3
        :param c_lr: classifier learning rate, default 1e-4
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
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.c_lr = c_lr

        self.d_loss = 0.
        self.g_loss = 0.
        self.c_loss = 0.

        self.g = None

        self.d_op = None
        self.g_op = None
        self.c_op = None

        self.merged = None
        self.saver = None
        self.writer = None

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")  # (-1, 784)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')    # (-1, 100)

        self.build_adagan()  # build AdaGAN model

    def classifier(self, x, reuse=None):
        with tf.variable_scope("classifier", reuse=reuse):
            pass

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            for i in range(1, 3):
                x = t.conv2d(x, self.df_dim * i, 5, 2, name='disc-conv2d-%d' % i)
                x = t.batch_norm(x, name='disc-bn-%d' % i)
                x = tf.nn.leaky_relu(x, alpha=0.3)

            x = t.flatten(x)

            logits = t.dense(x, 1, name='disc-fc-1')
            prob = tf.nn.sigmoid(logits)
            return prob, logits

    def generator(self, z, reuse=None, is_train=True):
        with tf.variable_scope("generator", reuse=reuse):
            x = t.dense(z, self.gf_dim * 7 * 7, name='gen-fc-1')
            x = t.batch_norm(x, name='gen-bn-1')
            x = tf.nn.leaky_relu(x, alpha=0.3)

            x = tf.reshape(x, [-1, 7, 7, self.gf_dim])

            for i in range(1, 3):
                x = t.deconv2d(x, self.gf_dim, 5, 2, name='gen-deconv2d-%d' % (i + 1))
                x = t.batch_norm(x, is_train=is_train, name='gen-bn-%d' % (i + 1))
                x = tf.nn.leaky_relu(x, alpha=0.3)

            x = t.deconv2d(x, 1, 5, 1, name='gen-deconv2d-3')
            x = tf.nn.sigmoid(x)
            return x

    def build_adagan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real, _ = self.discriminator(self.x)
        d_fake, _ = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = -tf.reduce_mean(t.safe_log(d_real))
        d_fake_loss = -tf.reduce_mean(t.safe_log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(t.safe_log(d_fake))

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("loss/c_loss", self.c_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]
        c_params = [v for v in t_vars if v.name.startswith('c')]

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
