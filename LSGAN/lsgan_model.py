import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class LSGAN:

    def __init__(self, s, batch_size=64, height=64, width=64, channel=3, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 df_dim=64, gf_dim=64, fc_unit=1024, z_dim=128, lr=2e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 64
        :param width: input image width, default 64
        :param channel: input image channel, default 3 (gray-scale)
        :param n_classes: input DataSet's classes

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_unit: the number of fully connected filters, default 1024

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param lr: learning rate, default 2e-4
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

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.lr = lr

        # pre-defined
        self.g_loss = 0.
        self.d_loss = 0.

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel],
                                name="x-image")  # (-1, 64, 64, 3)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim],
                                name='z-noise')  # (-1, 128)

        self.build_lsgan()  # build LSGAN model

    def discriminator(self, x, reuse=None):
        """ Same as DCGAN Disc Net """
        with tf.variable_scope('discriminator', reuse=reuse):
            x = t.conv2d(x, self.df_dim * 1, 5, 2, name='disc-conv2d-1')
            x = tf.nn.leaky_relu(x)

            x = t.conv2d(x, self.df_dim * 2, 5, 2, name='disc-conv2d-2')
            x = t.batch_norm(x, name='disc-bn-1')
            x = tf.nn.leaky_relu(x)

            x = t.conv2d(x, self.df_dim * 4, 5, 2, name='disc-conv2d-3')
            x = t.batch_norm(x, name='disc-bn-2')
            x = tf.nn.leaky_relu(x)

            x = t.conv2d(x, self.df_dim * 8, 5, 2, name='disc-conv2d-4')
            x = t.batch_norm(x, name='disc-bn-3')
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            logits = t.dense(x, 1, name='disc-fc-1')
            prob = tf.nn.sigmoid(logits)

            return prob, logits

    def generator(self, z, reuse=None, is_train=True):
        """ Same as DCGAN Gen Net """
        with tf.variable_scope('generator', reuse=reuse):
            x = t.dense(z, self.gf_dim * 8 * 4 * 4, name='gen-fc-1')

            x = tf.reshape(x, [-1, 4, 4, self.gf_dim * 8])
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-1')
            x = tf.nn.relu(x)

            x = t.deconv2d(x, self.gf_dim * 4, 5, 2, name='gen-deconv2d-1')
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-2')
            x = tf.nn.relu(x)

            x = t.deconv2d(x,  self.gf_dim * 2, 5, 2, name='gen-deconv2d-2')
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-3')
            x = tf.nn.relu(x)

            x = t.deconv2d(x,  self.gf_dim * 1, 5, 2, name='gen-deconv2d-3')
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-4')
            x = tf.nn.relu(x)

            x = t.deconv2d(x, self.channel, 5, 2, name='gen-deconv2d-4')
            x = tf.nn.tanh(x)

            return x

    def build_lsgan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # LSGAN Loss
        d_real_loss = t.mse_loss(d_real, tf.ones_like(d_real), self.batch_size)
        d_fake_loss = t.mse_loss(d_fake, tf.zeros_like(d_fake), self.batch_size)
        self.d_loss = (d_real_loss + d_fake_loss) / 2.
        self.g_loss = t.mse_loss(d_fake, tf.ones_like(d_fake), self.batch_size)

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
