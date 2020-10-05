import tensorflow as tf

import sys

import vgg19

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)


class DeblurGAN:

    def __init__(self, s, batch_size=64, height=64, width=64, channel=3,
                 sample_num=8 * 8, sample_size=8,
                 z_dim=128, gf_dim=64, df_dim=64, lr=2e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 64
        :param width: input image width, default 64
        :param channel: input image channel, default 3 (RGB)
        - in case of CelebA, image size is 64x64x3(HWC).

        # Output Settings
        :param sample_num: the number of sample images, default 64
        :param sample_size: sample image size, default 8

        # Model Settings
        :param z_dim: z noise dimension, default 128
        :param gf_dim: the number of generator filters, default 64
        :param df_dim: the number of discriminator filters, default 64

        # Training Settings
        :param lr: learning rate, default 2e-4
        """

        self.s = s
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.image_shape = [self.height, self.width, self.channel]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.vgg19 = None
        self.vgg_image_shape = [224, 224, 3]
        self.vgg_mean = [103.939, 116.779, 123.68]

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
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')

        # Training Options
        self.beta1 = 0.5  # 0.9 is not good at oscillation & instability
        self.lr = lr      # 1e-3 is too high...

        self.bulid_deblurgan()  # build DeblurGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = t.conv2d(x, self.df_dim * 1, 4, 2, name='disc-conv2d-1')
            x = tf.nn.leaky_relu(x)

            for i in range(1, 3):
                x = t.conv2d(x, self.df_dim * (2 ** i), 4, 2, name='disc-conv2d-%d' % (i + 1))
                x = t.instance_norm(x, reuse=reuse, name='disc-inst_norm-%d' % i)
                x = tf.nn.leaky_relu(x)

            x = t.conv2d(x, self.df_dim * 8, 4, 1, name='disc-conv2d-4')
            x = t.instance_norm(x, reuse=reuse, name='disc-inst_norm-3')
            x = tf.nn.leaky_relu(x)

            x = t.conv2d(x, 1, 4, 1, name='disc-conv2d-5')

            return x

    def generator(self, x, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            def residual_block(x, f, name=""):
                with tf.variable_scope(name, reuse=reuse):
                    skip_connection = tf.identity(x, name='gen-skip_connection-1')

                    x = t.conv2d(x, f, 3, 1, name='gen-conv2d-1')
                    x = t.instance_norm(x, reuse=reuse, name='gen-inst_norm-1')
                    x = tf.nn.relu(x)
                    x = t.conv2d(x, f, 3, 1, name='gen-conv2d-2')
                    x = tf.nn.relu(x)

                    return skip_connection + x

            shortcut = tf.identity(x, name='shortcut-init')

            x = t.conv2d(x, self.gf_dim * 1, 7, 1, name='gen-conv2d-1')
            x = t.instance_norm(x, affine=False, reuse=reuse, name='gen-inst_norm-1')
            x = tf.nn.relu(x)

            for i in range(1, 3):
                x = t.conv2d(x, self.gf_dim * (2 ** i), 3, 2, name='gen-conv2d-%d' % (i + 1))
                x = t.instance_norm(x, affine=False, reuse=reuse, name='gen-inst_norm-%d' % (i + 1))
                x = tf.nn.relu(x)

            # 9 Residual Blocks
            for i in range(9):
                x = residual_block(x, self.gf_dim * 4, name='gen-residual_block-%d' % (i + 1))

            for i in range(1, 3):
                x = t.deconv2d(x, self.gf_dim * (2 ** i), 3, 2, name='gen-deconv2d-%d' % i)
                x = t.instance_norm(x, affine=False, reuse=reuse, name='gen-inst_norm-%d' % (i + 3))
                x = tf.nn.relu(x)

            x = t.conv2d(x, self.gf_dim * 1, 7, 1, name='gen-conv2d-4')
            x = tf.nn.tanh(x)
            return shortcut + x

    def build_vgg19(self, x, reuse=None):
        with tf.variable_scope("vgg19", reuse=reuse):
            # image re-scaling
            x = tf.cast((x + 1) / 2, dtype=tf.float32)  # [-1, 1] to [0, 1]
            x = tf.cast(x * 255., dtype=tf.float32)     # [0, 1]  to [0, 255]

            r, g, b = tf.split(x, 3, 3)
            bgr = tf.concat([b - self.vgg_mean[0],
                             g - self.vgg_mean[1],
                             r - self.vgg_mean[2]], axis=3)

            self.vgg19 = vgg19.VGG19(bgr)

            net = self.vgg19.vgg19_net['conv3_3']
            return net

    def bulid_deblurgan(self):
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
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

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
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
