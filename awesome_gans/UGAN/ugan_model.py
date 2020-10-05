import tensorflow as tf

import awesome_gans.tfutil as t

tf.set_random_seed(777)


class UGAN:
    def __init__(
        self,
        s,
        batch_size=64,
        height=32,
        width=32,
        channel=3,
        sample_num=8 * 8,
        sample_size=8,
        z_dim=256,
        gf_dim=64,
        df_dim=64,
        d_lr=2e-4,
        g_lr=1e-4,
    ):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 32
        :param width: input image width, default 32
        :param channel: input image channel, default 3 (RGB)

        # Output Settings
        :param sample_num: the number of sample images, default 64
        :param sample_size: sample image size, default 8

        # Model Settings
        :param z_dim: z noise dimension, default 256
        :param gf_dim: the number of generator filters, default 64
        :param df_dim: the number of discriminator filters, default 64

        # Training Settings
        :param d_lr: learning rate, default 2e-4
        :param g_lr: learning rate, default 1e-4
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

        # pre-defined
        self.d_loss = 0.0
        self.g_loss = 0.0

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')

        # Training Options
        self.beta1 = 0.5
        self.d_lr = d_lr
        self.g_lr = g_lr

        self.bulid_ugan()  # build UGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            for i in range(1, 4):
                x = t.conv2d(x, self.gf_dim * (2 ** (i - 1)), 3, 2, name='disc-conv2d-%d' % i)
                x = t.batch_norm(x, name='disc-bn-%d' % i)
                x = tf.nn.leaky_relu(x, alpha=0.3)

            x = tf.layers.flatten(x)

            x = t.dense(x, 1, name='disc-fc-1')
            return x

    def generator(self, z, reuse=None, is_train=True):
        with tf.variable_scope('generator', reuse=reuse):
            x = t.dense(z, self.gf_dim * 8 * 4 * 4, name='gen-fc-1')

            x = tf.reshape(x, [-1, 4, 4, self.gf_dim * 8])
            x = t.batch_norm(x, is_train=is_train, name='gen-bn-1')
            x = tf.nn.relu(x)

            for i in range(1, 4):
                x = t.deconv2d(x, self.gf_dim * 4, 3, 2, name='gen-deconv2d-%d' % i)
                x = t.batch_norm(x, is_train=is_train, name='gen-bn-%d' % (i + 1))
                x = tf.nn.relu(x)

            x = t.conv2d(x, self.channel, 3, name='gen-conv2d-1')
            x = tf.nn.sigmoid(x)
            return x

    def bulid_ugan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Losses
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
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=self.beta1).minimize(
            self.d_loss, var_list=d_params
        )
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=self.beta1).minimize(
            self.g_loss, var_list=g_params
        )

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
