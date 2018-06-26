import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)


class WGAN:

    def __init__(self, s, batch_size=32, height=28, width=28, channel=1, n_classes=10,
                 sample_num=8 * 8, sample_size=8,
                 z_dim=128, gf_dim=32, df_dim=32,
                 enable_bn=False, enable_adam=False, enable_gp=False):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (gray-scale)
        :param n_classes: input DatSset's classes

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param gf_dim: the number of generator filters, default 32
        :param df_dim: the number of discriminator filters, default 32
        :param enable_bn: enabling batch normalization, default False
        :param enable_adam: enabling adam optimizer, default False
        :param enable_gp: enabling gradient penalty, default False
        """

        self.s = s
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel
        self.n_classes = n_classes

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.image_shape = [self.height, self.width, self.channel]

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel],
                                name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim],
                                name='z-noise')

        # Training Options - based on the WGAN paper
        self.beta1 = 0.5
        self.learning_rate = 2e-4
        self.critic = 5
        self.clip = .01
        self.d_clip = []  # (-0.01 ~ 0.01)
        self.d_lambda = .25
        self.decay = .9

        self.EnableBN = enable_bn
        self.EnableAdam = enable_adam
        self.EnableGP = enable_gp

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.gradient_penalty = 0.

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        self.build_wgan()  # build WGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = t.conv2d(x, self.df_dim, 3, 2, name="disc-conv2d-1")
            if self.EnableBN:
                x = t.batch_norm(x, name="disc-bn-1")
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = t.conv2d(x, self.df_dim * 2, 3, 2, name="disc-conv2d-2")
            if self.EnableBN:
                x = t.batch_norm(x, name="disc-bn-2")
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = t.conv2d(x, self.df_dim * 4, k=3, s=1, name="disc-conv2d-3")
            if self.EnableBN:
                x = t.batch_norm(x, name="disc-bn-3")
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, 1)

            return x

    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            x = t.dense(z, self.gf_dim * 4 * 7 * 7, name="gen-fc-1")
            x = tf.reshape(x, [-1, 7, 7, self.gf_dim * 4])

            if self.EnableBN:
                x = t.batch_norm(x, name="gen-bn-1")
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = t.deconv2d(x, self.gf_dim * 2, k=3, s=2, name="gen-deconv2d-1")
            if self.EnableBN:
                x = t.batch_norm(x, name="gen-bn-2")
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = t.deconv2d(x, self.gf_dim * 1, k=3, s=2, name="gen-deconv2d-2")
            if self.EnableBN:
                x = t.batch_norm(x, name="gen-bn-3")
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = t.deconv2d(x, 1, k=3, s=1, name="gen-deconv2d-3")
            x = tf.nn.sigmoid(x)  # tf.nn.tanh(x)

            return x

    def build_wgan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # The WGAN losses
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        # The gradient penalty loss
        if self.EnableGP:
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1., name='alpha')
            diff = self.g - self.x  # fake data - real data
            interpolates = self.x + alpha * diff
            d_interp = self.discriminator(interpolates, reuse=True)
            gradients = tf.gradients(d_interp, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))

            # Update D loss
            self.d_loss += self.d_lambda * self.gradient_penalty

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        if self.EnableGP:
            tf.summary.scalar("misc/gp", self.gradient_penalty)

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
