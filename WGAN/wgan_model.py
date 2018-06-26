import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)


class WGAN:

    def __init__(self, s, batch_size=32, height=28, width=28, channel=1, n_classes=10,
                 sample_num=8 * 8, sample_size=8,
                 z_dim=128, gf_dim=32, df_dim=32, fc_unit=512,
                 enable_adam=False, enable_gp=False):

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
        :param fc_unit: the number of fc units, default 512
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
        self.fc_unit = fc_unit

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel],
                                name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim],
                                name='z-noise')

        # Training Options - based on the WGAN paper
        self.beta1 = 0.   # 0.5
        self.beta2 = 0.9  # 0.999
        self.lr = 1e-4
        self.critic = 5
        self.clip = .01
        self.d_clip = []  # (-0.01 ~ 0.01)
        self.d_lambda = 10.
        self.decay = .9

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

    def mean_pool_conv(self, x, f, name):
        with tf.name_scope("mean_pool_conv-%s" % name):
            x = tf.add_n([x[:, ::2, ::2, :], x[:, 1::2, ::2, :], x[:, ::2, 1::2, :], x[:, 1::2, 1::2, :]]) / 4.
            x = t.conv2d(x, f, 3, 1, name='conv2d-1' % name)
            return x

    def conv_mean_pool(self, x, f, name):
        with tf.name_scope("conv_mean_pool-%s" % name):
            x = t.conv2d(x, f, 3, 1, name='conv2d-1' % name)
            x = tf.add_n([x[:, ::2, ::2, :], x[:, 1::2, ::2, :], x[:, ::2, 1::2, :], x[:, 1::2, 1::2, :]]) / 4.
            return x

    def upsample_conv(self, x, f, name):
        with tf.name_scope("upsample_conv-%s" % name):
            x = tf.concat([x, x, x, x], axis=-1)
            x = tf.depth_to_space(x, 2)
            x = t.conv2d(x, f, 3, 1, name='conv2d-1' % name)
            return x

    def residual_block(self, x, f, sampling=None, name=""):
        with tf.name_scope(name):
            shortcut = tf.identity(x)

            if name.startswith('gen'):
                x = t.batch_norm(x, name='bn-1')
            x = tf.nn.relu(x)

            if sampling == 'up':
                x = self.upsample_conv(x, f, sampling + "-1")
            elif sampling == 'down' or sampling is None:
                x = t.conv2d(x, f, name='%s-conv2d-1' % sampling)

            if name.startswith('gen'):
                x = t.batch_norm(x, name='bn-2')
            x = tf.nn.relu(x)

            if sampling == 'up' or sampling is None:
                x = t.conv2d(x, f, name='%s-conv2d-1' % sampling)
            elif sampling == 'down':
                x = self.conv_mean_pool(x, f, sampling + "-1")

            if sampling == 'up':
                shortcut = self.upsample_conv(shortcut, f, sampling + "-2")
            elif sampling == 'down':
                shortcut = self.conv_mean_pool(shortcut, f, sampling + "-2")

            return shortcut + x

    def residual_block_init(self, x, f, name=""):
        with tf.name_scope("%s-res_block_init" % name):
            shortcut = tf.identity(x)

            x = t.conv2d(x, f, name='conv2d-1')
            x = tf.nn.relu(x)

            x = self.conv_mean_pool(x, f, 'rb_init-1')
            shortcut = self.mean_pool_conv(shortcut, 1, 'rb_init-1')

            return shortcut + x

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):

            x = self.residual_block_init(x, self.z_dim, name='disc')

            x = self.residual_block(x, self.z_dim, sampling='down', name='disc-res_block-1')
            x = self.residual_block(x, self.z_dim, name='disc-res_block-2')
            x = self.residual_block(x, self.z_dim, name='disc-res_block-3')

            x = tf.nn.relu(x)

            x = tf.reduce_mean(x, axis=[1, 2])

            x = t.dense(x, 1, name='disc-fc-1')
            return x

    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            x = t.dense(z, self.z_dim * 4 * 4, name="gen-fc-1")
            x = tf.reshape(x, [-1, 4, 4, self.z_dim])

            for i in range(1, 4):
                x = self.residual_block(x, self.z_dim, sampling='up', name='gen-res_block-%d' % i)

            x = t.batch_norm(x, name='gen-bn-1')
            x = tf.nn.relu(x)

            x = t.conv2d(x, 3, 3, 1, name="gen-conv2d-1")
            x = tf.nn.tanh(x)
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

        if not self.EnableGP:
            self.d_clip = [v.assign(tf.clip_by_value(v, -self.clip, self.clip)) for v in d_params]

        # Optimizer
        if self.EnableAdam:
            self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr * 2,
                                               beta1=self.beta1, beta2=self.beta2).minimize(loss=self.d_loss,
                                                                                            var_list=d_params)
            self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr * 2,
                                               beta1=self.beta1, beta2=self.beta2).minimize(loss=self.g_loss,
                                                                                            var_list=g_params)
        else:
            self.d_op = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                  decay=self.decay).minimize(self.d_loss, var_list=d_params)
            self.g_op = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                  decay=self.decay).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
