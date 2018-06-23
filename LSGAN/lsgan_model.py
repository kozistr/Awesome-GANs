import tensorflow as tf
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class LSGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 n_input=784, df_dim=64, gf_dim=64, fc_unit=1024,
                 z_dim=128, g_lr=8e-4, d_lr=8e-4, epsilon=1e-12):

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
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # For CNN model
        :param n_input: input image size, default 784(28x28)
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_unit: the number of fully connected filters, default 1024

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 8e-4
        :param d_lr: discriminator learning rate, default 8e-4
        :param epsilon: epsilon, default 1e-12
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
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon

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
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name="x-image")                                        # (-1, 28, 28, 1)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')  # (-1, 128)

        self.build_lsgan()  # build LSGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            x = t.conv2d(x, self.df_dim, name='d-conv-1')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.5, name='d_dropout-1')

            x = t.conv2d(x, self.df_dim * 2, name='d-conv-2')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.5, name='d-dropout-2')

            x = tf.layers.flatten(x)

            x = t.dense(x, self.fc_unit / 2, name='d-fc-1')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.5, name='d-dropout-3')

            x = t.dense(x, 1, name='d-fc-2')  # logits

            return x

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            x = t.dense(z, self.gf_dim * 2 * 7 * 7, name='g-fc-1')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 7, 7, self.gf_dim * 2])

            x = t.deconv2d(x, self.gf_dim, k=2, s=2, name='g-deconv-1')
            x = tf.nn.leaky_relu(x)

            x = t.deconv2d(x, 1, k=2, s=2, name='g-deconv-2')
            x = tf.sigmoid(x)

            return x

    def build_lsgan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # LSGAN Loss
        d_real_loss = t.mse_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.mse_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = (d_real_loss + d_fake_loss) / 2.
        self.g_loss = t.mse_loss(d_fake, tf.ones_like(d_fake))

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
