import tensorflow as tf
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class CoGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1, n_classes=10,
                 sample_num=8 * 8, sample_size=8,
                 n_input=784, fc_unit=1024, df_dim=64, gf_dim=64,
                 z_dim=128, g_lr=2e-4, d_lr=2e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input DataSet's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param fc_unit: fully connected units, default 1024
        :param df_dim: the number of disc filters, default 64
        :param gf_dim: the number of gen filters, default 64

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 2e-4
        :param d_lr: discriminator learning rate, default 2e-4
        :param epsilon: epsilon, default 1e-9
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
        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.z_dim = z_dim
        self.beta1 = .5
        self.beta2 = .999
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon

        # pre-defined
        self.d_loss = 0.
        self.d_1_loss = 0.
        self.d_2_loss = 0.
        self.g_loss = 0.
        self.g_1_loss = 0.
        self.g_2_loss = 0.

        self.g_1 = None
        self.g_2 = None
        self.g_sample_1 = None
        self.g_sample_2 = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholder
        self.x_1 = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                     self.height, self.width, self.channel],
                                  name="x-image1")  # (-1, 28, 28, 1)
        self.x_2 = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                     self.height, self.width, self.channel],
                                  name="x-image2")  # (-1, 28, 28, 1)
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_classes],
                                name="y-label")   # (-1, 10)
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim],
                                name='z-noise')   # (-1, 128)

        self.build_cogan()  # build CoGAN model

    def discriminator(self, x, y=None, share_params=False, reuse=False, name=""):
        with tf.variable_scope("discriminator-%s" % name, reuse=reuse):
            if y is None:
                x = tf.layers.flatten(x)

                x = tf.concat([x, y], axis=1)

                x = t.dense(x, self.height * self.width * self.channel,
                            name='disc-' + name + '-dense-0-y')
                x = tf.reshape(x, self.image_shape)
            else:
                pass

            # Using conv2d pooling instead of max_pool2d because of the speed.
            # In the CoGAN paper, max_pool2d is used.

            x = t.conv2d(x, f=self.df_dim, k=5, s=2, reuse=False, name='disc-' + name + '-conv2d-0')
            x = t.prelu(x, reuse=False, name='disc-' + name + '-prelu-0')
            # x = tf.nn.max_pool(x, ksize=2, strides=2, padding='SAME', name='disc' + name + '-max_pool2d-0')

            x = t.conv2d(x, f=self.df_dim * 2, k=5, s=2, reuse=False, name='disc-' + name + '-conv2d-1')
            x = t.batch_norm(x, is_train=False, name='disc-bn-0')
            x = t.prelu(x, reuse=False, name='disc-' + name + '-prelu-1')
            # x = tf.nn.max_pool(x, ksize=2, strides=2, padding='SAME', name='disc' + name + '-max_pool2d-1')

            x = tf.layers.flatten(x)

        x = t.dense(x, self.fc_unit, reuse=share_params, name='disc-dense-0')
        x = t.batch_norm(x, is_train=share_params, name='disc-bn-1')
        x = t.prelu(x, reuse=share_params, name='disc-prelu-2')

        x = t.dense(x, 1, reuse=share_params, name='disc-dense-1')

        return x

    def generator(self, z, y=None, share_params=False, reuse=False, training=True, name=""):
        if y is None:
            x = tf.concat([z, y], axis=1)
        else:
            x = z

        x = tf.layers.flatten(x)

        x = tf.layers.dense(x, self.fc_unit, reuse=share_params, name='gen-dense-0')
        x = t.prelu(x, reuse=share_params, name='gen-prelu-0')

        x = tf.layers.dense(x, self.gf_dim * 8 * 7 * 7, reuse=share_params, name='gen-dense-1')
        x = t.batch_norm(x, reuse=share_params, is_train=training, name='gen-bn-0')
        x = t.prelu(x, reuse=share_params, name='gen-prelu-1')

        x = tf.reshape(x, (self.batch_size, 7, 7, self.gf_dim * 8))

        # x = deconv2d(x, f=self.gf_dim * 16, k=4, s=1, reuse=share_params, name='gen-deconv2d-0')
        # x = batch_norm(x, reuse=share_params, training=training, name="gen-bn-0")
        # x = prelu(x, reuse=share_params, name='gen-prelu-1')

        x = t.deconv2d(x, f=self.gf_dim * 4, k=3, s=2, reuse=share_params, name='gen-deconv2d-1')
        x = t.batch_norm(x, reuse=share_params, is_train=training, name="gen-bn-1")
        x = t.prelu(x, reuse=share_params, name='gen-prelu-2')

        x = t.deconv2d(x, f=self.gf_dim * 2, k=3, s=2, reuse=share_params, name='gen-deconv2d-2')
        x = t.batch_norm(x, reuse=share_params, is_train=training, name="gen-bn-2")
        x = t.prelu(x, reuse=share_params, name='gen-prelu-3')

        with tf.variable_scope("generator-%s" % name, reuse=reuse):
            x = t.deconv2d(x, f=self.channel, k=6, s=1, reuse=False, name='gen-' + name + '-deconv2d-3')
            x = tf.nn.sigmoid(x, name='gen' + name + '-sigmoid-0')

        return x

    def build_cogan(self):
        # Generator
        self.g_1 = self.generator(self.z, self.y, share_params=False, reuse=False, name='g1')
        self.g_2 = self.generator(self.z, self.y, share_params=True, reuse=False, name='g2')

        self.g_sample_1 = self.generator(self.z, self.y, share_params=True, reuse=True, training=False, name='g1')
        self.g_sample_2 = self.generator(self.z, self.y, share_params=True, reuse=True, training=False, name='g2')

        # Discriminator
        d_1_real = self.discriminator(self.x_1, self.y, share_params=False, reuse=False, name='d1')
        d_2_real = self.discriminator(self.x_2, self.y, share_params=True, reuse=False, name='d2')
        d_1_fake = self.discriminator(self.g_1, self.y, share_params=True, reuse=True, name='d1')
        d_2_fake = self.discriminator(self.g_2, self.y, share_params=True, reuse=True, name='d2')

        # Losses
        d_1_real_loss = t.sce_loss(d_1_real, tf.ones_like(d_1_real))
        d_1_fake_loss = t.sce_loss(d_1_fake, tf.zeros_like(d_1_fake))
        d_2_real_loss = t.sce_loss(d_2_real, tf.ones_like(d_2_real))
        d_2_fake_loss = t.sce_loss(d_2_fake, tf.zeros_like(d_2_fake))
        self.d_1_loss = d_1_real_loss + d_1_fake_loss
        self.d_2_loss = d_2_real_loss + d_2_fake_loss
        self.d_loss = self.d_1_loss + self.d_2_loss

        g_1_loss = t.sce_loss(d_1_fake, tf.ones_like(d_1_fake))
        g_2_loss = t.sce_loss(d_2_fake, tf.ones_like(d_2_fake))
        self.g_loss = g_1_loss + g_2_loss

        # Summary
        tf.summary.scalar("loss/d_1_real_loss", d_1_real_loss)
        tf.summary.scalar("loss/d_1_fake_loss", d_1_fake_loss)
        tf.summary.scalar("loss/d_2_real_loss", d_2_real_loss)
        tf.summary.scalar("loss/d_2_fake_loss", d_2_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_1_loss", g_1_loss)
        tf.summary.scalar("loss/g_2_loss", g_2_loss)
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
