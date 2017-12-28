import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


class AdaMaxOptimizer:

    def __init__(self, learning_rate=1e-3, beta1=0.5, beta2=0.9, use_locking=True):

        """
        :param learning_rate: training learning rate, default 1e-3
        :param beta1: ada optimizer beta1 (momentum), default 0.5
        :param beta2: ada optimizer beta2 (momentum), default 0.9
        :param use_locking: using locking, default True
        """

        self.lr = tf.convert_to_tensor(learning_rate, name='lr')
        self.beta1 = tf.convert_to_tensor(beta1, name='beta1')
        self.beta2 = tf.convert_to_tensor(beta2, name='beta2')
        self.use_locking = tf.convert_to_tensor(use_locking, name='lock')


class MAGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, channel=1, n_classes=10,
                 sample_num=64, sample_size=8, output_height=28, output_width=28,
                 n_input=784, df_dim=64, gf_dim=64, fc_d_unit=32, fc_g_unit=1024,
                 z_dim=128, g_lr=5e-4, d_lr=5e-4, epsilon=1e-12):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 28
        :param input_width: input image width, default 28
        :param channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input dataset's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8
        :param output_height: output images height, default 28
        :param output_width: output images width, default 28

        # For CNN model
        :param n_input: input image size, default 784(28x28)
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_d_unit: the number of fully connected filters used in D net, default 32
        :param fc_g_unit: the number of fully connected filters used in G net, default 1024

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 5e-4
        :param d_lr: discriminator learning rate, default 5e-4
        :param epsilon: epsilon, default 1e-12
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.channel = channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.channel]
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.n_input = n_input
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_d_unit = fc_d_unit
        self.fc_g_unit = fc_g_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon
        self.pt_lambda = 0.1

        self.g_loss = 0.
        self.d_loss = 0.
        self.d_real_loss = 0.
        self.d_fake_loss = 0.

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=self.image_shape, name="x-image")               # (-1, 28, 28, 1)
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='z-noise')  # (-1, 128)
        self.m = tf.placeholder(tf.float32, name='margin')

        self.build_magan()  # build MAGAN model

    def encoder(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: embeddings
        """
        with tf.variable_scope('encoder', reuse=reuse):
            x = tf.layers.conv2d(x,
                                 filters=self.fc_d_unit * 2,
                                 kernel_size=4, strides=2, padding='SAME', name='enc-conv-1')
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, units=self.fc_d_unit, name='enc-fc-1')

            return x

    def decoder(self, x, reuse=None):
        """
        :param x: embeddings
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope('decoder', reuse=reuse):
            x = tf.layers.dense(x, units=self.fc_d_unit * 2 * 14 * 14, name='dec-fc-1')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [self.batch_size, 14, 14, self.fc_d_unit * 2])

            x = tf.layers.conv2d_transpose(x, filters=1,
                                           kernel_size=4, strides=2, padding='SAME', name='dec-deconv-1')
            x = tf.nn.sigmoid(x)

            return x

    def discriminator(self, x, reuse=None):
        """
        # architecture
        : (64)4c2s-FC32-FC64*14*14_BR-(1)4dc2s_S
        :param x: images
        :param reuse: re-usable
        :return: prob, embeddings, gen-ed_image
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            embeddings = self.encoder(x, reuse=reuse)
            decoded = self.decoder(embeddings, reuse=reuse)

            return embeddings, decoded

    def generator(self, z, reuse=None):
        """
        # architecture
        : (FC1024_BR)-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        :param z: embeddings
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            # x = tf.layers.dense(z, units=self.fc_g_unit, name='g-fc-1')
            # x = tf.nn.leaky_relu(x)

            x = tf.layers.dense(z, units=7 * 7 * int(self.fc_g_unit / 4), name='g-fc-2')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [self.batch_size, 7, 7, int(self.fc_g_unit / 4)])

            x = tf.layers.conv2d_transpose(x, filters=self.gf_dim,
                                           kernel_size=4, strides=2, padding='SAME', name='g-deconv-1')
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d_transpose(x, filters=1,
                                           kernel_size=4, strides=2, padding='SAME', name='g-deconv-2')
            x = tf.nn.sigmoid(x)

            return x

    def build_magan(self):
        def mse_loss(pred, data, n=self.batch_size):
            """
            :param pred: prediction
            :param data: image
            :param n: batch_size
            :return: MSE(Mean Square Error) loss
            """
            return tf.sqrt(2. * tf.nn.l2_loss(pred - data)) / n

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        _, d_real = self.discriminator(self.x)
        _, d_fake = self.discriminator(self.g, reuse=True)

        self.d_real_loss = mse_loss(d_real, self.x)
        self.d_fake_loss = mse_loss(d_fake, self.g)
        self.d_loss = tf.reduce_mean(self.d_real_loss + tf.maximum(0., self.m - self.d_fake_loss))
        self.g_loss = tf.reduce_mean(self.d_fake_loss)

        # Summary
        tf.summary.histogram("z-noise", self.z)

        tf.summary.image("g", self.g)  # generated images by Generative Model
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("d_real_loss", self.d_real_loss)
        tf.summary.scalar("d_fake_loss", self.d_fake_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Optimizer
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]

        self.d_real_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                                beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.d_real_loss, var_list=d_params)

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1,
                                           beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1,
                                           beta2=self.beta2).minimize(self.g_loss, var_list=g_params)
        """
        To-Do
        : Implement AdaMaxOptimizer
        : Using AdaMax instead of Adam
        """

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
