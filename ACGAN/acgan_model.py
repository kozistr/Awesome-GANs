import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


def conv2d(input_, filter_=64, k=(5, 5), d=(1, 1), activation=tf.nn.leaky_relu, pad='same', name="Conv2D"):
    with tf.variable_scope(name):
        return tf.layers.conv2d(inputs=input_,
                                filters=filter_,
                                kernel_size=k,
                                strides=d,
                                padding=pad,
                                activation=activation,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                bias_initializer=tf.constant_initializer(0.),
                                name=name)


class ACGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1, n_classes=10,
                 sample_num=64, sample_size=8, output_height=28, output_width=28,
                 n_input=784, df_dim=64, gf_dim=64, fc_unit=1024,
                 z_dim=128, g_lr=1e-3, d_lr=2e-4, c_lr=1e-3, epsilon=1e-12):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 28
        :param input_width: input image width, default 28
        :param input_channel: input image channel, default 1 (gray-scale)
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
        :param fc_unit: the number of fully connected filters, default 1024

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 1e-3
        :param d_lr: discriminator learning rate, default 2e-4
        :param c_lr: classifier learning rate, default 1e-3
        :param epsilon: epsilon, default 1e-12
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.n_input = n_input
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.d_lr, self.g_lr, self.c_lr = d_lr, g_lr, c_lr
        self.eps = epsilon

        self.g_loss = 0.
        self.d_loss = 0.
        self.c_loss = 0.

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=self.image_shape, name="x-image")                   # (bs, 28, 28, 1)
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_classes], name="y-label")  # (bs, 10)
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name="z-noise")      # (bs, 128)

        self.build_acgan()  # build ACGAN model

    def classifier(self, x, reuse=None):
        """
        :param x: image, shape=(batch_size, 28, 28, 1)
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("classifier", reuse=reuse):
            x = tf.layers.dense(x, self.fc_unit / 4, activation=tf.nn.leaky_relu, name='c-fc-1')
            x = tf.layers.dense(x, self.n_classes, name='c-fc-2')

            return x

    def discriminator(self, x, reuse=None):
        """
        :param x: image, shape=(batch_size, 28, 28, 1)
        :param reuse: re-usable
        :return: logits, networks
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv2d(x, self.df_dim, name='d_conv_1')
            # x = tf.layers.dropout(x, 0.5, name='d_dropout_1')

            x = conv2d(x, self.df_dim * 2, name='d_conv_2')
            # x = tf.layers.dropout(x, 0.5, name='d_dropout_2')

            x = tf.layers.flatten(x)

            net = tf.layers.dense(x, self.fc_unit / 2, activation=tf.nn.leaky_relu, name='d_fc_1')
            # x = tf.layers.dropout(x, 0.5, name='d_dropout_3')

            x = tf.layers.dense(net, 1, name='d_fc_2')  # logits

            return x, net

    def generator(self, y, z, reuse=None):
        """
        :param y: image label
        :param z: image noise
        :param reuse: re-usable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([z, y], axis=1)

            x = tf.layers.dense(x, self.fc_unit / 2, name='g_fc_0')
            x = tf.layers.dense(x, self.gf_dim * 2 * 7 * 7, activation=tf.nn.leaky_relu, name='g_fc_1')
            x = tf.reshape(x, [self.batch_size, 7, 7, self.gf_dim * 2])

            # x = tf.layers.flatten(x)

            x = tf.layers.conv2d_transpose(x, filters=self.gf_dim,
                                           kernel_size=2, strides=2,
                                           activation=tf.nn.leaky_relu, name='g_deconv_1')
            x = tf.layers.conv2d_transpose(x, filters=1,
                                           kernel_size=2, strides=2,
                                           activation=tf.nn.sigmoid, name='g_deconv_2')

            return x

    def build_acgan(self):
        # Generator
        self.g = self.generator(self.y, self.z)

        # Discriminator
        d_real, d_real_net = self.discriminator(self.x)
        d_fake, d_fake_net = self.discriminator(self.g, reuse=True)

        # Classifier
        c_real = self.classifier(d_real_net)
        c_fake = self.classifier(d_fake_net, reuse=True)

        # sigmoid CE Loss
        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real,
                                                                             labels=tf.ones_like(d_real)))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                                             labels=tf.zeros_like(d_fake)))
        self.d_loss = d_real_loss + d_fake_loss

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                                             labels=tf.ones_like(d_fake)))

        c_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_real, labels=self.y))
        c_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_fake, labels=self.y))
        self.c_loss = c_real_loss + c_fake_loss

        # Summary
        tf.summary.histogram("z-noise", self.z)

        tf.summary.image("g", self.g)  # generated images by Generative Model
        tf.summary.histogram("d_real", d_real)
        tf.summary.histogram("d_fake", d_fake)
        tf.summary.histogram("c_real", c_real)
        tf.summary.histogram("c_fake", c_fake)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("c_loss", self.c_loss)

        # Optimizer
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]
        c_params = [v for v in vars if v.name.startswith('c')]

        self.d_op = tf.train.AdamOptimizer(self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)
        self.c_op = tf.train.AdamOptimizer(self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.c_loss, var_list=c_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
