import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


def conv2d(input_, filter_=64, k=5, d=2, activation=tf.nn.leaky_relu, pad='same', name="Conv2D"):
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


def deconv2d(input_, filter_=64, k=5, d=2, activation=tf.nn.leaky_relu, pad='same', name="DeConv2D"):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(inputs=input_,
                                          filters=filter_,
                                          kernel_size=k,
                                          strides=d,
                                          padding=pad,
                                          activation=activation,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          bias_initializer=tf.constant_initializer(0.),
                                          name=name)


class EBGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, channel=1, n_classes=10,
                 sample_num=64, sample_size=8, output_height=28, output_width=28,
                 n_input=784, df_dim=64, gf_dim=64, fc_unit=1024,
                 z_dim=128, g_lr=8e-4, d_lr=8e-4, enable_pull_away=True, epsilon=1e-12):

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
        :param fc_unit: the number of fully connected filters, default 1024

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 8e-4
        :param d_lr: discriminator learning rate, default 8e-4
        :param enable_pull_away: enabling PULLAWAY loss, default True
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
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.d_lr, self.g_lr = d_lr, g_lr  # 1e-3 in the paper
        self.EnablePullAway = enable_pull_away
        self.eps = epsilon
        self.pt_lambda = 0.1
        self.margin = 10  # 10, 20

        self.g_loss = 0.
        self.d_loss = 0.
        self.pt_loss = 0.

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=self.image_shape, name="x-image")               # (-1, 28, 28, 1)
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='z-noise')  # (-1, 128)

        self.build_ebgan()  # build EBGAN model

    def encoder(self, x, reuse=None):
        with tf.variable_scope('encoder', reuse=reuse):
            x = conv2d(x, filter_=self.df_dim, name='e-conv-1')
            x = conv2d(x, filter_=self.df_dim * 2, name='e-conv-2')
            x = conv2d(x, filter_=self.df_dim * 4, name='e-conv-3')

            return x

    def decoder(self, x, reuse=None):
        with tf.variable_scope('decoder', reuse=reuse):
            x = deconv2d(x, filter_=self.df_dim, name='d-deconv-1')
            x = deconv2d(x, filter_=int(self.df_dim / 2), name='d-deconv-2')
            x = deconv2d(x, filter_=3, activation=None, name='d-deconv-3')

            return x

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            encode = self.encoder(x, reuse=reuse)
            decode = self.decoder(encode, reuse=reuse)

            prob = tf.nn.sigmoid(decode)

            return prob, encode

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(z, self.gf_dim * 2 * 7 * 7, activation=tf.nn.leaky_relu, name='g_fc_1')
            x = tf.reshape(x, [self.batch_size, 7, 7, self.gf_dim * 2])

            x = deconv2d(x, filter_=self.df_dim, k=2, d=2, name='g_deconv_1')
            x = deconv2d(x, filter_=1, k=2, d=2, activation=tf.nn.sigmoid, name='g_deconv_2')

            return x

    def build_ebgan(self):
        def pullaway_loss(embeddings, n=self.batch_size):
            """
            # PullAway Loss # 2.4 Repelling Regularizer in 1609.03126.pdf
            :param embeddings: encoded images
            :param N: batch_size
            :return: PullAway loss
            """
            normalized = embeddings / tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            similarity = tf.matmul(normalized, normalized, transpose_b=True)

            return (tf.reduce_sum(similarity) - n) / (n * (n - 1))

        def mse_loss(pred, data, n=self.batch_size):
            """
            :param pred: prediction
            :param data: image
            :param n: batch_size
            :return: MSE(Mean Square Error) loss
            """
            return tf.sqrt(tf.nn.l2_loss(pred - data)) / n

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real, _ = self.discriminator(self.x)
        d_fake, d_embed_fake = self.discriminator(self.g, reuse=True)

        d_real_loss = tf.reduce_sum(mse_loss(d_real, tf.ones_like(d_real)))
        d_fake_loss = tf.reduce_sum(mse_loss(d_fake, tf.zeros_like(d_fake)))
        zero = tf.zeros_like(self.margin - d_fake_loss)
        self.d_loss = d_real_loss + tf.maximum(zero, self.margin - d_fake_loss)

        self.g_loss = tf.reduce_sum(mse_loss(d_fake, tf.ones_like(d_fake)))

        if self.EnablePullAway:
            self.pt_loss = pullaway_loss(d_embed_fake)
            self.g_loss += + self.pt_lambda * self.pt_loss  # g_loss = mse + lambda * pt_loss

        # Summary
        tf.summary.histogram("z-noise", self.z)

        tf.summary.image("g", self.g)  # generated images by Generative Model
        # tf.summary.histogram("d_real", d_real)
        # tf.summary.histogram("d_fake", d_fake)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("pt_loss", self.pt_loss)

        # Optimizer
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
