import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


class BGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1, n_classes=10,
                 sample_num=64, sample_size=8, output_height=28, output_width=28,
                 n_input=784, n_hidden_layer_1=128,
                 z_dim=100, g_lr=1e-3, d_lr=1e-3, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
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

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param n_hidden_layer_1: first NN hidden layer, default 128

        # Training Option
        :param z_dim: z dimension (kinda noise), default 100
        :param g_lr: generator learning rate, default 1e-3
        :param d_lr: discriminator learning rate, default 1e-3
        :param epsilon: epsilon, default 1e-9
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
        self.n_hl_1 = n_hidden_layer_1

        self.z_dim = z_dim
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon

        self.W = {
            # for discriminator
            'd_h1': tf.get_variable('d_h1',
                                    shape=[self.n_input, self.n_hl_1],
                                    initializer=tf.contrib.layers.variance_scaling_initializer()),
            'd_h_out': tf.get_variable('d_h_out',
                                       shape=[self.n_hl_1, 1],
                                       initializer=tf.contrib.layers.variance_scaling_initializer()),
            # for generator
            'g_h1': tf.get_variable('g_h1',
                                    shape=[self.z_dim, self.n_hl_1],
                                    initializer=tf.contrib.layers.variance_scaling_initializer()),
            'g_h_out': tf.get_variable('g_h_out',
                                       shape=[self.n_hl_1, self.n_input],
                                       initializer=tf.contrib.layers.variance_scaling_initializer()),
        }

        self.b = {
            # for discriminator
            'd_b1': tf.Variable(tf.zeros([self.n_hl_1])),
            'd_b_out': tf.Variable(tf.zeros([1])),
            # for generator
            'g_b1': tf.Variable(tf.zeros([self.n_hl_1])),
            'g_b_out': tf.Variable(tf.zeros([self.n_input])),
        }

        self.d_loss = 0.
        self.g_loss = 0.

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")  # (-1, 784)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')    # (-1, 100)

        self.build_bgan()  # build BGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.nn.bias_add(tf.matmul(x, self.W['d_h1']), self.b['d_b1'])
            net = tf.nn.leaky_relu(net)

            logits = tf.nn.bias_add(tf.matmul(net, self.W['d_h_out']), self.b['d_b_out'])
            prob = tf.nn.sigmoid(logits)

        return prob, logits

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            de_net = tf.nn.bias_add(tf.matmul(z, self.W['g_h1']), self.b['g_b1'])
            de_net = tf.nn.leaky_relu(de_net)

            logits = tf.nn.bias_add(tf.matmul(de_net, self.W['g_h_out']), self.b['g_b_out'])
            prob = tf.nn.sigmoid(logits)

        return prob

    def build_bgan(self):
        def log(x):
            return tf.log(x + self.eps)

        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real, _ = self.discriminator(self.x)
        d_fake, _ = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = -tf.reduce_mean(log(d_real))
        d_fake_loss = -tf.reduce_mean(log(1. - d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(tf.square(log(d_fake) + log(1. - d_fake))) / 2

        # Summary
        tf.summary.histogram("z-noise", self.z)

        g = tf.reshape(self.g, shape=self.image_shape)
        tf.summary.image("generated", g)  # generated images by Generative Model
        tf.summary.scalar("d_real_loss", d_real_loss)
        tf.summary.scalar("d_fake_loss", d_fake_loss)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Optimizer
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('d')]
        g_params = [v for v in vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(self.d_lr).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(self.g_lr).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
