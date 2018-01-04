import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


class GAN:

    def __init__(self, s, batch_size=32, input_height=28, input_width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10, output_height=28, output_width=28,
                 n_input=784, fc_unit=128,
                 z_dim=128, g_lr=8e-4, d_lr=8e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
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

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param fc_unit: fully connected units, default 128

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 8e-4
        :param d_lr: discriminator learning rate, default 8e-4
        :param epsilon: epsilon, default 1e-9
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.channel = channel
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.n_input = n_input
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon

        '''
        self.batch = tf.Variable(0)
        self.opt = lambda x: tf.train.exponential_decay(
            learning_rate=x,
            global_step=self.batch,
            decay_rate=0.95,
            decay_steps=100,
            staircase=True
        )
        '''
        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")  # (-1, 784)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')    # (-1, 100)

        self.build_gan()  # build GAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, self.fc_unit, activation=tf.nn.leaky_relu, name='d-fc-1')
            x = tf.layers.dense(x, 1, activation=None, name='d-fc-2')

        return x

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(z, self.fc_unit, activation=tf.nn.leaky_relu, name='g-fc-1')
            x = tf.layers.dense(x, self.n_input, activation=tf.nn.sigmoid, name='g-fc-2')

        return x

    def build_gan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # General GAN loss function referred in the paper
        """
        log = lambda x: tf.log(x + self.eps)

        self.g_loss = -tf.reduce_mean(log(d_fake))
        self.d_loss = -tf.reduce_mean(log(d_real) + log(1. - d_fake))
        """

        # Softmax Loss
        # Z_B = sigma x∈B exp(−μ(x)), −μ(x) is discriminator
        z_b = tf.reduce_sum(tf.exp(-d_real)) + tf.reduce_sum(tf.exp(-d_fake)) + self.eps

        b_plus = self.batch_size
        b_minus = self.batch_size * 2

        # L_G = sigma x∈B+ μ(x)/abs(B) + sigma x∈B- μ(x)/abs(B) + ln(Z_B), B+ : batch _size
        self.g_loss = tf.reduce_sum(d_real / b_plus) + tf.reduce_sum(d_fake / b_minus) + tf.log(z_b)

        # L_D = sigma x∈B+ μ(x)/abs(B) + ln(Z_B), B+ : batch _size
        self.d_loss = tf.reduce_sum(d_real / b_plus) + tf.log(z_b)

        # Summary
        tf.summary.histogram("z-noise", self.z)

        # g = tf.reshape(self.g, shape=[-1, self.output_height, self.output_height, self.channel])
        # tf.summary.image("generated", g)  # generated images by Generative Model
        tf.summary.histogram("d_real", d_real)
        tf.summary.histogram("d_fake", d_fake)
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)

        # Optimizer
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
