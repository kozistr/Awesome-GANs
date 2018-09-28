import tensorflow as tf
import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class GAN:

    def __init__(self, s, batch_size=32, height=28, width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 n_input=784, fc_unit=128, z_dim=128, g_lr=8e-4, d_lr=8e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input dataset's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param fc_unit: fully connected units, default 128

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 8e-4
        :param d_lr: discriminator learning rate, default 8e-4
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.n_input = n_input
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.d_lr, self.g_lr = d_lr, g_lr

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.

        self.g = None

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
            x = t.dense(x, self.fc_unit, name='disc-fc-1')
            x = tf.nn.leaky_relu(x)

            x = t.dense(x, 1, name='discd-fc-2')

            return x

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            x = t.dense(z, self.fc_unit, name='gen-fc-1')
            x = tf.nn.leaky_relu(x)

            x = t.dense(x, self.n_input, name='gen-fc-2')
            x = tf.nn.sigmoid(x)

            return x

    def build_gan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # General GAN loss function referred in the paper
        """
        self.g_loss = -tf.reduce_mean(t.safe_log(d_fake))
        self.d_loss = -tf.reduce_mean(t.safe_log(d_real) + t.safe_log(1. - d_fake))
        """

        # Softmax Loss
        # Z_B = sigma x∈B exp(−μ(x)), −μ(x) is discriminator
        z_b = tf.reduce_sum(tf.exp(-d_real)) + tf.reduce_sum(tf.exp(-d_fake)) + t.eps

        b_plus = self.batch_size
        b_minus = self.batch_size * 2

        # L_G = sigma x∈B+ μ(x)/abs(B) + sigma x∈B- μ(x)/abs(B) + ln(Z_B), B+ : batch _size
        self.g_loss = tf.reduce_sum(d_real / b_plus) + tf.reduce_sum(d_fake / b_minus) + t.safe_log(z_b)

        # L_D = sigma x∈B+ μ(x)/abs(B) + ln(Z_B), B+ : batch _size
        self.d_loss = tf.reduce_sum(d_real / b_plus) + t.safe_log(z_b)

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

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
