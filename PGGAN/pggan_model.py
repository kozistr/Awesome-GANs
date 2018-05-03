import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # reproducibility
np.random.seed(777)      # reproducibility


he_normal = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
l2_reg = tf.contrib.layers.l2_regularizer


def conv2d(x, f=64, k=3, d=1, reg=5e-4, act=None, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param d: strides, default 2
    :param reg: weight regularizer, default 5e-4
    :param act: activation function, default None
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: conv2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=d,
                            kernel_initializer=he_normal,
                            kernel_regularizer=l2_reg(reg),
                            bias_initializer=tf.zeros_initializer(),
                            activation=act,
                            padding=pad,
                            name=name)


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


class PGGAN:

    def __init__(self, s, batch_size=16, input_height=128, input_width=128, input_channel=3,
                 pg=1, pg_t=False, sample_num=1 * 1, sample_size=1, output_height=128, output_width=128,
                 df_dim=64, gf_dim=64, z_dim=512, lr=1e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 16
        :param input_height: input image height, default 128
        :param input_width: input image width, default 128
        :param input_channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 128x128x3(HWC).

        # Output Settings
        :param pg: size of the image for model?, default 1
        :param pg_t: pg status, default False
        :param sample_num: the number of output images, default 1
        :param sample_size: sample image size, default 1
        :param output_height: output images height, default 128
        :param output_width: output images width, default 128

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
        :param z_dim: z dimension (kinda noise), default 512
        :param lr: learning rate, default 1e-4
        :param epsilon: epsilon, default 1e-9
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.pg = pg
        self.pg_t = pg_t
        self.output_size = 4 * pow(2, self.pg - 1)

        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.z_dim = z_dim
        self.beta1 = 0.
        self.beta2 = .99
        self.lr = lr
        self.eps = epsilon

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.g_loss = 0.
        self.d_loss = 0.
        self.gp = 0.
        self.lambda_ = 10.
        self.k = 1e-3

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.output_size, self.output_size, self.input_channel],
                                name="x-image")
        self.z = tf.placeholder(tf.float32,
                                shape=[None, self.z_dim],
                                name='z-noise')
        self.alpha_trans = tf.Variable(initial_value=0., trainable=False, name='alpha_trans')

        self.build_pggan()  # build PGGAN model

    def discriminator(self, x, pg, pg_t, reuse=None):

        with tf.variable_scope("disc", reuse=reuse):

            return x

    def generator(self, z, pg, pg_t, reuse=None):

        with tf.variable_scope("gen", reuse=reuse):

            return z

    def build_pggan(self):
        def l1_loss(x, y):
            return tf.reduce_mean(tf.abs(x - y))

        # Generator
        self.g = self.generator(self.z, self.pg, self.pg_t)

        # Discriminator
        d_real = self.discriminator(self.x, self.pg, self.pg_t)
        d_fake = self.discriminator(self.g, self.pg, self.pg_t, reuse=True)

        # Loss
        d_real_loss = l1_loss(self.x, d_real)
        d_fake_loss = l1_loss(self.g, d_fake)
        self.d_loss = d_real_loss - d_fake_loss
        self.g_loss = d_fake_loss

        # Gradient Penalty
        diff = self.g - self.x
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interp = self.x + (alpha * diff)
        d_interp = self.discriminator(interp, self.pg, self.pg_t, reuse=True)
        grads = tf.gradients(d_interp, [interp])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1, 2, 3]))
        self.gp = tf.reduce_mean((slopes - 1.) ** 2)

        self.d_loss = self.lambda_ * self.gp + self.k

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("misc/gp", self.gp)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
