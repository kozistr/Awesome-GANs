import tensorflow as tf
import numpy as np


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, bias=True, padding="SAME", name="conv2d"):
    if padding == "VALID":
        pad_size = (k_w - 1) / 2
        input_ = tf.pad(input_, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "SYMMETRIC")
    else:
        pass

    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.variance_scaling_initializer())

        x = tf.nn.conv2d(input_, w, [1, d_h, d_w, 1], padding=padding)

        if bias:
            b = tf.get_variable('b', shape=output_dim,
                                initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)

    return x


def avgpool(input_, k_h=2, k_w=2, d_h=1, d_w=1):
    return tf.nn.avg_pool(input_, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding="SAME")


def fc(input_, output_dim, bias=True, name="fc"):
    input_dim = np.prod(input_.get_shape().as_list()[1:])

    x = tf.reshape(input_, [-1, input_dim])

    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[input_dim, output_dim],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.matmul(x, w)

        if bias:
            b = tf.get_variable('b', shape=[output_dim],
                                initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)

    return x


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


class BEGAN:

    def __init__(self, s, input_height=128, input_width=128, channel=3,
                 output_height=128, output_width=128, sample_size=128, sample_num=64, batch_size=64,
                 z_dim=128, filter_num=64, eps=1e-12, gamma=0.4):
        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.channel = channel
        self.image_shape = [self.input_height, self.input_height, self.channel]  # 128x128x3

        self.eps = eps
        self.gamma = gamma  # 0.3 ~ 0.5 # 0.7

        self.z_dim = z_dim
        self.filter_num = filter_num

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.learning_rate = tf.train.exponential_decay(
            learning_rate=1e-4,
            decay_rate=0.95,
            decay_steps=2000,
            global_step=2e5,
            staircase=False
        )

        self.build_bdgan()

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            f = self.filter_num
            w = self.sample_num

            x = fc(x, 8 * 8 * f)
            x = tf.reshape(x, [-1, 8, 8, f])

            x = conv2d(x, f)
            x = tf.nn.elu(x)
            x = conv2d(x, f)
            x = tf.nn.elu(x)

            if self.sample_size == 128:
                x = resize_nn(x, w / 8)
                x = conv2d(x, f)
                x = tf.nn.elu(x)
                x = conv2d(x, f)
                x = tf.nn.elu(x)

            x = resize_nn(x, w / 4)
            x = conv2d(x, f)
            x = tf.nn.elu(x)
            x = conv2d(x, f)
            x = tf.nn.elu(x)

            x = resize_nn(x, w / 2)
            x = conv2d(x, f)
            x = tf.nn.elu(x)
            x = conv2d(x, f)
            x = tf.nn.elu(x)

            x = resize_nn(x, w)
            x = conv2d(x, f)
            x = tf.nn.elu(x)
            x = conv2d(x, f)
            x = tf.nn.elu(x)

            x = conv2d(x, 3)

        return x

    def generator(self, x, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            f = self.filter_num
            w = self.sample_num



    def build_bdgan(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, "x-image")
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], "z-noise")

        self.lr = tf.placeholder(tf.float32, "learning-rate")
        self.kt = tf.placeholder(tf.float32, "kt")

