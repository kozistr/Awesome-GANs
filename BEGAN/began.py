import tensorflow as tf
import tensorflow.contrib.slim as slim
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

    def __init__(self, s, input_height=64, input_width=64, output_height=64, output_width=64, channel=3,
                 sample_size=64, sample_num=64, batch_size=16,
                 z_dim=128, filter_num=64, embedding=128,
                 epsilon=1e-12, gamma=0.4, lambda_k=1e-3, momentum1=0.5, momentum2=0.999,
                 g_lr=8e-5, d_lr=8e-5, lr_low_boundary=2e-5, lr_update_step=1e5):
        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.channel = channel
        self.conv_repeat_num = int(np.log2(input_height)) - 2
        self.image_shape = [self.input_height, self.input_height, self.channel]  # 64x64x3

        self.eps = epsilon
        self.gamma = gamma  # 0.3 ~ 0.5 # 0.7
        self.lambda_k = lambda_k
        self.mm1 = momentum1  # beta1
        self.mm2 = momentum2  # beta2

        self.z_dim = z_dim
        self.filter_num = filter_num
        self.embedding = embedding

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.g_lr = g_lr
        self.d_lr = d_lr

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, lr_low_boundary, name="g_lr_update"))
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, lr_low_boundary, name="d_lr_update"))

        self.learning_rate = tf.train.exponential_decay(
            learning_rate=1e-4,
            decay_rate=0.95,
            decay_steps=2000,
            global_step=2e5,  # 200k
            staircase=False
        )

        self.build_began()

    def encoder(self, x, embedding, reuse=None):
        with tf.variable_scope("encoder", reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                stride=1, activation_fn=tf.nn.elu, padding="SAME",
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                weights_regularizer=slim.l2_regularizer(5e-4),
                                bias_initializer=tf.zeros_initializer()):
                x = slim.conv2d(x, embedding, 3)

                for i in range(self.conv_repeat_num):
                    channel_num = embedding * (i + 1)
                    x = slim.repeat(x, 2, slim.conv2d, channel_num, 3)
                    if i < self.conv_repeat_num - 1:
                        # Is using stride pooling more better method than max pooling?
                        x = slim.conv2d(x, channel_num, kernel_size=3, stride=2)  # sub-sampling
                        # x = slim.max_pooling2d(x, 3, 2)

                x = tf.reshape(x, [-1, np.prod([8, 8, channel_num]))
        return x

    def decoder(self, z, embedding, reuse=None):
        with tf.variable_scope("decoder", reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                weights_regularizer=slim.l2_regularizer(5e-4),
                                bias_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d], padding="SAME",
                                    activation_fn=tf.nn.elu, stride=1):
                    x = slim.fully_connected(z, 8 * 8 * embedding, activation_fn=None)
                    x = tf.reshape(x, [-1, 8, 8, embedding])

                    for i in range(self.conv_repeat_num):
                        x = slim.repeat(x, 2, slim.conv2d, embedding, 3)
                        if i < self.conv_repeat_num - 1:
                            x = resize_nn(x, 2)  # NN up-sampling

                    x = slim.conv2d(x, 3, 3, activation_fn=None)
        return x

    def discriminator(self, x, embedding, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            x = self.encoder(x, embedding, reuse=reuse)
            x = self.decoder(x, embedding, reuse=reuse)

            return x

    def generator(self, z, embedding, reuse=None):
        with tf.variable_scope("generator/decoder", reuse=reuse):
            x = self.decoder(z, embedding)

            return x

    def build_began(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_width], "x-image")
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.sample_size, self.sample_size, self.channel])

        self.lr = tf.placeholder(tf.float32, "learning-rate")
        self.kt = tf.placeholder(tf.float32, "kt")

        # Generator Model
        self.G = self.generator(self.x)

        # Discriminator Model
        self.D = self.decoder(self.x, reuse=True)
        self.D_real = self.discriminator(self.y)
        self.D_fake = self.discriminator(self.G, reuse=True)

