import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


class BEGAN:

    def __init__(self, s, input_height=64, input_width=64, output_height=64, output_width=64, channel=3,
                 sample_size=64, sample_num=64, embedding=100, batch_size=16,
                 gamma=0.4, lambda_k=1e-3, momentum1=0.5, momentum2=0.999,
                 g_lr=8e-5, d_lr=8e-5, lr_low_boundary=2e-5):
        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.channel = channel
        self.conv_repeat_num = int(np.log2(input_height)) - 2
        self.image_shape = [self.input_height, self.input_height, self.channel]  # 64x64x3

        self.gamma = gamma  # 0.3 ~ 0.5 # 0.7
        self.lambda_k = lambda_k
        self.mm1 = momentum1  # beta1
        self.mm2 = momentum2  # beta2

        self.sample_size = sample_size
        self.sample_num = sample_num
        self.embedding = embedding

        self.g_lr = g_lr
        self.d_lr = d_lr

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, lr_low_boundary, name="g_lr_update"))
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, lr_low_boundary, name="d_lr_update"))

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
                        # or average pooling
                        # x = slim.conv2d(x, channel_num, kernel_size=3, stride=2)  # sub-sampling
                        x = slim.avg_pool2d(x, kernel_size=2, stride=2)
                        # x = slim.max_pooling2d(x, 3, 2)

                x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
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
            z = self.encoder(x, embedding, reuse=reuse)
            x = self.decoder(z, embedding, reuse=reuse)

            return x

    def generator(self, z, embedding, reuse=None):
        with tf.variable_scope("generator/decoder", reuse=reuse):
            x = self.decoder(z, embedding)

            return x

    def build_began(self):
        # x, z placeholder
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size] + self.image_shape, name='x-images')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.embedding], name='z-noise')

        self.lr = tf.placeholder(tf.float32, "learning-rate")
        self.kt = tf.placeholder(tf.float32, "k_t")

        # Generator Model
        self.G = self.generator(self.z, self.embedding)

        # Discriminator Model (Critic)
        self.D_real = self.discriminator(self.x, self.embedding)  # discriminator
        self.D_fake = self.discriminator(self.G, self.embedding, reuse=True)  # discriminate

        # Generator & Discriminator loss(l1 loss)
        # loss of D = loss of D_real - kt * loss of D_fake (0 < k_t < 1, k_0 = 0)
        # loss of G = loss of D_fake
        self.D_real_loss = l1_loss(self.x, self.D_real)
        self.D_fake_loss = l1_loss(self.G, self.D_fake)
        self.D_loss = self.D_real_loss - self.kt * self.D_fake_loss

        self.G_loss = self.D_fake_loss

        self.M_global = self.D_real_loss + (self.gamma * self.D_real_loss * self.D_fake_loss)

        # summary
        self.z_sum = tf.summary.histogram("z", self.z)
        self.lr_sum = tf.summary.histogram("lr", self.lr)
        self.kt_sum = tf.summary.histogram("k_t", self.kt)

        self.G_sum = tf.summary.image("G", self.G)  # generated image from G model
        self.D_real_sum = tf.summary.histogram("D_real", self.D_real)
        self.D_fake_sum = tf.summary.histogram("D_fake", self.D_fake)

        self.d_real_loss_sum = tf.summary.scalar("d_real_loss", self.D_real_loss)
        self.d_fake_loss_sum = tf.summary.scalar("d_fake_loss", self.D_fake_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)

        # collect trainer values
        vars = tf.trainable_variables()
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

        # model saver
        self.saver = tf.train.Saver()

        # optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr_update, beta1=self.mm1).\
            minimize(self.D_loss, var_list=self.d_vars)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr_update, beta1=self.mm1).\
            minimize(self.G_loss, var_list=self.g_vars)

        # merge summary
        self.g_sum = tf.summary.merge([self.z_sum, self.D_fake_sum, self.G_sum, self.d_fake_loss_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.D_real_sum, self.d_real_loss_sum, self.d_loss_sum])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
