import numpy as np
import tensorflow as tf

import awesome_gans.modules as t

tf.set_random_seed(777)
np.random.seed(777)


class DiscoGAN:
    def __init__(
        self,
        s,
        batch_size=64,
        height=64,
        width=64,
        channel=3,
        sample_size=32,
        sample_num=64,
        z_dim=128,
        gf_dim=32,
        df_dim=32,
        learning_rate=2e-4,
        beta1=0.5,
        beta2=0.999,
        eps=1e-9,
    ):

        self.s = s
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_num = sample_num

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = (self.height, self.width, self.channel)

        self.z_dim = z_dim

        self.eps = eps
        self.mm1 = beta1
        self.mm2 = beta2

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.lr = learning_rate

        self.build_discogan()

    def discriminator(self, x, scope_name, reuse=None):
        with tf.variable_scope("%s" % scope_name, reuse=reuse):
            x = t.conv2d(x, f=self.df_dim, k=4, s=1)  # 64 x 64 x 3
            x = tf.nn.leaky_relu(x)

            for i in range(np.log2(x.get_shape()[1]) - 2):  # 0 ~ 3
                x = t.conv2d(x, self.df_dim * (2 ** (i + 1)), k=4, s=2)
                x = t.batch_norm(x)
                x = tf.nn.leaky_relu(x)

            #  (-1, 4, 4, 512)
            x = tf.layers.flatten(x)

            x = t.dense(x, 512)
            x = tf.nn.leaky_relu(x)

            x = t.dense(x, 1)
            x = tf.sigmoid(x)

            return x

    def generator(self, z, scope_name, reuse=None, is_train=True):
        with tf.variable_scope("%s" % scope_name, reuse=reuse):
            x = t.dense(z, 4 * 4 * 8 * self.gf_dim)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)
            x = tf.reshape(x, (-1, 4, 4, 8))

            for i in range(np.log2(self.height) - 2):  # 0 ~ 3
                x = t.deconv2d(x, self.gf_dim * (2 ** (i + 1)), k=4, s=2)
                x = t.batch_norm(x, is_train=is_train)
                x = tf.nn.leaky_relu(x)

            x = t.conv2d(x, 3)

            return x

    def build_discogan(self):
        self.A = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name='trainA')
        self.B = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name='trainA')
        # generator
        # s : shoes, b : bags, 2 : to
        self.G_AB = self.generator(self.A, "generator_AB")
        self.G_BA = self.generator(self.B, "generator_BA")

        self.G_ABA = self.generator(self.G_AB, "generator_AB", reuse=True)
        self.G_BAB = self.generator(self.G_BA, "generator_BA", reuse=True)

        # discriminator
        self.D_s_real = self.discriminator(self.A, "discriminator_real")
        self.D_b_real = self.discriminator(self.B, "discriminator_fake")

        self.D_s_fake = self.discriminator(self.G_ABA, "discriminator_real", reuse=True)
        self.D_b_fake = self.discriminator(self.G_BAB, "discriminator_fake", reuse=True)

        # loss
        self.s_loss = tf.reduce_sum(tf.losses.mean_squared_error(self.A, self.G_ABA))
        self.b_loss = tf.reduce_sum(tf.losses.mean_squared_error(self.B, self.G_BAB))

        # self.g_shoes_loss = tf.reduce_sum(tf.square(self.D_s_fake - 1)) / 2
        # self.g_bags_loss = tf.reduce_sum(tf.square(self.D_b_fake - 1)) / 2

        # self.d_shoes_real_loss = tf.reduce_sum(tf.square(self.D_s_real - 1)) / 2
        # self.d_shoes_fake_loss = tf.reduce_sum(tf.square(self.D_s_fake)) / 2
        # self.d_bags_real_loss = tf.reduce_sum(tf.square(self.D_b_real - 1)) / 2
        # self.d_bags_fake_loss = tf.reduce_sum(tf.square(self.D_b_fake)) / 2

        # self.d_shoes_loss = self.d_shoes_real_loss + self.d_shoes_fake_loss
        # self.d_bags_loss = self.d_bags_real_loss + self.d_bags_fake_loss

        # self.g_loss = 10 * (self.s_loss + self.b_loss) + self.g_shoes_loss + self.g_bags_loss
        # self.d_loss = self.d_shoes_loss + self.d_bags_loss
        # sigmoid cross entropy loss
        self.g_s_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_s_fake, labels=tf.ones_like(self.D_s_fake))
        )
        self.g_b_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_b_fake, labels=tf.ones_like(self.D_b_fake))
        )

        self.d_s_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_s_real, labels=tf.ones_like(self.D_s_real))
        )
        self.d_b_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_b_real, labels=tf.ones_like(self.D_b_real))
        )
        self.d_s_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_s_fake, labels=tf.zeros_like(self.D_s_fake))
        )
        self.d_b_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_b_fake, labels=tf.zeros_like(self.D_b_fake))
        )

        self.g_loss = (self.s_loss + self.g_s_loss) + (self.b_loss + self.g_b_loss)

        self.d_s_loss = self.d_s_real_loss + self.d_s_fake_loss
        self.d_b_loss = self.d_b_real_loss + self.d_b_fake_loss
        self.d_loss = self.d_s_loss + self.d_b_loss

        # collect trainer values
        # vars = tf.trainable_variables()
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_*")
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator_*")

        # summary
        self.g_shoes_sum = tf.summary.histogram("G_A", self.g_s_loss)
        self.g_bags_sum = tf.summary.histogram("G_B", self.g_b_loss)

        self.d_shoes_sum = tf.summary.histogram("D_A", self.d_s_loss)
        self.d_bags_sum = tf.summary.histogram("D_B", self.d_b_loss)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.saver = tf.train.Saver()

        # train op
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.mm1, beta2=self.mm2).minimize(
            self.g_loss, var_list=self.g_vars
        )
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.mm1, beta2=self.mm2).minimize(
            self.d_loss, var_list=self.d_vars
        )

        # merge summary
        self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_shoes_sum, self.g_bags_sum])
        self.d_sum = tf.summary.merge([self.d_loss_sum, self.d_shoes_sum, self.d_bags_sum])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
