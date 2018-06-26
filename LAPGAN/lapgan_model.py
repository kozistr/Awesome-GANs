import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)


# In image_utils, up/down_sampling
def image_sampling(img, sampling_type='down'):
    shape = img.get_shape()  # [batch, height, width, channels]

    if sampling_type == 'down':
        h = int(shape[1] // 2)
        w = int(shape[2] // 2)
    else:  # 'up'
        h = int(shape[1] * 2)
        w = int(shape[2] * 2)

    return tf.image.resize_images(img, [h, w], tf.image.ResizeMethod.BILINEAR)


class LAPGAN:

    def __init__(self, s, batch_size=128, height=32, width=32, channel=3, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 z_dim=128, gf_dim=64, df_dim=64, d_fc_unit=512, g_fc_unit=1024):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 128
        :param height: input image height, default 32
        :param width: input image width, default 32
        :param channel: input image channel, default 3 (RGB)
        :param n_classes: the number of classes, default 10
        - in case of CIFAR, image size is 32x32x3(HWC), classes are 10.

        # Output Settings
        :param sample_size: sample image size, default 8
        :param sample_num: the number of sample images, default 64

        # Model Settings
        :param z_dim: z noise dimension, default 128
        :param gf_dim: the number of generator filters, default 64
        :param df_dim: the number of discriminator filters, default 64
        :param d_fc_unit: the number of fully connected filters used at Disc, default 512
        :param g_fc_unit: the number of fully connected filters used at Gen, default 1024
        """

        self.s = s
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel
        self.n_classes = n_classes

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.image_shape = [self.height, self.width, self.channel]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.d_fc_unit = d_fc_unit
        self.g_fc_unit = g_fc_unit

        # Placeholders
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes],
                                name='y-classes')  # one_hot
        self.x1_fine = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel],
                                      name='x-images')

        self.x1_scaled = image_sampling(self.x1_fine, 'down')
        self.x1_coarse = image_sampling(self.x1_scaled, 'up')
        self.x1_diff = self.x1_fine - self.x1_coarse

        self.x2_fine = self.x1_scaled  # [16, 16]
        self.x2_scaled = image_sampling(self.x2_fine, 'down')
        self.x2_coarse = image_sampling(self.x2_scaled, 'up')
        self.x2_diff = self.x2_fine - self.x2_coarse

        self.x3_fine = self.x2_scaled  # [8, 8]

        self.z = []
        self.z_noises = [32 * 32, 16 * 16, self.z_dim]
        for i in range(3):
            self.z.append(tf.placeholder(tf.float32,
                                         shape=[None, self.z_noises[i]],
                                         name='z-noise_{0}'.format(i)))

        self.do_rate = tf.placeholder(tf.float32, None, name='do-rate')

        self.g = []       # generators
        self.g_loss = []  # generator losses

        self.d_reals = []       # discriminator_real logits
        self.d_fakes = []       # discriminator_fake logits
        self.d_reals_prob = []  # discriminator_real probs
        self.d_fakes_prob = []  # discriminator_fake probs
        self.d_loss = []        # discriminator_real losses

        # Training Options
        self.d_op = []
        self.g_op = []

        self.beta1 = 0.5
        self.beta2 = 0.9
        self.lr = 8e-4

        self.saver = None
        self.merged = None
        self.writer = None

        self.bulid_lapgan()  # build LAPGAN model

    def discriminator(self, x1, x2, y, scale=32, reuse=None):
        """
        :param x1: image to discriminate
        :param x2: down-up sampling-ed images
        :param y: classes
        :param scale: image size
        :param reuse: variable re-use
        :return: logits
        """

        assert (scale % 8 == 0)  # 32, 16, 8

        with tf.variable_scope('discriminator_{0}'.format(scale), reuse=reuse):
            if scale == 8:
                x1 = tf.reshape(x1, [-1, scale * scale * 3])  # tf.layers.flatten(x1)

                h = tf.concat([x1, y], axis=1)

                h = t.dense(h, self.d_fc_unit, name='disc-fc-1')
                h = tf.nn.relu(h)
                h = tf.layers.dropout(h, 0.5, name='disc-dropout-1')

                h = t.dense(h, self.d_fc_unit // 2, name='d-fc-2')
                h = tf.nn.relu(h)
                h = tf.layers.dropout(h, 0.5, name='disc-dropout-2')

                h = t.dense(h, 1, name='disc-fc-3')
            else:
                x = x1 + x2

                y = t.dense(y, scale * scale, name='disc-fc-y')
                y = tf.nn.relu(y)

                y = tf.reshape(y, [-1, scale, scale, 1])

                h = tf.concat([x, y], axis=3)  # (-1, scale, scale, channel + 1)

                h = t.conv2d(h, self.df_dim * 1, 5, 1, pad='SAME', name='disc-conv2d-1')
                h = tf.nn.relu(h)
                h = tf.layers.dropout(h, 0.5, name='disc-dropout-1')

                h = t.conv2d(h, self.df_dim * 2, 5, 1, pad='SAME', name='disc-conv2d-2')
                h = tf.nn.relu(h)
                h = tf.layers.dropout(h, 0.5, name='disc-dropout-2')

                h = tf.layers.flatten(h)

                h = t.dense(h, 1, name='disc-fc-2')

            return h

    def generator(self, x, y, z, scale=32, reuse=None, do_rate=0.5):
        """
        :param x: images to fake
        :param y: classes
        :param z: noise
        :param scale: image size
        :param reuse: variable re-use
        :param do_rate: dropout rate
        :return: logits
        """

        assert(scale % 8 == 0)  # 32, 16, 8

        with tf.variable_scope('generator_{0}'.format(scale), reuse=reuse):
            if scale == 8:
                h = tf.concat([z, y], axis=1)

                h = t.dense(h, self.g_fc_unit, name='gen-fc-1')
                h = tf.nn.relu(h)
                h = tf.layers.dropout(h, do_rate, name='gen-dropout-1')

                h = t.dense(h, self.g_fc_unit, name='gen-fc-2')
                h = tf.nn.relu(h)
                h = tf.layers.dropout(h, do_rate, name='gen-dropout-2')

                h = t.dense(h, self.channel * 8 * 8, name='gen-fc-3')

                h = tf.reshape(h, [-1, 8, 8, self.channel])
            else:
                y = t.dense(y, scale * scale, name='gen-fc-0')

                y = tf.reshape(y, [-1, scale, scale, 1])
                z = tf.reshape(z, [-1, scale, scale, 1])

                h = tf.concat([z, y, x], axis=3)  # concat into 5 dims

                h = t.deconv2d(h, self.gf_dim * 2, 5, 1, name='gen-deconv2d-1')
                h = tf.nn.relu(h)

                h = t.deconv2d(h, self.gf_dim * 1, 5, 1, name='gen-deconv2d-2')
                h = tf.nn.relu(h)

                h = t.deconv2d(h, self.channel, 5, 1, name='gen-deconv2d-3')

            h = tf.nn.tanh(h)

            return h

    def bulid_lapgan(self):
        # Generator & Discriminator
        g1 = self.generator(x=self.x1_coarse, y=self.y, z=self.z[0], scale=32, do_rate=self.do_rate)
        d1_fake = self.discriminator(x1=g1, x2=self.x1_coarse, y=self.y, scale=32)
        d1_real = self.discriminator(x1=self.x1_diff, x2=self.x1_coarse, y=self.y, scale=32, reuse=True)

        g2 = self.generator(x=self.x2_coarse, y=self.y, z=self.z[1], scale=16, do_rate=self.do_rate)
        d2_fake = self.discriminator(x1=g2, x2=self.x2_coarse, y=self.y, scale=16)
        d2_real = self.discriminator(x1=self.x2_diff, x2=self.x2_coarse, y=self.y, scale=16, reuse=True)

        g3 = self.generator(x=None, y=self.y, z=self.z[2], scale=8, do_rate=self.do_rate)
        d3_fake = self.discriminator(x1=g3, x2=None, y=self.y, scale=8)
        d3_real = self.discriminator(x1=self.x3_fine, x2=None, y=self.y, scale=8, reuse=True)

        self.g = [g1, g2, g3]
        self.d_reals = [d1_real, d2_real, d3_real]
        self.d_fakes = [d1_fake, d2_fake, d3_fake]

        # Losses
        with tf.variable_scope('loss'):
            for i in range(len(self.g)):
                self.d_loss.append(t.sce_loss(self.d_reals[i], tf.ones_like(self.d_reals[i])) +
                                   t.sce_loss(self.d_fakes[i], tf.zeros_like(self.d_fakes[i])))
                self.g_loss.append(t.sce_loss(self.d_fakes[i], tf.ones_like(self.d_fakes[i])))

        # Summary
        for i in range(len(self.g)):
            tf.summary.scalar('loss/d_loss_{0}'.format(i), self.d_loss[i])
            tf.summary.scalar('loss/g_loss_{0}'.format(i), self.g_loss[i])

        # Optimizer
        t_vars = tf.trainable_variables()
        for idx, i in enumerate([32, 16, 8]):
            self.d_op.append(tf.train.AdamOptimizer(learning_rate=self.lr,
                                                    beta1=self.beta1, beta2=self.beta2).
                             minimize(loss=self.d_loss[idx],
                                      var_list=[v for v in t_vars if v.name.startswith('discriminator_{0}'.format(i))]))
            self.g_op.append(tf.train.AdamOptimizer(learning_rate=self.lr,
                                                    beta1=self.beta1, beta2=self.beta2).
                             minimize(loss=self.g_loss[idx],
                                      var_list=[v for v in t_vars if v.name.startswith('generator_{0}'.format(i))]))

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
