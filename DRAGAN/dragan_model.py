import tensorflow as tf

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)


class DRAGAN:

    def __init__(self, s, batch_size=16, height=28, width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 z_dim=128, fc_unit=512):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (Gray-Scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: the number of classes, default 10

        # Output Settings
        :param sample_num: the number of sample images, default 100
        :param sample_size: sample image size, default 10

        # Model Settings
        :param z_dim: z noise dimension, default 128
        :param fc_unit: the number of fc units, default 512
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
        self.n_input = self.height * self.width * self.channel

        self.z_dim = z_dim

        self.fc_unit = fc_unit

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.
        self.gp = 0.

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel],
                                name='x-images')
        self.x_p = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel],
                                  name='x-perturbed-images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')

        # Training Options
        self.lambda_ = 10.  # Higher lambda value, More stable. But slower...
        self.beta1 = .5
        self.beta2 = .9
        self.lr = 2e-4

        self.bulid_dragan()  # build DRAGAN model

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.layers.flatten(x)

            for i in range(1, 5):
                x = t.dense(x, self.fc_unit, name='disc-fc-%d' % i)
                x = tf.nn.leaky_relu(x)

            x = t.dense(x, 1, name='disc-fc-5')
            return x

    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            x = z
            for i in range(1, 5):
                x = t.dense(x, self.fc_unit, name='gen-fc-%d' % i)
                x = tf.nn.relu(x)

            x = t.dense(x, self.n_input, name='gen-fc-5')
            x = tf.nn.sigmoid(x)
            return x

    def bulid_dragan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # sce losses
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        # DRAGAN loss with GP (gradient penalty)
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1., name='alpha')
        diff = self.x_p - self.x
        interpolates = self.x + alpha * diff
        d_inter = self.discriminator(interpolates, reuse=True)
        grads = tf.gradients(d_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        self.gp = tf.reduce_mean(tf.square(slopes - 1.))

        # update d_loss with gp
        self.d_loss += self.lambda_ * self.gp

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("misc/gp", self.gp)

        # Collect trainer values
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        # Optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
