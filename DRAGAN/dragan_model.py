import tensorflow as tf
import tfutil as t


tf.set_random_seed(777)


class DRAGAN:

    def __init__(self, s, batch_size=16, height=28, width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 z_dim=128, gf_dim=64, df_dim=64, fc_unit=1024, eps=1e-9):

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
        :param gf_dim: the number of generator filters, default 64
        :param df_dim: the number of discriminator filters, default 64
        :param fc_unit: the number of fc units, default 1024

        # Training Settings
        :param eps: epsilon, default 1e-9
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

        self.eps = eps

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.fc_unit = fc_unit

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.
        self.gp = 0.

        self.g = None
        self.g_test = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name='x-images')
        self.x_ = tf.placeholder(tf.float32,
                                 shape=[None, self.height, self.width, self.channel],
                                 name='x-perturbed-images')
        self.z = tf.placeholder(tf.float32,
                                shape=[None, self.z_dim],
                                name='z-noise')

        # Training Options
        self.lambda_ = 10.  # Higher lambda value, More stable. But slower...
        self.beta1 = .5
        self.beta2 = .999
        self.lr = 2e-4

        self.bulid_dragan()  # build DRAGAN model

    def discriminator(self, x, reuse=None, is_train=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = t.conv2d(x, self.df_dim, s=1, name='disc-conv2d-0')
            x = tf.nn.leaky_relu(x)

            for i in range(1, 3):
                x = t.conv2d(x, self.df_dim * (2 ** i), name='disc-conv2d-%d' % i)
                x = t.batch_norm(x, is_train=is_train)
                x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            logits = t.dense(x, 1, name='disc-fc-0')
            prob = tf.nn.sigmoid(logits)

            return prob, logits

    def generator(self, z, reuse=None, is_train=True):
        with tf.variable_scope('generator', reuse=reuse):
            x = t.dense(z, self.fc_unit, name='gen-fc-0')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.leaky_relu(x)

            x = t.dense(x, self.gf_dim * 4 * 7 * 7, name='gen-fc-1')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 7, 7, self.gf_dim * 4])

            x = t.deconv2d(x, self.gf_dim * 2, name='gen-deconv2d-0')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.leaky_relu(x)

            logits = t.deconv2d(x, self.channel, name='gen-deconv2d-1')
            prob = tf.nn.sigmoid(logits)

            return prob

    def bulid_dragan(self):
        # Generator
        self.g = self.generator(self.z)
        self.g_test = self.generator(self.z, reuse=True, is_train=False)  # for test

        # Discriminator
        _, d_real = self.discriminator(self.x)
        _, d_fake = self.discriminator(self.g, reuse=True)

        # sce losses
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        # DRAGAN loss with GP (gradient penalty)
        alpha = tf.random_uniform(shape=[self.batch_size] + [1, 1, 1], minval=0., maxval=1., name='alpha')

        diff = self.x_ - self.x
        interpolates = self.x + alpha * diff
        _, d_inter = self.discriminator(interpolates, reuse=True)
        grads = tf.gradients(d_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        self.gp = tf.reduce_mean(tf.square((slopes - 1.)))

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
