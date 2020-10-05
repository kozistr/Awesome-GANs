import tensorflow as tf

import awesome_gans.modules as t

tf.set_random_seed(777)  # reproducibility


class CycleGAN:
    def __init__(
        self,
        s,
        batch_size=8,
        height=128,
        width=128,
        channel=3,
        sample_num=1 * 1,
        sample_size=1,
        df_dim=64,
        gf_dim=32,
        fd_unit=512,
        g_lr=2e-4,
        d_lr=2e-4,
        epsilon=1e-9,
    ):
        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 8
        :param height: input image height, default 128
        :param width: input image width, default 128
        :param channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 128x128x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 4
        :param sample_size: sample image size, default 2

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 32
        :param fd_unit: fully connected units, default 512

        # Training Option
        :param g_lr: generator learning rate, default 2e-4
        :param d_lr: discriminator learning rate, default 2e-4
        :param epsilon: epsilon, default 1e-9
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fd_unit = fd_unit

        self.beta1 = 0.5
        self.beta2 = 0.999
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.lambda_ = 10.0
        self.lambda_cycle = 10.0
        self.n_train_critic = 10
        self.eps = epsilon

        # pre-defined
        self.g_a2b = None
        self.g_b2a = None
        self.g_a2b2a = None
        self.g_b2a2b = None

        self.w_a = None
        self.w_b = None
        self.w = None

        self.gp_a = None
        self.gp_b = None
        self.gp = None

        self.g_loss = 0.0
        self.g_a_loss = 0.0
        self.g_b_loss = 0.0
        self.d_loss = 0.0
        self.cycle_loss = 0.0

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # placeholders
        self.a = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel], name='image-a')
        self.b = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel], name='image-b')
        self.lr_decay = tf.placeholder(tf.float32, None, name='learning_rate-decay')

        self.build_cyclegan()  # build CycleGAN

    def discriminator(self, x, reuse=None, name=""):
        """
        :param x: 128x128x3 images
        :param reuse: re-usability
        :param name: name

        :return: logits, prob
        """
        with tf.variable_scope('discriminator-%s' % name, reuse=reuse):

            def residual_block(x, f, name=''):
                x = t.conv2d(x, f=f, k=4, s=2, name='disc-conv2d-%s' % name)
                x = t.instance_norm(x, name='disc-ins_norm-%s' % name)
                x = tf.nn.leaky_relu(x, alpha=0.2)
                return x

            x = t.conv2d(x, f=self.df_dim, name='disc-conv2d-0')
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = residual_block(x, f=self.df_dim * 2, name='1')
            x = residual_block(x, f=self.df_dim * 4, name='2')
            x = residual_block(x, f=self.df_dim * 8, name='3')
            # for 256x256x3 images
            # x = residual_block(x, f=self.df_dim * 8, name='4')
            # x = residual_block(x, f=self.df_dim * 8, name='5')

            logits = t.conv2d(x, f=1, name='disc-con2d-last')
            # prob = tf.nn.sigmoid(logits)

            return logits

    def generator(self, x, reuse=None, name=""):
        """ The form of Auto-Encoder
        :param x: 128x128x3 images
        :param reuse: re-usability
        :param name: name

        :return: logits, prob
        """
        with tf.variable_scope('generator-%s' % name, reuse=reuse):

            def d(x, f, name=''):
                x = t.conv2d(x, f=f, k=3, s=2, name='gen-d-conv2d-%s' % name)
                x = t.instance_norm(x, name='gen-d-ins_norm-%s' % name)
                x = tf.nn.relu(x)
                return x

            def R(x, f, name=''):
                x = t.conv2d(x, f=f, k=3, s=1, name='gen-R-conv2d-%s-0' % name)
                x = t.conv2d(x, f=f, k=3, s=1, name='gen-R-conv2d-%s-1' % name)
                x = t.instance_norm(x, name='gen-R-ins_norm-%s' % name)
                x = tf.nn.relu(x)
                return x

            def u(x, f, name=''):
                x = t.deconv2d(x, f=f, k=3, s=2, name='gen-u-deconv2d-%s' % name)
                x = t.instance_norm(x, name='gen-u-ins_norm-%s' % name)
                x = tf.nn.relu(x)
                return x

            x = t.conv2d(x, f=self.gf_dim, k=7, s=1, name='gen-conv2d-0')

            x = d(x, self.gf_dim * 2, name='1')
            x = d(x, self.gf_dim * 4, name='2')

            for i in range(1, 7):
                x = R(x, self.gf_dim * 4, name=str(i))

            x = u(x, self.gf_dim * 4, name='1')
            x = u(x, self.gf_dim * 2, name='2')

            logits = t.conv2d(x, f=3, k=7, s=1, name='gen-conv2d-1')
            prob = tf.nn.tanh(logits)

            return prob

    def build_cyclegan(self):
        # Generator
        with tf.variable_scope("generator-a2b"):
            self.g_a2b = self.generator(self.a, name="a2b")  # a to b
        with tf.variable_scope("generator-b2a"):
            self.g_b2a = self.generator(self.b, name="b2a")  # b to a

        with tf.variable_scope("generator-b2a", reuse=True):
            self.g_a2b2a = self.generator(self.g_a2b, reuse=True, name="b2a")  # a to b to a
        with tf.variable_scope("generator-a2b", reuse=True):
            self.g_b2a2b = self.generator(self.g_b2a, reuse=True, name="a2b")  # b to a to b

        # Classifier
        with tf.variable_scope("discriminator-a"):
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            a_hat = alpha * self.a + (1.0 - alpha) * self.g_b2a

            d_a = self.discriminator(self.a)
            d_b2a = self.discriminator(self.g_b2a, reuse=True)
            d_a_hat = self.discriminator(a_hat, reuse=True)
        with tf.variable_scope("discriminator-b"):
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            b_hat = alpha * self.b + (1.0 - alpha) * self.g_a2b

            d_b = self.discriminator(self.b)
            d_a2b = self.discriminator(self.g_a2b, reuse=True)
            d_b_hat = self.discriminator(b_hat, reuse=True)

        # Training Ops
        self.w_a = tf.reduce_mean(d_a) - tf.reduce_mean(d_b2a)
        self.w_b = tf.reduce_mean(d_b) - tf.reduce_mean(d_a2b)
        self.w = self.w_a + self.w_b

        self.gp_a = tf.reduce_mean(
            (tf.sqrt(tf.reduce_sum(tf.gradients(d_a_hat, a_hat)[0] ** 2, reduction_indices=[1, 2, 3])) - 1.0) ** 2
        )
        self.gp_b = tf.reduce_mean(
            (tf.sqrt(tf.reduce_sum(tf.gradients(d_b_hat, b_hat)[0] ** 2, reduction_indices=[1, 2, 3])) - 1.0) ** 2
        )
        self.gp = self.gp_a + self.gp_b

        self.d_loss = self.lambda_ * self.gp - self.w

        cycle_a_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(self.a - self.g_a2b2a), reduction_indices=[1, 2, 3]))
        cycle_b_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(self.b - self.g_b2a2b), reduction_indices=[1, 2, 3]))
        self.cycle_loss = cycle_a_loss + cycle_b_loss

        # using adv loss
        self.g_a_loss = -1.0 * tf.reduce_mean(d_b2a)
        self.g_b_loss = -1.0 * tf.reduce_mean(d_a2b)
        self.g_loss = self.g_a_loss + self.g_b_loss + self.lambda_cycle * self.cycle_loss

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/cycle_loss", self.cycle_loss)
        tf.summary.scalar("loss/cycle_a_loss", cycle_a_loss)
        tf.summary.scalar("loss/cycle_b_loss", cycle_b_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("loss/g_a_loss", self.g_a_loss)
        tf.summary.scalar("loss/g_b_loss", self.g_b_loss)
        tf.summary.scalar("misc/gradient_penalty", self.gp)
        tf.summary.scalar("misc/g_lr", self.g_lr)
        tf.summary.scalar("misc/d_lr", self.d_lr)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(
            learning_rate=self.d_lr * self.lr_decay, beta1=self.beta1, beta2=self.beta2
        ).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(
            learning_rate=self.g_lr * self.lr_decay, beta1=self.beta1, beta2=self.beta2
        ).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
