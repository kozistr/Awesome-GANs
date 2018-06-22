import tensorflow as tf
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class ACGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1, n_classes=10,
                 sample_num=10 * 10, sample_size=10,
                 n_input=784, df_dim=16, gf_dim=128, fc_unit=512,
                 z_dim=128, g_lr=8e-4, d_lr=2e-4, c_lr=8e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: image height, default 28
        :param width: image width, default 28
        :param channel: image channel, default 1 (gray-scale)
        - in case of MNIST, image size is 28x28x1(HWC).
        :param n_classes: input dataset's classes
        - in case of MNIST, 10 (0 ~ 9)

        # Output Settings
        :param sample_num: the number of output images, default 128
        :param sample_size: sample image size, default 10

        # For CNN model
        :param n_input: input image size, default 784(28x28)
        :param df_dim: discriminator filter, default 16
        :param gf_dim: generator filter, default 128
        :param fc_unit: the number of fully connected filters, default 1024

        # Training Option
        :param z_dim: z dimension (kinda noise), default 100
        :param g_lr: generator learning rate, default 1e-3
        :param d_lr: discriminator learning rate, default 2e-4
        :param c_lr: classifier learning rate, default 1e-3
        :param epsilon: epsilon, default 1e-9
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.n_input = n_input
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.d_lr, self.g_lr, self.c_lr = d_lr, g_lr, c_lr
        self.eps = epsilon

        # pre-defined
        self.g_loss = 0.
        self.d_loss = 0.
        self.c_loss = 0.
        
        self.g = None
        self.g_test = None
        
        self.d_op = None
        self.g_op = None
        self.c_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name="x-image")  # (-1, 28, 28, 1)
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="y-label")  # (-1, 10)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="z-noise")      # (-1, 128)

        self.build_acgan()  # build ACGAN model

    def classifier(self, x, reuse=None):
        """
        # Following a C Network, CiFar-like-hood, referred in the paper
        :param x: image, shape=(-1, 28, 28, 1)
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("classifier", reuse=reuse):
            x = t.dense(x, self.fc_unit, name='c-fc-1')
            x = tf.nn.leaky_relu(x)
            x = t.dense(x, self.n_classes, name='c-fc-2')

            return x

    def discriminator(self, x, reuse=None):
        """
        # Following a D Network, CiFar-like-hood, referred in the paper
        :param x: image, shape=(-1, 28, 28, 1)
        :param reuse: re-usable
        :return: logits, networks
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = t.conv2d(x, self.df_dim, k=3, s=2, name='d-conv-0')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.5, name='d-dropout-0')

            for i in range(1, 2 * 2 + 1):
                f = self.df_dim * (i + 1)
                x = t.conv2d(x, f=f, k=3, s=(i % 2 + 1), name='d-conv-%d' % i)
                x = t.batch_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.dropout(x, 0.5, name='d-dropout-%d' % i)

            x = tf.layers.flatten(x)

            x = t.dense(x, self.fc_unit * 2, name='d-fc-1')
            net = tf.nn.leaky_relu(x)

            x = tf.layers.dense(net, 1, name='d-fc-2')  # logits

            return x, net

    def generator(self, y, z, reuse=None, is_train=True):
        """
        # Following a G Network, CiFar-like-hood, referred in the paper
        :param y: image label
        :param z: image noise
        :param reuse: re-usable
        :param is_train: trainable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([z, y], axis=1)

            x = tf.layers.dense(x, self.gf_dim * 2, name='g-fc-0')
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.gf_dim * 7 * 7, name='g-fc-1')
            x = tf.nn.relu(x)

            x = tf.reshape(x, [-1, 7, 7, self.gf_dim])

            x = t.deconv2d(x, f=self.gf_dim // 2, k=5, s=2, name='g-deconv-1')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.relu(x)

            x = t.deconv2d(x, f=1, k=5, s=2, name='g-deconv-2')  # channel
            x = tf.nn.sigmoid(x)  # x = tf.nn.tanh(x)

            return x

    def build_acgan(self):
        # Generator
        self.g = self.generator(self.y, self.z)
        self.g_test = self.generator(self.y, self.z, is_train=False)

        # Discriminator
        d_real, d_real_net = self.discriminator(self.x)
        d_fake, d_fake_net = self.discriminator(self.g, reuse=True)

        # Classifier
        c_real = self.classifier(d_real_net)
        c_fake = self.classifier(d_fake_net, reuse=True)

        # sigmoid CE Loss
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss

        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        c_real_loss = t.sce_loss(c_real, self.y)
        c_fake_loss = t.sce_loss(c_fake, self.y)
        self.c_loss = c_real_loss + c_fake_loss

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/c_real_loss", c_real_loss)
        tf.summary.scalar("loss/c_fake_loss", c_fake_loss)
        tf.summary.scalar("loss/c_loss", self.c_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]
        c_params = [v for v in t_vars if v.name.startswith('c')]

        self.d_op = tf.train.AdamOptimizer(self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)
        self.c_op = tf.train.AdamOptimizer(self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.c_loss, var_list=c_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
