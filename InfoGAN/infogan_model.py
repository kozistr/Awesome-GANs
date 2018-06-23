import tensorflow as tf
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class InfoGAN:

    def __init__(self, s, batch_size=64, height=28, width=28, channel=1,
                 sample_num=10 * 10, sample_size=10,
                 df_dim=64, gf_dim=64, fc_unit=1024, n_categories=10,
                 n_continous_factor=2, z_dim=62, g_lr=1e-3, d_lr=2e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param height: input image height, default 28
        :param width: input image width, default 28
        :param channel: input image channel, default 1 (RGB)
        - in case of MNIST, image size is 28x28x1(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 100
        :param sample_size: sample image size, default 10

        # Hyper-parameters
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_unit: fully connected unit, default 1024

        # Training Option
        :param n_categories: the number of categories, default 10 (For MNIST)
        :param n_continous_factor: the number of cont factors, default 2 (For MNIST)
        :param z_dim: z dimension (kinda noise), default 62 (For MNIST)
        :param g_lr: generator learning rate, default 1e-3
        :param d_lr: discriminator learning rate, default 2e-4
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
        self.fc_unit = fc_unit

        """
        - MNIST
        n_cat : 10, n_cont : 2, z : 62 => embeddings : 10 + 2 + 62 = 74
        - SVHN
        n_cat : 40, n_cont : 4, z : 124 => embeddings : 40 + 4 + 124 = 168
        - Celeb-A
        n_cat : 100, n_cont : 0, z : 128 => embeddings : 100 + 0 + 128 = 228
        """
        self.n_cat = n_categories         # category dist, label
        self.n_cont = n_continous_factor  # gaussian dist, rotate, etc
        self.z_dim = z_dim
        self.lambda_ = 1  # sufficient for discrete latent codes # less than 1

        self.beta1 = 0.5
        self.beta2 = 0.999
        self.d_lr = d_lr
        self.g_lr = g_lr

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.

        self.g_loss = 0.
        self.d_loss = 0.
        self.q_loss = 0.

        self.g = None
        self.g_test = None

        self.d_op = None
        self.g_op = None
        self.q_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name="x-image")                                                # (-1, 32, 32, 3)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')          # (-1, 128)
        self.c = tf.placeholder(tf.float32, shape=[None, self.n_cat + self.n_cont], name='c')  # (-1, 12)

        self.build_infogan()  # build InfoGAN model

    def classifier(self, x, reuse=None, is_train=True):
        """
        # This is a network architecture for MNIST DataSet referred in the paper
        :param x: ~ D
        :param reuse: re-usable
        :param is_train: trainable
        :return: prob, logits
        """
        with tf.variable_scope("classifier", reuse=reuse):
            x = t.dense(x, 128, name='d-fc-1')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            logits = t.dense(x, self.n_cat + self.n_cont, name='d-fc-2')
            prob = tf.nn.softmax(logits)

            return prob, logits

    def discriminator(self, x, reuse=None):
        """
        # This is a network architecture for MNIST DataSet referred in the paper
        :param x: 28x28x1 images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = t.conv2d(x, f=self.df_dim, name='d-conv2d-0')
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = t.conv2d(x, f=self.df_dim * 2, name='d-conv2d-1')
            x = t.batch_norm(x)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = tf.layers.flatten(x)

            x = t..dense(x, units=self.fc_unit, name='d-fc-0')
            x = t.batch_norm(x)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            logits = t.dense(x, 1, name='d-fc-1')
            prob = tf.nn.sigmoid(logits)

            return prob, logits, x

    def generator(self, z, c, reuse=None, is_train=True):
        """
        # This is a network architecture for MNIST DataSet referred in the paper
        :param z: 62 z-noise
        :param c: 10 categories + 2 dimensions
        :param reuse: re-usable
        :param is_train: trainable
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([z, c], axis=1)  # (-1, 74)

            x = t.dense(x, self.fc_unit, name='g-fc-0')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = t.dense(x, 7 * 7 * self.gf_dim * 2, name='g-fc-1')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = tf.reshape(x, shape=[-1, 7, 7, self.gf_dim * 2])

            x = t.deconv2d(x, f=self.gf_dim, name='g-conv2d-0')
            x = t.batch_norm(x, is_train=is_train)
            x = tf.nn.leaky_relu(x, alpha=0.1)

            x = t.deconv2d(x, f=1, name='g-conv2d-2')
            x = tf.nn.sigmoid(x)

            return x

    def build_infogan(self):
        # Generator
        self.g = self.generator(self.z, self.c)
        self.g_test = self.generator(self.z, self.c, is_train=False)

        # Discriminator
        d_real, d_real_logits, _ = self.discriminator(self.x)
        d_fake, d_fake_logits, d_fake_d = self.discriminator(self.g, reuse=True)

        # Classifier
        c_fake, c_fake_logits = self.classifier(d_fake_d)  # Q net

        # Losses
        d_real_loss = t.sce_loss(d_real_logits, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake_logits, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake_logits, tf.ones_like(d_fake))

        # categorical
        q_cat_logits = c_fake_logits[:, :self.n_cat]
        q_cat_labels = self.c[:, :self.n_cat]

        q_cat_loss = t.softce_loss(q_cat_logits, q_cat_labels)
        # gaussian
        q_cont_logits = c_fake[:, self.n_cat:]
        q_cont_labels = self.c[:, self.n_cat:]

        q_cont_loss = t.mse_loss(q_cont_labels, q_cont_logits)  # l2 loss
        self.q_loss = q_cat_loss + q_cont_loss

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("loss/q_loss", self.q_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]
        q_params = [v for v in t_vars if v.name.startswith('d') or v.name.startswith('g') or v.name.startswith('c')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)
        self.q_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.q_loss, var_list=q_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
