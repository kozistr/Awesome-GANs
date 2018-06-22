import tensorflow as tf
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class AnoGAN:

    def __init__(self, s, batch_size=16, height=64, width=64, channel=3, n_classes=41, sample_num=1, sample_size=1,
                 df_dim=64, gf_dim=64, fc_unit=1024, lambda_=1e-1, z_dim=128, g_lr=2e-4, d_lr=2e-4, epsilon=1e-9,
                 detect=False, use_label=False):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 16
        :param height: image height, default 64
        :param width: image width, default 64
        :param channel: image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 108x108x3(HWC).
        :param n_classes: the number of classes, default 41

        # Output Settings
        :param sample_num: the number of output images, default 1
        :param sample_size: sample image size, default 1

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_unit: fully connected units, default 1024

        # Training Option
        :param lambda_: anomaly lambda value, default 1e-1
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 1e-4
        :param d_lr: discriminator learning rate, default 1e-4
        :param epsilon: epsilon, default 1e-9
        :param detect: anomalies detection if True, just training a model, default False
        """

        self.s = s
        self.batch_size = batch_size
        self.test_batch_size = 1

        self.height = height
        self.width = width
        self.channel = channel
        self.n_classes = n_classes
        self.image_shape = [self.batch_size, self.height, self.width, self.channel]

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.fc_unit = fc_unit

        self.lambda_ = lambda_
        self.z_dim = z_dim
        self.beta1 = .5
        self.beta2 = .9
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.eps = epsilon

        self.detect = detect
        self.use_label = use_label

        # pre-defined
        self.d_fake_loss = 0.
        self.d_real_loss = 0.
        self.d_loss = 0.
        self.g_loss = 0.
        self.r_loss = 0.
        self.ano_loss = 0.

        self.g = None
        self.g_test = None

        self.d_op = None
        self.g_op = None
        self.ano_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name="x-image")                                                # (-1, 64, 64, 3)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')          # (-1, 128)
        if self.use_label:
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-label')  # (-1, 41)
        else:
            self.y = None

        self.build_anogan()  # build AnoGAN model

    def discriminator(self, x, y=None, reuse=None, is_train=True):
        """
        :param x: images
        :param y: labels
        :param reuse: re-usable
        :param is_train: en/disable batch_norm, default True
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            if y:
                raise NotImplemented("[-] Not Implemented Yet...")

            x = t.conv2d(x, f=self.gf_dim * 1, name="disc-conv2d-0")
            x = tf.nn.leaky_relu(x)

            for i in range(1, 4):
                x = t.conv2d(x, f=self.gf_dim * (2 ** i), name="disc-conv2d-%d" % i)
                x = t.batch_norm(x, is_train=is_train)
                x = tf.nn.leaky_relu(x)

            feature_match = x   # (-1, 8, 8, 512)

            x = tf.layers.flatten(x)

            x = t.dense(x, 1, name='disc-fc-0')

            return feature_match, x

    def generator(self, z, y=None, reuse=None, is_train=True):
        """
        :param z: embeddings
        :param y: labels
        :param reuse: re-usable
        :param is_train: en/disable batch_norm, default True
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            if y:
                raise NotImplemented("[-] Not Implemented Yet...")

            x = t.dense(z, f=self.fc_unit, name='gen-fc-0')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 8, 8, self.fc_unit // (8 * 8)])

            for i in range(1, 4):
                x = t.deconv2d(x, f=self.gf_dim * (2 ** i), name="gen-conv2d-%d" % i)
                x = t.batch_norm(x, is_train=is_train)
                x = tf.nn.leaky_relu(x)

            x = t.deconv2d(x, f=3, s=1, name="gen-conv2d-4")  # (-1, 64, 64, 3)
            x = tf.sigmoid(x)  # [0, 1]

            return x

    def build_anogan(self):
        # Generator
        self.g = self.generator(self.z, self.y)
        self.g_test = self.generator(self.z, self.y, reuse=True, is_train=False)

        # Discriminator
        d_real_fm, d_real = self.discriminator(self.x)
        d_fake_fm, d_fake = self.discriminator(self.g_test, reuse=True)

        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('disc')]
        g_params = [v for v in t_vars if v.name.startswith('gen')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(loss=self.d_loss,
                                                                                        var_list=d_params)

        if self.detect:
            self.d_loss = t.l1_loss(d_fake_fm, d_real_fm)  # disc     loss
            self.r_loss = t.l1_loss(self.x, self.g)        # residual loss
            self.ano_loss = (1. - self.lambda_) * self.r_loss + self.lambda_ * self.d_loss

            tf.summary.scalar("loss/d_loss", self.d_loss)
            tf.summary.scalar("loss/r_loss", self.r_loss)
            tf.summary.scalar("loss/ano_loss", self.ano_loss)

            self.ano_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                                 beta1=self.beta1, beta2=self.beta2).minimize(loss=self.ano_loss,
                                                                                              var_list=g_params)
        else:
            self.d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
            self.d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
            self.d_loss = self.d_real_loss + self.d_fake_loss
            self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

            # Summary
            tf.summary.scalar("loss/d_fake_loss", self.d_fake_loss)
            tf.summary.scalar("loss/d_real_loss", self.d_real_loss)
            tf.summary.scalar("loss/d_loss", self.d_loss)
            tf.summary.scalar("loss/g_loss", self.r_loss)

            self.g_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                               beta1=self.beta1, beta2=self.beta2).minimize(loss=self.d_loss,
                                                                                            var_list=d_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        if not self.detect:
            self.writer = tf.summary.FileWriter('./orig-model/', self.s.graph)
        else:
            self.writer = tf.summary.FileWriter('./ano-model/', self.s.graph)
