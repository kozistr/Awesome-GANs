import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


def conv2d(x, f=64, k=5, s=2, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: conv2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad,
                            name=name)


def deconv2d(x, f=64, k=5, s=2, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: decovn2d net
    """
    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad,
                                      name=name)


def batch_norm(x, momentum=0.9, eps=1e-5, train=True):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         training=train)


class AnoGAN:

    def __init__(self, s, batch_size=16, input_height=108, input_width=108, input_channel=3,
                 sample_num=1 * 1, sample_size=1, output_height=108, output_width=108,
                 df_dim=64, gf_dim=64, fc_unit=1024,
                 lambda_=1e-1, z_dim=128, g_lr=2e-4, d_lr=2e-4, epsilon=1e-12,
                 detect=False):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 16
        :param input_height: input image height, default 108
        :param input_width: input image width, default 108
        :param input_channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 108x108x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 1
        :param sample_size: sample image size, default 1
        :param output_height: output images height, default 108
        :param output_width: output images width, default 108

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64
        :param fc_unit: fully connected units, default 1024

        # Training Option
        :param lambda_: anomaly lambda value, default 1e-1
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 1e-4
        :param d_lr: discriminator learning rate, default 1e-4
        :param epsilon: epsilon, default 1e-12
        :param detect: anomalies detection if True, just training a model, default False
        """

        self.s = s
        self.batch_size = batch_size
        self.test_batch_size = 1

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

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

        # pre-defined
        self.d_loss = 0.
        self.g_loss = 0.
        self.anomaly_loss = 0.

        self.g = None
        self.g_test = None

        self.d_op = None
        self.ano_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.input_height, self.input_width, self.input_channel],
                                name="x-image")                                        # (-1, 108, 108, 3)
        # self.y = tf.placeholder(tf.float32, shape=[None, 41], name='y-label')        # (-1, 41)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')  # (-1, 128)

        self.build_anogan()  # build AnoGAN model

    def discriminator(self, x, reuse=None, is_bn_train=True):
        """
        :param x: images
        :param reuse: re-usable
        :param is_bn_train: en/disable batch_norm, default True
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv2d(x, f=self.gf_dim * 1, name="disc-conv2d-0")
            x = tf.nn.leaky_relu(x)

            for i in range(1, 4):
                x = conv2d(x, f=self.gf_dim * (2 ** i), name="disc-conv2d-%d" % i)
                x = batch_norm(x, train=is_bn_train)
                x = tf.nn.leaky_relu(x)

            feature_match = x   # (-1, 8, 8, 512)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, 1, name='disc-fc-0')

            return feature_match, x

    def generator(self, z, reuse=None, is_bn_train=True):
        """
        :param z: embeddings
        :param reuse: re-usable
        :param is_bn_train: en/disable batch_norm, default True
        :return: prob
        """
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(z, units=self.fc_unit, name='gen-fc-0')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 8, 8, self.fc_unit // (8 * 8)])

            for i in range(1, 4):
                x = deconv2d(x, f=self.gf_dim * (2 ** i), name="gen-conv2d-%d" % i)
                x = batch_norm(x, train=is_bn_train)
                x = tf.nn.leaky_relu(x)

            x = deconv2d(x, f=3, s=1, name="gen-conv2d-4")  # (-1, 64, 64, 3)
            x = tf.nn.tanh(x)

            return x

    def build_anogan(self):
        def l1_loss(x, y):
            return tf.reduce_mean(tf.reduce_sum(tf.abs(x - y)))

        def sce_loss(logits_, labels_):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_, labels=labels_))

        # Generator
        self.g = self.generator(self.z)
        self.g_test = self.generator(self.z, reuse=True, is_bn_train=False)

        # Discriminator
        d_real_fm, d_real = self.discriminator(self.x, reuse=True)
        d_fake_fm, d_fake = self.discriminator(self.g_test, reuse=True)

        # Loss
        self.d_loss = l1_loss(d_fake_fm, d_real_fm)
        self.g_loss = l1_loss(self.x, self.g_test)
        self.anomaly_loss = (1. - self.lambda_) * self.g + self.lambda_ * self.d_loss

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("loss/ano_loss", self.anomaly_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('disc')]
        g_params = [v for v in t_vars if v.name.startswith('gen')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(loss=self.d_loss,
                                                                                        var_list=d_params)
        self.ano_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                             beta1=self.beta1, beta2=self.beta2).minimize(loss=self.anomaly_loss,
                                                                                          var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
