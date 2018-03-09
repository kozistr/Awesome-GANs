import tensorflow as tf


tf.set_random_seed(777)  # reproducibility


def sub_pixel_conv2d(x, f, s=2):
    """
    # ref : https://github.com/tensorlayer/SRGAN/blob/master/tensorlayer/layers.py
    """
    if f is None:
        f = int(int(x.get_shape()[-1]) / (s ** 2))

    bsize, a, b, c = x.get_shape().as_list()
    bsize = tf.shape(x)[0]

    x_s = tf.split(x, s, 3)
    x_r = tf.concat(x_s, 2)

    return tf.reshape(x_r, (bsize, s * a, s * b, f))


def conv2d(x, f=64, k=3, s=1, act=None, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 1
    :param act: activation function, default None
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: covn2d net
    """
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.zeros_initializer(),
                            activation=act,
                            padding=pad,
                            name=name)


def batch_norm(x, momentum=0.9, eps=1e-5):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True)


class SRGAN:

    def __init__(self, s, batch_size=16, input_height=384, input_width=384, input_channel=3,
                 sample_num=1 * 1, sample_size=1, output_height=384, output_width=384,
                 df_dim=64, gf_dim=64,
                 g_lr=1e-4, d_lr=1e-4):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 16
        :param input_height: input image height, default 384
        :param input_width: input image width, default 384
        :param input_channel: input image channel, default 3 (RGB)
        - in case of DIV2K-HR, image size is 384x384x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 1
        :param sample_size: sample image size, default 1
        :param output_height: output images height, default 384
        :param output_width: output images width, default 384

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
        :param g_lr: generator learning rate, default 1e-4
        :param d_lr: discriminator learning rate, default 1e-4
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

        self.lr_image_shape = [None, self.input_height // 4, self.input_width // 4, self.input_channel]
        self.hr_image_shape = [None, self.input_height, self.input_width, self.input_channel]

        self.vgg_image_shape = [224, 224, 3]

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.d_lr = d_lr
        self.g_lr = g_lr

        self.vgg_mean = [103.939, 116.779, 123.68]

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.d_loss = 0.
        self.g_loss = 0.

        self.g = None
        self.content_loss_weight = 1e-3
        self.vgg_loss_weight = 2e-6

        self.d_op = None
        self.g_op = None
        self.g_init_op = None
        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x_hr = tf.placeholder(tf.float32, shape=self.hr_image_shape, name="x-image-hr")  # (-1, 384, 384, 3)
        self.x_lr = tf.placeholder(tf.float32, shape=self.lr_image_shape, name="x-image-lr")  # (-1, 96, 96, 3)

        self.build_srgan()  # build SRGAN model

    def discriminator(self, x, reuse=None):
        """
        # Following a network architecture referred in the paper
        :param x: Input images (-1, 384, 384, 3)
        :param reuse: re-usability
        :return: HR (High Resolution) or SR (Super Resolution) images
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv2d(x, self.df_dim, act=tf.nn.leaky_relu, name='n64s1-1')

            strides = [2, 1]
            filters = [1, 2, 2, 4, 4, 8, 8]

            for i, f in enumerate(filters):
                x = conv2d(x, f=f, s=strides[i % 2], name='n%ds%d-%d' % (f, strides[i % 2], i + 1))
                x = batch_norm(x)
                x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)  # (-1, 96 * 96 * 64)

            x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu, name='d-fc-0')
            x = tf.layers.dense(x, 1, name='d-fc-1')

            return x

    def generator(self, x, reuse=None):
        """
        # For MNIST
        :param x: LR (Low Resolution) images, (-1, 96, 96, 3)
        :param reuse: scope re-usability
        :return: SR (Super Resolution) images, (-1, 384, 384, 3)
        """

        with tf.variable_scope("generator", reuse=reuse):
            def residual_block(x, name):
                with tf.variable_scope(name):
                    x = conv2d(x, self.gf_dim, name="n64s1-2")
                    x = batch_norm(x)
                    x = tf.nn.relu(x)

                    return x

            x = conv2d(x, self.gf_dim, act=tf.nn.relu, name='n64s1-1')
            x_ = x  # for later, used at layer concat

            # B residual blocks
            for i in range(1, 17):  # (1, 9)
                xx = residual_block(x, name='b-residual_block_%d' % (i * 2 - 1))
                xx = residual_block(xx, name='b-residual_block_%d' % (i * 2))
                xx = tf.add(x, xx)
                x = xx

            x = conv2d(x, self.gf_dim, name='n64s1-3')
            x = batch_norm(x)

            x = tf.add(x_, x)

            for i in range(1, 3):
                x = conv2d(x, self.gf_dim * 4, name='n256s1-%d' % (i + 2))
                x = sub_pixel_conv2d(x, f=None, s=2)
                x = tf.nn.relu(x)

            x = conv2d(x, self.input_channel, act=tf.nn.tanh, k=1, name='n3s1')  # (-1, 384, 384, 3)

            return x

    def vgg19(self, x, reuse=None):
        """
        Download pre-trained model for tensorflow

        Credited by https://github.com/tensorlayer
        Link : https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs

        :param x: 224x224x3 HR images
        :param reuse: re-usability
        :return: prob
        """
        with tf.variable_scope("vgg19", reuse=reuse):
            rgb_scaled = x * 255.  # inverse_transform

            # rgb to bgr
            r, g, b = tf.split(3, 3, rgb_scaled)
            bgr = tf.concat([b - self.vgg_mean[0],
                             g - self.vgg_mean[1],
                             r - self.vgg_mean[2]], axis=3)

            # VGG19
            """ Conv1 """
            x = conv2d(bgr, f=64, k=3, s=1, act=tf.nn.relu, name='conv1_1')
            x = conv2d(x, f=64, k=3, s=1, act=tf.nn.relu, name='conv1_2')
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME', name='pool1')

            """ Conv2 """
            x = conv2d(x, f=128, k=3, s=1, act=tf.nn.relu, name='conv2_1')
            x = conv2d(x, f=128, k=3, s=1, act=tf.nn.relu, name='conv2_2')
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME', name='pool2')

            """ Conv3 """
            x = conv2d(x, f=256, k=3, s=1, act=tf.nn.relu, name='conv3_1')
            x = conv2d(x, f=256, k=3, s=1, act=tf.nn.relu, name='conv3_2')
            x = conv2d(x, f=256, k=3, s=1, act=tf.nn.relu, name='conv3_3')
            x = conv2d(x, f=256, k=3, s=1, act=tf.nn.relu, name='conv3_4')
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME', name='pool3')

            """ Conv4 """
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv4_1')
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv4_2')
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv4_3')
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv4_4')
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME', name='pool4')

            bottle_neck = x

            """ Conv5 """
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv5_1')
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv5_2')
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv5_3')
            x = conv2d(x, f=512, k=3, s=1, act=tf.nn.relu, name='conv5_4')
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME', name='pool5')

            """ fc layers """
            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, units=4096, activation=tf.nn.relu, name='fc6')
            x = tf.layers.dense(x, units=4096, activation=tf.nn.relu, name='fc7')
            x = tf.layers.dense(x, units=1000, name='fc8')

            return tf.identity(x), bottle_neck

    def build_srgan(self):
        def mse_loss(pred, data):
            return tf.reduce_mean(tf.nn.l2_loss(pred - data))

        def sigmoid_loss(logits, label):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label)

        # Generator
        self.g = self.generator(self.x_lr)

        # Discriminator
        d_real = self.discriminator(self.x_hr)
        d_fake = self.discriminator(self.g, reuse=True)

        # VGG19
        x_vgg_real = tf.image.resize_images(self.x_hr, size=self.vgg_image_shape[:2])  # default BILINEAR method
        x_vgg_fake = tf.image.resize_images(self.g, size=self.vgg_image_shape[:2])

        vgg_net_real, vgg_bottle_real = self.vgg19((x_vgg_real + 1.) / 2.)
        _, vgg_bottle_fake = self.vgg19((x_vgg_fake + 1.) / 2., reuse=True)

        # Losses
        d_real_loss = sigmoid_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = sigmoid_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss

        g_cnt_loss = self.content_loss_weight * sigmoid_loss(d_fake, tf.ones_like(d_fake))
        g_mse_loss = mse_loss(self.g, self.x_hr)
        g_vgg_loss = self.vgg_loss_weight * mse_loss(vgg_bottle_fake, vgg_bottle_real)
        self.g_loss = g_cnt_loss + g_mse_loss + g_vgg_loss

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_cnt_loss", g_cnt_loss)
        tf.summary.scalar("loss/g_mse_loss", g_mse_loss)
        tf.summary.scalar("loss/g_vgg_loss", g_vgg_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(loss=self.d_loss,
                                                                                        var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(loss=self.g_loss,
                                                                                        var_list=g_params)

        # pre-train
        self.g_init_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                                beta1=self.beta1, beta2=self.beta2).minimize(loss=g_mse_loss,
                                                                                             var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
