import tensorflow as tf

import vgg19

import sys

sys.path.append('../')
import tfutil as t


tf.set_random_seed(777)  # reproducibility


class SRGAN:

    def __init__(self, s, batch_size=16, height=384, width=384, channel=3,
                 sample_num=1 * 1, sample_size=1,
                 df_dim=64, gf_dim=64, lr=1e-4):

        """ Super-Resolution GAN Class
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 16
        :param height: input image height, default 384
        :param width: input image width, default 384
        :param channel: input image channel, default 3 (RGB)
        - in case of DIV2K-HR, image size is 384x384x3(HWC).

        # Output Settings
        :param sample_num: the number of output images, default 1
        :param sample_size: sample image size, default 1

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
        :param lr: learning rate, default 1e-4
        :param use_vgg19: using pre-trained vgg19 bottle-neck features, default False
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel

        self.lr_image_shape = [None, self.height // 4, self.width // 4, self.channel]
        self.hr_image_shape = [None, self.height, self.width, self.channel]

        self.vgg_image_shape = [224, 224, 3]

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.beta1 = 0.9
        self.beta2 = 0.999

        self.lr_decay_rate = 1e-1
        self.lr_low_boundary = 1e-5
        self.lr_update_step = 1e5
        self.lr_update_epoch = 1000

        self.vgg_mean = [103.939, 116.779, 123.68]

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.d_loss = 0.
        self.g_adv_loss = 0.
        self.g_mse_loss = 0.
        self.g_cnt_loss = 0.
        self.g_loss = 0.
        self.psnr = 0.

        self.vgg19 = None

        self.g = None

        self.adv_scaling = 1e-3
        self.vgg_scaling = 3e-6  # 1. / 12.75  # 6e-3

        self.d_op = None
        self.g_op = None
        self.g_init_op = None
        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholders
        self.x_hr = tf.placeholder(tf.float32, shape=self.hr_image_shape, name="x-image-hr")  # (-1, 384, 384, 3)
        self.x_lr = tf.placeholder(tf.float32, shape=self.lr_image_shape, name="x-image-lr")  # (-1, 96, 96, 3)

        self.lr = tf.placeholder(tf.float32, name='lr')

        self.build_srgan()  # build SRGAN model

    def discriminator(self, x, reuse=None):
        """
        # Following a network architecture referred in the paper
        :param x: Input images (-1, 384, 384, 3)
        :param reuse: re-usability
        :return: HR (High Resolution) or SR (Super Resolution) images
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            x = t.conv2d(x, self.df_dim, 3, 1, name='n64s1-1')
            x = tf.nn.leaky_relu(x)

            strides = [2, 1]
            filters = [1, 2, 2, 4, 4, 8, 8]

            for i, f in enumerate(filters):
                x = t.conv2d(x, f=f, k=3, s=strides[i % 2], name='n%ds%d-%d' % (f, strides[i % 2], i + 1))
                x = t.batch_norm(x, name='n%d-bn-%d' % (f, i + 1))
                x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)  # (-1, 96 * 96 * 64)

            x = t.dense(x, 1024, name='disc-fc-1')
            x = tf.nn.leaky_relu(x)

            x = t.dense(x, 1, name='disc-fc-2')
            # x = tf.nn.sigmoid(x)
            return x

    def generator(self, x, reuse=None, is_train=True):
        """
        :param x: LR (Low Resolution) images, (-1, 96, 96, 3)
        :param reuse: scope re-usability
        :param is_train: is trainable, default True
        :return: SR (Super Resolution) images, (-1, 384, 384, 3)
        """

        with tf.variable_scope("generator", reuse=reuse):
            def residual_block(x, f, name="", _is_train=True):
                with tf.variable_scope(name):
                    shortcut = tf.identity(x, name='n64s1-shortcut')

                    x = t.conv2d(x, f, 3, 1, name="n64s1-1")
                    x = t.batch_norm(x, is_train=_is_train, name="n64s1-bn-1")
                    x = t.prelu(x, reuse=reuse, name='n64s1-prelu-1')
                    x = t.conv2d(x, f, 3, 1, name="n64s1-2")
                    x = t.batch_norm(x, is_train=_is_train, name="n64s1-bn-2")
                    x = tf.add(x, shortcut)

                    return x

            x = t.conv2d(x, self.gf_dim, 9, 1, name='n64s1-1')
            x = t.prelu(x, name='n64s1-prelu-1')

            skip_conn = tf.identity(x, name='skip_connection')

            # B residual blocks
            for i in range(1, 17):  # (1, 9)
                x = residual_block(x, self.gf_dim, name='b-residual_block_%d' % i, _is_train=is_train)

            x = t.conv2d(x, self.gf_dim, 3, 1, name='n64s1-3')
            x = t.batch_norm(x, is_train=is_train, name='n64s1-bn-3')

            x = tf.add(x, skip_conn)

            # sub-pixel conv2d blocks
            for i in range(1, 3):
                x = t.conv2d(x, self.gf_dim * 4, 3, 1, name='n256s1-%d' % (i + 2))
                x = t.sub_pixel_conv2d(x, f=None, s=2)
                x = t.prelu(x, name='n256s1-prelu-%d' % i)

            x = t.conv2d(x, self.channel, 9, 1, name='n3s1')  # (-1, 384, 384, 3)
            x = tf.nn.tanh(x)
            return x

    def build_vgg19(self, x, reuse=None):
        with tf.variable_scope("vgg19", reuse=reuse):
            # image re-scaling
            x = tf.cast((x + 1) / 2, dtype=tf.float32)  # [-1, 1] to [0, 1]
            x = tf.cast(x * 255., dtype=tf.float32)     # [0, 1]  to [0, 255]

            r, g, b = tf.split(x, 3, 3)
            bgr = tf.concat([b - self.vgg_mean[0],
                             g - self.vgg_mean[1],
                             r - self.vgg_mean[2]], axis=3)

            self.vgg19 = vgg19.VGG19(bgr)

            net = self.vgg19.vgg19_net['conv5_4']

            return net  # last layer

    def build_srgan(self):
        # Generator
        self.g = self.generator(self.x_lr)

        # Discriminator
        d_real = self.discriminator(self.x_hr)
        d_fake = self.discriminator(self.g, reuse=True)

        # Losses
        # d_real_loss = -tf.reduce_mean(t.safe_log(d_real))
        # d_fake_loss = -tf.reduce_mean(t.safe_log(1. - d_fake))
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss

        x_vgg_real = tf.image.resize_images(self.x_hr, size=self.vgg_image_shape[:2], align_corners=False)
        x_vgg_fake = tf.image.resize_images(self.g, size=self.vgg_image_shape[:2], align_corners=False)

        vgg_bottle_real = self.build_vgg19(x_vgg_real)
        vgg_bottle_fake = self.build_vgg19(x_vgg_fake, reuse=True)

        self.g_cnt_loss = self.vgg_scaling * t.mse_loss(vgg_bottle_fake, vgg_bottle_real, self.batch_size)

        self.g_mse_loss = t.mse_loss(self.g, self.x_hr, self.batch_size)

        # self.g_adv_loss = self.adv_scaling * tf.reduce_mean(-1. * t.safe_log(d_fake))
        self.g_adv_loss = self.adv_scaling * t.sce_loss(d_fake, tf.ones_like(d_fake))
        self.g_loss = self.g_adv_loss + self.g_mse_loss + self.g_cnt_loss

        def inverse_transform(img):
            return (img + 1.) * 127.5

        # calculate PSNR
        g, x_hr = inverse_transform(self.g), inverse_transform(self.x_hr)
        self.psnr = t.psnr_loss(g, x_hr, self.batch_size)

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_cnt_loss", self.g_cnt_loss)
        tf.summary.scalar("loss/g_mse_loss", self.g_mse_loss)
        tf.summary.scalar("loss/g_adv_loss", self.g_adv_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("misc/psnr", self.psnr)
        tf.summary.scalar("misc/lr", self.lr)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(loss=self.d_loss,
                                                                                        var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(loss=self.g_loss,
                                                                                        var_list=g_params)

        # pre-train
        self.g_init_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                beta1=self.beta1, beta2=self.beta2).minimize(loss=self.g_mse_loss,
                                                                                             var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=2)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
