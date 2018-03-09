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
        self.lr_decay_rate = 1e-1
        self.lr_decay_epoch = 100

        self.vgg_params = []
        self.vgg_weights = '/home/zero/hdd/vgg19.npy'  # for temporary path; it'll be moved to another place...
        self.vgg_mean = [103.939, 116.779, 123.68]

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.d_loss = 0.
        self.g_mse_loss = 0.
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

    def vgg19(self, x, weights, reuse=None):
        """
        Download pre-trained model for tensorflow

        Credited by https://github.com/tensorlayer
        Link : https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs

        :param x: 224x224x3 HR images
        :param weights: vgg19 pre-trained weight
        :param reuse: re-usability
        :return: prob
        """
        with tf.variable_scope("vgg19", reuse=reuse):
            rgb_scaled = tf.cast(x * 255., dtype=tf.float32)  # inverse_transform

            # rgb to bgr
            r, g, b = tf.split(rgb_scaled, 3, 3)
            bgr = tf.concat([b - self.vgg_mean[0],
                             g - self.vgg_mean[1],
                             r - self.vgg_mean[2]], axis=3)

            std = 1e-1

            # VGG19
            """ Conv1 """
            with tf.name_scope("conv1_1") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[64], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(bgr, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv1_1 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv1_2") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[64], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv1_2 = tf.nn.relu(out, name=scope)

            pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            """ Conv2 """
            with tf.name_scope("conv2_1") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[128], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv2_1 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv2_2") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[128], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv2_2 = tf.nn.relu(out, name=scope)

            pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            """ Conv3 """
            with tf.name_scope("conv3_1") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv3_1 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv3_2") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv3_2 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv3_3") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv3_3 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv3_4") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv3_3, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv3_4 = tf.nn.relu(out, name=scope)

            pool3 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            """ Conv4 """
            with tf.name_scope("conv4_1") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv4_1 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv4_2") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv4_2 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv4_3") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv4_3 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv4_4") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv4_3, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv4_4 = tf.nn.relu(out, name=scope)

            pool4 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

            bottle_neck = pool4

            """ Conv5 """
            with tf.name_scope("conv5_1") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv5_1 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv5_2") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv5_2 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv5_3") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv5_3 = tf.nn.relu(out, name=scope)

            with tf.name_scope("conv5_4") as scope:
                kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(0., shape=[512], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                conv = tf.nn.conv2d(conv5_3, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, bias)
                conv5_4 = tf.nn.relu(out, name=scope)

            pool5 = tf.nn.max_pool(conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

            """ fc layers """
            import numpy as np

            with tf.name_scope("fc6") as scope:
                pool5_size = int(np.prod(pool5.get_shape()[1:]))  # (-1, x)

                weight = tf.Variable(tf.truncated_normal(shape=[pool5_size, 4096], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(1., shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                x = tf.reshape(pool5, (-1, pool5_size))

                out = tf.nn.bias_add(tf.matmul(x, weight), bias, name=scope)
                fc6 = tf.nn.relu(out)

            with tf.name_scope("fc7") as scope:
                fc6_size = int(np.prod(fc6.get_shape()[1:]))  # (-1, x)

                weight = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(1., shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                x = tf.reshape(fc6, (-1, fc6_size))

                out = tf.nn.bias_add(tf.matmul(x, weight), bias, name=scope)
                fc7 = tf.nn.relu(out)

            with tf.name_scope("fc8") as scope:
                fc7_size = int(np.prod(fc6.get_shape()[1:]))  # (-1, x)

                weight = tf.Variable(tf.truncated_normal(shape=[4096, 1000], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(1., shape=[1000], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([kernel, bias])

                x = tf.reshape(fc7, (-1, fc7_size))

                out = tf.nn.bias_add(tf.matmul(x, weight), bias, name=scope)
                prob = tf.nn.softmax(out)

        # Loading vgg19-pre_trained.npz weights
        if reuse is None:
            from collections import OrderedDict

            vgg19_model = np.load(weights, encoding='latin1').item()
            vgg19_model = OrderedDict(sorted(vgg19_model.items()))

            for i, k in enumerate(vgg19_model.keys()):
                print("[+] Loading VGG19 - %-2d layer : %-8s" % (i, k))

                try:
                    self.s.run(self.vgg_params[i][0].assign(tf.convert_to_tensor(vgg19_model[k][0], dtype=tf.float32)))
                except ValueError:
                    print("[-] model weight's shape :", self.vgg_params[i][0].get_shape())
                    print("[-] file  weight's shape :", vgg19_model[k][0].shape)
                    raise ValueError

                try:
                    self.s.run(self.vgg_params[i][1].assign(tf.convert_to_tensor(vgg19_model[k][1], dtype=tf.float32)))
                except ValueError:
                    print("[-] model bias's shape :", self.vgg_params[i][1].get_shape())
                    print("[-] file  bias's shape :", vgg19_model[k][1].shape)
                    raise ValueError

        return tf.identity(fc8), bottle_neck

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

        vgg_net_real, vgg_bottle_real = self.vgg19((x_vgg_real + 1) / 2, weights=self.vgg_weights)
        _, vgg_bottle_fake = self.vgg19((x_vgg_fake + 1) / 2, reuse=True, weights=self.vgg_weights)

        # Losses
        d_real_loss = sigmoid_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = sigmoid_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss

        g_cnt_loss = self.content_loss_weight * sigmoid_loss(d_fake, tf.ones_like(d_fake))
        self.g_mse_loss = mse_loss(self.g, self.x_hr)
        g_vgg_loss = self.vgg_loss_weight * mse_loss(vgg_bottle_fake, vgg_bottle_real)
        self.g_loss = g_cnt_loss + self.g_mse_loss + g_vgg_loss

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_cnt_loss", g_cnt_loss)
        tf.summary.scalar("loss/g_mse_loss", self.g_mse_loss)
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
                                                beta1=self.beta1, beta2=self.beta2).minimize(loss=self.g_mse_loss,
                                                                                             var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
