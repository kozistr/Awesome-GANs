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
                            use_bias=False,
                            activation=act,
                            padding=pad,
                            name=name)


def batch_norm(x, momentum=0.9, eps=1e-5, is_train=True):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         trainable=is_train)


class SRGAN:

    def __init__(self, s, batch_size=16, input_height=384, input_width=384, input_channel=3,
                 sample_num=1 * 1, sample_size=1, output_height=384, output_width=384,
                 df_dim=64, gf_dim=64, g_lr=1e-4, d_lr=1e-4):

        """ Super-Resolution GAN Class
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
        self.vgg_weights = '/home/zero/hdd/vgg19.npy'  # TF Slim VGG19 pre-trained model
        self.vgg_mean = [103.939, 116.779, 123.68]

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.d_loss = 0.
        self.g_adv_loss = 0.
        self.g_cnt_loss = 0.
        self.g_mse_loss = 0.
        self.g_loss = 0.

        self.g = None
        self.g_test = None
        self.adv_scaling = 1e-3
        self.vgg_scaling = 2e-6

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
            logits = tf.layers.dense(x, 1, name='d-fc-1')
            prob = tf.nn.sigmoid(logits)

            return logits, prob

    def generator(self, x, reuse=None, is_train=True):
        """
        :param x: LR (Low Resolution) images, (-1, 96, 96, 3)
        :param reuse: scope re-usability
        :param is_train: is trainable, default True
        :return: SR (Super Resolution) images, (-1, 384, 384, 3)
        """

        with tf.variable_scope("generator", reuse=reuse):
            def residual_block(x, name="", _is_train=True):
                with tf.variable_scope(name):
                    x = conv2d(x, self.gf_dim, name="n64s1-2")
                    x = batch_norm(x, is_train=_is_train)
                    x = tf.nn.relu(x)

                    return x

            x = conv2d(x, self.gf_dim, act=tf.nn.relu, name='n64s1-1')
            x_ = x  # for later, used at layer concat

            # B residual blocks
            for i in range(1, 17):  # (1, 9)
                xx = residual_block(x, name='b-residual_block_%d' % (i * 2 - 1), _is_train=is_train)
                xx = residual_block(xx, name='b-residual_block_%d' % (i * 2), _is_train=is_train)
                xx = tf.add(x, xx)
                x = xx

            x = conv2d(x, self.gf_dim, name='n64s1-3')
            x = batch_norm(x, is_train=is_train)

            x = tf.add(x_, x)

            # subpixel conv blocks
            for i in range(1, 3):
                x = conv2d(x, self.gf_dim * 4, name='n256s1-%d' % (i + 2))
                x = sub_pixel_conv2d(x, f=None, s=2)
                x = tf.nn.relu(x)

            x = conv2d(x, self.input_channel, act=tf.nn.tanh, k=1, name='n3s1')  # (-1, 384, 384, 3)

            return x

    def vgg_model(self, x, weights=None, reuse=None):
        import numpy as np

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
            # image re-scaling
            x = tf.cast((x + 1) / 2, dtype=tf.float32)  # [-1, 1] to [0, 1]
            x = tf.cast(x * 255., dtype=tf.float32)     # [0, 1]  to [0, 255]

            r, g, b = tf.split(x, 3, 3)
            bgr = tf.concat([b - self.vgg_mean[0],
                             g - self.vgg_mean[1],
                             r - self.vgg_mean[2]], axis=3)

            # size re-checking...
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

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

            bottle_neck = tf.identity(pool4)

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
            """
            with tf.name_scope("fc6") as scope:
                pool5_size = int(np.prod(pool5.get_shape()[1:]))  # (-1, x)

                weight = tf.Variable(tf.truncated_normal(shape=[pool5_size, 4096], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(1., shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([weight, bias])

                x = tf.reshape(pool5, (-1, pool5_size))

                out = tf.nn.bias_add(tf.matmul(x, weight), bias, name=scope)
                fc6 = tf.nn.relu(out)

            with tf.name_scope("fc7") as scope:
                fc6_size = int(np.prod(fc6.get_shape()[1:]))  # (-1, x)

                weight = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(1., shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([weight, bias])

                x = tf.reshape(fc6, (-1, fc6_size))

                out = tf.nn.bias_add(tf.matmul(x, weight), bias, name=scope)
                fc7 = tf.nn.relu(out)

            with tf.name_scope("fc8") as scope:
                fc7_size = int(np.prod(fc6.get_shape()[1:]))  # (-1, x)

                weight = tf.Variable(tf.truncated_normal(shape=[4096, 1000], stddev=std, dtype=tf.float32),
                                     name='weights')
                bias = tf.Variable(tf.constant(1., shape=[1000], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.vgg_params.append([weight, bias])

                x = tf.reshape(fc7, (-1, fc7_size))

                fc8 = tf.nn.bias_add(tf.matmul(x, weight), bias, name=scope)
                prob = tf.nn.softmax(fc8)
            """

        # Loading vgg19-pre_trained.npz weights
        if reuse is None:
            from collections import OrderedDict

            vgg19_model = np.load(weights, encoding='latin1').item()
            vgg19_model = OrderedDict(sorted(vgg19_model.items()))

            removed_layers = ["fc6", "fc7", "fc8"]
            for rl in removed_layers:
                vgg19_model.pop(rl, None)

            for i, k in enumerate(vgg19_model.keys()):  # skip fc layers
                print("[+] Loading VGG19 - %2d layer : %8s " % (i, k),
                      self.vgg_params[i][0].get_shape(), self.vgg_params[i][1].get_shape())

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

        return bottle_neck

    def build_srgan(self):
        def mse_loss(pred, data):
            return tf.reduce_mean(tf.reduce_sum(tf.square(pred - data), axis=[3]))

        def sigmoid_loss(logits, label):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label))

        # Generator
        self.g = self.generator(self.x_lr)
        self.g_test = self.generator(self.x_lr, reuse=True, is_train=False)

        # Discriminator
        d_real, d_real_prob = self.discriminator(self.x_hr)
        d_fake, d_fake_prob = self.discriminator(self.g, reuse=True)

        # VGG19
        # x_vgg_real = tf.image.resize_images(self.x_hr, size=self.vgg_image_shape[:2])  # default BILINEAR method
        # x_vgg_fake = tf.image.resize_images(self.g, size=self.vgg_image_shape[:2])

        # vgg_bottle_real = self.vgg_model(x_vgg_real, weights=self.vgg_weights)
        # vgg_bottle_fake = self.vgg_model(x_vgg_fake, weights=self.vgg_weights, reuse=True)

        # Losses
        d_real_loss = sigmoid_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = sigmoid_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss

        self.g_adv_loss = self.adv_scaling * sigmoid_loss(d_fake, tf.ones_like(d_fake))
        self.g_mse_loss = mse_loss(self.g, self.x_hr)
        # tf.losses.mean_squared_error(self.g, self.x_hr, reduction=tf.losses.Reduction.MEAN)
        # self.g_cnt_loss = self.vgg_scaling * mse_loss(vgg_bottle_real, vgg_bottle_fake)
        # tf.losses.mean_squared_error(vgg_bottle_fake, vgg_bottle_real, reduction=tf.losses.Reduction.MEAN)
        self.g_loss = self.g_adv_loss + self.g_mse_loss + self.g_cnt_loss

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        # tf.summary.scalar("loss/g_cnt_loss", self.g_cnt_loss)
        tf.summary.scalar("loss/g_mse_loss", self.g_mse_loss)
        tf.summary.scalar("loss/g_adv_loss", self.g_adv_loss)
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
