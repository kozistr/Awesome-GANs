import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # reproducibility
np.random.seed(777)      # reproducibility


he_normal = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
l2_reg = tf.contrib.layers.l2_regularizer


def conv2d(x, f=64, k=4, s=1, reg=5e-4, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 4
    :param s: strides, default 1
    :param reg: weight regularizer, default 5e-4
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: conv2d net
    """

    if pad == 'other':
        pad = 'VALID'
        x = tf.pad(x, [[0, 0], [3, 3], [0, 3], [3, 3]], "CONSTANT")

    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=he_normal,
                            kernel_regularizer=l2_reg(reg),
                            bias_initializer=tf.zeros_initializer(),
                            padding=pad,
                            name=name)


def deconv2d(x, f=64, k=4, s=1, reg=5e-4, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 4
    :param s: strides, default 1
    :param reg: weight regularizer, default 5e-4
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: deconv2d net
    """

    if pad == 'other':
        pad = 'VALID'
        x = tf.pad(x, [[0, 0], [3, 3], [0, 3], [3, 3]], "CONSTANT")

    return tf.layers.conv2d_transpose(x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=he_normal,
                                      kernel_regularizer=l2_reg(reg),
                                      bias_initializer=tf.zeros_initializer(),
                                      padding=pad,
                                      name=name)


def batch_norm(x, eps=1e-5, reuse=False, name='batch_norm'):
    return tf.layers.batch_normalization(x, epsilon=eps, momentum=0.9, scale=True, reuse=reuse, name=name)


def inst_norm(x, eps=1e-5, affine=True, name="instance_norm"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(x, [1, 2], keep_dims=True)

        normalized = tf.div(x - mean, tf.sqrt(variance + eps))

        if not affine:
            return normalized
        else:
            depth = x.get_shape()[3]  # input channel

            scale = tf.get_variable('scale', [depth],
                                    initializer=tf.random_normal_initializer(mean=1., stddev=.02, dtype=tf.float32))
            offset = tf.get_variable('offset', [depth],
                                     initializer=tf.zeros_initializer())

        return scale * normalized + offset


def pixel_norm(x, eps=1e-8):
    return x / tf.sqrt(tf.reduce_mean(x ** 2, axis=3, keep_dims=True) + eps)


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


def bacth_concat(x, eps=1e-8, averaging='all'):
    """
    ref : https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow/blob/master/ops.py#L145
    """
    adj_std = lambda x_, **kwargs: tf.sqrt(tf.reduce_mean((x_ - tf.reduce_mean(x_, **kwargs)) ** 2, **kwargs) + eps)

    val_ = adj_std(x, axis=0, keep_dims=True)
    if averaging == 'all':
        val_ = tf.reduce_mean(val_, keep_dims=True)
    val_ = tf.tile(val_, multiples=[tf.shape(x)[0], 4, 4, 1])
    return tf.concat([x, val_], axis=3)


class PGGAN:

    def __init__(self, s, batch_size=16, input_height=128, input_width=128, input_channel=3,
                 pg=1, pg_t=False, sample_num=1 * 1, sample_size=1, output_height=128, output_width=128,
                 df_dim=64, gf_dim=64, z_dim=512, lr=1e-4, epsilon=1e-9):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 16
        :param input_height: input image height, default 128
        :param input_width: input image width, default 128
        :param input_channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 128x128x3(HWC).

        # Output Settings
        :param pg: size of the image for model?, default 1
        :param pg_t: pg status, default False
        :param sample_num: the number of output images, default 1
        :param sample_size: sample image size, default 1
        :param output_height: output images height, default 128
        :param output_width: output images width, default 128

        # For CNN model
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
        :param z_dim: z dimension (kinda noise), default 512
        :param lr: learning rate, default 1e-4
        :param epsilon: epsilon, default 1e-9
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.pg = pg
        self.pg_t = pg_t
        self.output_size = 4 * pow(2, self.pg - 1)

        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.z_dim = z_dim
        self.beta1 = 0.  # 0 ??
        self.beta2 = .99
        self.lr = lr
        self.eps = epsilon

        # pre-defined
        self.d_real = 0.
        self.d_fake = 0.
        self.g_loss = 0.
        self.d_loss = 0.
        self.gp = 0.
        self.gp_target = 1.
        self.gp_lambda = 10.  # slower convergence but good
        self.gp_w = 1e-3

        self.g = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None
        self.r_saver = None
        self.out_saver = None

        # Placeholders
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.output_size, self.output_size, self.input_channel],
                                name="x-image")
        self.z = tf.placeholder(tf.float32,
                                shape=[None, self.z_dim],
                                name='z-noise')
        self.step_pl = tf.placeholder(tf.float32, shape=None)
        self.alpha_trans = tf.Variable(initial_value=0., trainable=False, name='alpha_trans')
        self.alpha_trans_update = None

        self.build_pggan()  # build PGGAN model

    def discriminator(self, x, pg=1, pg_t=False, reuse=None):
        def nf(n):
            return tf.cast(min(1024 / (2 ** n), self.z_dim), dtype=tf.int32)

        with tf.variable_scope("disc", reuse=reuse):
            if pg_t:
                x_out = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
                x_out = conv2d(x_out, nf(pg - 2), k=1, s=1, name='disc_out_conv2d-%d' % x_out.get_shape()[1])
                x_out = tf.nn.leaky_relu(x_out)

            x = conv2d(x, nf(pg - 1), k=1, s=1, name='disc_out_conv2d-%d' % x.get_shape()[1])
            x = tf.nn.leaky_relu(x)

            for i in range(pg - 1):
                x = conv2d(x, nf(pg - 1 - i), k=1, s=1, name='disc_n_1_conv2d-%d' % x.get_shape()[1])
                x = tf.nn.leaky_relu(x)

                x = conv2d(x, nf(pg - 2 - i), k=1, s=1, name='disc_n_2_conv2d-%d' % x.get_shape()[1])
                x = tf.nn.leaky_relu(x)

                x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)

                if i == 0 and pg_t:
                    x = (1. - self.alpha_trans) * x_out + self.alpha_trans * x

            x = bacth_concat(x)

            x = conv2d(x, nf(1), k=3, s=1, name='disc_n_1_conv2d-%d' % x.get_shape()[1])
            x = tf.nn.leaky_relu(x)

            x = conv2d(x, nf(1), k=4, s=1, pad='VALID', name='disc_n_2_conv2d-%d' % x.get_shape()[1])
            x = tf.nn.leaky_relu(x)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, 1, name='disc_n_fc')

            return x

    def generator(self, z, pg=1, pg_t=False, reuse=None):
        def nf(n):
            return tf.cast(min(1024 / (2 ** n), self.z_dim), dtype=tf.int32)

        with tf.variable_scope("gen", reuse=reuse):
            x = tf.reshape(z, [None, 1, 1, nf(1)])
            x = conv2d(x, 512, k=4, s=1, name='gen_n_1_conv2d')
            x = tf.nn.leaky_relu(x)
            x = pixel_norm(x)

            x = tf.reshape(z, [None, 4, 4, nf(1)])
            x = conv2d(x, 512, k=3, s=1, name='gen_n_2_conv2d')
            x = tf.nn.leaky_relu(x)
            x = pixel_norm(x)

            x_out = None
            for i in range(pg - 1):
                if i == pg - 2 and pg_t:
                    x_out = conv2d(x, 3, k=1, s=1, name='gen_out_conv2d-%d' % x.get_shape()[1])  # to RGB images
                    x_out = resize_nn(x_out, x_out.get_shape().as_list()[1] * 2)                 # up-sampling

                x = resize_nn(x, x.get_shape().as_list()[1] * 2)
                x = conv2d(x, nf(i + 1), k=3, s=1, name='gen_n_1_conv2d-%d' % x.get_shape()[1])
                x = tf.nn.leaky_relu(x)
                x = pixel_norm(x)

                x = resize_nn(x, x.get_shape().as_list()[1] * 2)
                x = conv2d(x, nf(i + 1), k=3, s=1, name='gen_n_2_conv2d-%d' % x.get_shape()[1])
                x = tf.nn.leaky_relu(x)
                x = pixel_norm(x)

            x = conv2d(x, 3, k=1, s=1, name='gen_out_conv2d-%d' % x.get_shape()[1])  # to RGB images

            if pg == 1:
                return x

            if pg_t:
                x = (1. - self.alpha_trans) * x_out + self.alpha_trans * x

            return x

    def build_pggan(self):
        self.alpha_trans_update = self.alpha_trans.assign(self.step_pl / 32000)

        # Generator
        self.g = self.generator(self.z, self.pg, self.pg_t)

        # Discriminator
        d_real = self.discriminator(self.x, self.pg, self.pg_t)
        d_fake = self.discriminator(self.g, self.pg, self.pg_t, reuse=True)

        # Loss (skipping label-penalty for G/D nets)
        d_real_loss = tf.reduce_mean(d_real)
        d_fake_loss = tf.reduce_mean(d_fake)
        self.d_loss = d_real_loss - d_fake_loss
        self.g_loss = -d_fake_loss

        # Gradient Penalty
        diff = self.g - self.x
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interp = self.x + (alpha * diff)
        d_interp = self.discriminator(interp, self.pg, self.pg_t, reuse=True)
        grads = tf.gradients(d_interp, [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1, 2, 3]))
        self.gp = tf.reduce_mean(tf.square(slopes - self.gp_target))

        self.d_loss += (self.gp_lambda / (self.gp_target ** 2)) * self.gp + \
            self.gp_w * tf.reduce_mean(tf.square(d_real - 0.))

        # Summary
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        tf.summary.scalar("misc/gp", self.gp)

        # Training Parameters
        t_vars = tf.trainable_variables()

        d_params = [v for v in t_vars if v.name.startswith('disc')]
        g_params = [v for v in t_vars if v.name.startswith('gen')]

        d_n_params = [v for v in d_params if v.name.startswith('disc_n')]
        g_n_params = [v for v in g_params if v.name.startswith('gen_n')]

        d_n_out_params = [v for v in d_params if v.name.startswith('disc_out')]
        g_n_out_params = [v for v in g_params if v.name.startswith('gen_out')]

        d_n_nwm_params = [v for v in d_n_params if not v.name.endswith('%d' % self.output_size)]  # nwm : not new model
        g_n_nwm_params = [v for v in g_n_params if not v.name.endswith('%d' % self.output_size)]  # nwm : not new model

        d_n_out_nwm_params = [v for v in d_n_out_params if not v.name.endswith('%d' % self.output_size)]
        g_n_out_nwm_params = [v for v in g_n_out_params if not v.name.endswith('%d' % self.output_size)]

        # Optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(d_params + g_params, max_to_keep=1)
        self.r_saver = tf.train.Saver(d_n_nwm_params + g_n_nwm_params, max_to_keep=1)
        if len(d_n_out_nwm_params + g_n_out_nwm_params):
            self.out_saver = tf.train.Saver(d_n_out_nwm_params + g_n_out_nwm_params, max_to_keep=1)

        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
