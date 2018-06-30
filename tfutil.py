"""
Inspired by https://github.com/tkarras/progressive_growing_of_gans/blob/master/tfutil.py
"""

import tensorflow as tf
import numpy as np


seed = 1337
np.random.seed(seed)
tf.set_random_seed(seed)


# ---------------------------------------------------------------------------------------------
# For convenience :)


def run(*args, **kwargs):
    return tf.get_default_session().run(*args, **kwargs)


def is_tf_expression(x):
    return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable) or isinstance(x, tf.Operation)


def safe_log(x, eps=1e-12):
    with tf.name_scope("safe_log"):
        return tf.log(x + eps)


def safe_log2(x, eps=1e-12):
    with tf.name_scope("safe_log2"):
        return tf.log(x + eps) * np.float32(1. / np.log(2.))


def lerp(a, b, t):
    with tf.name_scope("lerp"):
        return a + (b - a) * t


def lerp_clip(a, b, t):
    with tf.name_scope("lerp_clip"):
        return a + (b - a) * tf.clip_by_value(t, 0., 1.)


def gaussian_noise(x, std=5e-2):
    noise = tf.random_normal(x.get_shape(), mean=0., stddev=std, dtype=tf.float32)
    return x + noise


# ---------------------------------------------------------------------------------------------
# Image Sampling with TF


def down_sampling(img, interp=tf.image.ResizeMethod.BILINEAR):
    shape = img.get_shape()  # [batch, height, width, channels]

    h2 = int(shape[1] // 2)
    w2 = int(shape[2] // 2)

    return tf.image.resize_images(img, [h2, w2], interp)


def up_sampling(img, interp=tf.image.ResizeMethod.BILINEAR):
    shape = img.get_shape()  # [batch, height, width, channels]

    h2 = int(shape[1] * 2)
    w2 = int(shape[2] * 2)

    return tf.image.resize_images(img, [h2, w2], interp)


# ---------------------------------------------------------------------------------------------
# Optimizer


class Optimizer(object):

    def __init__(self,
                 name='train',
                 optimizer='tf.train.AdamOptimizer',
                 learning_rate=1e-3,
                 use_loss_scaling=False,
                 loss_scaling_init=64.,
                 loss_scaling_inc=5e-4,
                 loss_scaling_dec=1.,
                 use_grad_scaling=False,
                 grad_scaling=7.,
                 **kwargs):

        self.name = name
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.use_loss_scaling = use_loss_scaling
        self.loss_scaling_init = loss_scaling_init
        self.loss_scaling_inc = loss_scaling_inc
        self.loss_scaling_dec = loss_scaling_dec

        self.use_grad_scaling = use_grad_scaling
        self.grad_scaling = grad_scaling


# ---------------------------------------------------------------------------------------------
# Network


class Network:

    def __init__(self):
        pass


# ---------------------------------------------------------------------------------------------
# Functions


w_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
b_init = tf.zeros_initializer()

reg = 5e-4
w_reg = tf.contrib.layers.l2_regularizer(reg)

eps = 1e-5


# Layers


def conv2d_alt(x, f=64, k=3, s=1, pad=0, pad_type='zero', use_bias=True, sn=False, name='conv2d'):
    with tf.variable_scope(name):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        elif pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        else:
            raise NotImplementedError("[-] Only 'zero' & 'reflect' are supported :(")

        if sn:
            w = tf.get_variable('kernel', shape=[k, k, x.get_shape()[-1], f],
                                initializer=w_init, regularizer=w_reg)
            x = tf.nn.conv2d(x, filter=spectral_norm(w), strides=[1, s, s, 1], padding='VALID')

            if use_bias:
                b = tf.get_variable('bias', shape=[f], initializer=b_init)
                x = tf.nn.bias_add(x, b)
        else:
            x = conv2d(x, f, k, s, name=name)

        return x


def conv2d(x, f=64, k=3, s=1, pad='SAME', reuse=None, is_train=True, name='conv2d'):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param reuse: reusable
    :param is_train: trainable
    :param name: scope name
    :return: net
    """
    return tf.layers.conv2d(inputs=x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=w_init,
                            kernel_regularizer=w_reg,
                            bias_initializer=b_init,
                            padding=pad,
                            reuse=reuse,
                            name=name)


def conv1d(x, f=64, k=3, s=1, pad='SAME', reuse=None, is_train=True, name='conv1d'):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param reuse: reusable
    :param is_train: trainable
    :param name: scope name
    :return: net
    """
    return tf.layers.conv1d(inputs=x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=w_init,
                            kernel_regularizer=w_reg,
                            bias_initializer=b_init,
                            padding=pad,
                            reuse=reuse,
                            name=name)


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


def deconv2d_alt(x, f=64, k=3, s=1, use_bias=True, sn=False, name='deconv2d'):
    with tf.variable_scope(name):
        if sn:
            w = tf.get_variable('kernel', shape=[k, k, x.get_shape()[-1], f],
                                initializer=w_init, regularizer=w_reg)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), strides=[1, s, s, 1], padding='SAME',
                                       output_shape=[x.get_shape()[0], x.get_shape()[1] * s, x.get_shape()[2] * s, f])

            if use_bias:
                b = tf.get_variable('bias', shape=[f], initializer=b_init)
                x = tf.nn.bias_add(x, b)
        else:
            x = deconv2d(x, f, k, s, name=name)

        return x


def deconv2d(x, f=64, k=3, s=1, pad='SAME', reuse=None, is_train=True, name='deconv2d'):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param reuse: reusable
    :param is_train: trainable
    :param name: scope name
    :return: net
    """
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=f, kernel_size=k, strides=s,
                                      kernel_initializer=w_init,
                                      kernel_regularizer=w_reg,
                                      bias_initializer=b_init,
                                      padding=pad,
                                      reuse=reuse,
                                      name=name)


def dense_alt(x, f=1024, sn=False, use_bias=True, name='fc'):
    with tf.variable_scope(name):
        x = flatten(x)

        if sn:
            w = tf.get_variable('kernel', shape=[x.get_shape()[-1], f],
                                initializer=w_init, regularizer=w_reg, dtype=tf.float32)
            x = tf.matmul(x, spectral_norm(w))

            if use_bias:
                b = tf.get_variable('bias', shape=[f], initializer=b_init)
                x = tf.nn.bias_add(x, b)
        else:
            x = dense(x, f, name=name)

        return x


def dense(x, f=1024, reuse=None, is_train=True, name='fc'):
    """
    :param x: input
    :param f: fully connected units
    :param reuse: reusable
    :param name: scope name
    :param is_train: trainable
    :return: net
    """
    return tf.layers.dense(inputs=x,
                           units=f,
                           kernel_initializer=w_init,
                           kernel_regularizer=w_reg,
                           bias_initializer=b_init,
                           reuse=reuse,
                           name=name)


def flatten(x):
    return tf.layers.flatten(x)


def hw_flatten(x):
    if is_tf_expression(x):
        return tf.reshape(x, shape=[x.get_shape()[0], -1, x.get_shape()[-1]])
    else:
        return np.reshape(x, [x.shape[0], -1, x.shape[-1]])


# Normalize


def l2_norm(x, eps=1e-12):
    return x / (tf.sqrt(tf.reduce_sum(tf.square(x))) + eps)


def batch_norm(x, momentum=0.9, scaling=True, is_train=True, reuse=None, name="bn"):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=scaling,
                                         training=is_train,
                                         reuse=reuse,
                                         name=name)


def instance_norm(x, affine=True, reuse=None, name=""):
    with tf.variable_scope('instance_normalize-%s' % name, reuse=reuse):
        mean, variance = tf.nn.moments(x, [1, 2], keepdims=True)

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


def pixel_norm(x):
    return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2, 3]) + eps)


def spectral_norm(x, n_iter=1):
    x_shape = x.get_shape()

    x = tf.reshape(x, (-1, x_shape[-1]))  # (n * h * w, c)

    u = tf.get_variable('u', shape=(1, x_shape[-1]), initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for _ in range(n_iter):
        v_ = tf.matmul(u_hat, tf.transpose(x))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, x)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, x), tf.transpose(u_hat))
    x_norm = x / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        x_norm = tf.reshape(x_norm, x_shape)

    return x_norm


# Activations


def prelu(x, stddev=1e-2, reuse=False, name='prelu'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        _alpha = tf.get_variable('_alpha',
                                 shape=[1],
                                 initializer=tf.constant_initializer(stddev),
                                 # initializer=tf.random_normal_initializer(stddev)
                                 dtype=x.dtype)

        return tf.maximum(_alpha * x, x)


# Pooling


def global_avg_pooling(x):
    return tf.reduce_mean(x, axis=[1, 2])


# Losses


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def l2_loss(x, y):
    return tf.nn.l2_loss(y - x)


def mse_loss(x, y, n, is_mean=False):  # ~ l2_loss
    if is_mean:
        return tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x, y)))
    else:
        return tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, y)))


def rmse_loss(x, y, n):
    return tf.sqrt(mse_loss(x, y, n))


def psnr_loss(x, y, n):
    return 20. * tf.log(tf.reduce_max(x) / mse_loss(x, y, n))


def sce_loss(data, label):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=data, labels=label))


def softce_loss(data, label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=data, labels=label))


def ssoftce_loss(data, label):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=data, labels=label))
