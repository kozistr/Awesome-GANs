import tensorflow as tf


class VBN(object):
    """
    Virtual Batch Normalization
    (modified from https://github.com/openai/improved-gan/ definition)
    """

    def __init__(self, x, name, epsilon=1e-5):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        shape = x.get_shape().as_list()
        assert len(shape) == 3, shape
        with tf.variable_scope(name):
            assert name.startswith("d") or name.startswith("g")
            self.epsilon = epsilon
            self.name = name
            self.mean = tf.reduce_mean(x, [0, 1], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(x), [0, 1], keep_dims=True)
            self.batch_size = int(x.get_shape()[0])

            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None

            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            self.reference_output = out

    def __call__(self, x):
        with tf.variable_scope(self.name):
            new_coeff = 1.0 / (self.batch_size + 1.0)
            old_coeff = 1.0 - new_coeff
            new_mean = tf.reduce_mean(x, [0, 1], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [0, 1], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")

            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 3
        self.gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1.0, 0.02))
        gamma = tf.reshape(self.gamma, [1, 1, -1])
        self.beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.0))
        beta = tf.reshape(self.beta, [1, 1, -1])

        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None

        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))

        out = x - mean
        out = out / std
        out = out * gamma
        out = out + beta

        return out


def gaussian_noise_layer(input_layer, std=0.5):
    noise = tf.random_normal(shape=input_layer.get_shape().as_list(), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def conv1d(x, f=64, k=1, s=1, reuse=False, bias=False, pad='SAME', name='conv1d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 1
    :param s: strides, default 1
    :param reuse: param re-usability, default False
    :param bias: using bias, default False
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: covn2d net
    """
    return tf.layers.conv1d(
        x,
        filters=f,
        kernel_size=k,
        strides=s,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
        use_bias=bias,
        padding=pad,
        reuse=reuse,
        name=name,
    )


def conv2d(x, f=64, k=5, s=2, reuse=False, bias=False, pad='SAME', name='conv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param reuse: param re-usability, default False
    :param bias: using bias, default False
    :param pad: padding (valid or same), default same
    :param name: scope name, default conv2d
    :return: covn2d net
    """
    return tf.layers.conv2d(
        x,
        filters=f,
        kernel_size=k,
        strides=s,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
        use_bias=bias,
        padding=pad,
        reuse=reuse,
        name=name,
    )


def deconv2d(x, f=64, k=5, s=2, reuse=False, bias=False, pad='SAME', name='deconv2d'):
    """
    :param x: input
    :param f: filters, default 64
    :param k: kernel size, default 3
    :param s: strides, default 2
    :param reuse: param re-usability, default False
    :param bias: using bias, default False
    :param pad: padding (valid or same), default same
    :param name: scope name, default deconv2d
    :return: decovn2d net
    """
    return tf.layers.conv2d_transpose(
        x,
        filters=f,
        kernel_size=k,
        strides=s,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
        use_bias=bias,
        padding=pad,
        reuse=reuse,
        name=name,
    )
