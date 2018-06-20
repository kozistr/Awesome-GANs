"""
Inspired by https://github.com/tkarras/progressive_growing_of_gans/blob/master/tfutil.py
"""

import tensorflow as tf
import numpy as np


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
