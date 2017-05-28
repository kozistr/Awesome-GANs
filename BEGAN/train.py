from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import began
import image_utils as iu
from dataset import Dataset, DataIterator


dirs = {
    # 'cifar-10': '/home/zero/cifar/cifar-10-batches-py/',
    # 'cifar-100': '/home/zero/cifar/cifar-100-python/',
    'celeb-a': '/home/zero/celeba/',
    'sample_output': './BEGAN/',
    'checkpoint': './model/checkpoint',
    'model': './model/BEGAN-model.ckpt'
}
paras = {
    'epoch': 150,  # with GTX 1080 11gb,
    'batch_size': 64,
    'logging_interval': 1000
}


def conv2d(x, filter_shape, bias=True, stride=1, padding="SAME", name="conv2d"):
    kw, kh, nin, nout = filter_shape
    pad_size = (kw - 1) / 2

    if padding == "VALID":
        x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "SYMMETRIC")

    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable("weight", shape=filter_shape, initializer=initializer)
        x = tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding=padding)

        if bias:
            b = tf.get_variable("bias", shape=filter_shape[-1], initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)
    return x


def fc(x, output_shape, bias=True, name='fc'):
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(x, [-1, dim])
    input_shape = dim

    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable("weight", shape=[input_shape, output_shape], initializer=initializer)
        x = tf.matmul(x, weight)

        if bias:
            b = tf.get_variable("bias", shape=[output_shape], initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)
    return x


def pool(x, r=2, s=1):
    return tf.nn.avg_pool(x, ksize=[1, r, r, 1], strides=[1, s, s, 1], padding="SAME")


# tf.nn.l2_loss


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


def main():
    start_time = time.time()  # clocking start

    with tf.device('/gpu:1'):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=config) as s:

            end_time = time.time() - start_time
            # elapsed time
            print("[+] Elapsed time {:.8f}s".format(end_time))

            # close tf.Session
            s.close()

if __name__ == '__main__':
    main()
