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


def main():
    start_time = time.time()  # clocking start

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:

        end_time = time.time() - start_time

        # elapsed time
        print("[+] Elapsed time {:.8f}s".format(end_time))

        # close tf.Session
        s.close()

if __name__ == '__main__':
    main()
