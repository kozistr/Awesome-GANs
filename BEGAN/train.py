from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import began


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
