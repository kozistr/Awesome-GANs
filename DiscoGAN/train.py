from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import discogan

import sys
sys.path.insert(0, '../')

from datasets import DataIterator, DataSet
import image_utils as iu


dirs = {
    'sample_output': './DiscoGAN/',
    'checkpoint': './model/checkpoint',
    'model': './model/DiscoGAN-model.ckpt'
}
paras = {
    'epoch': 250,
    'batch_size': 256,
    'logging_interval': 1000
}


def main():
    start_time = time.time()  # clocking start

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        # DiscoGAN model
        model = discogan.DiscoGAN(s)

        # load model & graph & weight
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            model.saver.restore(s, ckpt.model_checkpoint_path)

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("[+] global step : %s" % global_step, " successfully loaded")
        else:
            global_step = 0
            print('[-] No checkpoint file found')

        # initializing variables
        tf.global_variables_initializer().run()

    end_time = time.time() - start_time

    # elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))

    # close tf.Session
    s.close()

if __name__ == '__main__':
    main()



