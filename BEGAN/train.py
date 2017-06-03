from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import began
import dataset
import image_utils as iu


dirs = {
    'celeb-a': '/home/zero/celeba/img_align_celeba/',
    'sample_output': './BEGAN/',
    'checkpoint': './model/checkpoint',
    'model': './model/BEGAN-model.ckpt'
}
paras = {
    'epoch': 250,
    'batch_size': 16,
    'logging_interval': 1000
}


def main():
    start_time = time.time()  # clocking start

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=config) as s:
        end_time = time.time() - start_time

        # BEGAN Model
        model = began.BEGAN(s)

        # initializing
        s.run(tf.global_variables_initializer())

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
            # return

        # initializing variables
        tf.global_variables_initializer().run()

        # load Celeb-A dataset
        batch = dataset(dirs['celeb-a'])

        kt = tf.Variable(0., dtype=tf.float32)  # init K_0 value, 0

        d_overpowered = False
        for epoch in range(paras['epoch']):
            # k_t update
            # k_t+1 = K_t + lambda_k * (gamma * d_real - d_fake)
            kt = kt + model.lambda_k * (model.gamma * model.D_real - model.D_fake)

            # z update
            z = np.random.uniform(-1., 1., [paras['batch_size'], model.z_dim]).astype(np.float32)

            # update D network
            if not d_overpowered:
                s.run(model.d_op, feed_dict={model.x: 0, model.z: z, model.kt: kt})

            # update G network
            s.run(model.g_op, feed_dict={model.z: z, model.kt: kt})

            if global_step % paras['logging_interval'] == 0:

                pass

            global_step += 1

    # elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))

    # close tf.Session
    s.close()

if __name__ == '__main__':
    main()
