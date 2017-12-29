from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import sys
import time

import magan_model as magan

sys.path.append('../')
import image_utils as iu

results = {
    'output': './gen_img/',
    'checkpoint': './model/checkpoint',
    'model': './model/MAGAN-model.ckpt'
}

train_step = {
    'epoch': 120,
    'n_iter': 1000,
    'logging_interval': 2000,
}


def main():
    start_time = time.time()  # Clocking start

    # MNIST Dataset load
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # MAGAN Model
        model = magan.MAGAN(s)

        # Initializing
        s.run(tf.global_variables_initializer())

        sample_x, _ = mnist.train.next_batch(model.sample_num)
        sample_x = np.reshape(sample_x, model.image_shape)
        sample_z = np.random.uniform(-1., 1., [model.sample_num, model.z_dim]).astype(np.float32)

        global_step = 0
        N = mnist.train.images.shape[0] // model.batch_size  # training set size

        # Pre-Train
        print("[+] pre-train")
        for _ in range(2):
            for iter_ in range(N):
                batch_x, _ = mnist.train.next_batch(model.batch_size)
                batch_x = np.reshape(batch_x, model.image_shape)

                s.run([model.d_real_op, model.d_real_loss],
                      feed_dict={
                          model.x: batch_x,
                      })

        # Initial margin value
        margin = s.run(model.d_real_loss,
                       feed_dict={
                           model.x: sample_x,
                       })

        s_g_0 = np.inf  # Sg_0 = infinite
        d_overpowered = False

        for epoch in range(train_step['epoch']):
            s_d, s_g = 0., 0.
            for i in range(N):
                batch_x, _ = mnist.train.next_batch(model.batch_size)  # with batch_size, 64
                batch_x = np.reshape(batch_x, model.image_shape)

                batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)  # 64 x 128

                # Update D network
                if not d_overpowered:
                    _, d_loss, d_real_loss = s.run([model.d_op, model.d_loss, model.d_real_loss],
                                                   feed_dict={
                                                       model.x: batch_x,
                                                       model.z: batch_z,
                                                       model.m: margin,
                                                   })

                # Update D real sample
                s_d += np.sum(d_real_loss)

                # Update G network
                _, g_loss, d_fake_loss = s.run([model.g_op, model.g_loss, model.d_fake_loss],
                                               feed_dict={
                                                   model.x: sample_x,
                                                   model.z: sample_z,
                                                   model.m: margin,
                                               })

                # Update G fake sample
                s_g += np.sum(d_fake_loss)

                # Logging
                if global_step % train_step['logging_interval'] == 0:
                    batch_x, _ = mnist.test.next_batch(model.batch_size)
                    batch_x = np.reshape(batch_x, model.image_shape)

                    batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)

                    d_loss, g_loss, summary = s.run([model.d_loss, model.g_loss, model.merged],
                                                    feed_dict={
                                                        model.x: batch_x,
                                                        model.z: batch_z,
                                                        model.m: margin,
                                                    })

                    d_overpowered = d_loss < g_loss / 2

                    # Print loss
                    print("[+] Epoch %03d Global Step %05d => " % (epoch, global_step),
                          " D loss : {:.8f}".format(d_loss),
                          " G loss : {:.8f}".format(g_loss))

                    # Training G model with sample image and noise
                    samples = s.run(model.g,
                                    feed_dict={
                                        model.x: batch_x,
                                        model.z: batch_z,
                                        model.m: margin,
                                    })

                    # Summary saver
                    model.writer.add_summary(summary, global_step)

                    # Export image generated by model G
                    sample_image_height = model.sample_size
                    sample_image_width = model.sample_size
                    sample_dir = results['output'] + 'train_{:08d}.png'.format(global_step)

                    # Generated image save
                    iu.save_images(samples,
                                   size=[sample_image_height, sample_image_width],
                                   image_path=sample_dir)

                    # Model save
                    model.saver.save(s, results['model'], global_step=global_step)

                global_step += 1

            # Update margin
            if s_d / N < margin and s_d < s_g and s_g_0 <= s_g:
                margin = s_d / N

            s_g_0 = s_g

            # Convergence Measure
            # e_d = s_d / N
            # e_g = s_g / N
            # L = e_d + np.abs(e_d - e_g)

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))

    # Close tf.Session
    s.close()


if __name__ == '__main__':
    main()
