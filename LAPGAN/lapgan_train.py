from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import sys
import time

import lapgan_model as lapgan

sys.path.append('../')
import image_utils as iu
from datasets import DataIterator
from datasets import CiFarDataSet as DataSet


results = {
    'output': './gen_img/',
    'checkpoint': './model/checkpoint',
    'model': './model/LAPGAN-model.ckpt'
}

train_step = {
    'epoch': 300,
    'batch_size': 128,
    'logging_interval': 5000,
}


def main():
    start_time = time.time()  # Clocking start

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # LAPGAN model # D/G Models are same as DCGAN
        model = lapgan.LAPGAN(s, batch_size=train_step['batch_size'])

        # Initializing variables
        s.run(tf.global_variables_initializer())

        # Training, test data set
        dataset = DataSet(input_height=32,
                          input_width=32,
                          input_channel=3,
                          name='cifar-10')
        dataset_iter = DataIterator(dataset.train_images, dataset.train_labels, train_step['batch_size'])

        step = 0
        cont = int(step / 750)
        for epoch in range(cont, cont + train_step['epoch']):
            for batch_images, batch_labels in dataset_iter.iterate():
                batch_images = batch_images.astype(np.float32) / 225.

                z = []
                for i in range(3):
                    z.append(np.random.uniform(-1., 1.,
                                               [train_step['batch_size'], model.z_noises[i]]).astype(np.float32))

                # Update D/G networks
                img_fake, _, _, img_coarse, d_loss_1, g_loss_1, \
                _, _, _, _, _, d_loss_2, g_loss_2, \
                _, _, _, _, d_loss_3, g_loss_3, \
                _, _, _, _, _, _ = s.run([
                    model.g[0], model.d_reals_prob[0], model.d_fakes_prob[0], model.x1_coarse,
                    model.d_loss[0], model.g_loss[0],

                    model.x2_fine, model.g[1], model.d_reals_prob[1], model.d_fakes_prob[1], model.x2_coarse,
                    model.d_loss[1], model.g_loss[1],

                    model.x3_fine, model.g[2], model.d_reals_prob[2], model.d_fakes_prob[2],
                    model.d_loss[2], model.g_loss[2],

                    model.d_op[0], model.g_op[0], model.d_op[1], model.g_op[1], model.d_op[2], model.g_op[2],  # D/G ops
                ],
                    feed_dict={
                        model.x1_fine: batch_images,  # images
                        model.y: batch_labels,        # classes
                        model.z[0]: z[0], model.z[1]: z[1], model.z[2]: z[2]  # z-noises
                    })

                # Logging
                if step % train_step['logging_interval'] == 0:
                    batch_x = batch_images[:model.sample_num]
                    batch_y = batch_labels[:model.sample_num]

                    z = []
                    for i in range(3):
                        z.append(np.random.uniform(-1., 1., [model.sample_num, model.z_noises[i]]).astype(np.float32))

                    # Update D/G networks
                    img_fake, _, _, img_coarse, d_loss_1, g_loss_1, \
                    _, _, _, _, _, d_loss_2, g_loss_2, \
                    _, _, _, _, d_loss_3, g_loss_3, \
                    _, _, _, _, _, _, summary = s.run([
                        model.g[0], model.d_reals_prob[0], model.d_fakes_prob[0], model.x1_coarse,
                        model.d_loss[0], model.g_loss[0],

                        model.x2_fine, model.g[1], model.d_reals_prob[1], model.d_fakes_prob[1], model.x2_coarse,
                        model.d_loss[1], model.g_loss[1],

                        model.x3_fine, model.g[2], model.d_reals_prob[2], model.d_fakes_prob[2],
                        model.d_loss[2], model.g_loss[2],

                        model.d_op[0], model.g_op[0], model.d_op[1], model.g_op[1], model.d_op[2], model.g_op[2],
                        model.merged,
                    ],
                        feed_dict={
                            model.x1_fine: batch_x,  # images
                            model.y: batch_y,        # classes
                            model.z[0]: z[0], model.z[1]: z[1], model.z[2]: z[2]  # z-noises
                        })

                    # Print loss
                    print("[+] Epoch %03d Step %05d => " % (epoch, step),
                          " D loss : {:.8f}".format(d_loss_1.mean()),
                          " G loss : {:.8f}".format(g_loss_1.mean()))

                    # Training G model with sample image and noise
                    samples = img_fake + img_coarse

                    # Summary saver
                    model.writer.add_summary(summary, step)  # time saving

                    # Export image generated by model G
                    sample_image_height = model.sample_size
                    sample_image_width = model.sample_size
                    sample_dir = results['output'] + 'train_{0}_{1}.png'.format(epoch, step)

                    # Generated image save
                    iu.save_images(samples, size=[sample_image_height, sample_image_width], image_path=sample_dir)

                    # Model save
                    model.saver.save(s, results['model'], global_step=step)  # time saving

                step += 1

        end_time = time.time() - start_time  # Clocking end

        # Elapsed time
        print("[+] Elapsed time {:.8f}s".format(end_time))

        # Close tf.Session
        s.close()


if __name__ == '__main__':
    main()
