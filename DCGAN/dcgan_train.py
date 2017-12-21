from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import sys
import time

import dcgan_model as dcgan
from dataset import DataIterator
from dataset import CiFarDataSet as DataSet

sys.path.append('../')
import image_utils as iu


results = {
    'output': './gen_img/',
    'checkpoint': './model/checkpoint',
    'model': './model/DCGAN-model.ckpt'
}

train_step = {
    'epoch': 150,
    'batch_size': 64,
    'logging_interval': 1000,
    'update_overpowered': 100,
}


def main():
    start_time = time.time()  # Clocking start

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # DCGAN model
        model = dcgan.DCGAN(s, batch_size=train_step['batch_size'])

        # Load model & Graph & Weights
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            model.saver.restore(s, ckpt.model_checkpoint_path)

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("[+] global step : %s" % global_step, " successfully loaded")
        else:
            global_step = 0
            print('[-] No checkpoint file found')

        # Initializing variables
        s.run(tf.global_variables_initializer())

        # Training, test data set
        dataset = DataSet(input_height=32, input_width=32, input_channel=3, name='cifar-100')
        dataset_iter = DataIterator(dataset.train_images, dataset.train_labels, train_step['batch_size'])

        sample_images = dataset.valid_images[:model.sample_num].astype(np.float32) / 255.0
        sample_z = np.random.uniform(-1., 1.,  # range -1 ~ 1
                                     size=(model.sample_num, model.z_dim))

        # Export real image
        valid_image_height = model.sample_size
        valid_image_width = model.sample_size
        sample_dir = results['output'] + 'valid.png'

        # Generated image save
        iu.save_images(sample_images, size=[valid_image_height, valid_image_width], image_path=sample_dir)

        d_overpowered = False  # G loss > D loss * 2

        step = int(global_step)
        cont = int(step / 750)
        for epoch in range(cont, cont + train_step['epoch']):
            for batch_images, _ in dataset_iter.iterate():
                batch_images = batch_images.astype(np.float32) / 255.0
                batch_z = np.random.uniform(-1., 1.,  # range -1 ~ 1
                                            [train_step['batch_size'], model.z_dim]).astype(np.float32)

                # Update D network
                if not d_overpowered:
                    _, d_loss = s.run([model.d_op, model.d_loss],
                                      feed_dict={
                                          model.x: batch_images,
                                          model.z: batch_z
                                      })

                # Update G network
                _, g_loss = s.run([model.g_op, model.g_loss],
                                  feed_dict={
                                      model.z: batch_z
                                  })
                # Logging
                if step % train_step['update_overpowered'] == 0:
                    d_overpowered = d_loss < (g_loss / 2)

                    if step % train_step['logging_interval'] == 0:
                        batch_images = dataset.valid_images[:train_step['batch_size']].astype(np.float32) / 255.0
                        batch_z = np.random.uniform(-1., 1.,  # range -1 ~ 1
                                                    [train_step['batch_size'], model.z_dim]).astype(np.float32)

                        d_loss, g_loss, summary = s.run([model.d_loss, model.g_loss, model.merged],
                                                        feed_dict={
                                                            model.x: batch_images,
                                                            model.z: batch_z
                                                        })

                        # Print loss
                        print("[+] Epoch %03d Step %05d => " % (epoch, step),
                              "D loss : {:.8f}".format(d_loss), " G loss : {:.8f}".format(g_loss),
                              "Overpowered :", d_overpowered)

                        # Update overpowered
                        d_overpowered = d_loss < (g_loss / 2)

                        # Training G model with sample image and noise
                        samples = s.run(model.g,
                                        feed_dict={
                                            model.x: sample_images,
                                            model.z: sample_z
                                       })

                        # Summary saver
                        model.writer.add_summary(summary, step)

                        # Export image generated by model G
                        sample_image_height = model.sample_size
                        sample_image_width = model.sample_size
                        sample_dir = results['output'] + 'train_{0}_{1}.png'.format(epoch, step)

                        # Generated image save
                        iu.save_images(samples, size=[sample_image_height, sample_image_width], image_path=sample_dir)

                        # Model save
                        model.saver.save(s, results['model'], global_step=step)

                step += 1

        end_time = time.time() - start_time  # Clocking end

        # Elapsed time
        print("[+] Elapsed time {:.8f}s".format(end_time))  # took about 100s on my machine

        # Close tf.Session
        s.close()


if __name__ == '__main__':
    main()
