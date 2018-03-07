from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import sys
import time

import srgan_model as srgan

sys.path.append('../')
import image_utils as iu
from datasets import Div2KDataSet as DataSet


results = {
    'output': './gen_img/',
    'checkpoint': './model/checkpoint',
    'model': './model/SRGAN-model.ckpt'
}

train_step = {
    'epochs': 100,
    'batch_size': 8,
    'logging_interval': 50,
}


def resize(s, x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)  # ndarray to tensor

    x_small = tf.image.resize_images(x, [96, 96],
                                     tf.image.ResizeMethod.BICUBIC)  # LR image
    x_nearest = tf.image.resize_images(x_small, [384, 384],
                                       tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # HR image

    x_small = s.run(x_small)      # tensor to ndarray
    x_nearest = s.run(x_nearest)  # tensor to ndarray

    return x_small, x_nearest


def main():
    start_time = time.time()  # Clocking start

    # Div2K -  Track 1: Bicubic downscaling - x4 DataSet load
    ds = DataSet(mode='r')
    hr_lr_images = ds.images
    hr, lr = hr_lr_images[0],  hr_lr_images[1]

    print("[+] Loaded HR image ", hr.shape)
    print("[+] Loaded LR image ", lr.shape)

    # GPU configure
    gpu_config = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_config)

    with tf.Session(config=config) as s:
        # SRGAN Model
        model = srgan.SRGAN(s, batch_size=train_step['batch_size'])

        # Initializing
        s.run(tf.global_variables_initializer())

        sample_x_hr, sample_x_lr = hr[:model.sample_num], lr[:model.sample_num]
        sample_x_hr, sample_x_lr = \
            np.reshape(sample_x_hr, [model.sample_num] + model.hr_image_shape[1:]),\
            np.reshape(sample_x_lr, [model.sample_num] + model.lr_image_shape[1:])

        sample_x_hr = np.reshape(sample_x_hr, [model.sample_num] + model.hr_image_shape[1:])
        sample_x_lr = np.reshape(sample_x_lr, [model.sample_num] + model.lr_image_shape[1:])

        # Export real image
        valid_image_height = model.sample_size
        valid_image_width = model.sample_size
        sample_hr_dir, sample_lr_dir = results['output'] + 'valid_hr.png', results['output'] + 'valid_lr.png'

        # Generated image save
        iu.save_images(sample_x_hr, size=[valid_image_height, valid_image_width], image_path=sample_hr_dir)
        iu.save_images(sample_x_lr, size=[valid_image_height, valid_image_width], image_path=sample_lr_dir)

        global_step = 0
        for epoch in range(train_step['epochs']):

            pointer = 0
            for i in range(ds.num_images // train_step['batch_size']):
                start = pointer
                pointer += train_step['batch_size']

                if pointer > ds.num_images:  # if 1 epoch is ended
                    # Shuffle training DataSet
                    perm = np.arange(ds.num_images)
                    np.random.shuffle(perm)

                    hr, lr = hr[perm], lr[perm]

                    start = 0
                    pointer = train_step['batch_size']

                end = pointer

                batch_x_hr, batch_x_lr = hr[start:end], lr[start:end]

                # reshape
                batch_x_hr = np.reshape(batch_x_hr, [train_step['batch_size']] + model.hr_image_shape[1:])
                batch_x_lr = np.reshape(batch_x_lr, [train_step['batch_size']] + model.lr_image_shape[1:])

                # Update G/D network
                _, d_loss, _, g_loss = s.run([model.d_op, model.d_loss, model.g_op, model.g_loss],
                                             feed_dict={
                                                 model.x_hr: batch_x_hr,
                                                 model.x_lr: batch_x_lr,
                                             })

                if i % train_step['logging_interval'] == 0:
                    d_loss, g_loss, summary = s.run([model.d_loss, model.g_loss, model.merged],
                                                    feed_dict={
                                                        model.x_hr: batch_x_hr,
                                                        model.x_lr: batch_x_lr,
                                                    })
                    # Print loss
                    print("[+] Step %08d => " % global_step,
                          " D loss : {:.8f}".format(d_loss),
                          " G loss : {:.8f}".format(g_loss))

                    # Training G model with sample image and noise
                    samples = s.run(model.g,
                                    feed_dict={
                                        model.x_lr: sample_x_lr,
                                    })

                    samples = np.reshape(samples, [model.sample_num] + model.hr_image_shape[1:])

                    # Summary saver
                    model.writer.add_summary(summary, global_step)

                    # Export image generated by model G
                    sample_image_height = model.output_height
                    sample_image_width = model.output_width
                    sample_dir = results['output'] + 'train_{:08d}.png'.format(global_step)

                    # Generated image save
                    iu.save_images(samples,
                                   size=[sample_image_height, sample_image_width],
                                   image_path=sample_dir)

                    # Model save
                    model.saver.save(s, results['model'], global_step=global_step)

                global_step += 1

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))

    # Close tf.Session
    s.close()


if __name__ == '__main__':
    main()
