from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize

import sys
import time
import random

import pggan_model as pggan

sys.path.append('../')
import image_utils as iu
from datasets import DataIterator
from datasets import CelebADataSet as DataSet


results = {
    'output': './gen_img/',
    'checkpoint': './model/checkpoint-',
    'model': './model/PGGAN-model-'
}

train_step = {
    'epoch': 10000,
    'batch_size': 16,
    'logging_step': 1000,
}

pg = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
assert len(pg) == 11

r_pg = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
assert len(r_pg) == 11


def image_resize(x, s=128):
    imgs = []
    for i in range(x.shape[0]):
        imgs.append(resize(x[i, :, :, :], output_shape=(s, s), preserve_range=True))
    return np.asarray(imgs)


def main():
    start_time = time.time()  # Clocking start

    # Celeb-A DataSet images
    ds = DataSet(input_height=1024,
                 input_width=1024,
                 input_channel=3,
                 ds_type="CelebA-HQ",
                 ds_path="/home/zero/hdd/DataSet/CelebA-HQ").images
    n_ds = 30000
    dataset_iter = DataIterator(ds, None, train_step['batch_size'],
                                label_off=True)

    rnd = random.randint(0, n_ds)
    sample_x = ds[rnd]
    sample_x = np.reshape(sample_x, [-1, 1024, 1024, 3])

    # Export real image
    valid_image_height = 1
    valid_image_width = 1
    sample_dir = results['output'] + 'valid.png'

    # Generated image save
    iu.save_images(sample_x, size=[valid_image_height, valid_image_width], image_path=sample_dir,
                   inv_type='127')
    print("[+] sample image saved!")

    print("[+] pre-processing took {:.8f}s".format(time.time() - start_time))

    # GPU configure
    gpu_config = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_config)

    for idx, n_pg in enumerate(pg):

        with tf.Session(config=config) as s:
            pg_t = False if idx % 2 == 0 else True

            # PGGAN Model
            model = pggan.PGGAN(s, pg=n_pg, pg_t=pg_t)  # PGGAN

            # Initializing
            s.run(tf.global_variables_initializer())

            if not n_pg == 1 and not n_pg == 7:
                if pg_t:
                    model.r_saver.restore(s, results['model'] + '%d-%d.ckpt' % (idx, r_pg[idx]))
                    model.out_saver.restore(s, results['model'] + '%d-%d.ckpt' % (idx, r_pg[idx]))
                else:
                    model.saver.restore(s, results['model'] + '%d-%d.ckpt' % (idx, r_pg[idx]))

            global_step = 0
            for epoch in range(train_step['epoch']):
                # Later, adding n_critic for optimizing D net
                for batch_images in dataset_iter.iterate():
                    batch_x = np.reshape(batch_images, (-1, 128, 128, 3))
                    batch_x = (batch_x + 1.) * 127.5  # re-scaling to (0, 255)
                    batch_x = image_resize(batch_x, s=model.output_size)
                    batch_x = (batch_x / 127.5) - 1.  # re-scaling to (-1, 1)
                    batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)

                    if pg_t and not pg == 0:
                        alpha = global_step / 32000.
                        low_batch_x = zoom(batch_x, zoom=[1., .5, .5, 1.])
                        low_batch_x = zoom(low_batch_x, zoom=[1., 2., 2., 1.])
                        batch_x = alpha * batch_x + (1. - alpha) * low_batch_x

                    # Update D network
                    _, d_loss = s.run([model.d_op, model.d_loss],
                                      feed_dict={
                                          model.x: batch_x,
                                          model.z: batch_z,
                                      })

                    # Update G network
                    _, g_loss = s.run([model.g_op, model.g_loss],
                                      feed_dict={
                                          model.z: batch_z,
                                      })

                    # Update alpha_trans
                    s.run(model.alpha_trans_update,
                          feed_dict={
                              model.step_pl: global_step
                          })

                    if global_step % train_step['logging_step'] == 0:
                        gp, d_loss, g_loss, summary = s.run([model.gp,
                                                             model.d_loss, model.g_loss, model.merged],
                                                            feed_dict={
                                                                model.x: batch_x,
                                                                model.z: batch_z,
                                                            })

                        # Print loss
                        print("[+] PG %d Epoch %03d Step %07d =>" % (n_pg, epoch, global_step),
                              " D loss : {:.6f}".format(d_loss),
                              " G loss : {:.6f}".format(g_loss),
                              " GP     : {:.6f}".format(gp),
                              )

                        # Summary saver
                        model.writer.add_summary(summary, global_step)

                        # Training G model with sample image and noise
                        sample_z = np.random.uniform(-1., 1., [model.sample_num, model.z_dim]).astype(np.float32)

                        samples = s.run(model.g,
                                        feed_dict={
                                            model.z: sample_z,
                                        })
                        samples = np.clip(samples, -1, 1)

                        # Export image generated by model G
                        sample_image_height = 1
                        sample_image_width = 1
                        sample_dir = results['output'] + 'train_{0}.png'.format(global_step)

                        # Generated image save
                        iu.save_images(samples,
                                       size=[sample_image_height, sample_image_width],
                                       image_path=sample_dir,
                                       inv_type='127')

                        # Model save
                        model.saver.save(s, results['model'] + '%d-%d.ckpt' % (idx, n_pg), global_step=global_step)

                    global_step += 1

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))


if __name__ == '__main__':
    main()
