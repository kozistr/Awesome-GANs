from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import sys
import time

import magan_model as magan

sys.path.append('../')
import image_utils as iu
from datasets import DataIterator
from datasets import CelebADataSet as DataSet


results = {
    'output': './gen_img/',
    'model': './model/MAGAN-model.ckpt'
}

train_step = {
    'epochs': 50,
    'batch_size': 64,
    'global_step': 200001,
    'logging_interval': 1000,
}


def main():
    start_time = time.time()  # Clocking start

    # loading CelebA DataSet
    ds = DataSet(height=64,
                 width=64,
                 channel=3,
                 ds_image_path="D:/DataSet/CelebA/CelebA-64.h5",
                 ds_label_path="D:/DataSet/CelebA/Anno/list_attr_celeba.txt",
                 # ds_image_path="D:/DataSet/CelebA/Img/img_align_celeba/",
                 ds_type="CelebA",
                 use_save=False,
                 save_file_name="D:/DataSet/CelebA/CelebA-64.h5",
                 save_type="to_h5",
                 use_img_scale=False,
                 img_scale="-1,1")

    # saving sample images
    test_images = np.reshape(iu.transform(ds.images[:100], inv_type='127'), (100, 64, 64, 3))
    iu.save_images(test_images,
                   size=[10, 10],
                   image_path=results['output'] + 'sample.png',
                   inv_type='127')

    ds_iter = DataIterator(x=ds.images,
                           y=None,
                           batch_size=train_step['batch_size'],
                           label_off=True)

    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # MAGAN Model
        model = magan.MAGAN(s)

        # Initializing
        s.run(tf.global_variables_initializer())

        # Load model & Graph & Weights
        saved_global_step = 0
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(s, ckpt.model_checkpoint_path)

            saved_global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("[+] global step : %s" % saved_global_step, " successfully loaded")
        else:
            print('[-] No checkpoint file found')

        n_steps = ds.num_images // model.batch_size  # training set size

        # Pre-Train
        print("[*] pre-training - getting proper Margin")

        margin = 3.0585415484215974
        if margin == 0:
            sum_d_loss = 0.
            for i in range(2):
                for batch_x in ds_iter.iterate():
                    batch_x = np.reshape(iu.transform(batch_x, inv_type='127'),
                                         (model.batch_size, model.height, model.width, model.channel))
                    batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)

                    _, d_real_loss = s.run([model.d_op, model.d_real_loss],
                                           feed_dict={
                                               model.x: batch_x,
                                               model.z: batch_z,
                                               model.m: 0.,
                                           })
                    sum_d_loss += d_real_loss

                print("[*] Epoch {:1d} Sum of d_real_loss : {:.8f}".format(i + 1, sum_d_loss))

            # Initial margin value
            margin = (sum_d_loss / n_steps)

        print("[+] Margin : {0}".format(margin))

        old_margin = 0.
        s_g_0 = np.inf  # Sg_0 = infinite

        global_step = saved_global_step
        start_epoch = global_step // (ds.num_images // model.batch_size)           # recover n_epoch
        ds_iter.pointer = saved_global_step % (ds.num_images // model.batch_size)  # recover n_iter
        for epoch in range(start_epoch, train_step['epochs']):
            s_d, s_g = 0., 0.
            for batch_x in ds_iter.iterate():
                batch_x = iu.transform(batch_x, inv_type='127')
                batch_x = np.reshape(batch_x, (model.batch_size, model.height, model.width, model.channel))
                batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)

                # Update D network
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
                                                   model.x: batch_x,
                                                   model.z: batch_z,
                                                   model.m: margin,
                                               })

                # Update G fake sample
                s_g += np.sum(d_fake_loss)

                # Logging
                if global_step % train_step['logging_interval'] == 0:
                    summary = s.run(model.merged,
                                    feed_dict={
                                        model.x: batch_x,
                                        model.z: batch_z,
                                        model.m: margin,
                                    })

                    # Print loss
                    print("[+] Epoch %03d Global Step %05d => " % (epoch, global_step),
                          " D loss : {:.8f}".format(d_loss),
                          " G loss : {:.8f}".format(g_loss))

                    # Training G model with sample image and noise
                    sample_z = np.random.uniform(-1., 1., [model.sample_num, model.z_dim]).astype(np.float32)
                    samples = s.run(model.g,
                                    feed_dict={
                                        model.z: sample_z,
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
                                   image_path=sample_dir,
                                   inv_type='127')

                    # Model save
                    model.saver.save(s, results['model'], global_step)

                global_step += 1

            # Update margin
            if s_d / n_steps < margin and s_d < s_g and s_g_0 <= s_g:
                margin = s_d / n_steps
                print("[*] Margin updated from {:8f} to {:8f}".format(old_margin, margin))
                old_margin = margin

            s_g_0 = s_g

            # Convergence Measure
            e_d = s_d / n_steps
            e_g = s_g / n_steps
            l_ = e_d + np.abs(e_d - e_g)
            s.run(l_)

            print("[+] Epoch %03d " % epoch, " L : {:.8f}".format(l_))

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))

    # Close tf.Session
    s.close()


if __name__ == '__main__':
    main()
