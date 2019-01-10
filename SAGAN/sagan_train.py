from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import os
import sys
import time

import sagan_model as sagan

sys.path.append('../')
import tfutil as t
import image_utils as iu

from config import get_config
from datasets import DataIterator
from datasets import CelebADataSet as DataSet


cfg, _ = get_config()


train_step = {
    'epochs': 11,
    'batch_size': 64,
    'global_step': 10001,
    'logging_interval': 500,
}


def main():
    start_time = time.time()  # Clocking start

    height, width, channel = 128, 128, 3

    # loading CelebA DataSet # from 'raw images' or 'h5'
    use_h5 = False
    if not use_h5:
        ds = DataSet(height=height,
                     width=height,
                     channel=channel,
                     # ds_image_path="D:\\DataSet/CelebA/CelebA-%d.h5" % height,
                     ds_label_path=os.path.join(cfg.celeba, "Anno/list_attr_celeba.txt"),
                     ds_image_path=os.path.join(cfg.celeba, "Img/img_align_celeba/"),
                     ds_type="CelebA",
                     use_save=True,
                     save_file_name=os.path.join(cfg.celeba, "CelebA-%d.h5" % height),
                     save_type="to_h5",
                     use_img_scale=False,
                     )
    else:
        ds = DataSet(height=height,
                     width=height,
                     channel=channel,
                     ds_image_path=os.path.join(cfg.celeba, "CelebA-%d.h5" % height),
                     ds_label_path=os.path.join(cfg.celeba, "Anno/list_attr_celeba.txt"),
                     # ds_image_path=os.path.join(cfg.celeba, "Img/img_align_celeba/"),
                     ds_type="CelebA",
                     use_save=False,
                     # save_file_name=os.path.join(cfg.celeba, "CelebA-%d.h5" % height),
                     # save_type="to_h5",
                     use_img_scale=False,
                     )

    # saving sample images
    test_images = np.reshape(iu.transform(ds.images[:16], inv_type='127'), (16, height, width, channel))
    iu.save_images(test_images,
                   size=[4, 4],
                   image_path=os.path.join(cfg.output, "sample.png"),
                   inv_type='127')

    ds_iter = DataIterator(x=ds.images,
                           y=None,
                           batch_size=train_step['batch_size'],
                           label_off=True)

    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # SAGAN Model
        model = sagan.SAGAN(s,
                            height=height, width=width, channel=channel,
                            batch_size=train_step['batch_size'],
                            use_gp=False,
                            use_hinge_loss=True
                            )

        # Initializing
        s.run(tf.global_variables_initializer())

        print("[*] Reading checkpoints...")

        saved_global_step = 0
        ckpt = tf.train.get_checkpoint_state(cfg.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            model.saver.restore(s, ckpt.model_checkpoint_path)

            saved_global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("[+] global step : %d" % saved_global_step, " successfully loaded")
        else:
            print('[-] No checkpoint file found')

        global_step = saved_global_step
        start_epoch = global_step // (ds.num_images // model.batch_size)           # recover n_epoch
        ds_iter.pointer = saved_global_step % (ds.num_images // model.batch_size)  # recover n_iter
        for epoch in range(start_epoch, train_step['epochs']):
            for batch_x in ds_iter.iterate():
                batch_x = iu.transform(batch_x, inv_type='127')
                batch_x = np.reshape(batch_x, (model.batch_size, model.height, model.width, model.channel))
                batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)

                # Update D network
                _, d_loss = s.run([model.d_op, model.d_loss],
                                  feed_dict={
                                      model.x: batch_x,
                                      model.z: batch_z,
                                  })

                # Update G network
                _, g_loss = s.run([model.g_op, model.g_loss],
                                  feed_dict={
                                      model.x: batch_x,
                                      model.z: batch_z,
                                  })

                if global_step % train_step['logging_interval'] == 0:
                    summary = s.run(model.merged,
                                    feed_dict={
                                        model.x: batch_x,
                                        model.z: batch_z,
                                    })

                    # Training G model with sample image and noise
                    sample_z = np.random.uniform(-1., 1., [model.sample_num, model.z_dim]).astype(np.float32)
                    samples = s.run(model.g_test,
                                    feed_dict={
                                        model.z_test: sample_z,
                                    })

                    inception_score = t.inception_score(iu.inverse_transform(samples, inv_type='127'))
                    fid_score = t.fid_score(real_img=batch_x[:model.sample_num],
                                            fake_img=samples)

                    # Print loss
                    print("[+] Epoch %04d Step %08d => " % (epoch, global_step),
                          " D loss : {:.8f}".format(d_loss),
                          " G loss : {:.8f}".format(g_loss),
                          " Inception Score : {:.2f}".format(inception_score),
                          " FID Score : {:.2f}".format(fid_score))

                    # Summary saver
                    model.writer.add_summary(summary, global_step)

                    # Export image generated by model G
                    sample_image_height = model.sample_size
                    sample_image_width = model.sample_size
                    sample_dir = os.path.join(cfg.output, 'train_{:08d}.png'.format(global_step))

                    # Generated image save
                    iu.save_images(samples,
                                   size=[sample_image_height, sample_image_width],
                                   image_path=sample_dir,
                                   inv_type='127')

                    # Model save
                    model.saver.save(s, os.path.join(cfg.model_path, "SAGAN.ckpt"), global_step)

                global_step += 1

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))

    # Close tf.Session
    s.close()


if __name__ == '__main__':
    main()
