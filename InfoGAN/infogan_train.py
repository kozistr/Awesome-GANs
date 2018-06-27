from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import sys
import time

import infogan_model as infogan

sys.path.append('../')
import image_utils as iu
from datasets import DataIterator
from datasets import CelebADataSet as DataSet


results = {
    'output': './gen_img/',
    'model': './model/InfoGAN-model.ckpt'
}

train_step = {
    'epochs': 5,
    'batch_size': 16,
    'global_step': 200001,
    'logging_interval': 1000,
}


def gen_category(n_size, n_dim):
    return np.random.randn(n_size, n_dim) * .5  # gaussian


def gen_continuous(n_size, n_dim):
    code = np.zeros((n_size, n_dim))
    code[range(n_size), np.random.randint(0, n_dim, n_size)] = 1
    return code


def main():
    start_time = time.time()  # Clocking start

    # loading CelebA DataSet
    ds = DataSet(height=32,
                 width=32,
                 channel=3,
                 ds_image_path="D:\\DataSet/CelebA/CelebA-32.h5",
                 ds_label_path="D:\\DataSet/CelebA/Anno/list_attr_celeba.txt",
                 # ds_image_path="D:\\DataSet/CelebA/Img/img_align_celeba/",
                 ds_type="CelebA",
                 use_save=False,
                 save_file_name="D:\\DataSet/CelebA/CelebA-32.h5",
                 save_type="to_h5",
                 use_img_scale=False,
                 # img_scale="-1,1"
                 )

    # saving sample images
    test_images = np.reshape(iu.transform(ds.images[:16], inv_type='127'), (16, 32, 32, 3))
    iu.save_images(test_images,
                   size=[4, 4],
                   image_path=results['output'] + 'sample.png',
                   inv_type='127')

    ds_iter = DataIterator(x=ds.images,
                           y=ds.labels,
                           batch_size=train_step['batch_size'],
                           label_off=False)

    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # InfoGAN Model
        model = infogan.InfoGAN(s, batch_size=train_step['batch_size'],
                                n_categories=len(ds.labels))

        # Initializing
        s.run(tf.global_variables_initializer())

        # fixed z-noise
        sample_z = np.random.uniform(-1., 1., [model.sample_num, model.z_dim]).astype(np.float32)

        global_step = 0
        for epoch in range(train_step['epochs']):
            for batch_x, _ in ds_iter.iterate():
                batch_x = iu.transform(batch_x, inv_type='127')
                batch_x = np.reshape(batch_x, (model.batch_size, model.height, model.width, model.channel))

                batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)

                batch_z_con = gen_continuous(model.batch_size, model.n_continous_factor)
                batch_z_cat = gen_category(model.batch_size, model.n_categories)
                batch_c = np.concatenate((batch_z_con, batch_z_cat), axis=1)

                # Update D network
                _, d_loss = s.run([model.d_op, model.d_loss],
                                  feed_dict={
                                      model.c: batch_c,
                                      model.x: batch_x,
                                      model.z: batch_z,
                                })

                # Update G network
                _, g_loss = s.run([model.g_op, model.g_loss],
                                  feed_dict={
                                      model.c: batch_c,
                                      model.x: batch_x,
                                      model.z: batch_z,
                                  })

                # Logging
                if global_step % train_step['logging_interval'] == 0:
                    batch_z = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)

                    batch_z_con = gen_continuous(model.batch_size, model.n_continous_factor)
                    batch_z_cat = gen_category(model.batch_size, model.n_categories)
                    batch_c = np.concatenate((batch_z_con, batch_z_cat), axis=1)

                    d_loss, g_loss, summary = s.run([model.d_loss, model.g_loss, model.merged],
                                                    feed_dict={
                                                        model.c: batch_c,
                                                        model.x: batch_x,
                                                        model.z: batch_z,
                                                    })

                    # Print loss
                    print("[+] Step %08d => " % global_step,
                          " D loss : {:.8f}".format(d_loss),
                          " G loss : {:.8f}".format(g_loss))

                    # Training G model with sample image and noise
                    sample_z_con = np.zeros((model.sample_num, model.n_continous_factor))
                    for i in range(model.n_con):
                        sample_z_con[10 * i: 10 * (i + 1), 0] = np.linspace(-2, 2, 10)

                    sample_z_cat = np.zeros((model.sample_num, model.n_categories))
                    for i in range(model.n_cat):
                        sample_z_cat[10 * i: 10 * (i + 1), i] = 1

                    sample_c = np.concatenate((sample_z_con, sample_z_cat), axis=1)

                    samples = s.run(model.g_test,
                                    feed_dict={
                                        model.c: sample_c,
                                        model.z: sample_z,
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

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))

    # Close tf.Session
    s.close()


if __name__ == '__main__':
    main()
