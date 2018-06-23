from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import sys
import time

import stargan_model as stargan
from dataset import DataIterator
from dataset import CelebADataSet as DataSet

sys.path.append('../')
import image_utils as iu


results = {
    'output': './gen_img/',
    'model': './model/StarGAN-model.ckpt'
}

train_step = {
    'epoch': 100,
    'batch_size': 32,
    'logging_step': 500,
}


def main():
    start_time = time.time()  # Clocking start

    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # pre-chosen
        attr_labels = [
            'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Gray_Hair'
        ]

        # StarGAN Model
        model = stargan.StarGAN(s, attr_labels=attr_labels)  # StarGAN

        # Initializing
        s.run(tf.global_variables_initializer())

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

        # x_A # Celeb-A
        img_a = np.reshape(ds.images, [-1, 64, 64, 3])
        attr_a = ds.labels

        # x_B # Celeb-A # copied from x_A
        # later it'll be replaced to another DataSet like RaFD, used in the paper
        # but i can't find proper(good) DataSets, so i just do with single-domain (Celeb-A)
        # img_b = img_a[:]
        # attr_b = attr_a[:]

        # ds_a_iter = DataIterator(img_a, attr_a, train_step['batch_size'])
        # ds_b_iter = DataIterator(img_b, attr_b, train_step['batch_size'])

        print("[+] pre-processing elapsed time : {:.8f}s".format(time.time() - start_time))
        print("[*] image_A     :", img_a.shape, " attribute A :", attr_a.shape)

        global_step = 0
        for epoch in range(train_step['epoch']):
            # learning rate decay
            lr_decay = 1.
            if epoch >= train_step['epoch']:
                lr_decay = (train_step['epoch'] - epoch) / (train_step['epoch'] / 2.)

            # re-implement DataIterator for multi-input
            pointer = 0
            for i in range(ds.num_images // train_step['batch_size']):
                start = pointer
                pointer += train_step['batch_size']

                if pointer > ds.num_images:  # if ended 1 epoch
                    # Shuffle training DataSet
                    perm = np.arange(ds.num_images)
                    np.random.shuffle(perm)

                    # To-Do
                    # Getting Proper DataSet
                    img_a, img_b = img_a[perm], img_a[perm]
                    attr_a, attr_b = attr_a[perm], attr_a[perm]

                    start = 0
                    pointer = train_step['batch_size']

                end = pointer

                x_a, y_a = img_a[start:end], attr_a[start:end][:]
                x_b, y_b = img_a[start:end], attr_a[start:end][:]

                x_a = iu.transform(x_a, inv_type='127')
                x_b = iu.transform(x_b, inv_type='127')

                batch_a = ds.concat_data(x_a, y_a)
                batch_b = ds.concat_data(x_b, y_b)
                eps = np.random.rand(train_step['batch_size'], 1, 1, 1)

                # Generate fake_B
                fake_b = s.run(model.fake_B, feed_dict={model.x_A: batch_a})

                # Update D network - 5 times
                for _ in range(5):
                    _, d_loss = s.run([model.d_op, model.d_loss],
                                      feed_dict={
                                          model.x_B: batch_b,
                                          model.y_B: y_b,
                                          model.fake_x_B: fake_b,
                                          model.lr_decay: lr_decay,
                                          model.epsilon: eps,
                                    })

                # Update G network - 1 time
                _, g_loss = s.run([model.g_op, model.g_loss],
                                  feed_dict={
                                      model.x_A: batch_a,
                                      model.x_B: batch_b,
                                      model.y_B: y_b,
                                      model.lr_decay: lr_decay,
                                      model.epsilon: eps,
                                  })

                if global_step % train_step['logging_step'] == 0:
                    eps = np.random.rand(train_step['batch_size'], 1, 1, 1)

                    # Summary
                    samples, d_loss, g_loss, summary = s.run([model.fake_A, model.d_loss, model.g_loss, model.merged],
                                                             feed_dict={
                                                                 model.x_A: batch_a,
                                                                 model.x_B: batch_b,
                                                                 model.y_B: y_b,
                                                                 model.fake_x_B: fake_b,
                                                                 model.lr_decay: lr_decay,
                                                                 model.epsilon: eps,
                                                             })

                    # Print loss
                    print("[+] Epoch %04d Step %07d =>" % (epoch, global_step),
                          " D loss : {:.8f}".format(d_loss),
                          " G loss : {:.8f}".format(g_loss))

                    # Summary saver
                    model.writer.add_summary(summary, epoch)

                    # Export image generated by model G
                    sample_image_height = model.sample_size
                    sample_image_width = model.sample_size
                    sample_dir = results['output'] + 'train_{0}_{1}.png'.format(epoch, global_step)

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
