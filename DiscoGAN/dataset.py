from __future__ import division

import tensorflow as tf

dirs = {
    'pix2pix_shoes': '/home/zero/pix2pix/shoes/*.jpg',
    'pix2pix_bags': '/home/zero/pix2pix/bags/*.jpg'
}


class Dataset:

    def __init__(self, batch_size=256, input_height=64, input_width=64, input_channel=3,
                 num_threads=8):
        shoes_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(dirs['pix2pix_shoes']),
                                                              capacity=200)
        bags_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(dirs['pix2pix_bags']),
                                                             capacity=200)
        image_reader = tf.WholeFileReader()

        _, img_shoes = image_reader.read(shoes_filename_queue)
        _, img_bags = image_reader.read(bags_filename_queue)

        # decoding jpg images
        img_shoes, img_bags = tf.image.decode_jpeg(img_shoes), tf.image.decode_jpeg(img_bags)

        # image size : 64x64x3
        img_shoes = tf.cast(tf.reshape(img_shoes, shape=[input_height, input_width, input_channel]),
                            dtype=tf.float32) / 255.
        img_bags = tf.cast(tf.reshape(img_bags, shape=[input_height, input_width, input_channel]),
                           dtype=tf.float32) / 255.

        self.batch_shoes = tf.train.shuffle_batch([img_shoes],
                                                  batch_size=batch_size,
                                                  num_threads=num_threads,
                                                  capacity=1024, min_after_dequeue=256)

        self.batch_bags = tf.train.shuffle_batch([img_bags],
                                                 batch_size=batch_size,
                                                 num_threads=num_threads,
                                                 capacity=1024, min_after_dequeue=256)
