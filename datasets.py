from __future__ import division

import os
import h5py
import pickle
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from scipy.misc import imread, imresize
from sklearn.cross_validation import train_test_split
from tensorflow.examples.tutorials.mnist import input_data


# saved dataSets' paths
dirs = {
    'cifar-10': 'D:\\DataSet\\Cifar\\cifar-10-batches-py\\',
    'cifar-100': 'D:\\DataSet\\Cifar\\cifar-100-python\\',
    'celeb-a': 'D:\\DataSet\\Celeb-A\\img_align_celeba\\',
    'celeb-a-h5': 'D:\\DataSet\\Celeb-A\\Celeb-A.h5',
    'pix2pix_shoes': 'D:\\DataSet\\pix2pix\\edges2shoes\\',
    'pix2pix_bags': 'D:\\DataSet\\pix2pix\\edges2handbags\\',
    'pix2pix_monet': 'D:\\DataSet\\pix2pix\\monet2photo\\',
    'pix2pix_vangogh-A': 'D:\\DataSet\\pix2pix\\vangogh2photo\\trainA\\*.jpg',
    'pix2pix_vangogh-B': 'D:\\DataSet\\pix2pix\\vangogh2photo\\trainB\\*.jpg',
    # 'cifar-10': '/home/zero/cifar/cifar-10-batches-py/',
    # 'cifar-100': '/home/zero/cifar/cifar-100-python/',
    # 'celeb-a': '/home/zero/celeba/img_align_celeba/',
    # 'celeb-a-h5': '/home/zero/celeba/celeba.h5',
    # 'pix2pix_shoes': '/home/zero/pix2pix/edges2handbags/train/*.jpg',
    # 'pix2pix_bags': '/home/zero/pix2pix/edges2shoes/train/*.jpg'
}


def get_image(path, w, h):
    img = imread(path).astype(np.float)

    orig_h, orig_w = img.shape[:2]
    new_h = int(orig_h * w / orig_w)

    img = imresize(img, (new_h, w))
    margin = int(round((new_h - h) / 2))

    return img[margin:margin + h]


def unpickle(file):
    # python 3 version
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    return dict


def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class DataIterator:

    def __init__(self, x, y, batch_size, label_off=False):
        self.x = x
        self.label_off = label_off
        if not label_off:
            self.y = y
        self.batch_size = batch_size
        self.num_examples = num_examples = x.shape[0]
        self.num_batches = num_examples // batch_size
        self.pointer = 0

        assert self.batch_size <= self.num_examples

    def next_batch(self):
        start = self.pointer
        self.pointer += self.batch_size

        if self.pointer > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)

            self.x = self.x[perm]
            if not self.label_off:
                self.y = self.y[perm]

            start = 0
            self.pointer = self.batch_size

        end = self.pointer

        if not self.label_off:
            return self.x[start:end], self.y[start:end]
        else:
            return self.x[start:end]

    def iterate(self):
        for step in range(self.num_batches):
            yield self.next_batch()


class DataSet:
    '''
    Supporting DataSets
        - MNIST
        - Cifar-10, Cifar-100
        - Celeb-A
        - Pix2Pix-shoes, Pix2Pix-bags

    Download
        - MNIST
            There are internal(?) codes which download MNIST DataSets automatically

            By following codes,
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

        - CIFAR
            Cifar-10, 100 DataSets can be downloaded at https://www.cs.toronto.edu/~kriz/cifar.html

            Cifar-10-python link  : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
            Cifar-100-python link : https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

            OR you can download with 'wget' cmd
            Example : wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

        - Celeb-A
            Celeb-A DataSets can be downloaded at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

            Celeb-A link : https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg

            OR you can download following python code (but it doesn't work as well when i'm trying)
            code link : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py

        - Pix2Pix
            Pix2Pix DataSets can be downloaded at https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

            Choose any of them!
            DataSets used in this repo are 'edges2shoes' and 'edges2handbags'

            OR you can download with 'wget' cmd
            Example : wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz

        - Caltech
            Caltech-* DataSets can be downloaded at http://www.vision.caltech.edu/archive.html

            - Birds
                Caltech-CUB-200-2011 link : http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
                Caltech-CUB-200-2010 link : http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2010.tgz

            Choose any of them!
            DataSets used in this repo are 'Caltech-CUB-200-2011'

            OR you can download with 'wget' cmd
            Example : wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

    Unpack
        With following cmds...

        - .tar.gz
            tar zxvf [file-path]
        - .zip
            unzip -r [file-path]
        - .7z
            7z x [file-path]
    '''

    def __init__(self, batch_size=256, input_height=64, input_width=64, input_channel=3,
                 output_height=64, output_width=64, output_channel=3,
                 split_rate=0.2, random_state=42, num_threads=16, dataset_name="None"):
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

        self.output_height = output_height
        self.output_width = output_width
        self.output_channel = output_channel

        self.batch_size = batch_size
        self.split_rate = split_rate
        self.random_state = random_state
        self.dataset_name = dataset_name

        self.num_threads = num_threads  # change this value to the fitted value for ur system

        # image pre-processing
        if self.dataset_name in 'mnist':
            self.mnist()  # for CGAN, GAN, ...

        elif self.dataset_name in 'cifar-10':
            self.cifar_10()  # for DCGAN

        elif self.dataset_name in 'cifar-100':
            self.cifar_100()  # for DCGAN

        elif self.dataset_name in 'celeba-a':
            self.celeba()  # for BEGAN

        elif self.dataset_name in 'pix2pix_shoes_bags':
            self.pix2pix_shoes_bags()  # shoes & bags for DiscoGAN

        elif self.dataset_name in 'pix2pix_vangogh':
            self.pix2pix_vangogh()  # vangogh photos for DiscoGAN

        else:  # DataSets will be added more soon!
            pass

    def mnist(self):
        self.mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)  # download MNIST

    def cifar_10(self):
        dataset_path = dirs['cifar-10']  # DataSet Path
        self.n_classes = 10  # labels

        train_batch_1 = unpickle("{0}data_batch_1".format(dataset_path))
        train_batch_2 = unpickle("{0}data_batch_2".format(dataset_path))
        train_batch_3 = unpickle("{0}data_batch_3".format(dataset_path))
        train_batch_4 = unpickle("{0}data_batch_4".format(dataset_path))
        train_batch_5 = unpickle("{0}data_batch_5".format(dataset_path))

        # training data & label
        train_data = np.concatenate([
            train_batch_1[b'data'],
            train_batch_2[b'data'],
            train_batch_3[b'data'],
            train_batch_4[b'data'],
            train_batch_5[b'data']
        ], axis=0)

        train_labels = np.concatenate([
            train_batch_1[b'labels'],
            train_batch_2[b'labels'],
            train_batch_3[b'labels'],
            train_batch_4[b'labels'],
            train_batch_5[b'labels']
        ], axis=0)

        # image size : 32x32x3
        train_images = np.swapaxes(train_data.reshape([-1,
                                                       self.input_height,
                                                       self.input_width,
                                                       self.input_channel], order='F'), 1, 2)

        test_batch = unpickle("{0}test_batch".format(dataset_path))

        # test data & label
        test_data = test_batch[b'data']
        test_labels = np.array(test_batch[b'labels'])

        # image size : 32x32x3
        test_images = np.swapaxes(test_data.reshape([-1,
                                                     self.input_height,
                                                     self.input_width,
                                                     self.input_channel], order='F'), 1, 2)
        # split training data set into train, valid
        train_images, valid_images, train_labels, valid_labels = \
            train_test_split(train_images, train_labels,
                             test_size=self.split_rate, random_state=self.random_state)

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images

        self.train_labels = one_hot(train_labels, self.n_classes)
        self.valid_labels = one_hot(valid_labels, self.n_classes)
        self.test_labels = one_hot(test_labels, self.n_classes)

    def cifar_100(self):
        dataset_path = dirs['cifar-100']  # DataSet Path
        self.n_classes = 100  # labels

        train_batch = unpickle("{0}train".format(dataset_path))

        # training data & label
        train_data = np.concatenate([
            train_batch[b'data']
        ], axis=0)

        train_labels = np.concatenate([
            train_batch[b'fine_labels']
        ], axis=0)

        train_images = np.swapaxes(train_data.reshape([-1,
                                                       self.input_height,
                                                       self.input_width,
                                                       self.input_channel], order='F'), 1, 2)

        # test data & label
        test_batch = unpickle("{0}test".format(dataset_path))

        test_data = np.concatenate([
            test_batch[b'data']
        ], axis=0)

        test_labels = np.concatenate([
            test_batch[b'fine_labels']
        ], axis=0)

        test_images = np.swapaxes(test_data.reshape([-1,
                                                     self.input_height,
                                                     self.input_width,
                                                     self.input_channel], order='F'), 1, 2)

        # split training data set into train, valid
        train_images, valid_images, train_labels, valid_labels = \
            train_test_split(train_images, train_labels,
                             test_size=self.split_rate, random_state=self.random_state)

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images

        self.train_labels = one_hot(train_labels, self.n_classes)
        self.valid_labels = one_hot(valid_labels, self.n_classes)
        self.test_labels = one_hot(test_labels, self.n_classes)

    def celeba(self):
        self.filenames = glob(os.path.join("img_align_celeba", "*.jpg"))
        self.filenames = np.sort(self.filenames)

        self.data = np.zeros((len(self.filenames),
                              self.input_height * self.input_width * self.input_channel), dtype=np.uint8)

        for n, fname in tqdm(enumerate(self.filenames)):
            image = get_image(fname, self.input_width, self.input_height)
            self.data[n] = image.flatten()

        # write .h5 file for reusing later...
        with h5py.File(''.join([dirs['celeb-a-h5']]), 'w') as f:
            f.create_dataset("images", data=self.data)

        self.num_image = 202599
        self.images = self.load_data(size=self.num_image)

    def pix2pix_shoes_bags(self):
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
        img_shoes = tf.cast(tf.reshape(img_shoes, shape=[self.input_height,
                                                         self.input_width,
                                                         self.input_channel]), dtype=tf.float32) / 255.
        img_bags = tf.cast(tf.reshape(img_bags, shape=[self.input_height,
                                                       self.input_width,
                                                       self.input_channel]), dtype=tf.float32) / 255.

        self.batch_shoes = tf.train.shuffle_batch([img_shoes],
                                                  batch_size=self.batch_size,
                                                  num_threads=self.num_threads,
                                                  capacity=1024, min_after_dequeue=256)

        self.batch_bags = tf.train.shuffle_batch([img_bags],
                                                 batch_size=self.batch_size,
                                                 num_threads=self.num_threads,
                                                 capacity=1024, min_after_dequeue=256)

    def pix2pix_vangogh(self):
        queue_A = tf.train.string_input_producer(tf.train.match_filenames_once(dirs['pix2pix_vangogh-A']),
                                                 capacity=200)
        queue_B = tf.train.string_input_producer(tf.train.match_filenames_once(dirs['pix2pix_vangogh-B']),
                                                 capacity=200)
        image_reader = tf.WholeFileReader()

        _, img_A = image_reader.read(queue_A)
        _, img_B = image_reader.read(queue_B)

        # decoding jpg images
        img_A, img_B = tf.image.decode_jpeg(img_A), tf.image.decode_jpeg(img_B)

        # image size : 64x64x3
        img_A = tf.cast(tf.reshape(img_A, shape=[self.input_height,
                                                 self.input_width,
                                                 self.input_channel]), dtype=tf.float32) / 255.
        img_B = tf.cast(tf.reshape(img_B, shape=[self.input_height,
                                                 self.input_width,
                                                 self.input_channel]), dtype=tf.float32) / 255.

        self.batch_A = tf.train.shuffle_batch([img_A],
                                              batch_size=self.batch_size,
                                              num_threads=self.num_threads,
                                              capacity=1024, min_after_dequeue=256)

        self.batch_B = tf.train.shuffle_batch([img_B],
                                              batch_size=self.batch_size,
                                              num_threads=self.num_threads,
                                              capacity=1024, min_after_dequeue=256)

    def load_data(self, size, offset=0):
        '''
            From great jupyter notebook by Tim Sainburg:
            http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
        '''
        with h5py.File(dirs['celeb-a-h5'], 'r') as hf:
            faces = hf['images']

            full_size = len(faces)
            if size is None:
                size = full_size

            n_chunks = int(np.ceil(full_size / size))
            if offset >= n_chunks:
                print("[*] Looping back to start.")
                offset = offset % n_chunks

            if offset == n_chunks - 1:
                print("[-] Not enough data available, clipping to end.")
                faces = faces[offset * size:]

            else:
                faces = faces[offset * size:(offset + 1) * size]

            faces = np.array(faces, dtype=np.float16)

        return faces / 255
