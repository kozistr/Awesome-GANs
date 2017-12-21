from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pickle as p
from sklearn.model_selection import train_test_split

'''
This dataset is for CIFAR-10, 100

- CIFAR
    Cifar-10, 100 DataSets can be downloaded at https://www.cs.toronto.edu/~kriz/cifar.html
    
    Cifar-10-python link  : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Cifar-100-python link : https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    
    OR you can download with 'wget' cmd
    Example : wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
'''

DataSets = {
    'cifar-10': "/home/zero/hdd/DataSet/Cifar/cifar-10-batches-py",
    'cifar-100': "/home/zero/hdd/DataSet/Cifar/cifar-100-python",
}


def unpickle(file):
    with open(file, 'rb') as f:
        return p.load(f, encoding='bytes')


def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class CiFarDataSet:

    def __init__(self, batch_size=128, epoch=250, input_height=64, input_width=64, input_channel=3,
                 output_height=64, output_width=64, output_channel=3,
                 split_rate=0.2, random_state=42, num_threads=8, name="None"):

        """
        # General Settings
        :param batch_size: training batch size, default 128
        :param epoch: training epoch, default 250
        :param input_height: input image height, default 64
        :param input_width: input image width, default 64
        :param channel: input image channel, default 3 (RGB)
        - in case of CIFAR, image size is 32x32x3(HWC).
        :param n_classes: input datasets' classes
        - in case of CIFAR, 10, 100

        # Output Settings
        :param output_height: output images height, default 28
        :param output_width: output images width, default 28
        :param output_channel: output images channel, default 3

        # Pre-Processing Option
        :param split_rate: image split rate (into train & test), default 0.2
        :param random_state: random seed for shuffling, default 42
        :param num_threads: the number of threads for multi-threading, default 8

        # DataSet Option
        :param name: DataSet name, default None
        """

        self.batch_size = batch_size
        self.epoch = epoch
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

        self.output_height = output_height
        self.output_width = output_width
        self.output_channel = output_channel

        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = num_threads  # change this value to the fitted value for ur system
        self.name = name

        self.path = ""  # DataSet path
        self.n_classes = 10  # DataSet the number of classes, default 10

        self.train_images = ''
        self.valid_images = ''
        self.test_images = ''

        self.train_labels = ''
        self.valid_labels = ''
        self.test_labels = ''

        if self.name == "cifar-10":
            self.cifar_10()   # load cifar-10
        elif self.name == "cifar-100":
            self.cifar_100()  # load cifar-100
        else:
            pass

    def cifar_10(self):
        self.path = DataSets['cifar-10']
        self.n_classes = 10  # labels

        train_batch_1 = unpickle("{0}/data_batch_1".format(self.path))
        train_batch_2 = unpickle("{0}/data_batch_2".format(self.path))
        train_batch_3 = unpickle("{0}/data_batch_3".format(self.path))
        train_batch_4 = unpickle("{0}/data_batch_4".format(self.path))
        train_batch_5 = unpickle("{0}/data_batch_5".format(self.path))

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

        # Image size : 32x32x3
        train_images = np.swapaxes(train_data.reshape([-1,
                                                       self.input_height,
                                                       self.input_width,
                                                       self.input_channel], order='F'), 1, 2)

        # test data & label
        test_batch = unpickle("{0}/test_batch".format(self.path))

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
                             test_size=self.split_rate,
                             random_state=self.random_state)

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images

        self.train_labels = one_hot(train_labels, self.n_classes)
        self.valid_labels = one_hot(valid_labels, self.n_classes)
        self.test_labels = one_hot(test_labels, self.n_classes)

    def cifar_100(self):
        self.path = DataSets['cifar-100']
        self.n_classes = 100  # labels

        # training data & label
        train_batch = unpickle("{0}/train".format(self.path))

        train_data = np.concatenate([train_batch[b'data']], axis=0)
        train_labels = np.concatenate([train_batch[b'fine_labels']], axis=0)
        train_images = np.swapaxes(train_data.reshape([-1,
                                                       self.input_height,
                                                       self.input_width,
                                                       self.input_channel], order='F'), 1, 2)

        # test data & label
        test_batch = p.Unpickler("{0}/test".format(self.path))

        test_data = np.concatenate([test_batch[b'data']], axis=0)
        test_labels = np.concatenate([test_batch[b'fine_labels']], axis=0)
        test_images = np.swapaxes(test_data.reshape([-1,
                                                     self.input_height,
                                                     self.input_width,
                                                     self.input_channel], order='F'), 1, 2)

        # Split training data set into train, valid
        train_images, valid_images, train_labels, valid_labels = \
            train_test_split(train_images, train_labels,
                             test_size=self.split_rate,
                             random_state=self.random_state)

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images

        self.train_labels = one_hot(train_labels, self.n_classes)
        self.valid_labels = one_hot(valid_labels, self.n_classes)
        self.test_labels = one_hot(test_labels, self.n_classes)
