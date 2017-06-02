from __future__ import division

from sklearn.cross_validation import train_test_split
import numpy as np
import pickle


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

    def __init__(self, x, y, batch_size):
        self.x = x
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
            self.y = self.y[perm]

            start = 0
            self.pointer = self.batch_size

        end = self.pointer

        return self.x[start:end], self.y[start:end]

    def iterate(self):
        for step in range(self.num_batches):
            yield self.next_batch()


class Dataset:
    '''
    Download
        Cifar-10, 100 Data set download at https://www.cs.toronto.edu/~kriz/cifar.html
        
        cifar-10-python link  : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        cifar-100-pytohn link : https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
        
        or with wget cmd
        example : wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Unpack
        with following cmd...
        tar zxvf [file-path]
    '''

    def __init__(self, dataset_path, split_rate=0.1, random_state=42, name='cifar-10'):
        self.n_classes = 10

        if name in 'cifar-10':  # cifar-10 images
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

            train_images = np.swapaxes(train_data.reshape([-1, 32, 32, 3], order='F'), 1, 2)

            test_batch = unpickle("{0}test_batch".format(dataset_path))

            # test data & label
            test_data = test_batch[b'data']
            test_labels = np.array(test_batch[b'labels'])

            test_images = np.swapaxes(test_data.reshape([-1, 32, 32, 3], order='F'), 1, 2)

            # split training data set into train, valid
            train_images, valid_images, train_labels, valid_labels = \
                train_test_split(train_images, train_labels,
                                 test_size=split_rate, random_state=random_state)

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images

            self.train_labels = one_hot(train_labels, self.n_classes)
            self.valid_labels = one_hot(valid_labels, self.n_classes)
            self.test_labels = one_hot(test_labels, self.n_classes)

        elif name in 'cifar-100':  # cifar-100 images
            self.n_classes = 100

            train_batch = unpickle("{0}train".format(dataset_path))

            # training data & label
            train_data = np.concatenate([
                train_batch[b'data']
            ], axis=0)

            train_labels = np.concatenate([
                train_batch[b'fine_labels']
            ], axis=0)

            train_images = np.swapaxes(train_data.reshape([-1, 32, 32, 3], order='F'), 1, 2)

            # test data & label
            test_batch = unpickle("{0}test".format(dataset_path))

            test_data = np.concatenate([
                test_batch[b'data']
            ], axis=0)

            test_labels = np.concatenate([
                test_batch[b'fine_labels']
            ], axis=0)

            test_images = np.swapaxes(test_data.reshape([-1, 32, 32, 3], order='F'), 1, 2)

            # split training data set into train, valid
            train_images, valid_images, train_labels, valid_labels = \
                train_test_split(train_images, train_labels,
                                 test_size=split_rate, random_state=random_state)

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images

            self.train_labels = one_hot(train_labels, self.n_classes)
            self.valid_labels = one_hot(valid_labels, self.n_classes)
            self.test_labels = one_hot(test_labels, self.n_classes)

        else:  # will be added
            pass
