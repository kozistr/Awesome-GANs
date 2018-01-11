from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import h5py
import pickle as p
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data


DataSets = {
    # Linux
    # MNIST
    # 'mnist': '/home/zero/hdd/DataSet/MNIST/',
    # CiFar-10, 100
    # 'cifar-10': '/home/zero/hdd/DataSet/Cifar/cifar-10-batches-py/',
    # 'cifar-100': '/home/zero/hdd/DataSet/Cifar/cifar-100-python/',
    # Celeb-A DataSet
    # 'celeb-a': '/home/zero/hdd/DataSet/Celeb-A/img_align_celeba/',
    # 'celeb-a-attr': '/home/zero/hdd/DataSet/Celeb-A/list_attr_celeba.txt',
    # 'celeb-a-32x32-h5': '/home/zero/hdd/DataSet/Celeb-A/celeb-a-32x32.h5',
    # 'celeb-a-64x64-h5': '/home/zero/hdd/DataSet/Celeb-A/celeb-a-64x64.h5',
    # Windows
    # MNIST
    'mnist': 'D:\\DataSet\\MNIST\\',
    # CiFar-10, 100
    'cifar-10': 'D:\\DataSet\\Cifar\\cifar-10-batches-py\\',
    'cifar-100': 'D:\\DataSet\\Cifar\\cifar-100-python\\',
    # Celeb-A DataSet
    'celeb-a': 'D:\\DataSet\\Celeb-A\\img_align_celeba\\',
    'celeb-a-attr': 'D:\\DataSet\\Celeb-A\\list_attr_celeba.txt',
    'celeb-a-32x32-h5': 'D:\\DataSet\\Celeb-A\\celeb-a-32x32.h5',
    'celeb-a-64x64-h5': 'D:\\DataSet\\Celeb-A\\celeb-a-64x64.h5',
    # pix2pix DataSets
    'ae_photos': 'D:\\DataSet\\pix2pix\\ae_photos\\',
    'apple2orange': 'D:\\DataSet\\pix2pix\\apple2orange\\',
    'cezanne2photo': 'D:\\DataSet\\pix2pix\\cezanne2photo\\',
    'cityscapes': 'D:\\DataSet\\pix2pix\\cityscapes\\',
    'edges2handbags': 'D:\\DataSet\\pix2pix\\edges2handbags\\',
    'edges2shoes': 'D:\\DataSet\\pix2pix\\edges2shoes\\',
    'facades': 'D:\\DataSet\\pix2pix\\facades\\',
    'horse2zebra': 'D:\\DataSet\\pix2pix\\horse2zebra\\',
    'iphone2dslr_flower': 'D:\\DataSet\\pix2pix\\iphone2dslr_flower\\',
    'maps': 'D:\\DataSet\\pix2pix\\maps\\',
    'monet2photo': 'D:\\DataSet\\pix2pix\\monet2photo\\',
    'summer2winter_yosemite': 'D:\\DataSet\\pix2pix\\summer2winter_yosemite\\',
    'ukiyoe2photo': 'D:\\DataSet\\pix2pix\\vukiyoe2photo\\',
    'vangogh2photo': 'D:\\DataSet\\pix2pix\\vangogh2photo\\',
}


def get_image(path, w, h):
    img = imread(path).astype(np.float)

    orig_h, orig_w = img.shape[:2]
    new_h = int(orig_h * w / orig_w)

    img = imresize(img, (new_h, w))
    margin = int(round((new_h - h) / 2))

    return img[margin:margin + h]


def unpickle(file):
    with open(file, 'rb') as f:
        return p.load(f, encoding='bytes')


def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class MNISTDataSet:

    def __init__(self, split_rate=0.2, random_state=42, num_threads=8, is_split=False):
        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = num_threads
        self.is_split = is_split

        self.data = input_data.read_data_sets(DataSets['mnist'], one_hot=True)  # download MNIST

        # training data
        self.train_data = self.data.train

        self.train_images = self.train_data.images
        self.train_labels = self.train_data.labels
        self.valid_images = None
        self.valid_labels = None

        # test data
        self.test_data = self.data.test

        self.test_images = self.test_data.images
        self.test_labels = self.test_data.labels

        # split training data set into train, valid
        if self.is_split:
            self.train_images, self.valid_images, self.train_labels, self.valid_labels = \
                train_test_split(self.train_images, self.train_labels,
                                 test_size=self.split_rate,
                                 random_state=self.random_state)


class CiFarDataSet:

    def __init__(self, batch_size=128, epoch=250, input_height=64, input_width=64, input_channel=3,
                 output_height=64, output_width=64, output_channel=3,
                 split_rate=0.2, random_state=42, num_threads=8, name="cifar-10"):

        """
        # General Settings
        :param batch_size: training batch size, default 128
        :param epoch: training epoch, default 250
        :param input_height: input image height, default 64
        :param input_width: input image width, default 64
        :param input_channel: input image channel, default 3 (RGB)
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
        :param name: DataSet name, default cifar-10
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
        test_batch = unpickle("{0}/test".format(self.path))

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


class CelebADataSet:

    def __init__(self, batch_size=128, input_height=64, input_width=64, input_channel=3, attr_labels=(),
                 output_height=64, output_width=64, output_channel=3,
                 split_rate=0.2, random_state=42, num_threads=8, mode='w'):

        """
        # General Settings
        :param batch_size: training batch size, default 128
        :param input_height: input image height, default 64
        :param input_width: input image width, default 64
        :param input_channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 64x64x3(HWC).
        :param attr_labels: attributes of Celeb-A image, default empty tuple
        - in case of Celeb-A, the number of attributes is 40

        # Output Settings
        :param output_height: output images height, default 64
        :param output_width: output images width, default 64
        :param output_channel: output images channel, default 3

        # Pre-Processing Option
        :param split_rate: image split rate (into train & test), default 0.2
        :param random_state: random seed for shuffling, default 42
        :param num_threads: the number of threads for multi-threading, default 8

        # DataSet Option
        :param mode: h5 file mode(RW), default w
        """

        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        '''
        # Available attributes
        [
         5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair,
         Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair,
         Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face,
         Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair,
         Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young
        ]
        '''
        self.attr_labels = attr_labels
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]

        self.output_height = output_height
        self.output_width = output_width
        self.output_channel = output_channel

        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = num_threads  # change this value to the fitted value for ur system
        self.mode = mode

        self.path = ""  # DataSet path
        self.files = ""  # files' name
        self.n_classes = 0  # DataSet the number of classes, default 10

        self.data = []  # loaded images
        self.attr = []
        self.num_images = 202599
        self.images = []
        self.labels = {}
        self.ds_name = ""  # DataSet Name (by image size)

        self.celeb_a(mode=self.mode)  # load Celeb-A

    def celeb_a(self, mode):
        def get_image(path, w, h):
            img = imread(path).astype(np.float)

            orig_h, orig_w = img.shape[:2]
            new_h = int(orig_h * w / orig_w)

            img = imresize(img, (new_h, w))
            margin = int(round((new_h - h) / 2))

            return img[margin:margin + h]

        if self.input_height == 32:
            self.ds_name = 'celeb-a-32x32-h5'
        elif self.input_height == 64:
            self.ds_name = 'celeb-a-64x64-h5'

        self.labels = self.load_attr()    # selected attributes info (list)

        if mode == 'w':
            self.files = glob(os.path.join(DataSets['celeb-a'], "*.jpg"))
            self.files = np.sort(self.files)

            self.data = np.zeros((len(self.files), self.input_height * self.input_width * self.input_channel),
                                 dtype=np.uint8)

            print("[*] Image size : ", self.data.shape)

            assert (len(self.files) == self.num_images)

            for n, f_name in tqdm(enumerate(self.files)):
                image = get_image(f_name, self.input_width, self.input_height)
                self.data[n] = image.flatten()

            # write .h5 file for reusing later...
            with h5py.File(''.join([DataSets[self.ds_name]]), 'w') as f:
                f.create_dataset("images", data=self.data)

        self.images = self.load_data(size=self.num_images)

    def load_data(self, size, offset=0):
        """
            From great jupyter notebook by Tim Sainburg:
            http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
        """
        with h5py.File(DataSets[self.ds_name], 'r') as hf:
            faces = hf['images']

            full_size = len(faces)
            if size is None:
                size = full_size

            n_chunks = int(np.ceil(full_size / size))
            if offset >= n_chunks:
                print("[*] Looping from back to start.")
                offset = offset % n_chunks

            if offset == n_chunks - 1:
                print("[-] Not enough data available, clipping to end.")
                faces = faces[offset * size:]
            else:
                faces = faces[offset * size:(offset + 1) * size]

            faces = np.array(faces, dtype=np.float16)

        print("[+] Image size : ", faces.shape)

        return faces / 255.

    def load_attr(self):
        with open(DataSets['celeb-a-attr'], 'r') as f:
            img_attr = []

            self.num_images = int(f.readline().strip())
            self.attr = (f.readline().strip()).split(' ')

            print("[*] the number of images     : %d" % self.num_images)
            print("[*] the number of attributes : %d/%d" % (len(self.attr_labels), len(self.attr)))

            for fn in f.readlines():
                row = fn.strip().split()
                # img_name = row[0]
                attr = [int(x) for x in row[1:]]

                tmp = [attr[self.attr.index(x)] for x in self.attr_labels]
                tmp = [1. if x == 1 else 0. for x in tmp]  # one-hot labeling

                img_attr.append(tmp)

            return np.asarray(img_attr)

    def concat_data(self, img, label):
        label = np.tile(np.reshape(label, [-1, 1, 1, len(self.attr_labels)]),
                        [1, self.input_height, self.input_width, 1])

        return np.concatenate([img, label], axis=3)


class Pix2PixDataSet:

    def __init__(self, batch_size=64, input_height=64, input_width=64, input_channel=3,
                 output_height=64, output_width=64, output_channel=3,
                 split_rate=0.2, random_state=42, num_threads=8, mode='w', name=''):

        """
        # General Settings
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 64
        :param input_width: input image width, default 64
        :param input_channel: input image channel, default 3 (RGB)

        # Output Settings
        :param output_height: output images height, default 64
        :param output_width: output images width, default 64
        :param output_channel: output images channel, default 3

        # Pre-Processing Option
        :param split_rate: image split rate (into train & test), default 0.2
        :param random_state: random seed for shuffling, default 42
        :param num_threads: the number of threads for multi-threading, default 8

        # DataSet Option
        :param mode: h5 file mode(RW), default w
        :param name: DataSet name
        """

        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

        self.image_shape = [-1, self.input_height, self.input_width, self.input_channel]

        self.output_height = output_height
        self.output_width = output_width
        self.output_channel = output_channel

        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = num_threads  # change this value to the fitted value for ur system
        self.mode = mode

        self.path = ""  # DataSet path
        self.files = ""  # files' name

        self.data = []  # loaded images
        self.images = []
        self.ds_name = name

        self.train_images = []
        self.train_labels = []

        self.valid_images = []
        self.valid_labels = []

        self.test_images = []
        self.test_labels = []

        # testA, testB, (trainA, trainB)
        if self.ds_name == "apple2orange" or self.ds_name == "horse2zebra" or self.ds_name == "monet2photo" or \
                self.ds_name == "summer2winter_yosemite" or self.ds_name == "vangogh2photo" or \
                self.ds_name == "ae_photos" or self.ds_name == "cezanne2photo" or self.ds_name == "ukiyoe2photo" or \
                self.ds_name == "iphone2dslr_flower":
            self.single_img_process()

        # train, val, (test, sample) # double grid
        elif self.ds_name == "cityscapes" or self.ds_name == "edges2handbags" or self.ds_name == "edges2shoes" or \
                self.ds_name == "facades" or self.ds_name == "maps":
            self.double_img_process()

    def single_img_process(self):
        pass

    def double_img_process(self):
        pass


class DataIterator:

    def __init__(self, x, y, batch_size, label_off=False):
        self.x = x
        self.label_off = label_off
        if not self.label_off:
            self.y = y
        self.batch_size = batch_size
        self.num_examples = num_examples = x.shape[0]
        self.num_batches = num_examples // batch_size
        self.pointer = 0

        assert (self.batch_size <= self.num_examples)

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
