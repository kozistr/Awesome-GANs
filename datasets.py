from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import h5py
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.model_selection import train_test_split


seed = 1337


def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSetLoader:

    @staticmethod
    def get_extension(ext):
        if ext in ['jpg', 'png']:
            return 'img'
        elif ext == 'tfr':
            return 'tfr'
        elif ext == 'h5':
            return 'h5'
        elif ext == 'npy':
            return 'npy'
        else:
            raise ValueError("[-] There'is no supporting file... [%s] :(" % ext)

    @staticmethod
    def get_img(path, size=(64, 64), interp=cv2.INTER_CUBIC):
        img = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB
        if img.shape[0] == size[0]:
            return img
        else:
            return cv2.resize(img, size, interp)

    @staticmethod
    def parse_tfr_tf(record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features['data'], tf.uint8)
        return tf.reshape(data, features['shape'])

    @staticmethod
    def parse_tfr_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value
        data = ex.features.feature['data'].bytes_list.value[0]
        return np.fromstring(data, np.uint8).reshape(shape)

    @staticmethod
    def img_scaling(img, scale='0,1'):
        if scale == '0,1':
            img /= 255.
        elif scale == '-1,1':
            img = (img / 127.5) - 1.
        else:
            raise ValueError("[-] Only '0,1' or '-1,1' please")
        return img

    def __init__(self, path, size=None, name='to_tfr', use_save=False, save_file_name='',
                 buffer_size=4096, n_threads=8,
                 use_image_scaling=True, image_scale='0,1', debug=True):

        self.op = name.split('_')
        self.debug = debug

        try:
            assert len(self.op) == 2
        except AssertionError:
            raise AssertionError("[-] Invalid Target Types :(")

        self.size = size

        try:
            assert self.size
        except AssertionError:
            raise AssertionError("[-] Invalid Target Sizes :(")

        # To-DO
        # Supporting 4D Image
        self.height = size[0]
        self.width = size[1]
        self.channel = size[2]

        self.path = path

        try:
            assert os.path.exists(self.path)
        except AssertionError:
            raise AssertionError("[-] Path(%s) does not exist :(" % self.path)

        self.buffer_size = buffer_size
        self.n_threads = n_threads

        if os.path.isfile(self.path):
            self.file_list = [self.path]
            self.file_ext = self.path.split('.')[-1]
            self.file_names = [self.path]
        else:
            self.file_list = sorted(os.listdir(self.path))
            self.file_ext = self.file_list[0].split('.')[-1]
            self.file_names = glob(self.path + '/*')
        self.raw_data = np.ndarray([], dtype=np.uint8)  # (N, H * W * C)

        if self.debug:
            print("[*[ Detected Path            is [%s]" % self.path)
            print("[*[ Detected File Extension  is [%s]" % self.file_ext)
            print("[*] Detected First File Name is [%s] (%d File(s))" % (self.file_names[0], len(self.file_names)))

        self.types = ('img', 'tfr', 'h5', 'npy')  # Supporting Data Types
        self.op_src = self.get_extension(self.file_ext)
        self.op_dst = self.op[1]

        try:
            chk_src, chk_dst = False, False
            for t in self.types:
                if self.op_src == t:
                    chk_src = True
                if self.op_dst == t:
                    chk_dst = True
            assert chk_src and chk_dst
        except AssertionError:
            raise AssertionError("[-] Invalid Operation Types (%s, %s) :(" % (self.op_src, self.op_dst))

        if self.op_src == self.types[0]:
            self.load_img()
        elif self.op_src == self.types[1]:
            self.load_tfr()
        elif self.op_src == self.types[2]:
            self.load_h5()
        elif self.op_src == self.types[3]:
            self.load_npy()
        else:
            raise NotImplementedError("[-] Not Supported Type :(")

        # Random Shuffle
        order = np.arange(self.raw_data.shape[0])
        np.random.RandomState(seed).shuffle(order)
        self.raw_data = self.raw_data[order]

        # Clip [0, 255]
        self.raw_data = np.rint(self.raw_data).clip(0, 255).astype(np.uint8)

        self.use_save = use_save
        self.save_file_name = save_file_name

        if self.use_save:
            try:
                assert self.save_file_name
            except AssertionError:
                raise AssertionError("[-] Empty save-file name :(")

            if self.op_dst == self.types[0]:
                self.convert_to_img()
            elif self.op_dst == self.types[1]:
                self.tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
                self.tfr_writer = tf.python_io.TFRecordWriter(self.save_file_name + ".tfrecords", self.tfr_opt)
                self.convert_to_tfr()
            elif self.op_dst == self.types[2]:
                self.convert_to_h5()
            elif self.op_dst == self.types[3]:
                self.convert_to_npy()
            else:
                raise NotImplementedError("[-] Not Supported Type :(")

        self.use_image_scaling = use_image_scaling
        self.image_scale = image_scale

        if self.use_image_scaling:
            self.raw_data = self.img_scaling(self.raw_data, self.image_scale)

    def load_img(self):
        self.raw_data = np.zeros((len(self.file_list), self.height * self.width * self.channel),
                                 dtype=np.uint8)

        for i, fn in tqdm(enumerate(self.file_names)):
            self.raw_data[i] = self.get_img(fn, (self.height, self.width)).flatten()
            if self.debug:  # just once
                print("[*] Image Shape   : ", self.raw_data[i].shape)
                print("[*] Image Size    : ", self.raw_data[i].size)
                print("[*] Image MIN/MAX :  (%d, %d)" % (np.min(self.raw_data[i]), np.max(self.raw_data[i])))
                self.debug = False

    def load_tfr(self):
        self.raw_data = tf.data.TFRecordDataset(self.file_names, compression_type='', buffer_size=self.buffer_size)
        self.raw_data = self.raw_data.map(self.parse_tfr_tf, num_parallel_calls=self.n_threads)

    def load_h5(self, size=0, offset=0):
        init = True

        for fl in self.file_list:  # For multiple .h5 files
            with h5py.File(fl, 'r') as hf:
                data = hf['images']
                full_size = len(data)

                if size == 0:
                    size = full_size

                n_chunks = int(np.ceil(full_size / size))
                if offset >= n_chunks:
                    print("[*] Looping from back to start.")
                    offset %= n_chunks
                if offset == n_chunks - 1:
                    print("[-] Not enough data available, clipping to end.")
                    data = data[offset * size:]
                else:
                    data = data[offset * size:(offset + 1) * size]

                data = np.array(data, dtype=np.uint8)
                print("[+] ", fl, " => Image size : ", data.shape)

                if init:
                    self.raw_data = data
                    init = False

                    if self.debug:  # just once
                        print("[*] Image Shape   : ", self.raw_data[0].shape)
                        print("[*] Image Size    : ", self.raw_data[0].size)
                        print("[*] Image MIN/MAX :  (%d, %d)" % (np.min(self.raw_data[0]), np.max(self.raw_data[0])))
                        self.debug = False

                    continue
                else:
                    self.raw_data = np.concatenate((self.raw_data, data))

    def load_npy(self):
        self.raw_data = np.rollaxis(np.squeeze(np.load(self.file_names), axis=0), 0, 3)

        if self.debug:  # just once
            print("[*] Image Shape   : ", self.raw_data[0].shape)
            print("[*] Image Size    : ", self.raw_data[0].size)
            print("[*] Image MIN/MAX :  (%d, %d)" % (np.min(self.raw_data[0]), np.max(self.raw_data[0])))
            self.debug = False

    def convert_to_img(self):
        def to_img(i):
            cv2.imwrite('imgHQ%05d.png' % i, cv2.COLOR_BGR2RGB)
            return True

        raw_data_shape = self.raw_data.shape  # (N, H * W * C)

        try:
            assert os.path.exists(self.save_file_name)
        except AssertionError:
            print("[-] There's no %s :(" % self.save_file_name)
            print("[*] Make directory at %s... " % self.save_file_name)
            os.mkdir(self.save_file_name)

        ii = [i for i in range(raw_data_shape[0])]

        pool = Pool(self.n_threads)
        print(pool.map(to_img, ii))

    def convert_to_tfr(self):
        for data in self.raw_data:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=data.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tostring()]))
            }))
            self.tfr_writer.write(ex.SerializeToString())

    def convert_to_h5(self):
        with h5py.File(self.save_file_name, 'w') as f:
            f.create_dataset("images", data=self.raw_data)

    def convert_to_npy(self):
        np.save(self.save_file_name, self.raw_data)


class MNISTDataSet:

    def __init__(self, use_split=False, split_rate=0.15, random_state=42, ds_path=None):
        self.use_split = use_split
        self.split_rate = split_rate
        self.random_state = random_state

        self.ds_path = ds_path

        try:
            assert self.ds_path
        except AssertionError:
            raise AssertionError("[-] MNIST DataSet Path is required!")

        from tensorflow.examples.tutorials.mnist import input_data
        self.data = input_data.read_data_sets(self.ds_path, one_hot=True)  # download MNIST

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
        if self.use_split:
            self.train_images, self.valid_images, self.train_labels, self.valid_labels = \
                train_test_split(self.train_images, self.train_labels,
                                 test_size=self.split_rate,
                                 random_state=self.random_state)


class CiFarDataSet:

    @staticmethod
    def unpickle(file):
        import pickle as p

        # WARN: Only for python3, NOT FOR python2
        with open(file, 'rb') as f:
            return p.load(f, encoding='bytes')

    def __init__(self, height=32, width=32, channel=3,
                 use_split=False, split_rate=0.2, random_state=42, ds_name="cifar-10", ds_path=None):

        """
        # General Settings
        :param height: input image height, default 32
        :param width: input image width, default 32
        :param channel: input image channel, default 3 (RGB)
        - in case of CIFAR, image size is 32 x 32 x 3 (HWC).

        # Pre-Processing Option
        :param use_split: training DataSet splitting, default True
        :param split_rate: image split rate (into train & test), default 0.2
        :param random_state: random seed for shuffling, default 42

        # DataSet Option
        :param ds_name: DataSet's name, default cifar-10
        :param ds_path: DataSet's path, default None
        """

        self.height = height
        self.width = width
        self.channel = channel

        self.use_split = use_split
        self.split_rate = split_rate
        self.random_state = random_state

        self.ds_name = ds_name
        self.ds_path = ds_path  # DataSet path
        self.n_classes = 10     # DataSet the number of classes, default 10

        self.train_images = None
        self.valid_images = None
        self.test_images = None

        self.train_labels = None
        self.valid_labels = None
        self.test_labels = None

        try:
            assert self.ds_path
        except AssertionError:
            raise AssertionError("[-] CIFAR10/100 DataSets' Path is required!")

        if self.ds_name == "cifar-10":
            self.cifar_10()   # loading Cifar-10
        elif self.ds_name == "cifar-100":
            self.cifar_100()  # loading Cifar-100
        else:
            raise NotImplementedError("[-] Only 'cifar-10' or 'cifar-100'")

    def cifar_10(self):
        self.n_classes = 10  # labels

        train_batch_1 = self.unpickle("{0}/data_batch_1".format(self.ds_path))
        train_batch_2 = self.unpickle("{0}/data_batch_2".format(self.ds_path))
        train_batch_3 = self.unpickle("{0}/data_batch_3".format(self.ds_path))
        train_batch_4 = self.unpickle("{0}/data_batch_4".format(self.ds_path))
        train_batch_5 = self.unpickle("{0}/data_batch_5".format(self.ds_path))

        # training data & label
        train_data = np.concatenate([
            train_batch_1[b'data'],
            train_batch_2[b'data'],
            train_batch_3[b'data'],
            train_batch_4[b'data'],
            train_batch_5[b'data'],
        ], axis=0)

        train_labels = np.concatenate([
            train_batch_1[b'labels'],
            train_batch_2[b'labels'],
            train_batch_3[b'labels'],
            train_batch_4[b'labels'],
            train_batch_5[b'labels'],
        ], axis=0)

        # Image size : 32x32x3
        train_images = np.swapaxes(train_data.reshape([-1,
                                                       self.height,
                                                       self.width,
                                                       self.channel], order='F'), 1, 2)

        # test data & label
        test_batch = self.unpickle("{0}/test_batch".format(self.ds_path))

        test_data = test_batch[b'data']
        test_labels = np.array(test_batch[b'labels'])

        # image size : 32x32x3
        test_images = np.swapaxes(test_data.reshape([-1,
                                                     self.height,
                                                     self.width,
                                                     self.channel], order='F'), 1, 2)

        # split training data set into train / val
        if self.use_split:
            train_images, valid_images, train_labels, valid_labels = \
                train_test_split(train_images, train_labels,
                                 test_size=self.split_rate,
                                 random_state=self.random_state)

            self.valid_images = valid_images
            self.valid_labels = one_hot(valid_labels, self.n_classes)

        self.train_images = train_images
        self.test_images = test_images

        self.train_labels = one_hot(train_labels, self.n_classes)
        self.test_labels = one_hot(test_labels, self.n_classes)

    def cifar_100(self):
        self.n_classes = 100  # labels

        # training data & label
        train_batch = self.unpickle("{0}/train".format(self.ds_path))

        train_data = np.concatenate([train_batch[b'data']], axis=0)
        train_labels = np.concatenate([train_batch[b'fine_labels']], axis=0)
        train_images = np.swapaxes(train_data.reshape([-1,
                                                       self.height,
                                                       self.width,
                                                       self.channel], order='F'), 1, 2)

        # test data & label
        test_batch = self.unpickle("{0}/test".format(self.ds_path))

        test_data = np.concatenate([test_batch[b'data']], axis=0)
        test_labels = np.concatenate([test_batch[b'fine_labels']], axis=0)
        test_images = np.swapaxes(test_data.reshape([-1,
                                                     self.height,
                                                     self.width,
                                                     self.channel], order='F'), 1, 2)

        # split training data set into train / val
        if self.use_split:
            train_images, valid_images, train_labels, valid_labels = \
                train_test_split(train_images, train_labels,
                                 test_size=self.split_rate,
                                 random_state=self.random_state)

            self.valid_images = valid_images
            self.valid_labels = one_hot(valid_labels, self.n_classes)

        self.train_images = train_images
        self.test_images = test_images

        self.train_labels = one_hot(train_labels, self.n_classes)
        self.test_labels = one_hot(test_labels, self.n_classes)


class CelebADataSet:

    """
    This Class for CelebA & CelebA-HQ DataSets.
        - saving images as .h5 file for more faster loading.
        - Actually, CelebA-HQ DataSet is kinda encrypted. So if u wanna use it, decrypt first!
            There're a few codes that download & decrypt CelebA-HQ DataSet.
    """

    def __init__(self,
                 height=64, width=64, channel=3, attr_labels=(),
                 n_threads=30, use_split=False, split_rate=0.2, random_state=42,
                 ds_image_path=None, ds_label_path=None, ds_type="CelebA", use_img_scale=True, img_scale="-1,1",
                 use_save=False, save_type='to_h5', save_file_name=None,
                 use_concat_data=False):

        """
        # General Settings
        :param height: image height
        :param width: image width
        :param channel: image channel
        - in case of CelebA,    image size is  64  x  64  x 3 (HWC)
        - in case of CelebA-HQ, image size is 1024 x 1024 x 3 (HWC)
        :param attr_labels: attributes of CelebA DataSet
        - in case of CelebA,    the number of attributes is 40

        # Pre-Processing Option
        :param n_threads: the number of threads
        :param use_split: splitting train DataSet into train/val
        :param split_rate: image split rate (into train & val)
        :param random_state: random seed for shuffling, default 42

        # DataSet Settings
        :param ds_image_path: DataSet's Image Path
        :param ds_label_path: DataSet's Label Path
        :param ds_type: which DataSet is
        :param use_img_scale: using img scaling?
        :param img_scale: img normalize
        :param use_save: saving into another file format
        :param save_type: file format to save
        :param save_file_name: file name to save
        :param use_concat_data: concatenate images & labels
        """

        self.height = height
        self.width = width
        self.channel = channel
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
        self.image_shape = (self.height, self.width, self.channel)  # (H, W, C)

        self.n_threads = n_threads
        self.use_split = use_split
        self.split_rate = split_rate
        self.random_state = random_state

        self.attr = []      # loaded labels
        self.images = []
        self.labels = {}

        """
        Expected DataSet's Path Example
        CelebA    : CelebA/ (sub-folder : Anno/..., Img/... )
        CelebA-HQ : CelebA-HQ/ (sub-folder : ...npy, ...png )
        Labels    : CelebA/Anno/...txt
        
        Expected DatSet's Type
        'CelebA' or 'CelebA-HQ'
        """
        self.ds_image_path = ds_image_path
        self.ds_label_path = ds_label_path
        self.ds_type = ds_type

        self.use_img_scale = use_img_scale
        self.img_scale = img_scale

        try:
            assert self.ds_image_path and self.ds_label_path
        except AssertionError:
            raise AssertionError("[-] CelebA/CelebA-HQ DataSets' Path is required! (%s)")

        if self.ds_type == "CelebA":
            self.num_images = 202599  # the number of CelebA    images
        elif self.ds_type == "CelebA-HQ":
            self.num_images = 30000   # the number of CelebA-HQ images

            tmp_path = self.ds_image_path + "/imgHQ00000."
            if os.path.exists(tmp_path + "dat"):
                raise FileNotFoundError("[-] You need to decrypt .dat file first!\n" +
                                        "[-] plz, use original PGGAN repo or"
                                        " this repo https://github.com/nperraud/download-celebA-HQ")
        else:
            raise NotImplemented("[-] 'ds_type' muse be 'CelebA' or 'CelebA-HQ'")

        self.use_save = use_save
        self.save_type = save_type
        self.save_file_name = save_file_name

        self.use_concat_data = use_concat_data

        try:
            if self.use_save:
                assert self.save_file_name
        except AssertionError:
            raise AssertionError("[-] save-file/folder-name is required!")

        self.images = DataSetLoader(path=self.ds_image_path,
                                    size=self.image_shape,
                                    use_save=self.use_save,
                                    name=self.save_type,
                                    save_file_name=self.save_file_name,
                                    use_image_scaling=use_img_scale,
                                    image_scale=self.img_scale).raw_data  # numpy arrays
        self.labels = self.load_attr(path=self.ds_label_path)

        if self.use_concat_data:
            self.images = self.concat_data(self.images, self.labels)

        # split training data set into train / val
        if self.use_split:
            self.train_images, self.valid_images, self.train_labels, self.valid_labels = \
                train_test_split(self.images, self.labels,
                                 test_size=self.split_rate,
                                 random_state=self.random_state)

    def load_attr(self, path):
        with open(path, 'r') as f:
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
        label = np.tile(np.reshape(label, [-1, 1, 1, len(self.attr_labels)]), [1, self.height, self.width, 1])
        return np.concatenate([img, label], axis=3)


class Pix2PixDataSet:

    def __init__(self, height=64, width=64, channel=3,
                 use_split=False, split_rate=0.15, random_state=42, n_threads=8,
                 ds_path=None, ds_name=None, use_save=False, save_type='to_h5', save_file_name=None):

        """
        # General Settings
        :param height: image height, default 64
        :param width: image width, default 64
        :param channel: image channel, default 3 (RGB)

        # Pre-Processing Option
        :param use_split: using DataSet split, default False
        :param split_rate: image split rate (into train & test), default 0.2
        :param random_state: random seed for shuffling, default 42
        :param n_threads: the number of threads for multi-threading, default 8

        # DataSet Option
        :param ds_path: DataSet's Path, default None
        :param ds_name: DataSet's Name, default None
        :param use_save: saving into another file format
        :param save_type: file format to save
        :param save_file_name: file name to save
        """

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = (self.height, self.width, self.channel)

        self.use_split = use_split
        self.split_rate = split_rate
        self.random_state = random_state
        self.n_threads = n_threads  # change this value to the fitted value for ur system

        """
        Expected ds_path : pix2pix/...
        Expected ds_name : apple2orange
        """
        self.ds_path = ds_path
        self.ds_name = ds_name
        # single grid : testA, testB, (trainA, trainB)
        # double grid : train, val, (test, sample)
        self.ds_single_grid = ['apple2orange', 'horse2zebra', 'monet2photo', 'summer2winter_yosemite', 'vangogh2photo',
                               'ae_photos', 'cezanne2photo', 'ukivoe2photo', 'iphone2dslr_flower']
        self.ds_double_grid = ['cityscapes', 'edges2handbags', 'edges2shoes', 'facades', 'maps']

        # Single Grid DatSet - the number of images
        self.n_sg_images_a = 400
        self.n_sg_images_b = 6287
        # Double Grid DatSet - the number of images
        self.n_dg_images_a = 0
        self.n_dg_images_b = 0

        self.use_save = use_save
        self.save_type = save_type
        self.save_file_name = save_file_name

        try:
            if self.use_save:
                assert self.save_file_name
        except AssertionError:
            raise AssertionError("[-] save-file/folder-name is required!")

        if self.ds_name in self.ds_single_grid:
            self.images_a = DataSetLoader(path=self.ds_path + "/" + self.ds_name + "/trainA/",
                                          size=self.image_shape,
                                          use_save=self.use_save,
                                          name=self.save_type,
                                          save_file_name=self.save_file_name,
                                          use_image_scaling=True,
                                          image_scale='0,1').raw_data  # numpy arrays

            self.images_b = DataSetLoader(path=self.ds_path + "/" + self.ds_name + "/trainB/",
                                          size=self.image_shape,
                                          use_save=self.use_save,
                                          name=self.save_type,
                                          save_file_name=self.save_file_name,
                                          use_image_scaling=True,
                                          image_scale='0,1').raw_data  # numpy arrays
            self.n_images_a = self.n_sg_images_a
            self.n_images_b = self.n_sg_images_b
        elif self.ds_name in self.ds_double_grid:
            # To-Do
            # Implement this!
            self.n_images_a = self.n_dg_images_a
            self.n_images_b = self.n_dg_images_b
        else:
            raise NotImplementedError("[-] Not Implemented yet")


class ImageNetDataSet:

    def __init__(self):
        pass


class Div2KDataSet:

    def __init__(self, hr_height=384, hr_width=384, lr_height=96, lr_width=96, channel=3,
                 use_split=False, split_rate=0.1, random_state=42, n_threads=8,
                 ds_path=None, ds_name=None, use_save=False, save_type='to_h5', save_file_name=None):

        """
        # General Settings
        :param hr_height: input HR image height, default 384
        :param hr_width: input HR image width, default 384
        :param lr_height: input LR image height, default 96
        :param lr_width: input LR image width, default 96
        :param channel: input image channel, default 3 (RGB)
        - in case of Div2K - ds x4, image size is 384 x 384 x 3 (HWC).

        # Pre-Processing Option
        :param split_rate: image split rate (into train & test), default 0.1
        :param random_state: random seed for shuffling, default 42
        :param n_threads: the number of threads for multi-threading, default 8

        # DataSet Option
        :param ds_path: DataSet's Path, default None
        :param ds_name: DataSet's Name, default None
        :param use_save: saving into another file format
        :param save_type: file format to save
        :param save_file_name: file name to save
        """

        self.hr_height = hr_height
        self.hr_width = hr_width
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.channel = channel
        self.hr_shape = (self.hr_height, self.hr_width, self.channel)
        self.lr_shape = (self.lr_height, self.lr_width, self.channel)

        self.use_split = use_split
        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = n_threads  # change this value to the fitted value for ur system

        """
        Expected ds_path : div2k/...
        Expected ds_name : X4
        """
        self.ds_path = ds_path
        self.ds_name = ds_name
        self.ds_hr_path = self.ds_path + "/DIV2K_train_HR/"
        self.ds_lr_path = self.ds_path + "/DIV2K_train_LR_bicubic/" + self.ds_name + "/"

        try:
            assert self.ds_path
        except AssertionError:
            raise AssertionError("[-] DIV2K DataSet Path is required!")

        self.use_save = use_save
        self.save_type = save_type
        self.save_file_name = save_file_name

        try:
            if self.use_save:
                assert self.save_file_name
        except AssertionError:
            raise AssertionError("[-] save-file/folder-name is required!")

        self.n_images = 800
        self.n_images_val = 100

        self.hr_images = DataSetLoader(path=self.ds_hr_path,
                                       size=self.hr_shape,
                                       use_save=self.use_save,
                                       name=self.save_type,
                                       save_file_name=self.save_file_name,
                                       use_image_scaling=True,
                                       image_scale='-1,1').raw_data  # numpy arrays

        self.lr_images = DataSetLoader(path=self.ds_lr_path,
                                       size=self.lr_shape,
                                       use_save=self.use_save,
                                       name=self.save_type,
                                       save_file_name=self.save_file_name,
                                       use_image_scaling=True,
                                       image_scale='-1,1').raw_data  # numpy arrays


class UrbanSoundDataSet:

    def __init__(self):
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
