from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import h5py
import pickle as p
import numpy as np
import tensorflow as tf

from PIL import Image
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split


DataSets = {
    # Linux
    # pix2pix DataSets
    # 'ae_photos': '/home/zero/hdd/DataSet/pix2pix/ae_photos/',
    # 'apple2orange': '/home/zero/hdd/DataSet/pix2pix/apple2orange/',
    # 'cezanne2photo': '/home/zero/hdd/DataSet/pix2pix/cezanne2photo/',
    # 'cityscapes': '/home/zero/hdd/DataSet/pix2pix/cityscapes/',
    # 'edges2handbags': '/home/zero/hdd/DataSet/pix2pix/edges2handbags/',
    # 'edges2shoes': '/home/zero/hdd/DataSet/pix2pix/edges2shoes/',
    # 'facades': '/home/zero/hdd/DataSet/pix2pix/facades/',
    # 'horse2zebra': '/home/zero/hdd/DataSet/pix2pix/horse2zebra/',
    # 'iphone2dslr_flower': 'D/home/zero/hdd/DataSet/pix2pix/iphone2dslr_flower/',
    # 'maps': '/home/zero/hdd/DataSet/pix2pix/maps/',
    # 'monet2photo': '/home/zero/hdd/DataSet/pix2pix/monet2photo/',
    # 'summer2winter_yosemite': '/home/zero/hdd/DataSet/pix2pix/summer2winter_yosemite/',
    # 'ukiyoe2photo': '/home/zero/hdd/DataSet/pix2pix/vukiyoe2photo/',
    'vangogh2photo': '/home/zero/hdd/DataSet/pix2pix/vangogh2photo/',
    'vangogh2photo-32x32-h5': '/home/zero/hdd/DataSet/pix2pix/vangogh2photo/v2p-32x32.h5',
    'vangogh2photo-64x64-h5': '/home/zero/hdd/DataSet/pix2pix/vangogh2photo/v2p-64x64.h5',
    # DIV2K DataSet
    'div2k-hr': '/home/zero/hdd/DataSet/DIV2K/DIV2K_train_HR/',
    'div2k-hr-h5': '/home/zero/hdd/DataSet/DIV2K/div2k-hr.h5',
    'div2k-lr': '/home/zero/hdd/DataSet/DIV2K/DIV2K_train_LR_bicubic/X4/',
    'div2k-lr-h5': '/home/zero/hdd/DataSet/DIV2K/div2k-lr.h5',
    'div2k-hr-val': '/home/zero/hdd/DataSet/DIV2K/DIV2K_valid_HR_bicubic/X4/',
    'div2k-hr-val.h5': '/home/zero/hdd/DataSet/DIV2K/div2k-hr-val.h5',
    'div2k-lr-val': '/home/zero/hdd/DataSet/DIV2K/DIV2K_valid_LR_bicubic/X4/',
    'div2k-lr-val.h5': '/home/zero/hdd/DataSet/DIV2K/div2k-lr-val.h5',
    # UrbanSound8K DataSet
    'urban_sound': '/home/zero/hdd/DataSet/UrbanSound/audio/',
    # Windows
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
    # 'vangogh2photo': 'D:\\DataSet\\pix2pix\\vangogh2photo\\',
    # 'vangogh2photo-32x32-h5': 'D:\\DataSet\\pix2pix\\vangogh2photo\\v2p-32x32-',
    # 'vangogh2photo-64x64-h5': 'D:\\DataSet\\pix2pix\\vangogh2photo\\v2p-64x64-',
}


def get_image(path, w, h):
    img = imread(path).astype(np.float)

    orig_h, orig_w = img.shape[:2]
    new_h = int(orig_h * w / orig_w)

    img = imresize(img, (new_h, w))
    margin = int(round((new_h - h) / 2))

    return img[margin:margin + h]


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
            raise ValueError("[-] There'is no supporting file... :(")

    @staticmethod
    def get_img(path, size=(64, 64), interp=cv2.INTER_CUBIC):
        img = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB
        if img.shape[0] == size[0]:
            return img
        else:
            return cv2.imresize(img, size, interp)

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
                 use_image_scaling=True, image_scale='0,1'):

        self.op = name.split('_')

        try:
            assert len(self.op) == 1
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
            raise AssertionError("[-] Path does not exist :(")

        self.buffer_size = buffer_size
        self.n_threads = n_threads

        self.file_list = sorted(os.listdir(self.path))
        self.file_ext = self.file_list[0].split('.')[-1]
        self.file_names = glob(os.path.join(self.path, '/*.%s' % self.file_ext))
        self.raw_data = np.ndarray([])  # (N, H * W * C)

        self.types = ('img', 'tfr', 'h5', 'npy')  # Supporting Data Types
        self.op_src = self.get_extension(self.file_ext)
        self.op_dst = self.op[0]

        try:
            assert (self.op_src in self.types and self.op_dst in self.types)
        except AssertionError:
            raise AssertionError("[-] Invalid Operation Types :(")

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
        np.random.RandomState(1337).shuffle(order)
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
                    continue
                else:
                    self.raw_data = np.concatenate((self.raw_data, data))

    def load_npy(self):
        self.raw_data = np.rollaxis(np.squeeze(np.load(self.file_names), axis=0), 0, 3)

    def convert_to_img(self):
        raw_data_shape = self.raw_data.shape  # (N, H * W * C)

        try:
            assert os.path.exists(self.save_file_name)
        except AssertionError:
            print("[-] There's no %s :(" % self.save_file_name)
            print("[*] Make directory at %s... " % self.save_file_name)
            os.mkdir(self.save_file_name)

        for idx in range(raw_data_shape[0]):
            cv2.imwrite("%d.png" % idx, cv2.COLOR_RGB2BGR)

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

        from tensorflow.examples.tutorials.mnist import data
        self.data = data.read_data_sets(self.ds_path, one_hot=True)  # download MNIST

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
        # WARN: Only for python3, NOT FOR python2
        with open(file, 'rb') as f:
            return p.load(f, encoding='bytes')

    @staticmethod
    def one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    def __init__(self, height=64, width=64, channel=3,
                 use_split=False, split_rate=0.2, random_state=42, ds_name="cifar-10", ds_path=None):

        """
        # General Settings
        :param height: input image height, default 64
        :param width: input image width, default 64
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
            self.valid_labels = self.one_hot(valid_labels, self.n_classes)

        self.train_images = train_images
        self.test_images = test_images

        self.train_labels = self.one_hot(train_labels, self.n_classes)
        self.test_labels = self.one_hot(test_labels, self.n_classes)

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
            self.valid_labels = self.one_hot(valid_labels, self.n_classes)

        self.train_images = train_images
        self.test_images = test_images

        self.train_labels = self.one_hot(train_labels, self.n_classes)
        self.test_labels = self.one_hot(test_labels, self.n_classes)


class CelebADataSet:

    """
    This Class for CelebA & CelebA-HQ DataSets.
        - saving images as .h5 file for more faster loading.
        - Actually, CelebA-HQ DataSet is kinda encrypted. So if u wanna use it, decrypt first!
            There're a few codes that download & decrypt CelebA-HQ DataSet.
    """

    def __init__(self,
                 height=64, width=64, channel=3, attr_labels=(),
                 n_threads=30, use_split=False, split_rate=0.2,
                 ds_path=None, ds_type="CelebA",
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

        # DataSet Settings
        :param ds_path: DataSet's Path
        :param ds_type: which DataSet is
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
        self.ds_path = ds_path
        self.ds_image_path = ds_path + "/Img/img_aling_celeba/"
        self.ds_label_path = ds_path + "/Anno/list_attr_celeba.txt"
        self.ds_type = ds_type

        try:
            assert self.ds_path
        except AssertionError:
            raise AssertionError("[-] CelebA/CelebA-HQ DataSets' Path is required!")

        if self.ds_type == "CelebA":
            self.num_images = 202599  # the number of CelebA    images
        elif self.ds_type == "CelebA-HQ":
            self.num_images = 30000   # the number of CelebA-HQ images

            tmp_path = self.ds_path + "/imgHQ00000."
            if os.path.exists(tmp_path + "dat"):
                raise FileNotFoundError("[-] You need to decrypt .dat file first!\n" +
                                        "[-] plz, use original PGGAN repo or"
                                        " this repo https://github.com/nperraud/download-celebA-HQ")
            elif os.path.exists(tmp_path + "npy"):
                def npy2png(i):
                    try:
                        data = np.load('imgHQ%05d.npy' % i)
                    except:
                        print("[-] imgHQ%05d.npy" % i)
                        return False
                    im = Image.fromarray(np.rollaxis(np.squeeze(data, axis=0), 0, 3))
                    im.save('imgHQ%05d.png' % i)
                    return True

                print("[*] You should convert .npy files to .png image files for comfort :)")
                print("[*] But, I'll do it for you :) It'll take some times~")

                # Converting...
                ii = [i for i in range(self.num_images)]

                pool = Pool(self.n_threads)
                print(pool.map(npy2png, ii))
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

        self.images = DataSetLoader(path=self.ds_path,
                                    size=self.image_shape,
                                    use_save=self.use_save,
                                    name=self.save_type,
                                    save_file_name=self.save_file_name,
                                    use_image_scaling=True,
                                    image_scale='0,1').raw_data  # numpy arrays
        self.labels = self.load_attr()

        if self.use_concat_data:
            self.images = self.concat_data(self.images, self.labels)

    def load_attr(self):
        with open(self.ds_label_path, 'r') as f:
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

    def __init__(self, batch_size=64, height=64, width=64, channel=3,
                 output_height=64, output_width=64, output_channel=3,
                 crop_size=128, split_rate=0.2, random_state=42, num_threads=8, name=''):

        """
        # General Settings
        :param batch_size: training batch size, default 64
        :param height: input image height, default 64
        :param width: input image width, default 64
        :param channel: input image channel, default 3 (RGB)

        # Output Settings
        :param output_height: output images height, default 64
        :param output_width: output images width, default 64
        :param output_channel: output images channel, default 3

        # Pre-Processing Option
        :param crop_size: image crop size, default 128
        :param split_rate: image split rate (into train & test), default 0.2
        :param random_state: random seed for shuffling, default 42
        :param num_threads: the number of threads for multi-threading, default 8

        # DataSet Option
        :param name: train/test DataSet, default train
        """

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel

        self.image_shape = [self.batch_size, self.height, self.width, self.channel]

        self.output_height = output_height
        self.output_width = output_width
        self.output_channel = output_channel

        self.crop_size = crop_size
        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = num_threads  # change this value to the fitted value for ur system
        self.mode = 'w'

        self.files_a = []
        self.files_b = []
        self.data_a = []
        self.data_b = []
        self.images_a = []
        self.images_b = []
        self.num_images_a = 400
        self.num_images_b = 6287
        self.ds_name = name

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
        def get_image(path, w, h):
            img = imread(path).astype(np.float)

            orig_h, orig_w = img.shape[:2]
            new_h = int(orig_h * w / orig_w)

            img = imresize(img, (new_h, w))
            margin = int(round((new_h - h) / 2))

            return img[margin:margin + h]

        size = self.height
        self.ds_name += '-' + str(size) + 'x' + str(size) + '-h5'

        if os.path.exists(DataSets[self.ds_name]):
            self.mode = 'r'

        if self.mode == 'w':
            data_set_name = self.ds_name.split('-')[0]

            self.files_a = glob(os.path.join(DataSets[data_set_name] + 'trainA/', "*.jpg"))
            self.files_b = glob(os.path.join(DataSets[data_set_name] + 'trainB/', "*.jpg"))
            self.files_a = np.sort(self.files_a)
            self.files_b = np.sort(self.files_b)

            self.data_a = np.zeros((len(self.files_a), self.height * self.width * self.channel),
                                   dtype=np.uint8)
            self.data_b = np.zeros((len(self.files_b), self.height * self.width * self.channel),
                                   dtype=np.uint8)

            print("[*] Image A size : ", self.data_a.shape)
            print("[*] Image B size : ", self.data_b.shape)

            assert (len(self.files_a) == self.num_images_a) and (len(self.files_b) == self.num_images_b)

            for n, f_name in tqdm(enumerate(self.files_a)):
                image = get_image(f_name, self.width, self.height)
                self.data_a[n] = image.flatten()

            for n, f_name in tqdm(enumerate(self.files_b)):
                image = get_image(f_name, self.width, self.height)
                self.data_b[n] = image.flatten()

            # write .h5 file for reusing later...
            with h5py.File(''.join([DataSets[self.ds_name] + 'a.h5']), 'w') as f:
                f.create_dataset("images", data=self.data_a)

            with h5py.File(''.join([DataSets[self.ds_name] + 'b.h5']), 'w') as f:
                f.create_dataset("images", data=self.data_b)

        self.images_a = self.load_data(size=self.num_images_a, name='a.h5')
        self.images_b = self.load_data(size=self.num_images_b, name='b.h5')

    def double_img_process(self):
        pass

    def load_data(self, size, offset=0, name=""):
        with h5py.File(DataSets[self.ds_name] + name, 'r') as hf:
            pix2pix = hf['images']

            full_size = len(pix2pix)
            if size is None:
                size = full_size

            n_chunks = int(np.ceil(full_size / size))
            if offset >= n_chunks:
                print("[*] Looping from back to start.")
                offset = offset % n_chunks

            if offset == n_chunks - 1:
                print("[-] Not enough data available, clipping to end.")
                pix2pix = pix2pix[offset * size:]
            else:
                pix2pix = pix2pix[offset * size:(offset + 1) * size]

                pix2pix = np.array(pix2pix, dtype=np.float16)

        print("[+] Image size : ", pix2pix.shape)

        return pix2pix / 255.  # (pix2pix / 127.5) - 1.


class ImageNetDataSet:

    def __init__(self):
        pass


class Div2KDataSet:

    def __init__(self, batch_size=128,
                 hr_height=384, hr_width=384, lr_height=96, lr_width=96, channel=3,
                 use_split=False, split_rate=0.2, random_state=42, n_threads=8):

        """
        # General Settings
        :param batch_size: training batch size
        :param hr_height: input HR image height, default 384
        :param hr_width: input HR image width, default 384
        :param lr_height: input LR image height, default 96
        :param lr_width: input LR image width, default 96
        :param channel: input image channel, default 3 (RGB)
        - in case of Div2K - ds x4, image size is 384 x 384 x 3 (HWC).

        # Pre-Processing Option
        :param split_rate: image split rate (into train & test), default 0.2
        :param random_state: random seed for shuffling, default 42
        :param num_threads: the number of threads for multi-threading, default 8
        """

        self.batch_size = batch_size
        self.hr_height = hr_height
        self.hr_width = hr_width
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.channel = channel

        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = num_threads  # change this value to the fitted value for ur system
        self.mode = 'w'

        self.path = ""   # DataSet path
        self.files_hr, self.files_lr = [], []  # HR/LR files' name
        self.data_hr, self.data_lr = [], []   # loaded images
        self.images = []
        self.num_images = 800
        self.num_images_val = 100
        self.hr_ds_name = "div2k-hr-h5"  # DataSet Name
        self.lr_ds_name = "div2k-lr-h5"  # DataSet Name

        self.div2k()  # load DIV2K DataSet

    def div2k(self):

        def _get_image(path):
            img = cv2.imread(path)
            # return scipy.misc.imread(path, mode='RGB')
            return img

        def hr_pre_process(img):
            img = imresize(img, size=(self.hr_height, self.hr_width), interp='bilinear')
            # img = cv2.resize(img, (self.hr_height, self.hr_width), interpolation=cv2.INTER_AREA)
            return img

        def lr_pre_process(img):
            img = imresize(img, size=(self.lr_height, self.lr_width), interp='bicubic')
            # img = cv2.resize(img, (self.lr_height, self.lr_width), interpolation=cv2.INTER_CUBIC)
            return img

        if os.path.exists(DataSets[self.hr_ds_name]) and os.path.exists(DataSets[self.lr_ds_name]):
            self.mode = 'r'

        if self.mode == 'w':
            self.files_hr = np.sort(glob(os.path.join(DataSets['div2k-hr'], "*.png")))
            self.files_lr = np.sort(glob(os.path.join(DataSets['div2k-lr'], "*.png")))

            self.data_hr = np.zeros((len(self.files_hr),
                                     self.hr_height * self.hr_width * self.channel),
                                    dtype=np.uint8)
            self.data_lr = np.zeros((len(self.files_lr),
                                     self.lr_height * self.lr_width * self.channel),
                                    dtype=np.uint8)

            print("[*] HR Image size : ", self.data_hr.shape)
            print("[*] LR Image size : ", self.data_lr.shape)

            try:
                assert ((len(self.files_hr) == self.num_images) and (len(self.files_lr) == self.num_images))
            except AssertionError:
                print("[-] The number of HR images : %d" % len(self.files_hr))
                print("[-] The number of LR images : %d" % len(self.files_lr))
                raise AssertionError

            for n, f_name in tqdm(enumerate(self.files_hr)):
                self.data_hr[n] = hr_pre_process(_get_image(f_name)).flatten()

            for n, f_name in tqdm(enumerate(self.files_lr)):
                self.data_lr[n] = lr_pre_process(_get_image(f_name)).flatten()

            # write .h5 file for reusing later...
            with h5py.File(''.join([DataSets[self.hr_ds_name]]), 'w') as f:
                f.create_dataset("images", data=self.data_hr)

            with h5py.File(''.join([DataSets[self.lr_ds_name]]), 'w') as f:
                f.create_dataset("images", data=self.data_lr)

        self.images = self.load_data(size=self.num_images)

    def load_data(self, size, offset=0):
        """
            From great jupyter notebook by Tim Sainburg:
            http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
        """
        ds_names = [self.hr_ds_name, self.lr_ds_name]
        hr_lr_images = []

        for ds_name in ds_names:
            with h5py.File(DataSets[ds_name], 'r') as hf:
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

                # [0, 255] to [-1, 1]
                faces = np.array(faces, dtype=np.float32)
                faces = (faces / (255 / 2.)) - 1.

                print("[+] Image size : ", faces.shape)

                hr_lr_images.append(faces)

        return hr_lr_images


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
