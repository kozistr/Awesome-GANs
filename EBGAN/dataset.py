from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.model_selection import train_test_split

'''
This dataset is for Celeb-A

- Celeb-A
    Celeb-A DataSets can be downloaded at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Celeb-A link : https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
        
    OR you can download following python code (but it doesn't work as well when i'm trying)
    code link : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
'''

DataSets = {
    # Linux
    # 'celeb-a': "/home/zero/hdd/DataSet/Celeb-A/img_align_celeba",
    # Windows
    'celeb-a': 'D:\\DataSet\\Celeb-A\\img_align_celeba',
}


class CelebADataSet:

    def __init__(self, batch_size=128, epoch=250, input_height=64, input_width=64, input_channel=3,
                 output_height=64, output_width=64, output_channel=3,
                 split_rate=0.2, random_state=42, num_threads=8, name="None"):

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

        self.celeb_a()   # load Celeb-A

    def celeb_a(self):
        pass
