import tensorflow as tf
import numpy as np

import keras as K

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.activations import *
from keras.layers.convolutional import Conv2D, Deconv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.datasets import mnist


# setting random seed
tf.set_random_seed(777)
np.random.seed(777)


def selu(x):
    # fixed point mean, var (0, 1)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return scale * K.elu(x, alpha)


class DataSet:

    def __init__(self, img_rows=28, img_cols=28, img_channel=1,
                 debug=True):
        print("[*] Loading MNIST DataSet")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        print("[+] Done!")

        # Reshaping (-1, 784) to (-1, 28, 28, 1)
        x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, img_channel)
        x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, img_channel)

        # Normalizing
        x_train /= 255.
        x_test /= 255.

        if debug:  # printing information
            print("\t[*] Training Data Shape :", x_train.shape[1:])
            print("\t[*]   Test   Data Shape :", x_test.shape[1:])
            print("\t[*] Training Data Samples :", x_train.shape[0])
            print("\t[*]   Test   Data Samples :", x_test.shape[0])


class GAN:

    def __init__(self, batch_size=32,
                 input_img_rows=28, input_img_cols=28, input_img_channel=1,
                 output_img_rows=28, output_img_cols=28, output_img_channel=1,
                 z_dim=100, nb_filter=128, fc_filer=512,
                 sample_num=64, sample_size=16,
                 g_lr=9.995e-3, d_lr=9.995e-4, epsilon=1e-9):
        self.batch_size = batch_size

        self.input_rows = input_img_rows
        self.input_cols = input_img_cols
        self.input_channel = input_img_channel

        self.output_rows = output_img_rows
        self.output_cols = output_img_cols
        self.output_channel = output_img_channel

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.z_dim = z_dim
        self.nb_filter = nb_filter
        self.fc_filter = fc_filer

        self.d_lr, self.g_lr = d_lr, g_lr

        self.eps = epsilon

        self.build_gan()

    def make_trainable(self, net, val):
        net.trainable = val
        for layer in net.layers:
            layer.trainable = val

    def conv2d_bn(self, nb_filter, weight_decay=5e-4, name=''):
        def f(x):
            x = Conv2D(nb_filter, kernel_size=(5, 5), activation=None,
                       kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay),
                       padding="same", name=name)(x)
            x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay))(x)
            x = Activation(selu)(x)
            return ZeroPadding2D((1, 1))(x)

        return f

    def deconv2d_bn(self, nb_filter, weight_decay=5e-4, name=''):
        def f(x):
            x = Deconv2D(nb_filter, kernel_size=(5, 5), activation=None,
                         kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay),
                         padding="same", name=name)(x)
            x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay))(x)
            x = Activation(selu)(x)
            return ZeroPadding2D((1, 1))(x)

        return f

    def discriminator(self, input_=(28, 28, 1)):  # with simple conv-bn-pool layers
        def conv_pool(x, F, name):
            for idx, f in enumerate(F):
                x = self.conv2d_bn(f, name='{}_conv{}'.format(name, str(idx)))(x)
            # also u can use conv pooling instead of max pooling if u have a memory problem.
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)
            return Dropout(0.25)(x)

        net_in = Input(shape=input_)

        # layer 1
        nb_filter = self.nb_filter
        net = conv_pool(net_in, [nb_filter], name="block1")

        # layer 2
        nb_filter *= 2
        net = conv_pool(net, [nb_filter], name="block2")

        # u can add more layers deeper, but it just called DCGAN not GAN :).
        # so i just stopped adding more layers!

        net = Flatten()(net)
        net = Dense(self.fc_filter, activation=selu, name='fc')(net)
        net = Dropout(0.5)(net)
        net = Dense(2, activation='softmax', name='predictions')(net)

        net = Model(inputs=net_in, outputs=net)

        return net

    def generator(self, input_=(100, )):
        def up_conv(x, weight_decay=5e-4, name=''):
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(nb_filter, kernel_size=(5, 5), activation=None,
                       kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay),
                       padding="same", name=name)(x)
            return Activation(selu)(x)

        def fc_bn(x, weight_decay=5e-4, bn=True):
            x = Dense(F)(x)
            if bn:
                x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                                       beta_regularizer=l2(weight_decay))(x)
            return Activation(selu)(x)

        nb_filter = self.nb_filter
        rows, cols = self.input_rows, self.input_cols

        net_in = Input(shape=input_)

        de_net = fc_bn(net_in, self.fc_filter, bn=False)
        de_net = fc_bn(de_net, (rows / 4) * (cols / 4) * nb_filter, bn=True)
        de_net = Reshape((7, 7, 128))(de_net)  # Reshaping (-1, 7 * 7 * 128) to (-1, 7, 7 128)

        # layer 1
        de_net = up_conv(de_net, name="de_block1")
        # layer 2
        de_net = up_conv(de_net, name="de_block2")

        de_net = Model(inputs=net_in, outputs=de_net)

        return de_net

    def build_gan(self):
        '''
            i use conv-layer for making discriminator
            u can change the net if u want,
            for example, simple neural net with max-out(my previous version), or anything
        '''

        d_net = self.discriminator()
        g_net = self.generator()

        # Discriminator & generator & GAN Model
        self.make_trainable(d_net, False)
        gan = Model(inputs=d_net, outputs=g_net)

        g_net.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=self.d_lr, beta_1=0.9, beta_2=0.999, epsilon=self.eps))

        gan.compile(loss='categoricial_crossentropy',
                    optimizer=Adam(lr=self.d_lr, beta_1=0.9, beta_2=0.999, epsilon=self.eps))

        self.make_trainable(d_net, True)
        d_net.compile(loss='categoricial_crossentropy',
                      optimizer=Adam(lr=self.d_lr, beta_1=0.9, beta_2=0.999, epsilon=self.eps))

        gan.summary()
