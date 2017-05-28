import tensorflow as tf
import numpy as np

class BEGAN:

    def __init__(self, s, input_height=128, input_width=128, channel=3,
                 output_height=128, output_width=128, sample_size=128, sample_num=64, batch_size=64,
                 z_dim=100, eps=1e-12):
        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.channel = channel
        self.image_shape = [self.input_height, self.input_height, self.channel]  # 128x128x3

        self.eps = eps

        self.z_dim = z_dim

        self.sample_size = sample_size
        self.sample_num = sample_num

        self.build_bdgan()

    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            pass

    def generator(self, x, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            pass

    def build_bdgan(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, "x-image")
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], "z-noise")

        self.lr = tf.placeholder(tf.float32, "learning-rate")
        self.kt = tf.placeholder(tf.float32, "kt")
