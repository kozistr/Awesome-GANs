import scipy.misc
import numpy as np

import tensorflow as tf


def downsample(I):
    shape = I.get_shape()  # [batch, height, width, channels]

    h2 = int(shape[1] // 2)
    w2 = int(shape[2] // 2)

    return tf.image.resize_images(I, h2, w2,
                                  tf.image.ResizeMethod.BILINEAR)


def upsample(I):
    shape = I.get_shape()  # [batch, height, width, channels]

    h2 = int(shape[1] * 2)
    w2 = int(shape[2] * 2)

    return tf.image.resize_images(I, h2, w2,
                                  tf.image.ResizeMethod.BILINEAR)


def inverse_transform(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
