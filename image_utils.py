from skimage.transform import resize

import tensorflow as tf
import numpy as np
import imageio


def down_sampling(img):
    shape = img.get_shape()  # [batch, height, width, channels]

    h2 = int(shape[1] // 2)
    w2 = int(shape[2] // 2)

    return tf.image.resize_images(img, [h2, w2], tf.image.ResizeMethod.BILINEAR)


def up_sampling(img):
    shape = img.get_shape()  # [batch, height, width, channels]

    h2 = int(shape[1] * 2)
    w2 = int(shape[2] * 2)

    return tf.image.resize_images(img, [h2, w2], tf.image.ResizeMethod.BILINEAR)


def inverse_transform(images):
    images *= 255.
    images[images > 255.] = 255.
    images[images < 0.] = 0.
    return images


def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def img_save(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)


def save_images(images, size, image_path):
    return img_save(inverse_transform(images), size, image_path)


def get_image(path, w, h):
    img = np.asarray(imageio.imread(path))

    orig_h, orig_w = img.shape[:2]
    new_h = int(orig_h * w / orig_w)

    img = resize(img, (new_h, w), anti_alising=True)
    margin = int(round((new_h - h) / 2))

    return img[margin:margin + h]
