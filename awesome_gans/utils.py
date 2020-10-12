import os
import random
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)


def normalize_image(images):
    return (images / 127.5) - 1.0


def denormalized_image(images):
    return (images + 1.0) * 127.5


def merge_images(
    images: Optional[np.ndarray, tf.Tensor],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    padding: int = 0,
    pad_value: int = 0,
):
    """Merge multiple images into one (squared) image.
    :param images: images to merge.
        all images need to have a scale between -1, 1
    :param n_rows: number of image per row.
    :param n_cols: number of image per col.
    :param padding: number of pads to fill left.
    :param pad_value: value of pad to fill with.
    :return: image. [0, 255] scale, np.uint8.
    """
    images = denormalized_image(images)

    if isinstance(images, tf.Tensor):
        images = images.numpy()

    images = images.astype(np.uint8)

    n, h, w, c = images.shape
    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    shape = (h * n_rows + padding * (n_rows - 1), w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:  # in case of not gray-scale image
        shape += (c,)

    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i, j = idx % n_cols, idx // n_cols
        img[j * (h + padding) : j * (h + padding) + h, i * (w + padding) : i * (w + padding) + w, ...] = image

    return img


def save_tensor_image(image: tf.Tensor, fn: str):
    tf.io.write_file(fn, tf.image.encode_png(tf.cast(image, tf.uint8)))


def save_numpy_image(image: np.ndarray, fn: str, is_rgb: bool):
    cv2.imwrite(fn, image[..., ::-1] if is_rgb else image)


def save_image(image: Optional[np.ndarray, tf.Tensor], fn: str, is_rgb: bool = True):
    if isinstance(image, tf.Tensor):
        save_tensor_image(image, fn)
    elif isinstance(image, np.ndarray):
        save_numpy_image(image, fn, is_rgb)
    else:
        raise NotImplementedError()
