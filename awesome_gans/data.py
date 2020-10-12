import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    def __init__(self, config):
        self.dataset: str = config.dataset
        self.epochs: int = config.epochs
        self.bs: int = config.bs
        self.width: int = config.width
        self.height: int = config.height
        self.use_crop: bool = config.use_crop
        self.buffer_size: int = config.buffer_size

    def preprocess_image(self, image):
        if self.use_crop:
            image = tf.image.central_crop(image, 0.5)
        image = tf.image.resize(image, (self.width, self.height), antialias=True)
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0
        return image

    def load_dataset(self):
        ds = tfds.load(name=self.dataset, split=tfds.Split.ALL, shuffle_files=True)
        ds = ds.map(lambda x: self.preprocess_image(x['image']), tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.batch(self.bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(self.buffer_size)
        return ds
