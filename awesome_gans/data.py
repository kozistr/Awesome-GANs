import tensorflow as tf
import tensorflow_datasets as tfds


class TFDatasets:
    """tensorflow_datasets package
    This opensource dataset contains lots of public datasets & loader.
     - CelebA
     - CelebA-HQ
     - DIV2K
     - MNIST
     - E-MNIST
     - Fashion-MNIST
     - CIFAR10
     - CIFAR100
     - lots of ...
    github : https://github.com/tensorflow/datasets
    """

    def __init__(self, config):
        self.dataset: str = config.dataset
        self.epochs: int = config.epochs
        self.bs: int = config.bs
        self.width: int = config.width
        self.height: int = config.height
        self.use_crop: bool = config.use_crop

    def preprocess_image(self, image: tf.Tensor) -> tf.Tensor:
        if self.use_crop:
            image = tf.image.central_crop(image, 0.5)
        image = tf.image.resize(image, (self.width, self.height), antialias=True)
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0
        return image

    def load_dataset(self, use_label: bool = False):
        ds = tfds.load(name=self.dataset, split='train', as_supervised=use_label, shuffle_files=True)
        ds = ds.map(lambda x: self.preprocess_image(x['image']), tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(self.bs * 16)
        ds = ds.batch(self.bs, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
