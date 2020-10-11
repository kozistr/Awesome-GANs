import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    Layer,
    LeakyReLU,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model


class WGAN:
    def __init__(self, config):
        self.config = config

        self.n_feats: int = self.config.n_feats
        self.width: int = self.config.width
        self.height: int = self.config.height
        self.n_channels: int = self.config.n_channels
        self.z_dims: int = self.config.z_dims

        self.verbose: bool = self.config.verbose

        self.discriminator: tf.keras.Model = self.build_discriminator()
        self.generator: tf.keras.Model = self.build_generator()

        if self.verbose:
            self.discriminator.summary()
            self.generator.summary()

    def build_discriminator(self) -> tf.keras.Model:
        inputs = Input((self.width, self.height, self.n_channels))

        x = Conv2D(self.n_feats, 5, 2)(inputs)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(3):
            x = Conv2D(self.n_feats * (2 ** (i + 1)), 5, 2)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)

        x = Dense(1)(x)

        return Model(inputs, x, name='discriminator')

    def build_generator(self) -> tf.keras.Model:
        inputs = Input((1, 1, self.z_dims))

        x = Dense(4 * 4 * 4 * self.z_dims)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Reshape((-1, 4, 4, 4 * self.z_dims))(x)

        for i in range(3):
            x = Conv2DTranspose(self.z_dims * 4 // (2 ** i), kernel_size=5, strides=2)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

        x = Conv2DTranspose(x, self.n_channels, 5, 1)(x)
        x = Layer('tanh')(x)

        return Model(inputs, x, name='generator')

    @tf.function
    def train_discriminator(self):
        pass

    @tf.function
    def train_generator(self):
        pass

    @tf.function
    def generate_samples(self, z: tf.Tensor):
        return self.generator(z, training=False)
