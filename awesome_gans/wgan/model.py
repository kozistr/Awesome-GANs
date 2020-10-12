import os

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
from tqdm import tqdm

from awesome_gans.losses import discriminator_loss, generator_loss
from awesome_gans.optimizers import build_discriminator_optimizer, build_generator_optimizer
from awesome_gans.utils import merge_images, save_image


class WGAN:
    def __init__(self, config):
        self.config = config

        self.bs: int = self.config.bs
        self.n_samples: int = self.config.n_samples
        self.epochs: int = self.config.epochs
        self.d_loss = self.config.d_loss
        self.g_loss = self.config.g_loss
        self.n_feats: int = self.config.n_feats
        self.width: int = self.config.width
        self.height: int = self.config.height
        self.n_channels: int = self.config.n_channels
        self.z_dims: int = self.config.z_dims
        self.n_critics: int = self.config.n_critics

        self.output_path: str = self.config.output_path
        self.verbose: bool = self.config.verbose
        self.log_interval: int = self.config.log_interval

        self.discriminator: tf.keras.Model = self.build_discriminator()
        self.generator: tf.keras.Model = self.build_generator()

        self.d_opt: tf.keras.optimizers = build_discriminator_optimizer(config)
        self.g_opt: tf.keras.optimizers = build_generator_optimizer(config)

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
    def train_discriminator(self, x_real: tf.Tensor):
        z = tf.random.uniform((self.bs, self.z_dims))
        with tf.GradientTape() as gt:
            x_fake = self.generator(z, training=True)
            d_fake = self.discriminator(x_fake, training=True)
            d_real = self.discriminator(x_real, training=True)

            d_loss = tf.reduce_mean(discriminator_loss(self.d_loss, d_real, d_fake))

        gradient = gt.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(gradient, self.discriminator.trainable_variables))

        return d_loss

    @tf.function
    def train_generator(self):
        z = tf.random.uniform((self.bs, self.z_dims))
        with tf.GradientTape() as gt:
            x_fake = self.generator(z, training=True)
            d_fake = self.discriminator(x_fake, training=True)

            g_loss = tf.reduce_mean(generator_loss(self.g_loss, d_fake))

        gradient = gt.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(gradient, self.generator.trainable_variables))

        return g_loss

    def train(self, dataset: tf.data.Dataset):
        z_samples = tf.random.normal((self.n_samples, self.z_dims))

        for epoch in range(self.epochs):
            loader = tqdm(dataset, desc=f'[*] Epoch {epoch} / {self.epochs}')
            for n_iter, batch in enumerate(loader):
                d_loss = 0.0
                for _ in range(self.n_critics):
                    d_loss += self.train_discriminator(batch)

                g_loss = self.train_generator()

                loader.set_postfix(
                    d_loss=f'{d_loss / self.n_critics:.6f}',
                    g_loss=f'{g_loss:.6f}',
                )

                global_steps: int = epoch * len(loader) + n_iter
                if global_steps and global_steps % self.log_interval == 0:
                    samples = self.generate_samples(z_samples)
                    samples = merge_images(samples, n_rows=int(self.n_samples ** 0.5))
                    save_image(samples, os.path.join(self.output_path, f'{global_steps}.png'))

    @tf.function
    def generate_samples(self, z: tf.Tensor):
        return self.generator(z, training=False)
