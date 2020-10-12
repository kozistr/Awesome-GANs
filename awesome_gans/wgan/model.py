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

from awesome_gans.losses import discriminator_loss, generator_loss, discriminator_wgan_loss, generator_wgan_loss
from awesome_gans.optimizers import build_optimizer
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
        self.grad_clip: float = self.config.grad_clip

        self.model_path: str = self.config.model_path
        self.output_path: str = self.config.output_path
        self.verbose: bool = self.config.verbose

        self.discriminator: tf.keras.Model = self.build_discriminator()
        self.generator: tf.keras.Model = self.build_generator()

        self.d_opt: tf.keras.optimizers = build_optimizer(config, config.d_opt)
        self.g_opt: tf.keras.optimizers = build_optimizer(config, config.g_opt)

        self.checkpoint = tf.train.Checkpoint(
            discriminator=self.discriminator,
            discriminator_optimzer=self.d_opt,
            generator=self.generator,
            generator_optimizer=self.g_opt,
        )

        if self.verbose:
            self.discriminator.summary()
            self.generator.summary()

    def build_discriminator(self) -> tf.keras.Model:
        inputs = Input((self.width, self.height, self.n_channels))

        x = Conv2D(self.n_feats, kernel_size=5, strides=2, padding='same')(inputs)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(3):
            x = Conv2D(self.n_feats * (2 ** (i + 1)), kernel_size=5, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)

        x = Dense(1)(x)

        return Model(inputs, x, name='discriminator')

    def build_generator(self) -> tf.keras.Model:
        inputs = Input((self.z_dims,))

        x = Dense(4 * 4 * 4 * self.z_dims)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Reshape((4, 4, 4 * self.z_dims))(x)

        for i in range(3):
            x = Conv2DTranspose(self.z_dims * 4 // (2 ** i), kernel_size=5, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

        x = Conv2DTranspose(self.n_channels, kernel_size=5, strides=1, padding='same')(x)
        x = Layer('tanh')(x)

        return Model(inputs, x, name='generator')

    @tf.function
    def train_discriminator(self, x: tf.Tensor):
        z = tf.random.uniform((self.bs, self.z_dims))
        with tf.GradientTape() as gt:
            x_fake = self.generator(z, training=True)
            d_fake = self.discriminator(x_fake, training=True)
            d_real = self.discriminator(x, training=True)

            d_loss = discriminator_wgan_loss(d_real, d_fake)

            gradients = gt.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -self.grad_clip, self.grad_clip))

            return d_loss

    @tf.function
    def train_generator(self):
        z = tf.random.uniform((self.bs, self.z_dims))
        with tf.GradientTape() as gt:
            x_fake = self.generator(z, training=True)
            d_fake = self.discriminator(x_fake, training=True)

            g_loss = generator_wgan_loss(d_fake)

            gradients = gt.gradient(g_loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(gradients, self.generator.trainable_variables))

            return g_loss

    def load(self) -> int:
        return 0

    def train(self, dataset: tf.data.Dataset):
        start_epoch: int = self.load()

        z_samples = tf.random.uniform((self.n_samples, self.z_dims))

        for epoch in range(start_epoch, self.epochs):
            loader = tqdm(dataset, desc=f'[*] Epoch {epoch} / {self.epochs}')
            for n_iter, batch in enumerate(loader):
                for _ in range(self.n_critics):
                    d_loss = self.train_discriminator(batch)

                g_loss = self.train_generator()

                loader.set_postfix(
                    d_loss=f'{d_loss:.5f}',
                    g_loss=f'{g_loss:.5f}',
                )

            # saving the generated samples
            samples = self.generate_samples(z_samples)
            samples = merge_images(samples, n_rows=int(self.n_samples ** 0.5))
            save_image(samples, os.path.join(self.output_path, f'{epoch}.png'))

            # saving the models & optimizers
            self.checkpoint.save(file_prefix=os.path.join(self.model_path, f'{epoch}'))

    @tf.function
    def generate_samples(self, z: tf.Tensor):
        return self.generator(z, training=False)
