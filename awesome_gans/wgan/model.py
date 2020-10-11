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

import awesome_gans.modules as t


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

    def build_wgan(self):
        # Generator
        self.g = self.generator(self.z)

        # Discriminator
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = t.sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = t.sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = t.sce_loss(d_fake, tf.ones_like(d_fake))

        # The gradient penalty loss
        if self.EnableGP:
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0, name='alpha')
            diff = self.g - self.x  # fake data - real data
            interpolates = self.x + alpha * diff
            d_interp = self.discriminator(interpolates, reuse=True)
            gradients = tf.gradients(d_interp, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))

            # Update D loss
            self.d_loss += self.d_lambda * self.gradient_penalty

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        if self.EnableGP:
            tf.summary.scalar("misc/gp", self.gradient_penalty)

        # Collect trainer values
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('discriminator')]
        g_params = [v for v in t_vars if v.name.startswith('generator')]

        if not self.EnableGP:
            self.d_clip = [v.assign(tf.clip_by_value(v, -self.clip, self.clip)) for v in d_params]

        # Optimizer
        if self.EnableGP:
            self.d_op = tf.train.AdamOptimizer(learning_rate=self.lr * 2, beta1=self.beta1, beta2=self.beta2).minimize(
                loss=self.d_loss, var_list=d_params
            )
            self.g_op = tf.train.AdamOptimizer(learning_rate=self.lr * 2, beta1=self.beta1, beta2=self.beta2).minimize(
                loss=self.g_loss, var_list=g_params
            )
        else:
            self.d_op = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=self.decay).minimize(
                self.d_loss, var_list=d_params
            )
            self.g_op = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=self.decay).minimize(
                self.g_loss, var_list=g_params
            )

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
