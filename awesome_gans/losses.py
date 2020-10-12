import tensorflow as tf


@tf.function
def discriminator_wgan_loss(real: tf.Tensor, fake: tf.Tensor):
    return tf.reduce_mean(fake) - tf.reduce_mean(real)


@tf.function
def generator_wgan_loss(fake: tf.Tensor):
    return -tf.reduce_mean(fake)


@tf.function
def discriminator_loss(loss_func: str, real: tf.Tensor, fake: tf.Tensor, use_ra: bool = False):
    real_loss: float = 0.0
    fake_loss: float = 0.0

    if use_ra:
        if not loss_func.__contains__('wgan'):
            real = real - tf.reduce_mean(fake)
            fake = fake - tf.reduce_mean(real)

    if loss_func.__contains__('wgan'):
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.math.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss


@tf.function
def generator_loss(loss_func: str, real: tf.Tensor, fake: tf.Tensor, use_ra: bool = False):
    fake_loss: float = 0.0
    real_loss: float = 0.0

    if use_ra:
        fake_logit = fake - tf.reduce_mean(real)
        real_logit = real - tf.reduce_mean(fake)

        if loss_func == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))
            real_loss = tf.reduce_mean(tf.square(real_logit + 1.0))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan':
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake_logit)
            )
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real_logit)
            )

        if loss_func == 'hinge':
            fake_loss = tf.reduce_mean(tf.nn.relu(1.0 - fake_logit))
            real_loss = tf.reduce_mean(tf.nn.relu(1.0 + real_logit))
    else:
        if loss_func == 'wgan' or loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
            fake_loss = -tf.reduce_mean(fake)

        if loss_func == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan':
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

        if loss_func == 'hinge':
            fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss + real_loss

    return loss
