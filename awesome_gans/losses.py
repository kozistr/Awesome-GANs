import tensorflow as tf


def discriminator_loss(loss_func: str, d_real: tf.Tensor, d_fake: tf.Tensor):
    real_loss, fake_loss = 0.0, 0.0
    if loss_func in {'wgan', 'wgan-gp'}:
        real_loss = -d_real
        fake_loss = d_fake
    if loss_func == 'lsgan':
        real_loss = tf.math.squared_difference(d_real, 1.0)
        fake_loss = tf.square(d_fake)
    if loss_func in {'dragan', 'gan'}:
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real)
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake)
    if loss_func == 'hinge':
        real_loss = tf.math.maximum(0.0, 1.0 - d_real)
        fake_loss = tf.math.maximum(0.0, 1.0 + d_fake)
    return fake_loss + real_loss


def generator_loss(loss_func: str, d_fake: tf.Tensor):
    fake_loss = 0.0
    if loss_func in {'wgan', 'wgan-gp'}:
        fake_loss = -d_fake
    if loss_func == 'lsgan':
        fake_loss = tf.math.squared_difference(d_fake, 1.0)
    if loss_func in {'dragan', 'gan'}:
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake)
    if loss_func == 'hinge':
        fake_loss = -d_fake
    return fake_loss
