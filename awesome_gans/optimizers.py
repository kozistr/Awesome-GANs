from tensorflow.keras.optimizers import SGD, Adam


def build_discriminator_optimizer(config):
    optimizer_name: str = config.d_opt

    if optimizer_name == 'adam':
        return Adam(
            learning_rate=config.d_lr,
            beta_1=config.beta1,
            beta_2=config.beta2,
        )
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=config.d_lr)
    else:
        raise NotImplementedError()


def build_generator_optimizer(config):
    optimizer_name: str = config.g_opt

    if optimizer_name == 'adam':
        return Adam(
            learning_rate=config.g_lr,
            beta_1=config.beta1,
            beta_2=config.beta2,
        )
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=config.g_lr)
    else:
        raise NotImplementedError()
