from tensorflow.keras.optimizers import SGD, Adam, RMSprop


def build_optimizer(config, optimizer_name: str = 'adam'):
    if optimizer_name == 'adam':
        return Adam(
            learning_rate=config.d_lr,
            beta_1=config.beta1,
            beta_2=config.beta2,
        )
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=config.d_lr)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=config.d_lr)
    else:
        raise NotImplementedError()
