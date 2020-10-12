import tensorflow as tf

from awesome_gans.data import TFDatasets
from awesome_gans.utils import initialize, set_seed
from awesome_gans.wgan.config import get_config
from awesome_gans.wgan.model import WGAN


def main():
    config = get_config()

    # initial tf settings
    initialize()

    # reproducibility
    set_seed(config.seed)

    # load the data
    dataset: tf.data.Dataset = TFDatasets(config).load_dataset(use_label=False)

    if config.mode == 'train':
        model = WGAN(config)
        model.train(dataset)
    elif config.mode == 'inference':
        pass
    else:
        raise ValueError()


main()
