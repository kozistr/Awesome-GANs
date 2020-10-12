import tensorflow as tf

from awesome_gans.config import parse_args
from awesome_gans.data import TFDatasets
from awesome_gans.utils import initialize, set_seed
from awesome_gans.wgan.model import WGAN


def main():
    config = parse_args()

    # initial tf settings
    initialize()

    # reproducibility
    set_seed(config.seed)

    # load the data
    dataset: tf.data.Dataset = TFDatasets(config).load_dataset()

    if config.mode == 'train':
        model = WGAN(config)
        model.train(dataset)
    elif config.mode == 'inference':
        pass
    else:
        raise ValueError()


main()
