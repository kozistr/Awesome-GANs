from awesome_gans.config import parse_args
from awesome_gans.utils import initialize, set_seed


def main():
    config = parse_args()

    # initial tf settings
    initialize()

    # reproducibility
    set_seed(config.seed)

    if config.mode == 'train':
        pass
    else:
        raise ValueError()


main()
