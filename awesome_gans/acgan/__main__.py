from awesome_gans.config import parse_args
from awesome_gans.utils import set_seed


def main():
    config = parse_args()

    set_seed(config.seed)

    if config.mode == 'train':
        pass
    elif config.mode == 'inference':
        pass
    else:
        raise ValueError()


main()
