from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Awesome-GANs Arguments')

    # dataset
    parser.add_argument('--width', type=int, default=64, help='width of image')
    parser.add_argument('--height', type=int, default=64, help='height of image')
    parser.add_argument('--n_channels', type=int, default=3, help='number of channel of image')
    parser.add_argument('--root_path', type=str, default='./', help='root path')
    parser.add_argument('--mnist_path', type=str, default='mnist')
    parser.add_argument('--fashion_mnist_path', type=str, default='fashion-mnist')
    parser.add_argument('--cifar10_path', type=str, default='cifar10')
    parser.add_argument('--cifar100_path', type=str, default='cifar100')
    parser.add_argument('--celeba_path', type=str, default='CelebA')
    parser.add_argument('--celeba_hq_path', type=str, default='CelebA-HQ')
    parser.add_argument('--div2k_path', type=str, default='DIV2K')
    parser.add_argument('--pix2pix_path', type=str, default='pix2pix')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--output_path', type=str, default='outputs')

    # misc
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'inference'])
    parser.add_argument('--device', default='cuda', type=str, help='type of device', choices=['cpu', 'cuda'])
    parser.add_argument('--n_threads', default=8, type=int, help='number of threads')
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
    parser.add_argument('--log_interval', default=1000, type=int, help='intervals to log')
    parser.add_argument('--verbose', type=bool, default=True)

    return parser.parse_args()
