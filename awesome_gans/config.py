from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Awesome-GANs Arguments')

    # dataset
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

    # model
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='epochs to train')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for generic somethings')
    parser.add_argument('--d_lr', default=1e-4, type=float, help='learning rate of discriminator')
    parser.add_argument('--g_lr', default=1e-4, type=float, help='learning rate of generator')
    parser.add_argument('--c_lr', default=1e-4, type=float, help='learning rate of classifier')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd'], help='optimizer for generic')
    parser.add_argument('--d_optimizer', default='adam', type=str, choices=['adam', 'sgd'], help='disc optimizer')
    parser.add_argument('--g_optimizer', default='adam', type=str, choices=['adam', 'sgd'], help='gen optimizer')
    parser.add_argument('--c_optimizer', default='adam', type=str, choices=['adam', 'sgd'], help='cls optimizer')
    parser.add_argument('--loss', default='bce', type=str, choices=['bce', 'cce', 'l1', 'l2'])
    parser.add_argument(
        '--gan_loss', default='gan', type=str,
        choices=['gan', 'lsgan', 'ragan', 'wgan', 'wgan-gp', 'hinge', 'spheregan']
    )

    # misc
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'inference'])
    parser.add_argument('--device', default='cuda', type=str, help='type of device', choices=['cpu', 'cuda'])
    parser.add_argument('--n_threads', default=8, type=int, help='number of threads')
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
    parser.add_argument('--verbose', type=bool, default=True)

    return parser.parse_args()
