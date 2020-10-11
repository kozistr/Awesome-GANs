from awesome_gans.config import parse_args


def get_config():
    parser = parse_args()

    # Model
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='epochs to train')
    parser.add_argument('--global_steps', default=5e4, type=int, help='iterations to train')
    parser.add_argument('--n_feats', default=64, type=int, help='number of convolution filters')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for generic somethings')
    parser.add_argument('--d_lr', default=1e-4, type=float, help='learning rate of discriminator')
    parser.add_argument('--g_lr', default=1e-4, type=float, help='learning rate of generator')
    parser.add_argument('--d_opt', default='adam', type=str, choices=['adam', 'sgd'], help='disc optimizer')
    parser.add_argument('--beta1', default=0.0, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--g_opt', default='adam', type=str, choices=['adam', 'sgd'], help='gen optimizer')
    parser.add_argument('--d_loss', default='bce', type=str, choices=['bce', 'cce', 'l1', 'l2'])
    parser.add_argument('--g_loss', default='bce', type=str, choices=['bce', 'cce', 'l1', 'l2'])
    parser.add_argument('--grad_clip', default=1e-2, type=float)
    parser.add_argument(
        '--n_critics', default=5, type=int, help='number of times to train critic(discriminator) per 1-iter generator'
    )
    parser.add_argument('--z_dims', default=128, type=int, help='batch size')

    return parser.parse_args()
