# global configuration for GAN training

import argparse


args_list = []
parser = argparse.ArgumentParser()


def add_arg_group(name):
    """
    :param name: A str. Argument group.
    :return: An list. Arguments.
    """
    arg = parser.add_argument_group(name)
    args_list.append(arg)
    return arg


def get_config():
    cfg, un_parsed = parser.parse_known_args()
    return cfg, un_parsed


# Model
model_arg = add_arg_group('Model')
model_arg.add_argument('--model_path', type=str, default="./model/")
model_arg.add_argument('--output', type=str, default="./gen_img/")

# DataSet
data_arg = add_arg_group('DataSet')
data_arg.add_argument('--mnist', type=str, default="./mnist/")
data_arg.add_argument('--fashion_mnist', type=str, default="./fashion-mnist/")
data_arg.add_argument('--cifar10', type=str, default="./cifar10/")
data_arg.add_argument('--cifar100', type=str, default="./cifar100/")
data_arg.add_argument('--celeba', type=str, default="/media/zero/data/CelebA/")
data_arg.add_argument('--celeba-hq', type=str, default="./CelebA-HQ/")
data_arg.add_argument('--div2k', type=str, default="./DIV2K/")
data_arg.add_argument('--pix2pix', type=str, default="./pix2pix/")

# Misc
misc_arg = add_arg_group('Misc')
misc_arg.add_argument('--device', type=str, default='gpu')
misc_arg.add_argument('--n_threads', type=int, default=8,
                      help='the number of workers for speeding up')
misc_arg.add_argument('--seed', type=int, default=1337)
misc_arg.add_argument('--verbose', type=bool, default=True)
