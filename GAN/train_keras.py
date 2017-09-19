from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import argparse
import gan


model_path = {
    'sample_output': './GAN/',
    'model': './model/gan_generate.h5'
}

def generate():



def train():



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_size", type=int, default=16)
    parser.add_argument("--sample_num", type=int, default=64)
    parser.add_argument("--nb_filter", type=int, default=128)
    parser.add_argument("--fc_filter", type=int, default=512)
    parser.add_argument("--z_dim", type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.mode == "train":
        train()
    elif args.mode == "gen":
        generate()
