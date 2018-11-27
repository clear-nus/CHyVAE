from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
from models import CHyVAE


def set_args(parser):
    parser.add_argument('--dataset', required=True, help='2dshapes | correlated_ellipses')
    parser.add_argument('--batch_size', type=int,
                        default=50, help='input batch size')
    parser.add_argument('--image_size', type=int, default=64,
                        help='the dim of the input image to network')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of image channels')
    parser.add_argument('--z_dim', type=int, default=10,
                        help='latent vector dim')
    parser.add_argument('--nu', type=int, default=33,
                        help='degrees of freedom (> z_dim + 1)')
    parser.add_argument('--n_steps', type=int, default=25000,
                        help='numbers of training steps')
    parser.add_argument('--run', type=int, default=None,
                        help='run number')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = set_args(parser)
    args = parser.parse_args()
    print(args)
    model = CHyVAE(args.dataset, args.z_dim, args.image_size, args.channels, args.batch_size,
                   args.n_steps, args.nu, np.eye(args.z_dim), args.run)
    model.train()
    model.generate()
