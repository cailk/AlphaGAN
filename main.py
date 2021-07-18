# -*- coding: utf-8 -*-
# @Date    : 2019-08-04
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk
import argparse, os, torch
import numpy as np
from GAN import GAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from AlphaGAN import AlphaGAN

'''parsing and configuration'''
def parse_args():
    desc = 'Pytorch implementation of GAN collections'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='WDCGAN', choices=['GAN', 'WGAN', 'WGAN_GP', 'AlphaGAN'], help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='celeba', choices=['mnist', 'fashion-mnist', 'svhn', 'cifar10', 'celeba'], help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=60, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=32, help='The size of input image')
    parser.add_argument('--nz', type=int, default=62, help='The size of latent z vector')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='result', help='Directory name to save the generated images')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--param_a', type=float, default=0.5, help='Parameter a for AlphaGAN')
    parser.add_argument('--param_b', type=float, default=1.0, help='Parameter b for AlphaGAN')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=666)

    return check_args(parser.parse_args())

'''checking arguments'''
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.nz >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

'''main'''
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'WGAN':
        gan = WGAN(args)
    elif args.gan_type == 'WGAN_GP':
        gan = WGAN_GP(args)
    elif args.gan_type == 'AlphaGAN':
        gan = AlphaGAN(args)
    else:
        raise Exception('[!] There is no option for ' + args.gan_type)

    gan.train()

if __name__ == '__main__':
    main()