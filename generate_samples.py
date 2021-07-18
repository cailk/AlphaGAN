# -*- coding: utf-8 -*-
# @Date    : 2020-01-22
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk


import os
import argparse
import numpy as np
import torch
import utils
from model import generator_b

parser = argparse.ArgumentParser()
parser.add_argument('--gan_type', type=str, default='WGAN_GP', choices=['WGAN', 'WGAN_GP', 'AlphaGAN'], help='The type of GAN')
parser.add_argument('--dataset', type=str, default='CelebA', choices=['SVHN', 'CelebA'], help='The dataset')
parser.add_argument('--save_dir', type=str, default='model', help='Directory name to save the model')
parser.add_argument('--sample_size', type=int, default=10000)
parser.add_argument('--benchmark_mode', type=bool, default=True)
parser.add_argument('--seed', type=int, default=666)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.benchmark_mode:
    torch.backends.cudnn.benchmark = True

G = generator_b(128)

if args.dataset == 'SVHN':
    dataset_dir = 'model/svhn/'
elif args.dataset == 'CelebA':
    dataset_dir = 'model/celeba/'
elif args.dataset == 'MNIST':
    dataset_dir = 'model/mnist/'

if args.gan_type == 'WGAN':
    model_dir = 'WGAN/WGAN_G_wc.pkl'
elif args.gan_type == 'WGAN_GP':
    model_dir = 'WGAN_GP/WGAN_GP_G.pkl'
elif args.gan_type == 'AlphaGAN':
    model_dir = 'AlphaGAN_0.5_1.0/AlphaGAN_0.5_1.0_G.pkl'
    
print(dataset_dir + model_dir)

G.load_state_dict(torch.load(dataset_dir + model_dir))
G.cuda()
G.eval()

if not os.path.exists('data/' + args.gan_type):
    os.makedirs('data/' + args.gan_type)

z = torch.randn(64, 128)
z = z.cuda()
image = G(z).detach().cpu()

image = (image + 1) / 2

path = args.gan_type + '.png'
utils.save_images(image, 8, path)