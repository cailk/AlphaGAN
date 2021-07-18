# -*- coding: utf-8 -*-
# @Date    : 2019-07-31
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk
import os
import imageio
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    images = F.interpolate(images, scale_factor=2, mode='bilinear')
    vutils.save_image(images, image_path, nrow=size, padding=0)

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.style.use('ggplot')
    plt.plot(x, y1, label='D_loss', linewidth=1.0)
    plt.plot(x, y2, label='G_loss', linewidth=1.0)

    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.xlim(left=0)

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)
    plt.close()

def IS_plot(IS, path='IS.png', model_name=''):
    N = len(IS)
    x = np.linspace(0, 5 * N - 5, N)

    plt.style.use('ggplot')
    plt.plot(x, IS, lw=1.0)

    plt.xlabel('Epoch')
    plt.ylabel('Inception Score')
    plt.xlim(left=0)

    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path, model_name + '_IS.png')

    plt.savefig(path)
    plt.close()