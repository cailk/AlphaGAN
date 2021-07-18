# -*- coding: utf-8 -*-
# @Date    : 2019-07-31
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk


from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = Compose([Resize(input_size),
                         CenterCrop(input_size), 
    					 ToTensor(), 
    					 Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_gray = Compose([Resize(input_size),
                              CenterCrop(input_size),
    						  ToTensor(),
    						  Normalize((0.5, ), (0.5, ))])

    kwargs = {'num_workers': 20, 'pin_memory': False}
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform_gray),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform_gray),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif dataset == 'celeba':
        data_loader = DataLoader(
            datasets.ImageFolder('data/celeba', transform=transform),
            batch_size=batch_size, shuffle=True, **kwargs)

    return data_loader