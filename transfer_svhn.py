# -*- coding: utf-8 -*-
# @Date    : 2020-01-22
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk
 
import os
import utils
from tqdm import tqdm
from dataloader import dataloader

data_loader = dataloader('svhn', 32, 1)
data_bar = tqdm(data_loader)

if not os.path.exists('data/svhn_images'):
	os.makedirs('data/svhn_images')
    
for idx, (data, _) in enumerate(data_bar):
    path = 'data/svhn_images/image_' + str(idx) + '.png'
    utils.save_images(data, 1, path)