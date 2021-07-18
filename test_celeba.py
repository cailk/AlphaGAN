from dataloader import dataloader
import utils

data_loader = dataloader('celeba', 64, 64)

for idx, (data, _) in enumerate(data_loader):
    if idx > 0:
        break
    utils.save_images(data, 8, 'celeba.png')