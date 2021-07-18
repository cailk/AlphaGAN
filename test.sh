#!/bin/zsh

python fid_score.py 'data/WGAN' 'data/celeba/img_align_celeba' --gpu 0
python fid_score.py 'data/WGAN_GP' 'data/celeba/img_align_celeba' --gpu 0