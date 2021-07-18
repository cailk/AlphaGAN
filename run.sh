#!/bin/zsh

CUDA_VISIBLE_DEVICES=0 python main.py --gan_type GAN --dataset mnist --epoch 50 --input_size 28 --lrG 0.001
#CUDA_VISIBLE_DEVICES=0 python main.py --gan_type GAN --dataset fashion-mnist --epoch 50 --input_size 28