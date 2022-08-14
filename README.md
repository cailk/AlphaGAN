# Utilizing Amari-Alpha Divergence to Stabilize the Training of Generative Adversarial Networks

By Likun Cai, Yanjie Chen, Ning Cai, Wei Cheng, Hao Wang.

This repo is the official implementation of [AlphaGAN](https://www.mdpi.com/1099-4300/22/4/410).

## Introduction

We propose the Alpha-divergence Generative Adversarial Net (Alpha-GAN). 
It adopts the alpha divergence as the minimization objective function of generators. 
The alpha divergence can be regarded as a generalization of the Kullback–Leibler divergence, Pearson divergence, Hellinger divergence, etc. 
Our Alpha-GAN employs the power function as the form of adversarial loss for the discriminator with two order indices, and these hyper-parameters make our model more flexible to balance between the generated and target distributions. 