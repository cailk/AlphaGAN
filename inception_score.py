# -*- coding: utf-8 -*-
# @Date    : 2020-01-08
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=10):
	'''
	Computes the Inception score of the generated images
	'''

	N = imgs.size(0)

	assert batch_size > 0
	assert N > batch_size

	# Set up dataloader
	dataset = TensorDataset(imgs)
	dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=20)

	# Load Inception model
	inception_model = inception_v3(pretrained=True, transform_input=False)
	if cuda:
		inception_model.cuda()
	inception_model.eval()
	up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)

	def get_pred(x):
		if resize:
			x = up(x)
		x = inception_model(x)
		return F.softmax(x, dim=1).detach().cpu().numpy()

	# Get predictions
	preds = np.zeros((N, 1000))

	for i, (data, ) in enumerate(dataloader, 0):
		if cuda:
			data = data.cuda()
		batch_size_i = data.size(0)

		preds[i*batch_size : i*batch_size + batch_size_i] = get_pred(data)

	# Compute the mean kl-div
	split_scores = []

	for k in range(splits):
		part = preds[k * (N // splits) : (k+1) * (N // splits), :]
		p_y = np.mean(part, axis=0)
		scores = []
		for i in range(part.shape[0]):
			p_yx = part[i, :]
			scores.append(entropy(p_yx, p_y))
		split_scores.append(np.exp(np.mean(scores)))

	return np.mean(split_scores), np.std(split_scores)
