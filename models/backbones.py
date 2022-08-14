# -*- coding: utf-8 -*-
# @Date    : 2019-12-17
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk

import torch.nn as nn

from ..utils.utils import initialize_weights

class generator_a(nn.Module):
	def __init__(self, nz=64, nc=1):
		super(generator_a, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(nz, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),

			nn.Linear(1024, 128 * 7 * 7),
			nn.BatchNorm1d(128 * 7 * 7),
			nn.ReLU(True),
		)
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),

			nn.ConvTranspose2d(64, nc, 4, 2, 1),
			nn.Tanh(),
		)
		initialize_weights(self)

	def forward(self, input):
		x = self.fc(input)
		x = x.view(-1, 128, 7, 7)
		x = self.deconv(x)

		return x

class discriminator_a(nn.Module):
	def __init__(self, nc=1):
		super(discriminator_a, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(nc, 64, 4, 2, 1),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(64, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.fc = nn.Sequential(
			nn.Linear(128 * 7 * 7, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 1),
		)
		initialize_weights(self)

	def forward(self, input):
		x = self.conv(input)
		x = x.view(-1, 128 * 7 * 7)
		x = self.fc(x)

		return x.view(-1)

class generator_b(nn.Module):
	def __init__(self, nz=128, nc=3):
		super(generator_b, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(nz, 448 * 2 * 2),
			nn.BatchNorm1d(448 * 2 * 2),
			nn.ReLU(True),
		)
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(448, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),

			nn.ConvTranspose2d(256, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),

			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.ReLU(True),

			nn.ConvTranspose2d(64, nc, 4, 2, 1),
			nn.Tanh(),
		)
		initialize_weights(self)

	def forward(self, input):
		x = self.fc(input)
		x = x.view(-1, 448, 2, 2)
		x = self.deconv(x)

		return x

class discriminator_b(nn.Module):
	def __init__(self, nc=3):
		super(discriminator_b, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(nc, 64, 4, 2, 1),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(64, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(128, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(256, 256, 4, 1, 0),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.fc = nn.Sequential(
			nn.Linear(256, 1),
		)
		initialize_weights(self)

	def forward(self, input):
		x = self.conv(input)
		x = x.view(-1, 256)
		x = self.fc(x)

		return x.view(-1)