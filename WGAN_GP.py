# -*- coding: utf-8 -*-
# @Date    : 2019-08-11
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import utils
from inception_score import inception_score
from dataloader import dataloader
from tqdm import tqdm
from model import *

class WGAN_GP(object):
	def __init__(self, args):
		# paramters
		self.epoch = args.epoch
		self.sample_num = 100
		self.batch_size = args.batch_size
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.gpu_mode = args.gpu_mode
		self.model_name = args.gan_type
		self.input_size = args.input_size
		self.z_dim = args.nz
		self.lambda_ = 10
		self.n_critic = 5

		# load dataset
		self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)

		# network init
		if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
			self.G = generator_mnist(nz=self.z_dim)
			self.D = discriminator_mnist()
		else:
			self.G = generator_celeba(nz=self.z_dim)
			self.D = discriminator_celeba()

		self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.G.cuda()
			self.D.cuda()

		print('---------- Networks architecture -------------')
		utils.print_network(self.G)
		utils.print_network(self.D)
		print('-----------------------------------------------')

		# fixed noise
		self.sample_z_ = torch.randn(self.batch_size, self.z_dim)
		if self.gpu_mode:
			self.sample_z_ = self.sample_z_.cuda()

	def train(self):
		self.train_hist = {'D_loss': [], 'G_loss': []}
		self.IS = []

		self.D.train()
		print('[*] Start training %s on %s!' % (self.model_name, self.dataset))
		for epoch in range(1, self.epoch+1):
			self.G.train()
			train_bar = tqdm(self.data_loader)
			for idx, (x_, _) in enumerate(train_bar):
				z_ = torch.randn(x_.size(0), self.z_dim)
				if self.gpu_mode:
					x_, z_ = x_.cuda(), z_.cuda()

				# update D network
				self.D_optimizer.zero_grad()

				D_real = self.D(x_)
				D_real_loss = -torch.mean(D_real)

				G_ = self.G(z_)
				D_fake = self.D(G_.detach())
				D_fake_loss = torch.mean(D_fake)

				# gradient penalty
				alpha = torch.rand(x_.size(0), 1, 1, 1)
				if self.gpu_mode:
					alpha = alpha.cuda()

				x_hat = alpha * x_.detach() + (1 - alpha) * G_.detach()
				x_hat.requires_grad = True

				pred_hat = self.D(x_hat)
				if self.gpu_mode:
					gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(), 
									 create_graph=True, retain_graph=True, only_inputs=True)[0]
				else:
					gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()), 
									 create_graph=True, retain_graph=True, only_inputs=True)[0]

				gradient_penalty = self.lambda_ * ((gradients.view(gradients.size(0), -1).norm(2, 1) - 1)**2).mean()

				D_loss = D_real_loss + D_fake_loss + gradient_penalty
				D_loss.backward()
				self.D_optimizer.step()

				if ((idx+1) % self.n_critic) == 0:
					# update G network
					self.G_optimizer.zero_grad()

					D_fake = self.D(G_)
					G_loss = -torch.mean(D_fake)
					G_loss.backward()
					self.G_optimizer.step()

					self.train_hist['D_loss'].append(D_loss.item())
					self.train_hist['G_loss'].append(G_loss.item())

					train_bar.set_description(desc="Epoch: [%2d] D_loss: %.8f, G_loss: %.8f" % (epoch,  D_loss.item(), G_loss.item()))

			with torch.no_grad():
				self.visualize_results(epoch)

				if epoch % 25 == 0:
					is_mean, is_std = self.calculate_is()
					self.IS.append(is_mean)
					print("The inception score of epoch %d is [%.4f +- %.4f]. " % (epoch, is_mean, is_std))

		print('[*] Training finished!... save training results')
		self.save()
		utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
		utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
		utils.IS_plot(self.IS, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

	def visualize_results(self, epoch, fix=True):
		self.G.eval()

		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		if fix:
			" fixed noise "
			samples = self.G(self.sample_z_)
		else:
			" random noise "
			sample_z_ = torch.randn(self.batch_size, self.z_dim)
			if self.gpu_mode:
				sample_z_ = sample_z_.cuda()

			samples = self.G(sample_z_)

		if self.gpu_mode:
			samples = samples.cpu().detach()
		else:
			samples = samples.detach()

		samples = (samples + 1) / 2
		utils.save_images(samples[:image_frame_dim ** 2], image_frame_dim, 
						  self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

	def calculate_is(self):
		self.G.eval()

		imgs = torch.zeros(self.batch_size*100, 3, 32, 32)
		for i in range(100):
			z = torch.randn(self.batch_size, self.z_dim)
			if self.gpu_mode:
				z = z.cuda()
			image = self.G(z).cpu().detach()
			imgs[self.batch_size*i : self.batch_size*(i+1), :, :, :] = image

		return inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10)

	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))