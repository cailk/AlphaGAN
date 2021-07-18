# -*- coding: utf-8 -*-
# @Date    : 2019-08-13
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk
import utils, time, os, pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader

class generator(nn.Module):
	# Network Architecture is exactly same as in DCGAN
	def __init__(self, input_dim=100, output_dim=3, d=64):
		super(generator, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(self.input_dim, d*8, kernel_size=4, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(d*8),
			nn.ReLU(True),

			nn.ConvTranspose2d(d*8, d*4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(d*4),
			nn.ReLU(True),

			nn.ConvTranspose2d(d*4, d*2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(d*2),
			nn.ReLU(True),

			nn.ConvTranspose2d(d*2, d, 4, 2, 1, bias=False),
			nn.BatchNorm2d(d),
			nn.ReLU(True),

			nn.ConvTranspose2d(d, self.output_dim, 4, 2, 1, bias=False),
			nn.Tanh()
		)
		utils.initialize_weights(self)

	def forward(self, input):
		x = self.deconv(input)
		
		return x

class discriminator(nn.Module):
	# Network Architecture is exactly same as in DCGAN
	def __init__(self, input_dim=3, output_dim=1, d=64):
		super(discriminator, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, d, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(d, d*2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(d*2),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(d*4),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(d*8),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(d*8, self.output_dim, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)
		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv(input)
		
		return x.view(-1)

class DCGAN(object):
	def __init__(self, args):
		# parameters
		self.epoch = args.epoch
		self.sample_num = 100
		self.batch_size = args.batch_size
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.log_dir = args.log_dir
		self.gpu_mode = args.gpu_mode
		self.model_name = args.gan_type
		self.input_size = args.input_size
		self.z_dim = 100

		self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
		data = next(iter(self.data_loader))[0]

		self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1])
		self.D = discriminator(input_dim=data.shape[1], output_dim=1)
		self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
		self.G_scheduler = optim.lr_scheduler.MultiStepLR(self.G_optimizer, milestones=[10, 20], gamma=0.1)
		self.D_scheduler = optim.lr_scheduler.MultiStepLR(self.D_optimizer, milestones=[10, 20], gamma=0.1)

		if self.gpu_mode:
			self.G.cuda()
			self.D.cuda()
			self.BCE_loss = nn.BCELoss().cuda()
		else:
			self.BCE_loss = nn.BCELoss()

		print('---------- Networks architecture ----------')
		utils.print_network(self.G)
		utils.print_network(self.D)
		print('-------------------------------------------')

		# fixed noise
		self.sample_z_ = torch.randn(self.batch_size, self.z_dim, 1, 1)
		if self.gpu_mode:
			self.sample_z_ = self.sample_z_.cuda()

	def train(self):
		self.train_hist = {}
		self.train_hist['D_loss'] = []
		self.train_hist['G_loss'] = []
		self.train_hist['per_epoch_time'] = []
		self.train_hist['total_time'] = []

		self.y_real_, self.y_fake_ = torch.ones(self.batch_size), torch.zeros(self.batch_size)
		if self.gpu_mode:
			self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

		self.D.train()
		print('Training start!')
		start_time = time.time()
		for epoch in range(1, self.epoch+1):
			self.G.train()
			epoch_start_time = time.time()
			for idx, (x_, _) in enumerate(self.data_loader):
				if idx == len(self.data_loader.dataset) // self.batch_size:
					break

				z_ = torch.randn(self.batch_size, self.z_dim, 1, 1)
				if self.gpu_mode:
					x_, z_ = x_.cuda(), z_.cuda()

				# update D network
				self.D_optimizer.zero_grad()

				D_real = self.D(x_)
				D_real_loss = self.BCE_loss(D_real, self.y_real_)

				G_ = self.G(z_)
				D_fake = self.D(G_.detach())
				D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

				D_loss = D_real_loss + D_fake_loss
				self.train_hist['D_loss'].append(D_loss.item())

				D_loss.backward()
				self.D_optimizer.step()

				# update G network
				self.G_optimizer.zero_grad()

				D_fake = self.D(G_)
				G_loss = self.BCE_loss(D_fake, self.y_real_)
				self.train_hist['G_loss'].append(G_loss.item())

				G_loss.backward()
				self.G_optimizer.step()

				if ((idx + 1) % 100) == 0:
					print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" % 
						  (epoch, (idx + 1), len(self.data_loader.dataset) // self.batch_size, D_loss.item(), G_loss.item()))

			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			with torch.no_grad():
				self.visualize_results(epoch)

			self.D_scheduler.step()
			self.G_scheduler.step()

		self.train_hist['total_time'].append(time.time() - start_time)
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
			  self.epoch, self.train_hist['total_time'][0]))
		print("Training finish!... save training results")

		self.save()
		utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
		utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

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
			samples_z_ = torch.randn(self.batch_size, self.z_dim, 1, 1)
			if self.gpu_mode:
				samples_z_ = samples_z_.cuda()

			samples = self.G(samples_z_)

		if self.gpu_mode:
			samples = samples.cpu().data
		else:
			samples = samples.data

		samples = (samples + 1) / 2
		utils.save_images(samples[:image_frame_dim * image_frame_dim], image_frame_dim,
						  self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
	
	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))