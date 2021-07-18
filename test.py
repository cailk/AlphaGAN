# -*- coding: utf-8 -*-
# @Date    : 2020-01-09
# @Author  : cailk (cailikun007@gmail.com)
# @Link    : https://github.com/cailk

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def IS_plot(IS, path='IS.png', model_name=''):
	N = len(IS)
	x = np.linspace(0, 5 * N - 5, N)

	plt.style.use('ggplot')
	plt.plot(x, IS, lw=1.0)

	plt.xlabel('Epoch')
	plt.ylabel('Inception Score')
	plt.xlim(left=0)

	plt.grid(True)
	plt.tight_layout()
	path = os.path.join(path, model_name + '_IS.png')

	plt.savefig(path)
	plt.close()

IS = [2.6905, 3.0502, 3.0990, 3.3809, 3.4097, 3.4886, 3.6513, 3.8720, 3.8531, 3.9686, 3.9826, 4.1017]

IS_plot(IS, path='./', model_name='WGAN')