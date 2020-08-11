####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class DiceLoss2d(nn.Module):
	def __init__(self, gamma=0, weight=None, size_average=True):
		super(DiceLoss2d, self).__init__()
		self.size_average = size_average

	def forward(self, input, target):

		#N = target.size(0)
		#smooth = 0.000001
		#input = F.sigmoid(input)

		#input_flat = input.view(N, -1)
		#target_flat = target.view(N, -1)

		#intersection = input_flat*target_flat

		#loss = 2*(intersection.sum(1))/(input_flat.sum(1)+target_flat.sum(1)+smooth)

		#loss = 1-loss.sum()/N

		N = target.size(0)
		smooth = 1
		input = F.sigmoid(input)
		loss = 0
		for i in range(N):
			input_flat = input[i-1].view(1, -1)
			target_flat = target[i-1].view(1, -1)

			intersection = input_flat * target_flat

			dice = 2 * (intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1) + smooth)

			input_flat = 1 - input_flat
			target_flat = 1 - target_flat
			intersection = input_flat * target_flat

			dice2 = 2 * (intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
			dice = dice*0.7+dice2*0.3

			loss = 1 - dice.sum() + loss

		loss = loss/N
		return loss


