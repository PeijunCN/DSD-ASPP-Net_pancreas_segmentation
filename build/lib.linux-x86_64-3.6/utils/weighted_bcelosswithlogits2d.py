####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class WeightedBCEloss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(WeightedBCEloss, self).__init__()
		self.size_average = size_average
		self.weight = weight
	def forward(self, input, target):
		input = F.softmax(input)
		weight = self.weight
		if weight is not None:
			assert  len(weight) == 2

			loss = weight[1] * (target * torch.log(input)) + weight[0] * ((1 - target) * torch.log(1-input))
		else:
			loss = target * torch.log(input) + (1 - target) * torch.log(1-input)
		loss_sum = torch.sum(loss)
		#print(loss_sum)
		return torch.neg(loss_sum)





