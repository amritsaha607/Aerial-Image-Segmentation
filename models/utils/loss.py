import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelLoss(nn.Module):

	def __init__(self, num_classes=3, loss_weights=None):
		super(PixelLoss, self).__init__()
		self.num_classes = num_classes
		self.loss_weights = loss_weights

	def forward(self, y_pred, y, weights=None):
		'''
			y_pred 	: batch, n_classes, h, w
			y 		: batch, 1, h, w
		'''

		y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
		y = y.view(-1)
		if self.loss_weights is not None:
			if y.is_cuda:
				loss = F.cross_entropy(y_pred, y, self.loss_weights.cuda())
			else:
				loss = F.cross_entropy(y_pred, y, self.loss_weights.detach().cpu())
		else:
			loss = F.cross_entropy(y_pred, y)
		return loss