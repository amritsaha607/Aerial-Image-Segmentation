import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelLoss(nn.Module):

	def __init__(self, num_classes=3):
		super(PixelLoss, self).__init__()
		self.num_classes = num_classes

	def forward(self, y_pred, y, weights=None):
		'''
			y_pred 	: batch, n_classes, h, w
			y 		: batch, 1, h, w
		'''

		y_pred = y_pred.view(-1, self.num_classes)
		y = y.view(-1)
		print(y_pred.shape, y.shape)
		print(torch.unique(y))
		print("is it?")
		loss = F.cross_entropy(y_pred, y)
		print("done..")
		return loss