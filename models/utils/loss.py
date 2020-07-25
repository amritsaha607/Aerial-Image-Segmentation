import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelLoss(nn.Module):

	def __init__(self, num_classes=3, loss_weights=None, hnm=None):
		'''
			hnm : Hard negative mining factor
						None => no hard negative mining
						int  => factor of hrd negative mining
		'''

		super(PixelLoss, self).__init__()
		self.num_classes = num_classes
		self.loss_weights = loss_weights
		self.hnm = hnm

	def forward(self, y_pred, y):
		'''
			y_pred 	: batch, n_classes, h, w
			y 		: batch, 1, h, w
		'''

		eps = 1e-6

		y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
		y_pred += eps
		y = y.view(-1)

		if self.hnm is not None:
			y, y_pred = self.hard_negative_mining(y, y_pred, factor=self.hnm, ret_mask=False)

		if self.loss_weights is not None:
			if y.is_cuda:
				loss = F.cross_entropy(y_pred, y, self.loss_weights.cuda())
			else:
				loss = F.cross_entropy(y_pred, y, self.loss_weights.detach().cpu())
		else:
			loss = F.cross_entropy(y_pred, y)
		return loss

	def hard_negative_mining(self, y, y_pred, factor=1, ret_mask=False):
		'''
			In case number of negative examples are too large as compared to positive,
			apply hard negative mining. Consider (factor * #(+ve)) number of negative examples
			instead of all negative examples to calculate loss.

			Args:
				y, y_pred: parameters
				factor : #neg/#pos needed
				ret_mask : To return mask or masked values
		'''

		mask_neg = torch.nonzero(y==0)
		mask_pos = torch.nonzero(y!=0)
		n_neg, n_pos = len(mask_neg), len(mask_pos)
		n_neg_need = int(n_pos*factor)
		mask_mask_neg = torch.randperm(len(mask_neg))[:n_neg_need]
		mask_neg = mask_neg[mask_mask_neg]
		mask = (y!=0)
		mask[mask_neg] = True
		if ret_mask:
			return mask
		else:
			return y[mask], y_pred[mask]