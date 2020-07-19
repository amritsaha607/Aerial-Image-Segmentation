import numpy as np
import torch

# import time


def getMask(y_pred, softmax=False):
	'''
		Build mask from prediction of model (Single image)
		Args:
			y_pred : prediction of model (Single image) => 3, h, w
			softmax: if softmax=False, then you've to apply softmax here
		Returns:
			predicted mask
	'''

	if not softmax:
		y_pred = torch.nn.Softmax(dim=0)(y_pred)

	return torch.argmax(y_pred, dim=0)

def predict(x, model, use_cache=False, params=None):
	'''
		Build mask from input & output
		Args:
			x : input
			model : model
			use_cache : to skip forward pass and use cached prediction
			params : (y_pred, softmax) in case use_cache=True
	'''

	if use_cache:
		y_pred, softmax = params
		if not softmax:
			y_pred = torch.nn.Softmax(dim=1)(y_pred)
	else:
		y_pred = model(x)
		y_pred = torch.nn.Softmax(dim=1)(y_pred)

	# Torch argmax is slow (compared with 18 examples, torch => 23.5 secs, numpy => 1.6 secs)
	masks = torch.tensor(np.argmax(y_pred.numpy(), axis=1))
	return masks
