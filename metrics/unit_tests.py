import torch
import numpy as np
from .metrics import pixelAccuracy, pixelConfusion, getProbabilityConf
from utils.parameters import *

################################

def matchVal(v1, v2):
	return (v1-v2)<v1*1e-4

def matchArr(l1, l2):
	l1, l2 = list(l1), list(l2)
	for i in range(l1):
		if not matchVal(l1[i], l2[i]):
			return False
	return True

def matchDict(d1, d2):
	if d1.keys()!=d2.keys():
		return False
	for key in d1.keys():
		if isinstance(d1[key], dict):
			if not matchDict(d1[key], d2[key]):
				return False
		elif isinstance(d1[key], list):
			if not matchArr(d1[key], d2[key]):
				return False
		else:
			if not matchVal(d1[key], d2[key]):
				return False
	return True


################################

def test_case_getProbabilityConf(test_id=1):

	if test_id==1:
		mask = torch.LongTensor(np.array([[[0, 1], [0, 1]]]))
		y_prob = torch.FloatTensor(np.array([[
			[[0.8, 0.2], 	[0.42, 0.58]],
			[[0.69, 0.31], 	[0.11, 0.89]],
		]])).permute(0, 3, 1, 2)
		conf = dict(getProbabilityConf(mask, y_prob, is_softmax=True, ret_type=None, i2n=index2name_street))
		conf_ = {
			'Background': {'Background': 0.745, 'Street': 0.255},
            'Street': {'Background': 0.265, 'Street': 0.735},
        }
	
	if not matchDict(conf, conf_):
		return False
	return True