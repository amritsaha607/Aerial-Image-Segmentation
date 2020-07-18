import numpy as np

def pixelAccuracy(img1, img2):

	n = np.prod(img1.shape)
	n_match = n-np.count_nonzero(img1-img2)
	return n_match/n
