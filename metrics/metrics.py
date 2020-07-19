import numpy as np
from metrics.pred import predict
# import time

def pixelAccuracy(img1, img2):

    n = np.prod(img1.shape)
    n_match = n-np.count_nonzero(img1-img2)
    return n_match/n


def gatherMetrics(params, metrics=['acc'], mode='val'):

    mask, y_pred = params
    mask_pred = predict(None, None, use_cache=True, params=(y_pred, False))

    logg = {}

    if 'acc' in metrics:
        logg['{}_acc'.format(mode)] = pixelAccuracy(mask_pred, mask)

    return logg