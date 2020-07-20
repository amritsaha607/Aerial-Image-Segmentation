import wandb
import numpy as np
from collections import defaultdict
# import time

from metrics.pred import predict
from utils.parameters import index2name


def pixelAccuracy(img1, img2):
    '''
        Calculates pixelwise accuracy between two images
    '''

    n = np.prod(img1.shape)
    n_match = n-np.count_nonzero(img1-img2)
    return n_match/n


def pixelConfusion(img1, img2, heatmap=False):
    '''
        Calculates pixelwise confusion matrix between 2 images
    '''

    conf = defaultdict(dict)
    for key1, val1 in index2name.items():
        mask1 = img1==key1
        conf[val1] = {val2: np.count_nonzero(mask1 & (img2==key2)) for key2, val2 in index2name.items()}

    if heatmap:
        conf = [list(val.values()) for val in conf.values()]
        conf = wandb.plots.HeatMap(index2name.keys(), index2name.keys(), conf, show_text=True)

    return conf
        


def gatherMetrics(params, metrics=['acc'], mode='val'):

    mask, y_pred = params
    mask_pred = predict(None, None, use_cache=True, params=(y_pred, False))

    logg = {}

    if 'acc' in metrics:
        logg['{}_acc'.format(mode)] = pixelAccuracy(mask_pred, mask)

    if 'conf' in metrics:
        logg['{}_conf'.format(mode)] = pixelConfusion(mask_pred, mask)

    return logg