import wandb
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
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


def pixelConfusion(img1, img2, heatmap=None, debug=False):
    '''
        Calculates pixelwise confusion matrix between 2 images
    '''

    conf = defaultdict(dict)
    for key1, val1 in index2name.items():
        mask1 = img1==key1
        conf[val1] = {val2: np.count_nonzero(mask1 & (img2==key2)) for key2, val2 in index2name.items()}

    if heatmap:
        conf = [list(val.values()) for val in conf.values()]
        if heatmap=='image':
            df_cm = pd.DataFrame(conf, index=index2name.values(), columns=index2name.values())
            conf = plt.figure(figsize=(8, 8))
            sn.heatmap(df_cm, annot=True)
            plt.close()
            if not debug:
                conf = wandb.Image(conf)
        else:
            conf = wandb.plots.HeatMap(index2name.values(), index2name.values(), conf, show_text=True)

    return conf


def gatherMetrics(params, metrics=['acc'], mode='val', debug=False):

    mask, y_pred = params
    mask_pred = predict(None, None, use_cache=True, params=(y_pred, False))

    logg = {}

    if 'acc' in metrics:
        logg['{}_acc'.format(mode)] = pixelAccuracy(mask, mask_pred)

    if 'conf' in metrics:
        if mode=='eval':
            heatmap = True
        else:
            heatmap = 'image'
        logg['{}_conf'.format(mode)] = pixelConfusion(mask, mask_pred, heatmap=heatmap, debug=debug)

    return logg