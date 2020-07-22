import wandb
import numpy as np
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
# import time

from metrics.pred import predict
from utils.parameters import index2name
from .utils import conf_operations


def pixelAccuracy(img1, img2):
    '''
        Calculates pixelwise accuracy between two images
    '''

    n = np.prod(img1.shape)
    n_match = n-np.count_nonzero(img1-img2)
    return n_match/n


def pixelConfusion(img1, img2, mode='val', splits=False, heatmap=None, debug=False, i2n=index2name):
    '''
        Calculates pixelwise confusion matrix between 2 images
        Args:
            heatmap :   None for no heatmap
                        image for wandb image
                        else wandb heatmap
    '''

    conf = defaultdict(dict)
    for key1, val1 in i2n.items():
        mask1 = img1==key1
        conf[val1] = {val2: np.count_nonzero(mask1 & (img2==key2)) for key2, val2 in i2n.items()}

    tp, fp, tn, fn = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    prec, rec, acc = defaultdict(float), defaultdict(float), defaultdict(float)
    if splits:
        ret_dict = {}
        tot = np.prod(img1.shape)
        for key in conf.keys():
            tp_ = conf[key][key]
            fp_ = sum([conf[k][key] for k in conf.keys() if k!=key])
            fn_ = sum([conf[key][k] for k in conf.keys() if k!=key])
            tn_ = tot - tp_ - fp_ - fn_

            tp['{}_tp_{}'.format(mode, key)] = tp_
            fp['{}_fp_{}'.format(mode, key)] = fp_
            fn['{}_fn_{}'.format(mode, key)] = fn_
            tn['{}_tn_{}'.format(mode, key)] = tn_

            prec['{}_prec_{}'.format(mode, key)] = tp_/(tp_+fp_) if (tp_+fp_)>0 else 0
            rec['{}_rec_{}'.format(mode, key)] = tp_/(tp_+fn_) if (tp_+fn_)>0 else 0
            acc['{}_acc_{}'.format(mode, key)] = (tp_+tn_)/tot if tot>0 else 0


    if heatmap:
        conf = conf_operations(conf, ret_type=heatmap, debug=False, i2n=i2n)
        # conf = [list(val.values()) for val in conf.values()]
        # if heatmap=='image':
        #     df_cm = pd.DataFrame(conf, index=i2n.values(), columns=i2n.values())
        #     conf = plt.figure(figsize=(8, 8))
        #     sn.heatmap(df_cm, annot=True)
        #     plt.close()
        #     if not debug:
        #         conf = wandb.Image(conf)
        # else:
        #     conf = wandb.plots.HeatMap(i2n.values(), i2n.values(), conf, show_text=True)

    if splits:
        ret_dict.update(prec)
        ret_dict.update(rec)
        ret_dict.update(acc)
        return conf, ret_dict

    return conf


def getProbabilityConf(y, y_pred, is_softmax=False, ret_type=None, i2n=index2name):
    '''
        Gives confusion matrix that shows average probability of one class being predicted as same/different class
        Args:
            y : ground truth masks          =>      (batch, h, w)
            y_pred : Prediction by model    =>      (batch, n_classes, h, w)
            is_softmax : Set to True if softmax is already done
            ret_type :      None => dict
                            plt_fig => plt figure
                            image => wandb image
                            heatmap => wandb heatmap
    '''

    if not is_softmax:
        y_pred = torch.nn.Softmax(dim=1)(y_pred)

    conf = defaultdict(dict)

    for (key1, val1) in i2n.items():
        y_gt_mask = y==key1
        for (key2, val2) in i2n.items():
            y_pred_val = y_pred[:, key2, :, :][y_gt_mask]
            avg_prob = np.mean(y_pred_val.numpy())
            conf[val1][val2] = avg_prob

    if ret_type is not None:
        conf = conf_operations(conf, ret_type=ret_type, debug=False, i2n=i2n)

    return conf



def gatherMetrics(params, metrics=['acc'], mode='val', debug=False, i2n=index2name):

    mask, y_pred = params
    y_pred = torch.nn.Softmax(dim=1)(y_pred)
    mask_pred = predict(None, None, use_cache=True, params=(y_pred, True))

    logg = {}

    if 'acc' in metrics:
        logg['{}_acc'.format(mode)] = pixelAccuracy(mask, mask_pred)

    if 'conf' in metrics:
        if mode=='eval':
            heatmap = True
        else:
            heatmap = 'image'

        splits = False
        if 'splits' in metrics:
            conf, ret_dict = pixelConfusion(mask, mask_pred, mode=mode, splits=True, heatmap=heatmap, debug=debug, i2n=i2n)
            logg.update(ret_dict)
        else:
            conf = pixelConfusion(mask, mask_pred, mode=mode, splits=False, heatmap=heatmap, debug=debug, i2n=i2n)

        logg['{}_conf'.format(mode)] = conf

    if 'prob_conf' in metrics:
        prob_conf = getProbabilityConf(mask, y_pred, is_softmax=True, ret_type='image', i2n=i2n)
        logg['{}_prob_conf'.format(mode)] = prob_conf

    return logg