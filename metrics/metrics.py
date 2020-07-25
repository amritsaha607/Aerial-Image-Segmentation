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
from .utils import conf_operations, dictAdd, dictRatio



def bakeWeight(metrics_arr, weights_arr):
    '''
        Apply weights of different metrics & return final metrics
    '''
    tot_metrics = dictAdd(metrics_arr, weights=weights_arr)
    tot_weights = dictAdd(weights_arr)
    metrics = dictRatio(tot_metrics, tot_weights)
    return metrics



def pixelAccuracy(img1, img2, get_weights=False):
    '''
        Calculates pixelwise accuracy between two images
    '''

    n = np.prod(img1.shape)
    n_match = n-np.count_nonzero(img1-img2)
    if get_weights:
        return n_match/n, n
    return n_match/n


def pixelConfusion(img1, img2, mode='val', 
    splits=False, heatmap=None, debug=False, i2n=index2name, get_weights=False):
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
    prec, rec, f1, acc = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    if splits:
        ret_dict = {}
        if get_weights:
            ret_dict_weights = {}
            prec_weights, rec_weights, acc_weights = defaultdict(float), defaultdict(float), defaultdict(float)

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

            prec_ = tp_/(tp_+fp_) if (tp_+fp_)>0 else 0
            rec_ = tp_/(tp_+fn_) if (tp_+fn_)>0 else 0
            f1_den = (tp_+(fp_+fn_)/2)

            prec['{}_prec_{}'.format(mode, key)] = prec_
            rec['{}_rec_{}'.format(mode, key)] = rec_
            f1['{}_F1_{}'.format(mode, key)] = 1/f1_den if f1_den>0 else 0
            acc['{}_acc_{}'.format(mode, key)] = (tp_+tn_)/tot if tot>0 else 0

            if get_weights:
                prec_weights['{}_prec_{}'.format(mode, key)] = (tp_+fp_) if (tp_+fp_)>0 else 0
                rec_weights['{}_rec_{}'.format(mode, key)] = (tp_+fn_) if (tp_+fn_)>0 else 0
                f1_weights['{}_F1_{}'.format(mode, key)] = f1_den if f1_den>0 else 0
                acc_weights['{}_acc_{}'.format(mode, key)] = tot if tot>0 else 0


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
        ret_dict.update(f1)
        ret_dict.update(acc)
        if get_weights:
            ret_dict_weights.update(prec_weights)
            ret_dict_weights.update(rec_weights)
            ret_dict_weights.update(f1_weights)
            ret_dict_weights.update(acc_weights)
            return conf, (ret_dict, ret_dict_weights)
        return conf, ret_dict

    return conf


def getProbabilityConf(y, y_pred, is_softmax=False, ret_type=None, i2n=index2name, get_weights=False):
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
    if get_weights:
        weight = defaultdict(dict)

    for (key1, val1) in i2n.items():

        y_gt_mask = y==key1
        for (key2, val2) in i2n.items():
            y_pred_val = (y_pred[:, key2, :, :][y_gt_mask]).numpy()
            avg_prob = np.mean(y_pred_val)
            conf[val1][val2] = avg_prob
            if get_weights:
                weight[val1][val2] = np.prod(y_pred_val.shape)

    if ret_type is not None:
        conf = conf_operations(conf, ret_type=ret_type, debug=False, i2n=i2n)

    if get_weights:
        return conf, weight
    return conf



def gatherMetrics(params, metrics=['acc'], 
    mode='val', debug=False, i2n=index2name, get_weights=False):

    mask, y_pred = params
    if isinstance(y_pred, list):
        y_pred = torch.cat(y_pred, dim=0)
    if isinstance(mask, list):
        mask = torch.cat(mask, dim=0)

    y_pred = torch.nn.Softmax(dim=1)(y_pred)
    mask_pred = predict(None, None, use_cache=True, params=(y_pred, True))

    logg = {}
    if get_weights:
        weights = {}

    if 'acc' in metrics:
        acc = pixelAccuracy(mask, mask_pred, get_weights=get_weights)
        if get_weights:
            acc, acc_weight = acc
            weights['{}_acc'.format(mode)] = acc_weight
        logg['{}_acc'.format(mode)] = acc

    if 'prob_conf' in metrics:
        prob_conf = getProbabilityConf(
            mask, y_pred, is_softmax=True, 
            ret_type=None, i2n=i2n, get_weights=get_weights
        )
        if get_weights:
            prob_conf, prob_conf_weight = prob_conf
            weights['{}_prob_conf'.format(mode)] = prob_conf_weight
        logg['{}_prob_conf'.format(mode)] = prob_conf

    if 'conf' in metrics:
        if get_weights:
            heatmap = None
        elif mode=='eval':
            heatmap = True
        else:
            heatmap = 'image'

        if 'splits' in metrics:
            conf, ret_dict = pixelConfusion(
                mask, mask_pred, mode=mode, 
                splits=True, heatmap=heatmap, debug=debug, i2n=i2n, get_weights=get_weights,
            )
            if get_weights:
                ret_dict, ret_dict_weights = ret_dict
                weights.update(ret_dict_weights)
            logg.update(ret_dict)
        else:
            conf = pixelConfusion(
                mask, mask_pred, mode=mode, 
                splits=False, heatmap=heatmap, debug=debug, i2n=i2n,
            )

        logg['{}_conf'.format(mode)] = conf


    if get_weights:
        return logg, weights

    return logg