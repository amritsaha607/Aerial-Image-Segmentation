import matplotlib.pyplot as plt
import wandb
import pandas as pd
import seaborn as sn
from utils.parameters import index2name


def listAdd(lists):
    '''
        Add corresponding values of list of lists
    '''
    res = np.array([0]*len(lists[0]))
    lists = np.array(lists)
    for elem in lists:
        res += elem
    return res

def dictAdd(ld, weights=None):
    '''
        Add corresponding keys of list of dicts
        no matter how nested the dict is
        Args:
            ld : List of dicts
            weights : for weighted sum
        Returns:
            res : Final dict with added values
    '''
    n = len(ld)
    keys = ld[0].keys()
    res = {}

    if weights is not None:
        for i in range(n):
            ld[i] = dictMultiply(ld[i], weights[i])

    for key in keys:
        if isinstance(ld[0][key], dict):
            res[key] = dictAdd([ld[i][key] for i in range(n)])
        elif isinstance(ld[0][key], list):
            res[key] = listAdd([ld[i][key] for i in range(n)])
        else:
            res[key] = sum([ld[i][key] for i in range(n)])

    return res

def dictMultiply(d1, d2):
    '''
        in case any key is not found in d2, the d1 value will be kept in res
    '''
    keys = d1.keys()
    res = {}
    for key in keys:
        if not key in d2.keys():
            res[key] = d1[key]
        elif isinstance(d1[key], dict):
            res[key] = dictMultiply(d1[key], d2[key])
        elif isinstance(d1[key], list):
            res[key] = np.array(d1[key])*np.array(d2[key])
        else:
            res[key] = d1[key]*d2[key]
    return res



def dictRatio(d1, d2):
    '''
        Remember d1/d2 , not the other way around
        in case any key is not found in d2, the d1 value will be kept in res
    '''
    keys = d1.keys()
    res = {}
    for key in keys:
        if not key in d2.keys():
            res[key] = d1[key]
        elif isinstance(d1[key], dict):
            res[key] = dictRatio(d1[key], d2[key])
        elif isinstance(d1[key], list):
            res[key] = np.array(d1[key])/np.array(d2[key])
        else:
            res[key] = d1[key]/d2[key] if d2[key]!=0 else 0
    return res


def conf_operations(conf, ret_type=None, debug=False, i2n=index2name, size=(8, 8)):
    '''
        Args:
            conf : confusion matrix (dict of dict)
            ret_type : Return type
                        wandb Image (image)
                        wandb Heatmap (heatmap)
                        plt figure (plt_fig)
            debug:
                Set True for debugging in notebook
    '''

    if ret_type=='image' or ret_type=='plt_fig' or ret_type=='heatmap' or debug:

        if ret_type=='heatmap':
            conf = [list(c.values()) for c in conf.values()]
            conf = wandb.plots.HeatMap(i2n.values(), i2n.values(), conf, show_text=True)
            return conf

        conf = [list(val.values()) for val in conf.values()]
        df_cm = pd.DataFrame(conf, index=i2n.values(), columns=i2n.values())
        conf = plt.figure(figsize=size)
        sn.heatmap(df_cm, annot=True)
        plt.close()

        if ret_type=='plt_fig' or debug:
            return conf

        elif ret_type=='image':
            conf = wandb.Image(conf)

    else:
        raise ValueError("Unknown return type {}".format(ret_type))

    return conf