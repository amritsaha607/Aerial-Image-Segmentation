import matplotlib.pyplot as plt
import wandb
import pandas as pd
import seaborn as sn
from utils.parameters import index2name

def conf_operations(conf, ret_type=None, debug=False, i2n=index2name):
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
            conf = wandb.plots.HeatMap(i2n.values(), i2n.values(), conf, show_text=True)

        conf = [list(val.values()) for val in conf.values()]
        df_cm = pd.DataFrame(conf, index=i2n.values(), columns=i2n.values())
        conf = plt.figure(figsize=(8, 8))
        sn.heatmap(df_cm, annot=True)
        plt.close()

        if ret_type=='plt_fig' or debug:
            return conf

        elif ret_type=='image':
            conf = wandb.Image(conf)

    else:
        raise ValueError("Unknown return type {}".format(ret_type))

    return conf