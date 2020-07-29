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

def predict(x, model, use_cache=False, params=None, thres=None):

    if use_cache:
        y_pred, softmax = params
        if not softmax:
            y_pred = torch.nn.Softmax(dim=1)(y_pred)
    else:
        y_pred = model(x)
        y_pred = torch.nn.Softmax(dim=1)(y_pred)

    if thres is None or thres=='auto':
        # Torch argmax is slow (compared with 18 examples, torch => 23.5 secs, numpy => 1.6 secs)
        masks = torch.tensor(np.argmax(y_pred.numpy(), axis=1))
    else:
        masks = np.zeros((y_pred.size(0), y_pred.size(2), y_pred.size(3)))
        for cls_ in reversed(list(thres.keys())):
            masks[y_pred[:, cls_, :, :] > thres[cls_]] = cls_
        masks = torch.LongTensor(masks)

    return masks
