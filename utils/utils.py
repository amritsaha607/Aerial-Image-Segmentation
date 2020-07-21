from PIL import Image
import matplotlib.pyplot as plt
from .parameters import color2index
import numpy as np
import cv2

def processMask(mask, color2index=color2index, use_path=False, bake_anomaly=True, ret='image'):
    '''
        Removes all anomalies from the mask
        Args:
            masks: (h, w, 3)
            use_path: To use path of mask
            ret : image => returns PIL image
                    arr => return numpy array
    '''
    if use_path:
        mask = cv2.imread(mask)

    # Put labels
    for k in color2index:
        mask[(mask == k).all(axis=2)] = color2index[k]
    mask = mask[:, :, 0]

    # Set all the anomalies to background
    if bake_anomaly:
        anomaly_mask = np.logical_and.reduce([mask!=val for _, val in color2index.items()])
        mask[anomaly_mask] = 0

    if ret=='image':
        return Image.fromarray(mask)
    else:
        return mask


def pointAnomaly(mask, use_path=True, allow=[0, 255]):
    '''
        Shows if the mask contains abnormal numbers
    '''
    
    if use_path:
        mask = cv2.imread(mask)
    all_val = set(np.unique(mask))-set(allow)
    mask_anomaly = np.logical_and((mask!=allow[0]), (mask!=allow[1]))
    return mask_anomaly.astype(np.int)


'''
def changeAnomaly(mask, use_path=True, allow=[0, 255], change_to=0):
    if use_path:
        mask = cv2.imread(mask)
    mask_mask = np.logical_and(mask!=allow[0], mask!=allow[1])
    mask[mask_mask] = 0
    return mask
'''