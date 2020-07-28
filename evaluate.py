# Import modules

import os
import argparse
import yaml
import wandb
import random
import time

import math
import numpy as np
import torch
import torch.nn as nn

from data.dataset import SegmentDataset
from data.transforms import transform
from data.collate import collate
from torch.utils.data import DataLoader
from models.model import SegmentModel
from models.unet import UNet
from utils.parameters import *

from eval.utils import evaluate


# Extract argparses

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to be loaded')
parser.add_argument('--thresholds', type=str, default=None, 
    help='score thresholds of different classes (write in comma seperated without spaces) [eg : 0.3,0.4,0.3 for 3 classes]')
parser.add_argument('--mode', type=str, default='run', 
    help='Mode of evaluation ("run": Run, "save": save cache predictions, "load": load cached predictions)')
args = parser.parse_args()

version = args.version

# Extract Configurations

cfg_path = 'configs/{}.yml'.format(version.replace('_', '/').replace('-', '/'))
all_configs = yaml.safe_load(open(cfg_path))

random_seed = int(all_configs['random_seed'])
batch_size = int(all_configs['batch_size'])
num_classes = int(all_configs['num_classes'])
if num_classes==2:
    ftr = all_configs['ftr']
    if ftr.lower()=='street':
        index2name = index2name_street
        color2index = color2index_street
    elif ftr.lower()=='building':
        index2name = index2name_building
        color2index = color2index_building
    else:
        raise ValueError("Unknown feature found - {}".format(ftr))

val_annot = all_configs['val_annot']

H = all_configs['H'] if 'H' in all_configs else 2048
W = all_configs['W'] if 'W' in all_configs else 2048

model = all_configs['model'] if 'model' in all_configs else None
n_segment_layers = all_configs['n_segment_layers'] if 'n_segment_layers' in all_configs else None
tail = all_configs['tail'] if 'tail' in all_configs else None
pretrained = all_configs['pretrained'] if 'pretrained' in all_configs else None

CHCEKPOINT_DIR = all_configs['CHCEKPOINT_DIR']
ckpt_dir = os.path.join(CHCEKPOINT_DIR, version)

vis_batch = all_configs['vis_batch'] if ('vis_batch' in all_configs) else 4 
metric_batch = all_configs['metric_batch'] if ('metric_batch' in all_configs) else None


# Set random seeds

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)


# Load Model

if model=='unet':
    model = UNet(n_channels=3, n_classes=num_classes).cuda()
else:
    model = SegmentModel(num_features=num_classes, n_layers=n_segment_layers).cuda()

ckpt = args.ckpt
model.load_state_dict(torch.load(os.path.join(ckpt_dir, '{}.pth'.format(ckpt))))

# Get Dataloader

val_set = SegmentDataset(
    annot=val_annot, 
    transform=transform, 
    dim=(H, W), 
    c2i=color2index
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
)

n_batch = vis_batch
pred_fig_indices = list(range(0, len(val_loader)-1))
random.shuffle(pred_fig_indices)
pred_fig_indices = pred_fig_indices[:n_batch]


if __name__=='__main__':

    epoch = ckpt.split('_')[-1]

    # Get thresholds
    thresholds = 'auto'
    if args.thresholds is not None:
        thresholds = [float(thres) for thres in args.thresholds.split(',')]

    if thresholds=='auto':
        run_name = 'eval_{}_{}_{}'.format(version, epoch, thresholds)
    else:
        run_name = 'eval_{}_{}_{}'.format(version, epoch, '_'.join([str(thr) for thr in thresholds]))
    # print("run_name : ", run_name)

    wandb.init(name=run_name, project="Street Segmentation", dir='/content/wandb/')
    config = wandb.config

    config.version = version
    config.epoch = epoch
    config.thresholds = thresholds

    params = {
        'metrics': ['acc', 'conf', 'prob_conf', 'splits', 'pred'],
        'metric_batch': metric_batch,
        'i2n': index2name,
        'pred_fig_indices': pred_fig_indices,
        'mode': args.mode,
    }
    logg = evaluate(val_loader, model, params)
    wandb.log(logg)

