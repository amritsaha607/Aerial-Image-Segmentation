import sys
sys.path.append('../')

import os
import argparse
import yaml
import wandb

import numpy as np
import torch
import torch.nn as nn

from data.dataset import SegmentDataset
from data.transforms import transform
from data.collate import collate
from torch.utils.data import DataLoader
from models.model import SegmentModel
from models.utils.loss import PixelLoss
from metrics.metrics import pixelAccuracy


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
args = parser.parse_args()

version = args.version
cfg_path = 'configs/{}.yml'.format(version)
all_configs = yaml.safe_load(open(cfg_path))

batch_size = int(all_configs['batch_size'])
num_classes = int(all_configs['num_classes'])
n_epoch = int(all_configs['n_epoch'])
train_annot = all_configs['train_annot']
val_annot = all_configs['val_annot']
n_segment_layers = all_configs['n_segment_layers']
optimizer = all_configs['optimizer']
lr = float(all_configs['lr'])
weight_decay = float(all_configs['weight_decay'])
adam_eps = float(all_configs['adam_eps'])
amsgrad = all_configs['amsgrad']
CHCEKPOINT_DIR = all_configs['CHCEKPOINT_DIR']
ckpt_dir = os.path.join(CHCEKPOINT_DIR, version)
use_augmentation = all_configs['use_augmentation']

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


model = SegmentModel(num_features=3, n_layers=n_segment_layers).cuda()
criterion = PixelLoss(num_classes=num_classes)

if optimizer=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=adam_eps, amsgrad=amsgrad)


train_set = SegmentDataset(annot=train_annot, transform=transform, dim=(2048, 2048))
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
)
val_set = SegmentDataset(annot=val_annot, transform=transform, dim=(2048, 2048))
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
)


def train(epoch, loader, optimizer):
    n = len(loader)
    tot_loss = 0.0

    model.train()
    for batch_idx, (_, _, image, mask) in enumerate(loader):
        y_pred = model(image.cuda())
        image = image.detach().cpu()
        loss = criterion(y_pred, mask.cuda())
        loss.backward()
        optimizer.step()
        y_pred = y_pred.detach().cpu()
        tot_loss += loss.item()

        n_arr = (50*(batch_idx+1))//n
        print('Training : [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}'.format(
            '='*n_arr, '-'*(50-n_arr), (batch_idx+1), n, loss.item(), tot_loss/(batch_idx+1)), end='\r')

    print("\n")
    logg = {
        'training_loss': tot_loss/n,
    }
    return logg


def validate(epoch, loader, optimizer):
    n = len(loader)
    tot_loss = 0.0

    model.eval()
    for batch_idx, (image_path, mask_path, image, mask) in enumerate(loader):
        y_pred = model(image.cuda()).detach().cpu()
        image = image.detach().cpu()
        loss = criterion(y_pred, mask)#.cuda())
        # y_pred = y_pred.detach().cpu()
        tot_loss += loss.item()

        # y_pred, _ = y_pred.max()
        # acc = pixelAccuracy(y_pred, mask)

        n_arr = (50*(batch_idx+1))//n
        print('Validation : [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}'.format(
            '='*n_arr, '-'*(50-n_arr), (batch_idx+1), n, loss.item(), tot_loss/(batch_idx+1)), end='\r')

    print("\n")
    logg = {
        'val_loss': tot_loss/n,
    }
    return logg


def run():

    run_name = 'overfit_train_{}'.format(version)
    wandb.init(name=run_name, project="Street Segmentation", dir='/content/wandb/')
    wandb.watch(model, log='all')
    config = wandb.config

    config.version = version
    config.batch_size = batch_size
    config.num_classes = num_classes
    config.n_epoch = n_epoch
    config.train_annot = train_annot
    config.val_annot = val_annot
    config.n_segment_layers = n_segment_layers
    config.optimizer = all_configs['optimizer']
    config.lr = lr
    config.weight_decay = weight_decay
    config.adam_eps = adam_eps
    config.amsgrad = amsgrad
    config.CHCEKPOINT_DIR = CHCEKPOINT_DIR
    config.use_augmentation = use_augmentation
    config.log_interval = 1

    for epoch in range(1, n_epoch+1):
        print("Epoch {}".format(epoch))
        logg_train = train(epoch, train_loader, optimizer)
        logg_val = validate(epoch, val_loader, optimizer)
        logg = {}
        logg.update(logg_train)
        logg.update(logg_val)
        wandb.log(logg)


if __name__=='__main__':
    run()
