# import sys
# sys.path.append('../')

import os
import argparse
import yaml
import wandb
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data.dataset import SegmentDataset
from data.transforms import transform
from data.collate import collate
from torch.utils.data import DataLoader
from models.model import SegmentModel
from models.utils.loss import PixelLoss
from metrics.metrics import pixelAccuracy, gatherMetrics
from metrics.pred import predict, getMask
from utils.vis import showPredictions
from utils.decorators import timer



parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
args = parser.parse_args()

version = args.version
cfg_path = 'configs/{}.yml'.format(version)
all_configs = yaml.safe_load(open(cfg_path))


random_seed = int(all_configs['random_seed'])
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
loss_weights = None
if 'loss_weights' in all_configs:
    loss_weights = torch.FloatTensor(all_configs['loss_weights'])

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)


model = SegmentModel(num_features=3, n_layers=n_segment_layers).cuda()
criterion = PixelLoss(num_classes=num_classes, loss_weights=loss_weights)

if optimizer=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=adam_eps, amsgrad=amsgrad)

scheduler = None
if 'scheduler' in all_configs:
    train_losses, val_losses = [], []
    sch_factor = all_configs['scheduler']
    lr_lambda = lambda epoch: sch_factor**epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

n_batch = 2
pred_fig_indices = list(range(0, len(val_loader)-1))
random.shuffle(pred_fig_indices)
pred_fig_indices = pred_fig_indices[:n_batch]


@timer
def train(epoch, loader, optimizer, metrics=[]):
    n = len(loader)
    tot_loss = 0.0
    masks, mask_preds = [], []
    y_preds = []
    if 'pred' in metrics:
        vis_img, vis_mask, vis_y_pred = [], [], []

    model.train()
    for batch_idx, (_, _, image, mask) in enumerate(loader):
        y_pred = model(image.cuda())
        image = image.detach().cpu()
        loss = criterion(y_pred, mask.cuda())
        loss.backward()
        optimizer.step()

        y_pred = y_pred.detach().cpu()
        tot_loss += loss.item()

        train_losses.append(loss.item())
        y_preds.append(y_pred)
        masks.append(mask)

        if 'pred' in metrics:
            if batch_idx in pred_fig_indices:
                vis_img.append(image)
                vis_mask.append(mask)
                vis_y_pred.append(y_pred)

        n_arr = (50*(batch_idx+1))//n
        progress = 'Training : [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}'.format(
            '='*n_arr, '-'*(50-n_arr), (batch_idx+1), n, loss.item(), tot_loss/(batch_idx+1))
        # if 'acc' in metrics:
        #     progress = '{}, acc : {:.4f}, avg_acc : {:.4f}'.format(progress, acc, tot_acc/(batch_idx+1))
        print(progress, end='\r')

    print("\n")
    logg = {
        'training_loss': tot_loss/n,
    }

    # Metrics
    masks = torch.cat(masks, dim=0)
    y_preds = torch.cat(y_preds, dim=0)
    logg_metrics = gatherMetrics(
        params=(masks, y_preds),
        metrics=metrics,
        mode='train',
    )
    logg.update(logg_metrics)

    # Visualizations
    if 'pred' in metrics:
        vis_img = torch.cat(vis_img, dim=0)
        vis_mask = torch.cat(vis_mask, dim=0)
        vis_y_pred = torch.cat(vis_y_pred, dim=0)
        vis_mask_pred = predict(None, None, use_cache=True, params=(vis_y_pred, False))
        pred_fig = showPredictions(
            vis_img, vis_mask, vis_mask_pred, 
            use_path=False, ret='fig', debug=False, size=(8, 8)
        )
        logg.update({'train_prediction': wandb.Image(pred_fig)})

    return logg


@timer
def validate(epoch, loader, optimizer, metrics=[]):
    n = len(loader)
    tot_loss = 0.0
    masks, mask_preds = [], []
    y_preds = []
    if 'pred' in metrics:
        vis_img, vis_mask, vis_y_pred = [], [], []

    model.eval()
    for batch_idx, (_, _, image, mask) in enumerate(loader):
        y_pred = model(image.cuda()).detach().cpu()
        image = image.detach().cpu()
        loss = criterion(y_pred, mask)
        tot_loss += loss.item()

        val_losses.append(loss.item())
        y_preds.append(y_pred)
        masks.append(mask)

        if 'pred' in metrics:
            if batch_idx in pred_fig_indices:
                vis_img.append(image)
                vis_mask.append(mask)
                vis_y_pred.append(y_pred)

        n_arr = (50*(batch_idx+1))//n
        progress = 'Validation : [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}'.format(
            '='*n_arr, '-'*(50-n_arr), (batch_idx+1), n, loss.item(), tot_loss/(batch_idx+1))
        # if 'acc' in metrics:
        #     progress = '{}, acc : {:.4f}, avg_acc : {:.4f}'.format(progress, acc, tot_acc/(batch_idx+1))
        print(progress, end='\r')

    print("\n")
    logg = {
        'val_loss': tot_loss/n,
    }

    # Metrics
    masks = torch.cat(masks, dim=0)
    y_preds = torch.cat(y_preds, dim=0)
    logg_metrics = gatherMetrics(
        params=(masks, y_preds),
        metrics=metrics,
        mode='val',
    )
    logg.update(logg_metrics)

    # Visualizations
    if 'pred' in metrics:
        vis_img = torch.cat(vis_img, dim=0)
        vis_mask = torch.cat(vis_mask, dim=0)
        vis_y_pred = torch.cat(vis_y_pred, dim=0)
        vis_mask_pred = predict(None, None, use_cache=True, params=(vis_y_pred, False))
        pred_fig = showPredictions(
            vis_img, vis_mask, vis_mask_pred, 
            use_path=False, ret='fig', debug=False, size=(8, 8)
        )
        logg.update({'val_prediction': wandb.Image(pred_fig)})

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
    config.loss_weights = loss_weights
    config.log_interval = 1

    for epoch in range(1, n_epoch+1):
        print("Epoch {}".format(epoch))
        logg_train = train(epoch, train_loader, optimizer, metrics=['acc', 'conf', 'pred'])
        # logg_val = validate(epoch, val_loader, optimizer, metrics=['acc', 'conf', 'pred'])
        if scheduler:
            if epoch>5:
                # Apply lr scheduler if training loss isn't decreasing since last 4 epochs
                if train_losses[-1]>=train_losses[-4]:
                    print("applying scheduler")
                    scheduler.step()
        logg = {}
        logg.update(logg_train)
        # logg.update(logg_val)
        wandb.log(logg)


if __name__=='__main__':
    run()
