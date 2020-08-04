# import sys
# sys.path.append('../')

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
from models.utils.loss import PixelLoss
from metrics.metrics import pixelAccuracy, gatherMetrics
from metrics.pred import predict, getMask
from utils.vis import showPredictions
from utils.decorators import timer
from utils.parameters import *
from metrics.metrics import bakeWeight
from metrics.utils import conf_operations


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
parser.add_argument('--cont', type=int, default=None, help='to continue training from a specific epoch')
parser.add_argument('--wid', type=str, default=None, help='For continuing runs, provide the id of wandb run')
parser.add_argument(
    '--BEST_VAL_LOSS', 
    type=float, default=None, 
    help="For continuing runs, provide the best loss that you've found till now",
)
args = parser.parse_args()

version = args.version
cont = args.cont
wid = args.wid

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

n_epoch = int(all_configs['n_epoch'])
train_annot = all_configs['train_annot']
val_annot = all_configs['val_annot']

H = all_configs['H'] if 'H' in all_configs else 2048
W = all_configs['W'] if 'W' in all_configs else 2048

model = all_configs['model'] if 'model' in all_configs else None
n_segment_layers = all_configs['n_segment_layers'] if 'n_segment_layers' in all_configs else None
tail = all_configs['tail'] if 'tail' in all_configs else None
pretrained = all_configs['pretrained'] if 'pretrained' in all_configs else None

optimizer = all_configs['optimizer']
lr = float(all_configs['lr'])
weight_decay = float(all_configs['weight_decay'])
adam_eps = float(all_configs['adam_eps'])
amsgrad = all_configs['amsgrad']

CHCEKPOINT_DIR = all_configs['CHCEKPOINT_DIR']
ckpt_dir = os.path.join(CHCEKPOINT_DIR, version)

vis_batch = all_configs['vis_batch'] if ('vis_batch' in all_configs) else 4 
metric_batch = all_configs['metric_batch'] if ('metric_batch' in all_configs) else None
use_augmentation = all_configs['use_augmentation']
loss_weights, hnm = None, None

if 'hnm' in all_configs:
    hnm = float(all_configs['hnm'])

if 'loss_weights' in all_configs:
    loss_weights = torch.FloatTensor(all_configs['loss_weights'])

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

if model=='unet':
    model = UNet(n_channels=3, n_classes=num_classes).cuda()
else:
    model = SegmentModel(num_features=num_classes, n_layers=n_segment_layers).cuda()
criterion = PixelLoss(num_classes=num_classes, loss_weights=loss_weights, hnm=hnm)

if cont is not None:
    cont = int(cont)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'latest_{}.pth'.format(cont))))

if optimizer=='adam':
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, weight_decay=weight_decay, eps=adam_eps, amsgrad=amsgrad
    )

scheduler = None
train_losses, val_losses = [], []
if 'scheduler' in all_configs:
    sch_factor = all_configs['scheduler']
    lr_lambda = lambda epoch: sch_factor**epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_set = SegmentDataset(
    annot=train_annot, 
    transform=transform, 
    dim=(H, W), 
    c2i=color2index
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
)
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
print("pred_fig_indices : ", pred_fig_indices)

@timer
def train(epoch, loader, optimizer, metrics=[]):
    n = len(loader)
    tot_loss, loss_count = 0.0, 0
    masks = []
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
        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

        train_losses.append(loss.item())
        if (('pred' in metrics) and len(metrics)>1) or (not('pred' in metrics) and len(metrics)):
            y_preds.append(y_pred)
            masks.append(mask)

        if 'pred' in metrics:
            if batch_idx in pred_fig_indices:
                vis_img.append(image)
                vis_mask.append(mask)
                vis_y_pred.append(y_pred)

        n_arr = (50*(batch_idx+1))//n
        progress = 'Training : [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}'.format(
            '='*n_arr, '-'*(50-n_arr), (batch_idx+1), n, loss.item(), tot_loss/loss_count)
        # if 'acc' in metrics:
        #     progress = '{}, acc : {:.4f}, avg_acc : {:.4f}'.format(progress, acc, tot_acc/(batch_idx+1))
        print(progress, end='\r')

    print("\n")
    logg = {
        'training_loss': tot_loss/loss_count,
    }

    # Metrics
    if (('pred' in metrics) and len(metrics)>1) or (not('pred' in metrics) and len(metrics)):
        masks = torch.cat(masks, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        logg_metrics = gatherMetrics(
            params=(masks, y_preds),
            metrics=metrics,
            mode='train',
            i2n=index2name,
        )
        if 'train_prob_conf' in logg_metrics:
            logg_metrics['train_prob_conf'] = conf_operations(
                logg_metrics['train_prob_conf'], 
                ret_type='image', debug=False, i2n=index2name,
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
            use_path=False, ret='fig', debug=False, size='auto',
            getMatch=True,
        )
        logg.update({'train_prediction': wandb.Image(pred_fig)})

    return logg


@timer
def validate(epoch, loader, optimizer, metrics=[]):
    n = len(loader)
    tot_loss, loss_count = 0.0, 0
    masks = []
    y_preds = []
    if 'pred' in metrics:
        vis_img, vis_mask, vis_y_pred = [], [], []
    if metric_batch is not None:
        metrics_arr, weights_arr = [], []

    model.eval()
    for batch_idx, (_, _, image, mask) in enumerate(loader):
        y_pred = model(image.cuda()).detach().cpu()
        image = image.detach().cpu()
        loss = criterion(y_pred, mask)
        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

        val_losses.append(loss.item())
        y_preds.append(y_pred)
        masks.append(mask)

        if 'pred' in metrics:
            if batch_idx in pred_fig_indices:
                vis_img.append(image)
                vis_mask.append(mask)
                vis_y_pred.append(y_pred)

        if (metric_batch is not None) and (1+batch_idx)%metric_batch==0 or (1+batch_idx)==n:
            logg_metrics, weights = gatherMetrics(
                params=(masks, y_preds),
                metrics=metrics,
                mode='val',
                i2n=index2name,
                get_weights=True,
            )
            metrics_arr.append(logg_metrics)
            weights_arr.append(weights)

            y_preds, masks = [], []

        n_arr = (50*(batch_idx+1))//n
        progress = 'Validation : [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}'.format(
            '='*n_arr, '-'*(50-n_arr), (batch_idx+1), n, loss.item(), tot_loss/loss_count)
        # if 'acc' in metrics:
        #     progress = '{}, acc : {:.4f}, avg_acc : {:.4f}'.format(progress, acc, tot_acc/(batch_idx+1))
        print(progress, end='\r')

    print("\n")
    logg = {
        'val_loss': tot_loss/loss_count,
    }

    # Metrics
    if metric_batch is not None:
        # Calculate metrics from fly array
        logg_metrics = bakeWeight(metrics_arr, weights_arr)
        logg_metrics['val_conf'] = conf_operations(
            logg_metrics['val_conf'], 
            ret_type='image', debug=False, i2n=index2name,
        )
        logg_metrics['val_prob_conf'] = conf_operations(
            logg_metrics['val_prob_conf'], 
            ret_type='image', debug=False, i2n=index2name,
        )

    else:
        masks = torch.cat(masks, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        logg_metrics = gatherMetrics(
            params=(masks, y_preds),
            metrics=metrics,
            mode='val',
            i2n=index2name,
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
            use_path=False, ret='fig', debug=False, size='auto',
            getMatch=True,
        )
        logg.update({'val_prediction': wandb.Image(pred_fig)})

    return logg


def run():

    run_name = 'train_{}'.format(version)
    if cont is not None:
        wandb.init(id=wid, name=run_name, project="Street Segmentation", dir='/content/wandb/', resume=True)
    else:
        wandb.init(name=run_name, project="Street Segmentation", dir='/content/wandb/')
    wandb.watch(model, log='all')
    config = wandb.config

    config.version = version
    config.batch_size = batch_size
    config.num_classes = num_classes
    if config.num_classes==2:
        config.feature = ftr
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
    config.hnm = hnm
    config.loss_weights = loss_weights
    config.vis_batch = vis_batch
    config.log_interval = 1

    BEST_VAL_LOSS = float('inf')
    if cont:
        BEST_VAL_LOSS = float(args.BEST_VAL_LOSS) if args.BEST_VAL_LOSS is not None else float('inf')
        if scheduler:
            print("Setting up scheduler to continuing state...\n")
            for i in range(cont):
                scheduler.step()
    print("BEST_VAL_LOSS : ", BEST_VAL_LOSS)

    epoch_start = (cont+1) if cont is not None else 1
    for epoch in range(epoch_start, n_epoch+1):
        print("Epoch {}".format(epoch))
        logg_train = train(epoch, train_loader, optimizer, metrics=['pred'])
        logg_val = validate(epoch, val_loader, optimizer, metrics=['acc', 'conf', 'prob_conf', 'splits', 'pred'])
        
        # Save checkpoint
        os.system('rm {}'.format(os.path.join(ckpt_dir, 'latest*')))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'latest_{}.pth'.format(epoch)))
        if logg_val['val_loss']<BEST_VAL_LOSS:
            BEST_VAL_LOSS = logg_val['val_loss']
            os.system('rm {}'.format(os.path.join(ckpt_dir, 'best*')))
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_{}.pth'.format(epoch)))

        # Apply scheduler
        if scheduler:
            scheduler.step()
        #     if epoch>5:
        #         # Apply lr scheduler if training loss isn't decreasing since last 4 epochs
        #         if train_losses[-1]>=train_losses[-4]:
        #             print("applying scheduler")
        #             scheduler.step()

        # Logg results on wandb
        logg = {}
        logg.update(logg_train)
        logg.update(logg_val)
        wandb.log(logg)


if __name__=='__main__':
    run()
