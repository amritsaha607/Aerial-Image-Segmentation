import sys
sys.path.append('../')

import numpy as np
import torch.nn as nn

from data.dataset import SegmentDataset
from data.transforms import transform
from data.collate import collate
from utils.vis import showImageMask
from torch.utils.data import DataLoader
from models.model import SegmentModel
from models.blocks import DeepLayerBlock
from models.utils.loss import PixelLoss
from tqdm import tqdm

batch_size = 1
num_classes = 3

model = SegmentModel(num_features=3, n_layers=6).cuda()
criterion = PixelLoss(num_classes=num_classes)

dataset = SegmentDataset(annot='../assets/train_sample.txt', transform=transform, dim=(2048, 2048))
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
)
n = len(loader)

for batch_idx, (image_path, mask_path, image, mask) in enumerate(loader):
    y_pred = model(image.cuda())
    image = image.detach().cpu()
    loss = criterion(y_pred, mask.cuda())
    print("loss : {}".format(loss.item()))
    loss.backward()
    y_pred = y_pred.detach().cpu()
    n_arr = (100*(batch_idx+1))//n
    print('[{}>{}] ({}/{}) loss : {:.4f}'.format('='*n_arr, '-'*(100-n_arr), (batch_idx+1), n, loss.item()), end='\r')

