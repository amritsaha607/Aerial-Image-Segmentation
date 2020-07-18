from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Upsample, MaxPool2d, Dropout, ConvTranspose2d, BatchNorm2d

from .blocks import DeepLayerBlock


class SegmentModel(nn.Module):

    def __init__(self, num_features=3, dim=32, n_layers=6):
        super(SegmentModel, self).__init__()
        self.num_features = num_features
        self.dim = dim

        self.conv1 = Conv2d(3, self.dim, (1, 1))
        self.relu1 = ReLU()

        self.downs, self.ups, self.dropouts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            self.downs.append(DeepLayerBlock(self.dim, flag='down'))
            if i!=n_layers-1:
                self.ups.append(DeepLayerBlock(self.dim, flag='up'))
            if i%2:
                self.dropouts.append(Dropout(0.2))

        self.final = nn.Sequential(
            Conv2d(self.dim, 3, (1, 1)),
            ReLU(),
        )

        self.bn = BatchNorm2d(num_features=self.num_features)


    def forward(self, x):
        y_conv1 = self.conv1(x)
        y_relu1 = self.relu1(y_conv1)
        y_pred = self.build(y_relu1)
        y_pred = self.final(y_pred)
        y_pred = self.bn(y_pred)
        return y_pred


    def build(self, x):
        prev_op = x
        y_downs, l_y_downs = [], []

        for down in self.downs:
            # print(prev_op.shape)
            prev_op, [l_y_down] = down(prev_op)
            y_downs.append(prev_op)
            l_y_downs.append(l_y_down)

        prev_op = l_y_downs[-1]
        for up_idx, up in enumerate(self.ups):
            # print(prev_op.shape, l_y_downs[-(up_idx+1)].shape)
            prev_op, _ = up(prev_op, params=l_y_downs[-(up_idx+1)])

        # print(prev_op.shape)

        return prev_op