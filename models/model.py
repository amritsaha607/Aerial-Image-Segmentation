from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Upsample, MaxPool2d, Dropout, ConvTranspose2d, BatchNorm2d
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from .blocks import DeepLayerBlock


class SegmentModel(nn.Module):

    def __init__(self, num_features=3, dim=32, n_layers=6, tail='resnet18', pretrained=False):
        super(SegmentModel, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.tree_dim = 64

        self.conv1 = Conv2d(3, self.dim, (1, 1))
        self.relu1 = ReLU()

        self.downs, self.ups, self.dropouts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            self.downs.append(DeepLayerBlock(self.dim, flag='down'))
            if i!=n_layers-1:
                self.ups.append(DeepLayerBlock(self.dim, flag='up'))
            self.dropouts.append(Dropout(0.2))

        self.final = nn.Sequential(
            Conv2d(self.dim, self.num_features, (1, 1)),
            ReLU(),
        )

        self.bn = BatchNorm2d(num_features=self.num_features)


    def forward(self, x):
        y_conv1 = self.conv1(x)
        # print("\n\nprinting intermediates")
        # print("y_conv1 : ", y_conv1.max(), y_conv1.min())
        y_relu1 = self.relu1(y_conv1)
        # print("y_relu1 : ", y_relu1.max(), y_relu1.min())
        y_pred = self.build(y_relu1)
        # print("y_pred : ", y_pred.max(), y_pred.min())
        y_pred = self.final(y_pred)
        # print("y_pred : ", y_pred.max(), y_pred.min())
        y_pred = self.bn(y_pred)
        print("y_pred : ", y_pred.size())
        y_pred = self.tail(y_pred)
        # print("\n\ndone\n")
        return y_pred


    def build(self, x):
        prev_op = x
        y_downs, l_y_downs = [], []

        for i, down in enumerate(self.downs):
            # print(prev_op.shape)
            prev_op, [l_y_down] = down(prev_op)
            # if i==0:
            #     print("prev_op {}: ".format(i), prev_op.max(), prev_op.min())
            prev_op = self.dropouts[i](prev_op)
            y_downs.append(prev_op)
            l_y_downs.append(l_y_down)

        prev_op = l_y_downs[-1]
        for up_idx, up in enumerate(self.ups):
            # print(prev_op.shape, l_y_downs[-(up_idx+1)].shape)
            prev_op, _ = up(prev_op, params=l_y_downs[-(up_idx+1)])

        # print(prev_op.shape)

        return prev_op