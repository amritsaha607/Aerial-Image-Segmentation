import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Upsample

class DeepLayerBlock(nn.Module):

    def __init__(self, dim=32, flag='down'):
        super(DeepLayerBlock, self).__init__()
        self.flag = flag
        self.dim = dim

        self.relu1 = ReLU()
        self.conv1 = Conv2d(self.dim, self.dim, (1, 1))
        self.relu2 = ReLU()
        if self.flag=='up':
            self.conv2 = Conv2d(2*self.dim, self.dim, (1, 1))
            self.tail = Upsample(scale_factor=2)
        else:
            self.conv2 = Conv2d(self.dim, self.dim, (1, 1))
            self.tail = MaxPool2d((2, 2))

    def forward(self, x, params=None):
        l1 = self.relu1(x)                  # batch X dim X h X w
        l1 = self.conv1(l1)                 # batch X dim X h X w
        if self.flag=='up':
            l1 = torch.cat([l1, params], 1) # batch X 2*dim X h X w
        l1 = self.relu2(l1)                 # batch X 2*dim X h X w
        l1 = self.conv2(l1)                 # batch X 2*dim X h X w
        l1 = l1+x                           # batch X 2*dim X h X w
        y = self.tail(l1)                   # batch X 2*dim X 2h(h/2) X 2w(w/2)
        return y, [l1]
