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
        # print("step 0 : ", l1.max(), l1.min())
        l1 = self.conv1(l1)                 # batch X dim X h X w
        # print("step 1 : ", l1.max(), l1.min())
        if self.flag=='up':
            l1 = torch.cat([l1, params], 1) # batch X 2*dim X h X w
        # print("step 2 : ", l1.max(), l1.min())
        l1 = self.relu2(l1)                 # batch X 2*dim X h X w
        l1 = self.conv2(l1)                 # batch X 2*dim X h X w
        # print("step 3 : ", l1.max(), l1.min())
        l1 = l1+x                           # batch X 2*dim X h X w
        # print("step 4 : ", l1.max(), l1.min())
        y = self.tail(l1)                   # batch X 2*dim X 2h(h/2) X 2w(w/2)
        # print("step 5 : ", l1.max(), l1.min())
        return y, [l1]


class TreeBlock(nn.Module):

    def __init__(self, dim=256, n_tracks=32):
        super(TreeBlock, self).__init__()
        self.dim = dim
        self.n_tracks = n_tracks

        self.parallel1 = {}
        self.parallel2 = {}
        for i in range(n_tracks):
            self.parallel1['{}'.format(i+1)] = Conv2d(self.dim, 4, (1, 1))
            self.parallel2['{}'.format(i+1)] = Conv2d(4, 4, (3, 3), padding=1)

        self.tail = Conv2d(4*self.n_tracks, self.dim, (1, 1))

    def forward(self, x):
        y_pred = {key: val(x) for (key, val) in self.parallel1.items()}
        y_pred = [val(y_pred[key]) for (key, val) in self.parallel2.items()]
        y_pred = torch.cat(y_pred, dim=1)
        y_pred = self.tail(y_pred)
        y_pred += x
        return y_pred

    def cuda(self, *args, **kwargs):
        super(TreeBlock, self).cuda()
        for key in self.parallel1.keys():
            self.parallel1[key] = self.parallel1[key].cuda()
            self.parallel2[key] = self.parallel2[key].cuda()
        return self
