import torch
import torch.nn as nn


class SpatialAttention(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(SpatialAttention, self).__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, (1, 1))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x_high, x_low):
		x = self.conv(x_high)
		x = self.sigmoid(x)
		x += x_low
		# x = torch.cat([x, x_low], dim=1)
		x = torch.cat([x_high, x], dim=1)
		return x


class ChannelAttention(nn.Module):

	def __init__(self, n_channels):
		super(ChannelAttention, self).__init__()

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(n_channels, n_channels//16, 1)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Conv2d(n_channels//16, n_channels, 1)
		self.relu2 = nn.ReLU()

	def forward(self, x):
		x_branch = self.avg_pool(x)
		x_branch = self.relu1(self.fc1(x_branch))
		x_branch = self.relu2(self.fc2(x_branch))
		x += x_branch
		return x
