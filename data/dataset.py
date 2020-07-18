import glob
import os
from random import shuffle
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from utils.utils import processMask


class SegmentDataset(Dataset):

    def __init__(self, annot='assets/sample.txt', transform=None, dim=(2048, 2048)):
        self.annot = annot
        self.transform = transform
        self.dim = dim
        if isinstance(self.dim, int):
            self.dim = (dim, dim)
        lines = open(annot, 'r').read().strip().split('\n')
        shuffle(lines)
        self.image_paths = [line.split('\t')[0] for line in lines]
        self.mask_paths = [line.split('\t')[1] for line in lines]
        del lines

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path)
        mask = cv2.imread(mask_path)
        # print("b4 : ", np.unique(mask))
        mask = processMask(mask, use_path=False, bake_anomaly=True, ret='image')
        # print(np.unique(np.array(mask)))

        # print(np.array(mask).shape)
        # print(np.unique(np.array(mask)))

        if self.transform:
            image, mask = self.transform(image, mask, dim=self.dim)

        return (image_path, mask_path, image, mask)

