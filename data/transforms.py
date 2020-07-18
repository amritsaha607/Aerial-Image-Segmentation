import numpy as np
import torch
from torchvision import transforms

def transform(img, mask=None, mode='train', dim=(2000, 2000)):
    W, H = dim
    transformation_img = transforms.Compose([
        transforms.Resize((W, H)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225]
        # ),
    ])
    img = transformation_img(img)
    if mask:
        transformation_mask = transforms.Compose([
            transforms.Resize((W, H)),
        ])
        mask = np.array(transformation_mask(mask))
        mask = torch.LongTensor(mask)
    return img, mask

