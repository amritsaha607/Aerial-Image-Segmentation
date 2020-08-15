import torch
from .utils import splitImage

def collate(data):
    '''
        Args:
            0: image paths
            1: mask paths
            2: images
            3: masks
    '''
    batch_size = len(data)

    if batch_size==1 and isinstance(data[0], dict):
        images = splitImage(data[0]['data'][2], use_path=False, wh=data[0]['split'], mode='image')
        masks = splitImage(data[0]['data'][3], use_path=False, wh=data[0]['split'], mode='mask')
        n = len(images)
        image_paths = [data[0]['data'][0]]*n
        mask_paths = [data[0]['data'][1]]*n

    else:
        image_paths = [data[i][0] for i in range(batch_size)]
        mask_paths = [data[i][1] for i in range(batch_size)]
        images = torch.stack([data[i][2] for i in range(batch_size)])
        masks = torch.stack([data[i][3] for i in range(batch_size)])

    return (image_paths, mask_paths, images, masks)
