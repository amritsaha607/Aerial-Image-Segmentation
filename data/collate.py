import torch

def collate(data):
    '''
        Args:
            0: image paths
            1: mask paths
            2: images
            3: masks
    '''
    batch_size = len(data)

    image_paths = [data[i][0] for i in range(batch_size)]
    mask_paths = [data[i][1] for i in range(batch_size)]
    images = torch.stack([data[i][2] for i in range(batch_size)])
    masks = torch.stack([data[i][3] for i in range(batch_size)])

    return (image_paths, mask_paths, images, masks)