import torch
import itertools

def splitImage(image, use_path=False, wh=(512, 512), mode='image'):
    '''
        Splits a large image into smaller sized images
        e.g. 2048X2048 image => 16 X (512X512) images
        Args:
            image : tensor of image sized 3XwXh
            use_path : for using path or image
            wh : output dimension of splits
            mode : 'image' or 'mask'
    '''
    
    out_w, out_h = wh
    
    if mode=='image':
        _, w, h = image.shape
        image = torch.split(image, out_w, dim=1)
        image = list(itertools.chain.from_iterable([torch.split(img, out_h, dim=2) for img in image]))
    else:
        w, h = image.shape
        image = torch.split(image, out_w, dim=0)
        image = list(itertools.chain.from_iterable([torch.split(img, out_h, dim=1) for img in image]))
    return torch.stack(image)