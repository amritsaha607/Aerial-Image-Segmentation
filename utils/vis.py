import cv2
import matplotlib.pyplot as plt

def showImageMask(img, mask, use_path=True, debug=False):

    if use_path:
        img_path, mask_path = img.split('/')[-1], mask.split('/')[-1]
        img = cv2.imread(img)
        mask = cv2.imread(mask)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(mask)
    ax[1].axis('off')
    if use_path:
        ax[0].set_title(img_path)
        ax[1].set_title(mask_path)

    if debug:
        plt.show()
    else:
        plt.close()
        return fig

def multiShowImageMask(imgs, masks, use_path=True, debug=False):

    if debug:
        figs = []

    for (img, mask) in zip(imgs, masks):
        if debug:
            showImageMask(img, mask, use_path=use_path, debug=debug)
        else:
            fig = showImageMask(img, mask, use_path=use_path, debug=debug)
            figs.append(fig)

    if not debug:
        return figs


def showPredictions(imgs, masks, masks_pred, use_path=False, ret='arr', debug=False, size=(8, 8)):
    '''
        Show predictions of model, gt image & gt mask
        imgs: images/paths
        masks: masks/paths
        masks_pred: predicted masks
        ret:    'arr' => array of figures
                'fig' => single figure with subplots of images
        debug :  set True to debug on notebook
    '''
    if use_path:
        imgs = [cv2.imread(img) for img in imgs]
        masks = [cv2.imread(mask) for mask in masks]
    else:
        imgs = [img.permute(1, 2, 0) for img in imgs]

    n = len(imgs)

    if not debug:
        if ret=='fig':
            fig, ax = plt.subplots(n, 3, figsize=size)
        elif ret=='arr':
            figs = []

    for i, (img, mask, mask_pred) in enumerate(zip(imgs, masks, masks_pred)):

        if debug:
            fig, ax = plt.subplots(1, 3, figsize=size)
            ax[0].imshow(img)
            ax[0].axis('off')
            ax[1].imshow(mask)
            ax[1].axis('off')
            ax[2].imshow(mask_pred)
            ax[2].axis('off')
            plt.show()

        elif ret=='fig':
            ax[i, 0].imshow(img)
            ax[i, 0].axis('off')
            ax[i, 1].imshow(mask)
            ax[i, 1].axis('off')
            ax[i, 2].imshow(mask_pred)
            ax[i, 2].axis('off')

        elif ret=='arr':
            fig, ax = plt.subplots(1, 3, figsize=size)
            ax[0].imshow(img)
            ax[0].axis('off')
            ax[1].imshow(mask)
            ax[1].axis('off')
            ax[2].imshow(mask_pred)
            ax[2].axis('off')
            figs.append(fig)

    if debug:
        return

    if ret=='arr':
        return figs
    elif ret=='fig':
        return fig