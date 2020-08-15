import cv2
import matplotlib.pyplot as plt
import matplotlib
from utils.parameters import COLORS, COLORS_OVERLAY

def overlayPlot(img, mask, use_path=True, debug=False, size=(8, 8), transparency=0.5):
    
    if use_path:
        img = cv2.imread(img)
        mask = cv2.imread(mask)

    COLORS_OVERLAY_RELATIVE = [tuple([x/255. for x in y if x!=y[-1]]+[y[-1]]) for y in COLORS_OVERLAY]
    fig = plt.figure(figsize=size)
    plt.imshow(img)
    plt.imshow(mask, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(COLORS_OVERLAY_RELATIVE), alpha=transparency)
    plt.axis('off')
    if debug:
        plt.show()
    else:
        plt.close()
        return fig
        
def showImageMask(img, mask, use_path=True, debug=False, overlay=False):

    '''
        Visualizes image and mask simultaneously
        Args:
            img : image_path / image array
            mask : mask_path / mask array
            use_path : set True for using path
            debug : set True for debugging in notebook
            overlay : set True to overlay mask on image 
                        (better call overlayPlot function to set transperancy parameter)
    '''

    if use_path and not overlay:
        img_path, mask_path = img.split('/')[-1], mask.split('/')[-1]
        img = cv2.imread(img)
        mask = cv2.imread(mask)

    if not overlay:
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
    
    else:
        return overlayPlot(img, mask, use_path=use_path, debug=debug)


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


def showPredictions(imgs, masks, masks_pred, use_path=False, ret='arr', debug=False, size='auto', getMatch=False):
    '''
        Show predictions of model, gt image & gt mask
        Args:
            imgs: images/paths
            masks: masks/paths
            masks_pred: predicted masks
            ret:    'arr' => array of figures
                    'fig' => single figure with subplots of images
            debug :  set True to debug on notebook
            getMatch : set True to get match_mask of prediction & gt
    '''
    if use_path:
        imgs = [cv2.imread(img) for img in imgs]
        masks = [cv2.imread(mask) for mask in masks]
    else:
        imgs = [img.permute(1, 2, 0) for img in imgs]

    n = len(imgs)
    n_figs = 4 if getMatch else 3

    if size=='auto':
        size = (4*n_figs, 4*n)

    if not debug:
        if ret=='fig':
            fig, ax = plt.subplots(n, n_figs, figsize=size)
        elif ret=='arr':
            figs = []

    for i, (img, mask, mask_pred) in enumerate(zip(imgs, masks, masks_pred)):

        if debug:
            fig, ax = plt.subplots(1, n_figs, figsize=size)
            ax[0].imshow(img)
            ax[0].set_title('Image')
            ax[0].axis('off')
            ax[1].imshow(mask, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(COLORS))
            ax[1].set_title('Mask')
            ax[1].axis('off')
            ax[2].imshow(mask_pred, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(COLORS))
            ax[2].set_title('Pred Mask')
            ax[2].axis('off')
            if getMatch:
                ax[3].imshow(mask==mask_pred, cmap=plt.cm.gray)
                ax[3].set_title('Match')
                ax[3].axis('off')
            plt.show()

        elif ret=='fig':
            ax[i, 0].imshow(img)
            ax[i, 0].set_title('Image')
            ax[i, 0].axis('off')
            ax[i, 1].imshow(mask, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(COLORS))
            ax[i, 1].set_title('Mask')
            ax[i, 1].axis('off')
            ax[i, 2].imshow(mask_pred, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(COLORS))
            ax[i, 2].set_title('Pred Mask')
            ax[i, 2].axis('off')
            if getMatch:
                ax[i, 3].imshow(mask==mask_pred, cmap=plt.cm.gray)
                ax[i, 3].set_title('Match')
                ax[i, 3].axis('off')
            plt.close()

        elif ret=='arr':
            fig, ax = plt.subplots(1, 3, figsize=size)
            ax[0].imshow(img)
            ax[0].set_title('Image')
            ax[0].axis('off')
            ax[1].imshow(mask, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(COLORS))
            ax[1].set_title('Mask')
            ax[1].axis('off')
            ax[2].imshow(mask_pred, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(COLORS))
            ax[2].set_title('Pred Mask')
            ax[2].axis('off')
            if getMatch:
                ax[3].imshow(mask==mask_pred, cmap=plt.cm.gray)
                ax[3].set_title('Match')
                ax[3].axis('off')
            plt.close()
            figs.append(fig)

    if debug:
        return

    if ret=='arr':
        return figs
    elif ret=='fig':
        return fig