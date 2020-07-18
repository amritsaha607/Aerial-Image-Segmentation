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