import random
from random import shuffle
import glob
import os

def writeAnnot(images, masks, outfile='../assets/sample.txt'):
    f = open(outfile, 'w')
    for (img, mask) in zip(images, masks):
        f.write('{}\t{}\n'.format(img, mask))
    f.close()

def makeAnnotations(splits=[0.75, 0.2, 0.05], 
    src='/content/data/', image_key='image.png', mask_key='labels.png', random_state=0):

    random.seed(random_state)
    all_paths = glob.glob(os.path.join(src, '**', '*.png'))
    image_paths = [img for img in all_paths if img.__contains__(image_key)]
    shuffle(image_paths)

    n = len(image_paths)
    train_start, train_end = 0, int(n*splits[0])
    val_start, val_end = train_end, train_end+int(n*splits[1])
    test_start = val_end

    train_image_paths = image_paths[train_start:train_end]
    val_image_paths = image_paths[val_start:val_end]
    test_image_paths = image_paths[test_start:]

    train_mask_paths = [img.replace(image_key, mask_key) for img in train_image_paths]
    val_mask_paths = [img.replace(image_key, mask_key) for img in val_image_paths]
    test_mask_paths = [img.replace(image_key, mask_key) for img in test_image_paths]

    writeAnnot(train_image_paths, train_mask_paths, outfile='../assets/train.txt')
    writeAnnot(val_image_paths, val_mask_paths, outfile='../assets/val.txt')
    writeAnnot(test_image_paths, test_mask_paths, outfile='../assets/test.txt')




def createSmall(f_in, f_out, max_=100):

    f_in = open(f_in, 'r').read().strip().split('\n')[:max_]
    to_write = '\n'.join(f_in)
    f_out = open(f_out, 'w')
    f_out.write(to_write)
    f_out.close()

def makeSmallAnnot():
    createSmall(f_in='../assets/train.txt', f_out='../assets/train_100.txt')
    createSmall(f_in='../assets/val.txt', f_out='../assets/val_100.txt')
    createSmall(f_in='../assets/test.txt', f_out='../assets/test_100.txt')
    
makeAnnotations(
    splits=[0.75, 0.2, 0.05], 
    src='/content/data/', 
    image_key='image.png', mask_key='labels.png', 
    random_state=0
)
# makeAnnotations()
# makeSmallAnnot()