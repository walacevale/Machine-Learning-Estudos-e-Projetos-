import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2

image_dir = 'train_all'
mask_dir = 'mask_all'
img_size=(1024, 1024)
images = []
masks = []

image_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

for img_path, mask_path in zip(image_files, mask_files):
    image_name = os.path.basename(img_path).split('w1')[0]
    mask_name = os.path.basename(mask_path).split('_GT_')[0]
    
    
    #if image_name == mask_name:
    image = cv2.imread(mask_path)
    image_resize = cv2.resize(image, img_size, 
               interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('mask_end/' + mask_name + '.tif', image_resize)
        