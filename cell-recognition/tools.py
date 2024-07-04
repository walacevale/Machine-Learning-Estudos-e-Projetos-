import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def load_images_and_masks(image_dir, mask_dir):
    images_path = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    mask_path = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    return images_path, mask_path

def get_image_name(image_path):
    return os.path.basename(image_path).split('.')[0]

def resize_image(image, image_size, normalize):
    imagem_resized = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)
    if normalize:
        imagem_resized= cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)/255.0
    return imagem_resized

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def preprocess_image(image_path, input_size=(256, 256)):
    image = cv2.imread(image_path)
    image_gray = convert_to_grayscale(image)
    image_resize = resize_image(image_gray, input_size, True)
    image_expanded = np.expand_dims(image_resize, axis=0)
    return image_gray, image_expanded

def plot_image(image, prediction):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Previsão")
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.imshow(np.squeeze((prediction > 0.65).astype(np.uint8)), alpha=0.4)

    plt.show()
    return
    









