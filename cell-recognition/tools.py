import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_li
from skimage.measure import label, regionprops



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
    plt.imshow(np.squeeze((prediction > 0.5).astype(np.uint8)), alpha=0.4)

    plt.show()


def add_black_border(image):
    # Adiciona uma borda de 1 pixel com intensidade 0 (preto) à imagem
    bordered_image = cv2.copyMakeBorder(
        image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return bordered_image

def cortar_objetos(image, prediction_image, path_save_image, image_name):
    image_original = image.copy()
    thresh = threshold_li(prediction_image)
    image_threshd = (prediction_image > thresh).astype(np.uint8)*255
    # Rotulando objetos da imagem
    label_image = label(image_threshd)

    count = 0
    for region in regionprops(label_image):
        if region.area >= 8000:
            count += 1

            # Defino os pontos do recorte
            minr, minc, maxr, maxc = region.bbox
            # Cria uma máscara binária para a região atual
            mask = np.zeros_like(image_threshd)
            mask[minr:maxr, minc:maxc] = label_image[minr:maxr, minc:maxc] == region.label  

            # Aplica a máscara à imagem original
            image_cut = image_original * mask
            
            # Recorta a imagem com a máscara aplicada
            image_cut = image_cut[minr:maxr, minc:maxc]

            # Salvando imagem cortada na pasta previamente selecionada
            save_path = path_save_image + image_name + '-' +f"{count}.tif"
            cv2.imwrite(save_path, add_black_border(image_cut))

