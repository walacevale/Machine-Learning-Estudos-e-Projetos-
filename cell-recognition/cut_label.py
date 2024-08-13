import matplotlib.pyplot as plt

import cv2
import numpy as np
from tools import convert_to_grayscale, add_black_border


def cortar_imagens(image, path_save_image):
    image_original = image.copy()
    #image = convert_to_grayscale(image)
    thresh = threshold_li(image)
    image_threshd = (image > thresh).astype(np.uint8)*255
    # Rotulando regiões da imagem
    label_image = label(image_threshd)

    shift = 0
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
            image_cut = image_cut[minr-shift:maxr+shift, minc-shift:maxc+shift]

            # Salvando imagem cortada na pasta previamente selecionada
            save_path = path_save_image + f"{count}.tif"
            cv2.imwrite(save_path, add_black_border(image_cut))

