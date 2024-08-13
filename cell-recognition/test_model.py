from keras.models import load_model
from tools import preprocess_image, resize_image, add_black_border, cortar_objetos, get_image_name

import glob
import os

model = load_model('unet_model_3.0.keras')
path = 'validation'
path_save = 'validation_save/'



load_path_image = glob.glob(os.path.join(path, "*.tif"))

for path_image in load_path_image:

    image_name = get_image_name(path_image)
    image, image_prediction = preprocess_image(path_image, (256, 256))
    prediction = model.predict(image_prediction)
    prediction_image = resize_image(prediction[0, :, :, 0], (1024,1024), False)
    cortar_objetos(image, prediction_image, path_save, image_name)