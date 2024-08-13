from keras.models import load_model
from tools import preprocess_image, plot_image, resize_image

model = load_model('unet_model_3.0.keras')

path_image = 'validation/teste1.tif'

image, image_prediction = preprocess_image(path_image, (256, 256))
prediction = model.predict(image_prediction)
prediction_image = resize_image(prediction[0, :, :, 0], (1024,1024), False)

plot_image(image ,prediction_image)
