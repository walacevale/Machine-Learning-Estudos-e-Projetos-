from tools import *
from rede import *
from sklearn.model_selection import train_test_split


path_img  = 'train_end'
path_mask = 'mask_end'
image_size = (256,256)
images = []
masks = []

image, mask = load_images_and_masks(path_img, path_mask)

for image_path, mask_path in zip(image, mask):

    image_name =  get_image_name(image_path)
    mask_name  =  get_image_name(mask_path)

    if image_name == mask_name:
        
        img = cv2.imread(image_path)
        img_gray = convert_to_grayscale(img)
        img_resize = resize_image(img_gray, image_size , normalize=True)
        images.append(img_resize)

        mask = cv2.imread(mask_path)
        mask_gray = convert_to_grayscale(mask)
        mask_resize = resize_image(mask_gray, image_size, normalize= True)
        masks.append(np.expand_dims(mask_resize, axis=-1))
    else: 
        print(f"Image and mask names do not match: {image_name} != {mask_name}")

images, masks = np.array(images) , np.array(masks)


model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train,
                    epochs=15,
                    batch_size=8,
                    validation_data=(X_val, y_val))

model.save('unet_model_3.0.keras')