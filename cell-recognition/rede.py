from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

def conv_block(inputs, filters):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    return conv

def encoder_block(inputs, filters):
    conv = conv_block(inputs, filters)
    pool = MaxPooling2D((2, 2))(conv)
    return conv, pool

def decoder_block(inputs, skip_features, filters):
    upsample = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    merge = concatenate([upsample, skip_features])
    conv = conv_block(merge, filters)
    return conv

def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    c5 = conv_block(p4, 1024)

    c6 = decoder_block(c5, c4, 512)
    c7 = decoder_block(c6, c3, 256)
    c8 = decoder_block(c7, c2, 128)
    c9 = decoder_block(c8, c1, 64)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
