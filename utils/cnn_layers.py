import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1, padding='valid'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x