import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1, padding='valid', activation='linear'):
    """Read data from path and return data array.

    Parameters:
    ----------
    input_tensor: keras layer
        keras layer for Conv1DTransope
    filters: int > 0
        filter number
    kernel_size: int > 0
        kernel size
    strides: int > 0
        stride. Default value is 1
    padding: string
        padding type. Can be either 'valid' or 'same'. Default value is 'valid'
    activation: string
        activation function. Can be one of keras activation function. Default value is 'linear'
    Returns:
    data: keras layer
        Tensor after applying Conv1DTranspose

    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x