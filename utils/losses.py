import keras.backend as K

def rmse_loss(y_true, y_pred):
    """Calculate root mean square loss for training model

    Parameters:
    ----------
    y_true: keras layer
        Ground truth tensor
    y_pred: keras layer
        Prediction tensor
    Returns:
    rmse: tensor
        root mean square loss

    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def smape_loss(y_true, y_pred):
    """Calculate SMAPE loss for training model

    Parameters:
    ----------
    y_true: keras layer
        Ground truth tensor
    y_pred: keras layer
        Prediction tensor
    Returns:
    smape: tensor
        SMAPE loss

    """
    return K.mean(2.0*abs(y_true - y_pred)/(abs(y_true) + abs(y_pred)), axis=-1)