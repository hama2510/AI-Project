import keras.backend as K

def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def smape_loss(y_true, y_pred):
    return K.mean(2.0*abs(y_true - y_pred)/(abs(y_true) + abs(y_pred)), axis=-1)