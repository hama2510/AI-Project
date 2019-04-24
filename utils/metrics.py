import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.mean(abs(y_true-y_pred), axis=1), axis=1)

def mse(y_true, y_pred):
    return np.mean(np.power((y_true-y_pred), 2), axis=(1,2))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=(1,2)) 

def smape(y_true, y_pred):
    return np.mean(2.0*abs(y_true - y_pred)/(abs(y_true) + abs(y_pred)), axis=(1,2))