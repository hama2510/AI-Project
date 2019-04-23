from scipy.io import loadmat
import pandas as pd

def load_data(file_path):
    """Read data from path and return data array.

    Parameters:
    ----------
    file_path: string
        Path to data file

    Returns:
    data: array
        Array contains data

    """
    tmp = file_path.split('/')
    index = str(tmp[len(tmp)-1].split('.')[0])
    if len(index)<3:
        index = '0'+index
    
    mat = loadmat(file_path)
    
    data = mat['X%s_DE_time'%index]
    return data