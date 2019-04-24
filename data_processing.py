from scipy.io import loadmat
import pandas as pd
from glob import glob

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

def load_data_from_folder(folder_path):
    """Read data from folder and return pandas DataFrame.

    Parameters:
    ----------
    folder_path: string
        Path to folder contains files without trailing '/'

    Returns:
    data: array
        Array contains data

    """
    if folder_path.endswith('/'):
        folder_path+='*'
    else:
        folder_path+='/*'
    files = glob(folder_path)
    files.sort()
    
    arr = []
    for f in files:
        data = load_data(f)
        
        tmp = f.split('/')
        index = str(tmp[len(tmp)-1].split('.')[0])
        
        lb = tmp[len(tmp)-2]
        if lb=='OK':
            label = 'normal'
            label_n = 0
        else:
            label = 'abnormal'
            label_n = 1
        sr = 48000
        arr.append([index, data, sr, label, label_n])
        
    df = pd.DataFrame(arr)
    df.columns = ['id', 'acc', 'sampling_rate', 'label','label_number']
    return df