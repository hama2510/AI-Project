import numpy as np

def extract_fft(df, nfft=2048, frame_len=4096, slide_window=128, norm=None):
    """Read data from DataFrame calculate FFT and add fft to DataFrame as attribute 'fft'.

    Parameters:
    ----------
    df: DataFrame
        DataFrame contains data
    nfft: int > 0, optional
        Number FFT. Default value is 2048
    frame_len: int > 0, optional
        Length of frame to calculate FTT. Default value is 4096
    slide_window: int > 0, optional
        Length of window for sliding. Default value is 128
    norm: string, optional
        Normalization keyword. 
            None: No normalization
            'max': Normalizing by divide for max value
            'standard': Zero mean and stardard deviation

    """
    df['fft'] = None
    for i,row in df.iterrows():
        start = 0
        acc = row['acc']
        N = len(acc)
        features = []
        while start+frame_len <= N:
            tmp = acc[start:start+frame_len,:]
            tmp = np.reshape(tmp, (tmp.shape[0]))
            fft = abs(np.fft.fft(tmp, n=nfft))
            fft = fft[:len(fft)//2]
            if not norm is None:
                if norm=='max':
                    fft = fft/np.max(fft)
                elif norm=='standard':
                    fft = scale(fft)
            fft = np.reshape(fft, (fft.shape[0], 1))
            features.append(fft)
            start+=slide_window
        features = np.asarray(features)
        df.at[i, 'fft'] = features