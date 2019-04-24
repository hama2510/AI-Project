import librosa
import scipy
from matplotlib import pyplot as plt
import numpy as np
from obspy.signal.filter import bandpass
import seaborn as sns

def visualize_stft(df, idx=None, hop_length=512, nfft=2048, rows=15, cols=5, ylim=None, bandpass=None, y_axis='linear'):
    """Draw stft of each data in DataFrame

    Parameters:
    ----------
    df: DataFrame
        DataFrame contains data
    idx: int, optional
        Channel of data to visualize in case data contains many channels. If data have only one channel let it be None
    nfft: int > 0, optional
        FFT window size and also data window size. Default value is 2048
    hop_length: int >0, optional
        Ovelaping window size. Default value is 512
    rows: int > 0, optional
        Maximum rows in the graph. Default value is 15
    cols: int > 0, optional
        Maximum column in each row. Default value is 5
    ylim: tuple, optional
        y-axis limit
    bandpass: tuple (low, high), optional
        bandpass filter for data. If it is None, data will not be filtered
    y_axis: string, optional
        y-axis scale. Default is linear

    """
    fig = plt.figure(figsize=(8*cols, 6*rows))
    plt.subplots_adjust(left=0.8, right=2.0, top=2.0, bottom=0.5)
    for i, row in df.iterrows():
        title = '%s (%s)'%(str(row['id']), row['label'])
        ax = fig.add_subplot(rows, cols, i+1, title=title)
        if idx is None:
            acc = row['acc'].reshape(-1)
        else:
            acc = row['acc'][:,[idx]].reshape(-1)
        if not bandpass is None:
            low, high = bandpass
            acc = bandpass(acc, low, high, row['sampling_rate'])
            
        stft = np.abs(librosa.stft(acc, n_fft=nfft, hop_length=hop_length))
        librosa.display.specshow(stft, sr=row['sampling_rate'], hop_length=hop_length, 
                                 x_axis='time', y_axis=y_axis, cmap = None, ax=ax)
        if not ylim is None:
            plt.ylim(ylim)
    plt.show()
    
def visualize_fft(df, idx=None, nfft=2048, bandpass=None, xlim=None, ylim=None, rows=15, cols=5):
    """Draw fft of each data in DataFrame

    Parameters:
    ----------
    df: DataFrame
        DataFrame contains data
    idx: int, optional
        Channel of data to visualize in case data contains many channels. If data have only one channel let it be None
    nfft: int > 0, optional
        FFT window size and also data window size. Default value is 2048
    rows: int > 0, optional
        Maximum rows in the graph. Default value is 15
    cols: int > 0, optional
        Maximum column in each row. Default value is 5
    xlim: tuple, optional
        x-axis limit
    ylim: tuple, optional
        y-axis limit
    bandpass: tuple (low, high), optional
        bandpass filter for data. If it is None, data will not be filtered
    y_axis: string, optional
        y-axis scale. Default is linear

    """
    fig = plt.figure(figsize=(8*cols, 6*rows))
    plt.subplots_adjust(left=0.8, right=2.0, top=2.0, bottom=0.5)
    for i, row in df.iterrows():
        acc = row['acc']
        N = len(acc)
           
        if idx is None:
            acc = row['acc'].reshape(-1)
        else:
            acc = row['acc'][:,[idx]].reshape(-1)
        if not bandpass is None:
            low, high = bandpass
            acc = bandpass(acc, low, high, row['sampling_rate'])
        
        T = 1.0/row['sampling_rate']
        fft = abs(scipy.fftpack.fft(acc, n=nfft))
        title = '%s (%s)'%(str(row['id']), row['label'])
        ax = fig.add_subplot(rows, cols, i+1, title=title)
        xf = np.linspace(0.0, 1.0/(2.0*T), nfft//2)
        if row['label'] == "normal":
            ax.plot(xf, 2.0/nfft *fft[:len(fft)//2], c='blue')
        else:
            ax.plot(xf, 2.0/nfft *fft[:len(fft)//2], c='red')
        if not xlim is None:
            plt.xlim(xlim)
        if not ylim is None:
            plt.ylim(ylim)
        plt.ylabel('Amplitude')
        plt.xlabel('Hz')
    plt.show()
    
def set_font_size(font_size):
    """Set font size for graph

    Parameters:
    ----------
    font_size: int > 0
        Font size

    """
    font = {'size'   : 30}
    plt.rc('font', **font)