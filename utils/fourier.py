import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from scipy import signal
from scipy.fft import fft, fftfreq, ifft, fft2, ifft2
from scipy.signal import welch, butter, filtfilt


########################################
# Butterworth filter
########################################

def butterworth_filter(order, cutoff, fs, btype='lowpass', plot=True):
    """Creating the butterworth filter

    Args:
        order (int): The order of the filter.
        cutoff (float): Cutoff frequency in Hertz (Hz)
        fs (float): Sampling rate (samples/s or samples/m)
        btype (str, optional): Type of filter. Defaults to 'lowpass'.
        plot (bool, optional): Visualize the Butterworth filter. Defaults to True.

    Returns:
        b,a: Numerator (b) and denominator (a) polynomials of the IIR (infinite impulse response) filter. 
    """
    nyquist = fs/2
    normalized_cutoff = cutoff / nyquist
    b,a = signal.butter(N=order, Wn=normalized_cutoff, btype=btype, analog=False)
        
    if plot:
            
        w, h = signal.freqz(b,a, worN=1024)
        w = w*(fs/(2*np.pi))  # rad/amostra * (1/rad) = 1/amostra = Hz
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), tight_layout=True)

        plt.plot(w, abs(h))  
        plt.axhline(1/np.sqrt(2), color='red', linestyle='--')
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
            
        if btype == 'bandpass':
            plt.axvline(cutoff[0], color='green',linestyle='--')  
            plt.axvline(cutoff[1], color='green',linestyle='--') 
            plt.legend(['Butterworth filter - order {}'.format(order),
                            'Cutoff ampltitude - {}'.format(1/np.sqrt(2)), 'Lower cutoff frequency - {} Hz'.format(cutoff[0]),
                            'Upper cutoff frequency - {} Hz'.format(cutoff[1])])
        elif btype == 'lowpass':
            plt.axvline(cutoff, color='green', linestyle='--')
            plt.legend(['Butterworth filter - order {}'.format(order),'Cutoff ampltitude', 'Cutoff frequency - {} Hz'.format(cutoff)])
                
                
        plt.show()
            
    return b,a
    

def apply_butterfilter_df(df,order, cutoff, fs, btype='lowpass',subset=False):
    """Apply the butterworth filter on a dataset

    Args:
        df (pandas): Pandas dataset
        order (int): The order of the filter.
        cutoff (float): Cutoff frequency in Hertz (Hz)
        fs (float): Sampling rate (samples/s or samples/m)
        btype (str, optional): Type of filter. Defaults to 'lowpass'.
        subset (bool, optional): Specify a subset of the dataframe you wish to filter. Defaults to False.

    Returns:
        pandas: Filtered dataset
    """
    if subset:
        df = df[subset].T.to_numpy()
    
    b, a = butterworth_filter(order, cutoff, fs, btype=btype, plot=False)
    
    df_butter_filtered = []

    for idx2 in tqdm(range(len(df))):
        
        y = filtfilt(b, a, df[idx2])
        df_butter_filtered.append(y)
        
    return df_butter_filtered


def compare_filtered_signals(df, df_filtered, subset, legend, nrows, ncols, figsize):
    """Compare the original dataframe with the filtered one

    Args:
        df (pandas): Unfiltered dataframe of the dataset
        df_filtered (pandas): Filtered dataframe
        subset (list): List of columns in which to do the comparison
        legend (str): Legend label
        nrows (int): Number of rows
        ncols (int): Number of columns
        figsize (tuple): Figure size

    Returns:
        fig,ax:
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=figsize, tight_layout=True)
    ax = ax.reshape((nrows*ncols,))

    for idx2 in range(len(subset)):
        
        ax[idx2].plot(df[idx2])
        ax[idx2].plot(df_filtered[idx2], alpha=0.9)

        ax[idx2].legend([legend + ' denoised',legend + ' denoised + butterworth'])
        ax[idx2].set_xlabel('Index')
        ax[idx2].set_ylabel(str(subset[idx2]))
        
    return fig, ax






#############################
# Signal denoising
#############################



def unscaled_FFT(signal, dt):
    """Fourier transform without any normalization factors

    Args:
        signal (numpy array): signal 
        dt (float): sampling interval [seconds/sample] or [meters/sample]

    Returns:
        freq: frequency vector
        mag: magnitude of the FFT 
        fhat: FFT amplitude
        L: Index vector to represent the 1st half of the spectrum
        dc: DC component of the signal
        power: Power of the signal
    """
    N = len(signal)
    dc = np.mean(signal)                        # DC component
    signal = signal - np.mean(signal)           # Removing DC component 
    df = 1/(N*dt)

    L = np.arange(0, np.floor(N/2), dtype=int)  # 1st half of the spectrum
    
    # Frequency domain
    freq = df*np.arange(N)
    fhat = fft(signal, N)                       
    mag = abs(fhat)
    
    power = mag**2                             

    return freq, mag, fhat, L, dc, power


def noise_filter(signal, dt, threshold):
    """Filter a signal based on a noise threshold

    Args:
        signal (numpy array): Time series signal
        dt (float): Sampling interval (s/sample or m/sample)
        threshold (float): Threshold used to define the noise level. Values above this level are maintained while those below are pruned.

    Returns:
        clean_fhat: Denoised signal
        clean_power: Denoised power signal
        keep_idx: Index used to show which signal will be kept
        dc: DC component of the signal
    """

    freq, mag, fhat, L, dc, power = unscaled_FFT(signal.to_numpy(), dt)

    # Noise removal
    keep_idx = power > threshold
    clean_fhat = fhat * keep_idx
    clean_power = (mag*keep_idx)**2

    return clean_fhat, clean_power, keep_idx, dc


def noise_filter_df(df, dt, subset, threshold_list):
    """Filter an entire dataframe based on a list of thresholds

    Args:
        df (pandas): Pandas dataframe of the dataset
        dt (float): Sampling interval
        subset (list): List of columns that will be denoised
        threshold_list (list): List of thresholds to be applied for each individual signal

    Returns:
        df_denoised (pandas): Denoised dataframe
        keep_idx (numpy array): Index used to show which signal will be kept
    """

    df = df[subset]
    df_denoised = []

    for idx2, column in tqdm(enumerate(subset)):
        clean_fhat, clean_power, keep_idx, _ = noise_filter(signal=df[column], dt=dt, threshold=threshold_list[idx2])
        df_denoised.append(clean_power)

    return df_denoised, keep_idx

   
def plot_FFT(df, dt, nrows, ncols,  subset, legend, figsize=(20, 8), plot_filtered=False, threshold_list=[5, 10, 15]):
    """Plot the FFT of a dataframe and compare to its denoised version

    Args:
        df (pandas): Pandas dataframe of the dataset
        dt (float): Sampling interval
        nrows (int): Number of rows
        ncols (int): Number of columns
        subset (list): List of columns to apply the denoising threshold
        legend (list): Legend label
        figsize (tuple, optional): Figure size. Defaults to (20, 8).
        plot_filtered (bool, optional): Plot the filtered and unfiltered variables. Defaults to False.
        threshold_list (list, optional): List of thresholds to apply the filter. Defaults to [5, 10, 15].

    Returns:
        fig, ax: 
        df_FFT (numpy array): Denoised dataset in the frequency domain 
    """
    
    df = df[subset]
    df_FFT = []
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, tight_layout=True)
    ax = ax.reshape((nrows*ncols,))

    for idx2, column in enumerate(subset):

        signal = df[column].to_numpy()
        freq,mag,fhat,L,dc, power = unscaled_FFT(signal,dt)
        
        df_FFT.append(fhat)

        ax[idx2].plot(freq[L],power[L])
        ax[idx2].legend([legend])
        ax[idx2].set_xlabel('Frequency (Hz)')
        ax[idx2].set_ylabel(str(column))
        
        if plot_filtered:
        
            df_denoised, _ = noise_filter_df(df, dt, subset, threshold_list)
            ax[idx2].plot(freq[L], df_denoised[idx2][L])
            ax[idx2].axhline(y=threshold_list[idx2],color='r', linestyle='dashed')
        
            ax[idx2].legend([legend,'Denoised ' + legend,'Noise threshold'])
            ax[idx2].set_xlabel('Frequency (Hz)')
            ax[idx2].set_ylabel(str(column))
    
    return fig, ax, df_FFT


def ifft_df(df, df_FFT, dt, subset, threshold_list):
    """Apply the inverse Fourier transform

    Args:
        df (pandas): Pandas dataframe of the dataset
        df_FFT (numpy array): Dataset in the frequency domain
        dt (float): sampling interval
        subset (list): Dataframe columns
        threshold_list (list): List of threshold value to apply the filter

    Returns:
        [type]: [description]
    """
    
    df_filtered = []

    for idx2, column in tqdm(enumerate(subset)):

        clean_fhat, _, _, dc = noise_filter(signal=df[column], dt=dt, threshold=threshold_list[idx2])
        denoised_fhat = clean_fhat 
        
        df_filtered.append(dc + np.real(ifft(denoised_fhat)))

    return df_filtered


def compare_filtered_unfiltered(df, df_filtered, legend, subset, nrows, ncols, figsize):
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=figsize, tight_layout=True)
    ax = ax.reshape((nrows*ncols,))
    
    for idx2, column in enumerate(subset):
        ax[idx2].plot(df[column])
        ax[idx2].plot(df_filtered[idx2], alpha=0.9)
        
        ax[idx2].legend([legend,legend +' denoised'])
        ax[idx2].set_ylabel(str(column))
        ax[idx2].set_xlabel('Index')
        
    return fig,ax
    

#############################
# Welch's method
#############################

def Welch_PSD(signal, fs, window_size_frac=0.3, overlap_frac=0.8):
    """Welch's method for obtaining the Fourier spectrum

    Args:
        signal (pandas): time series signal
        fs (float): sampling rate in Hertz (Hz)
        window_size_frac (float, optional): Fraction of the dataset to use as a window for applying Welch's method. Defaults to 0.3.
        overlap_frac (float, optional): Permissible overlap between sucessive windows. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    #fs = sampling frequency - samples/meter

    segment_size = np.int32(window_size_frac*len(signal))
    fft_size = 2 ** (int(np.log2(segment_size)) + 1)

    overlap_size = overlap_frac*segment_size

    f, welch_coef = welch(x=signal,
                          fs=fs,
                          nperseg=segment_size,
                          noverlap=overlap_size,
                          nfft=fft_size,
                          return_onesided=True,
                          scaling='density',
                          detrend='constant',
                          window='hann',
                          average='mean')

    return f, welch_coef
