from scipy.signal import butter, filtfilt
from neurokit2 import signal_smooth
import numpy as np
from pandas import DataFrame, Series

def eda_digital_filter(eda_signal, sampling_rate, cut_off):
    """Takes an eda signal, the sampling rate, and a desired cut off
    frequency returns the clean signal
    """
    # Handle missing data
    eda_signal_fill = DataFrame.ffill(Series(eda_signal))

    # Parameters
    order = 4
    frequency = cut_off
    frequency = (
        2 * np.array(frequency) / sampling_rate
    )  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = butter(N=order, Wn=frequency, btype="lowpass", analog=False, output="ba")
    filtered = filtfilt(b, a, eda_signal_fill)

    # Smoothing
    clean = signal_smooth(
        filtered, method="convolution", kernel="boxzen", size=int(0.75 * sampling_rate)
    )

    return clean