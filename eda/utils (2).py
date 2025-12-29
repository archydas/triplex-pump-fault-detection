"""
EDA Utility Functions
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq
import data_processing.data_processing_config as config
import eda.config as eda_config


def time_plot(yf: np.ndarray, start: int, stop: int, fname: str):
    """
    Plots a time-domain waveform
    """
    time = np.linspace(0, config.DURATION, len(yf), endpoint=False)

    _, axs = plt.subplots(nrows=1, figsize=(11, 9))
    plt.rcParams['font.size'] = '14'

    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(14)

    plt.plot(time[start:stop], yf[start:stop])
    axs.set_title('Time-series signal')
    axs.set_ylabel('Pressure / Sensor Value')
    axs.set_xlabel('Time (s)')

    file_location = eda_config.OUTPUT_DATA_DIR / Path(f'{fname}.png')
    plt.savefig(file_location)
    plt.close()


def fft_plot(yf: np.ndarray, fname: str):
    """
    Plots FFT safely (handles shapes and matches freq axis to spectrum length)
    """
    # --- Handle multi-dimensional FFT arrays ---
    yf = np.asarray(yf)

    # If (channels x freq) take first channel
    if yf.ndim == 2:
        if yf.shape[0] < yf.shape[1]:
            yf = yf[0]
        else:
            yf = yf[:, 0]

    # Ensure 1-D
    yf = yf.flatten()

    # --- Build matching frequency axis ---
    effective_fs = int(config.SAMPLE_RATE / config.RESAMPLE_RATE)
    freqs = np.linspace(0, effective_fs / 2, len(yf))

    # --- Plot ---
    plt.figure(figsize=(11, 9))
    plt.plot(freqs, yf)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

    file_location = eda_config.OUTPUT_DATA_DIR / Path(f"{fname}.png")
    plt.savefig(file_location)
    plt.close()


