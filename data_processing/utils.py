"""
Utilities for Pump Fault Data Processing
"""

from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.fft import rfft
import logging
import pickle

from data_acquisition.custom_types import PumpDataset
from data_processing.custom_types import ProcessedPumpDataset
import data_processing.data_processing_config as config

logging.basicConfig(level=logging.INFO)


def _dataScaler(data):
    """
    Scale each channel using MinMaxScaler
    """
    data_temp = np.reshape(data, (-1, data.shape[2]))
    norm = MinMaxScaler().fit(data_temp)
    data_norm = norm.transform(data_temp)
    data_final = np.reshape(data_norm, (-1, data.shape[1], data.shape[2]))
    return data_final


def _downSampler(data, start_index, sample_rate):
    """
    Downsamples time-series using averaging windows
    """
    final_sequence = []

    for dataset in data:
        resampled = []
        start = start_index
        stop = sample_rate

        for _ in range(int(len(dataset) / sample_rate)):
            resampled.append(dataset[start:stop, :].mean(axis=0))
            start += sample_rate
            stop += sample_rate

        final_sequence.append(np.stack(resampled))

    return np.stack(final_sequence)


def _FFT(data):
    """
    Perform FFT with DC removal
    """
    data_fft = []

    for dataset in data:
        data_fft.append(np.stack(np.abs(rfft(dataset, axis=0))[1:, :]))

    return np.stack(data_fft)


def get_save_train_test_data(raw_data: PumpDataset) -> ProcessedPumpDataset:
    """
    Converts PumpDataset → scaled → FFT → labeled → train-test split
    """

    if Path.exists(config.OUTPUT_DATA_FILE):
        logging.info("Loading previously pickled processed pump data...")
        processed = pickle.load(open(config.OUTPUT_DATA_FILE, 'rb'))

    else:
        config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logging.info(f"Resampling data at rate: {config.RESAMPLE_RATE}")

        healthy = _downSampler(raw_data.healthy, 0, config.RESAMPLE_RATE)
        seal = _downSampler(raw_data.seal_leak, 0, config.RESAMPLE_RATE)
        blocked = _downSampler(raw_data.blocked_inlet, 0, config.RESAMPLE_RATE)
        bearing = _downSampler(raw_data.bearing_wear, 0, config.RESAMPLE_RATE)
        valve = _downSampler(raw_data.valve_leak, 0, config.RESAMPLE_RATE)
        plunger = _downSampler(raw_data.plunger_wear, 0, config.RESAMPLE_RATE)
        combined = _downSampler(raw_data.combined_fault, 0, config.RESAMPLE_RATE)

        logging.info("Scaling data...")
        healthy = _dataScaler(healthy)
        seal = _dataScaler(seal)
        blocked = _dataScaler(blocked)
        bearing = _dataScaler(bearing)
        valve = _dataScaler(valve)
        plunger = _dataScaler(plunger)
        combined = _dataScaler(combined)

        logging.info("Performing FFT...")
        healthy = _FFT(healthy)
        seal = _FFT(seal)
        blocked = _FFT(blocked)
        bearing = _FFT(bearing)
        valve = _FFT(valve)
        plunger = _FFT(plunger)
        combined = _FFT(combined)

        logging.info("Building labels...")

        y0 = np.zeros(len(healthy), dtype=int)
        y1 = np.full(len(seal), 1)
        y2 = np.full(len(blocked), 2)
        y3 = np.full(len(bearing), 3)
        y4 = np.full(len(valve), 4)
        y5 = np.full(len(plunger), 5)
        y6 = np.full(len(combined), 6)

        y = np.concatenate((y0, y1, y2, y3, y4, y5, y6))

        X = np.concatenate(
            (healthy, seal, blocked, bearing, valve, plunger, combined)
        )

        logging.info(f"Splitting test size = {config.DATA_TEST_SIZE}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.DATA_TEST_SIZE,
            random_state=42,
            stratify=y
        )

        processed = ProcessedPumpDataset(X_train, X_test, y_train, y_test)

        pickle.dump(processed, open(config.OUTPUT_DATA_FILE, 'wb'))

        logging.info("Processing complete. Ready for modelling!")

    return processed
