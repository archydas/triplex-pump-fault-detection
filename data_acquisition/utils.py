from pathlib import Path
import pandas as pd
import numpy as np
import logging
import pickle

import data_acquisition.data_acquisition_config as config
from data_acquisition.custom_types import PumpDataset

logging.basicConfig(level=logging.INFO)


def _data_reader(path_names: list) -> list:

    sequences = []

    for name in path_names:
        df = pd.read_csv(name)

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='any')

        sequences.append(df.values.astype(float))

    return sequences


def get_save_data() -> PumpDataset:

    if Path.exists(config.OUTPUT_DATA_FILE):
        logging.info("Loading previously saved raw pump dataset...")
        return pickle.load(open(config.OUTPUT_DATA_FILE, 'rb'))

    config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading raw pump fault data...")

    def safe_stack(files, label):
        if len(files) == 0:
            raise RuntimeError(f"No CSV files found for: {label}")
        return np.stack(_data_reader(files))

    healthy = safe_stack(config.HEALTHY_FILES, "healthy")
    seal = safe_stack(config.SEAL_LEAK_FILES, "seal_leak")
    blocked = safe_stack(config.BLOCKED_INLET_FILES, "blocked_inlet")
    bearing = safe_stack(config.BEARING_WEAR_FILES, "bearing_wear")
    valve = safe_stack(config.VALVE_LEAK_FILES, "valve_leak")
    plunger = safe_stack(config.PLUNGER_WEAR_FILES, "plunger_wear")
    combined = safe_stack(config.COMBINED_FAULT_FILES, "combined_fault")

    dataset = PumpDataset(
        healthy,
        seal,
        blocked,
        bearing,
        valve,
        plunger,
        combined
    )

    pickle.dump(dataset, open(config.OUTPUT_DATA_FILE, 'wb'))
    logging.info("Raw pump dataset saved to pickle.")

    return dataset

