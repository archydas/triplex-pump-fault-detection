from pathlib import Path
import pandas as pd
import numpy as np
import logging
import pickle

import data_acquisition.data_acquisition_config as config
from data_acquisition.custom_types import PumpDataset

logging.basicConfig(level=logging.INFO)


def _data_reader(path_names: list) -> list:
    """
    Reads raw CSV signals into list of numpy arrays
    """
    sequences = []

    for name in path_names:
        data = pd.read_csv(name, header=None)
        sequences.append(data.values)

    return sequences


def get_save_data() -> PumpDataset:
    """
    Reads datasets for all pump fault categories and stores them
    as a PumpDataset namedtuple. Uses cached pickle if available.
    """

    if Path.exists(config.OUTPUT_DATA_FILE):
        logging.info("Loading previously saved raw pump dataset...")
        dataset = pickle.load(open(config.OUTPUT_DATA_FILE, 'rb'))

    else:
        config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logging.info("Loading raw pump fault data...")

        healthy = np.stack(_data_reader(config.HEALTHY_FILES))
        seal_leak = np.stack(_data_reader(config.SEAL_LEAK_FILES))
        blocked_inlet = np.stack(_data_reader(config.BLOCKED_INLET_FILES))
        bearing_wear = np.stack(_data_reader(config.BEARING_WEAR_FILES))
        valve_leak = np.stack(_data_reader(config.VALVE_LEAK_FILES))
        plunger_wear = np.stack(_data_reader(config.PRUNGER_WEAR_FILES))
        combined_fault = np.stack(_data_reader(config.COMBINED_FAULT_FILES))

        logging.info("Load complete.")

        dataset = PumpDataset(
            healthy,
            seal_leak,
            blocked_inlet,
            bearing_wear,
            valve_leak,
            plunger_wear,
            combined_fault
        )

        pickle.dump(dataset, open(config.OUTPUT_DATA_FILE, 'wb'))
        logging.info("Raw pump dataset saved to pickle.")

    return dataset
