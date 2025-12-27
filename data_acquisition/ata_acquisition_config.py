"""
Configuration file for Triplex Pump Fault Data Acquisition
"""

from pathlib import Path
import glob

# Base directory
INPUT_DATA_DIR = Path('./data_acquisition/data')

# Input data paths
HEALTHY_FILES = glob.glob(str(INPUT_DATA_DIR) + '/healthy/*.csv')
SEAL_LEAK_FILES = glob.glob(str(INPUT_DATA_DIR) + '/seal_leak/*.csv')
BLOCKED_INLET_FILES = glob.glob(str(INPUT_DATA_DIR) + '/blocked_inlet/*.csv')
BEARING_WEAR_FILES = glob.glob(str(INPUT_DATA_DIR) + '/bearing_wear/*.csv')
VALVE_LEAK_FILES = glob.glob(str(INPUT_DATA_DIR) + '/valve_leak/*.csv')
PLUNGER_WEAR_FILES = glob.glob(str(INPUT_DATA_DIR) + '/plunger_wear/*.csv')
COMBINED_FAULT_FILES = glob.glob(str(INPUT_DATA_DIR) + '/combined_fault/*.csv')

# Output pickle directory
OUTPUT_DATA_DIR = Path('./checkpoints')
OUTPUT_DATA_FILE = OUTPUT_DATA_DIR / Path('raw_pump_data.p')
