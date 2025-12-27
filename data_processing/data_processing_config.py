"""
Config for Triplex Pump Fault Data Processing
"""

from pathlib import Path

DATA_TEST_SIZE = 0.2     # test split
SAMPLE_RATE = 2000       # your simulation fs
RESAMPLE_RATE = 100      # downsample factor
DURATION = 1             # your signal duration in seconds

OUTPUT_DATA_DIR = Path('./checkpoints')
OUTPUT_DATA_FILE = OUTPUT_DATA_DIR / Path('train_test_pump_data.p')
