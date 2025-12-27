"""
Configuration for Pump Fault Model Training
"""

from pathlib import Path

# General configuration
MONITOR = 'val_loss'
VAL_SPLIT = 0.2
LSTM_UNITS = 100
OPTIMIZER = 'adam'
EPOCHS = 80   # you can increase later

# Multi-class classification
OUTPUT_SIZE = 7   # Pump faults = 7 classes
ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'

# Output paths
OUTPUT_DATA_DIR = Path('./output')
