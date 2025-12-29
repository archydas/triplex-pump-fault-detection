"""
Configuration for Pump Fault Model Training
"""

from pathlib import Path

# General configuration
MONITOR = 'val_loss'
VAL_SPLIT = 0.2

# IMPORTANT — reduce model capacity to avoid overfitting
LSTM_UNITS = 64          # was 100 → too large for dataset
OPTIMIZER = 'adam'

# Training stability
EPOCHS = 60              # 80 was unnecessary + caused overfitting

# Multi-class classification
OUTPUT_SIZE = 7          # 7 Pump fault classes
ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'

# Output paths
OUTPUT_DATA_DIR = Path('./output')
