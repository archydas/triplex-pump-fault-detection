from collections import namedtuple

PumpDataset = namedtuple(
    'PumpDataset',
    'healthy seal_leak blocked_inlet bearing_wear valve_leak plunger_wear combined_fault'
)
