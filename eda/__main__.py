from data_acquisition.utils import get_save_data
from data_processing.utils import _downSampler, _FFT, _dataScaler
from eda.utils import time_plot, fft_plot
import eda.config as config

def main():

    dataset = get_save_data()

    config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # TIME DOMAIN VISUAL
    # ------------------------
    healthy = dataset.healthy.copy()
    seal = dataset.seal_leak.copy()
    blocked = dataset.blocked_inlet.copy()

    # Plot a single raw waveform sample each
    time_plot(healthy[0], 0, 2500, 'healthy_time')
    time_plot(seal[0], 0, 2500, 'seal_leak_time')
    time_plot(blocked[0], 0, 2500, 'blocked_inlet_time')

    # ------------------------
    # RESAMPLED + FFT VISUAL
    # ------------------------
    healthy_resampled = _downSampler(healthy, 0, 100)
    seal_resampled = _downSampler(seal, 0, 100)
    blocked_resampled = _downSampler(blocked, 0, 100)

    healthy_fft = _FFT(healthy_resampled)
    seal_fft = _FFT(seal_resampled)
    blocked_fft = _FFT(blocked_resampled)

    fft_plot(healthy_fft[0], 'fft_healthy')
    fft_plot(seal_fft[0], 'fft_seal_leak')
    fft_plot(blocked_fft[0], 'fft_blocked_inlet')

    # ------------------------
    # SCALED VISUAL
    # ------------------------
    healthy_scaled = _dataScaler(healthy_resampled)
    time_plot(healthy_scaled[0], 0, 2500, 'healthy_scaled')

    print("EDA plots generated successfully!")


if __name__ == "__main__":
    main()
