import numpy as np
from scipy.interpolate import interp1d

class BasePreprocessor:
    def __init__(self, window_size, valid_subcarrier_index):
        self.window_size = window_size
        self.valid_subcarrier_index = valid_subcarrier_index
        self.csi_buffer = []

    def add_data(self, csi_data):
        self.csi_buffer.append(csi_data)

    def is_ready(self):
        return len(self.csi_buffer) >= self.window_size

    def process(self):
        """
        Base processing without interpolation.
        Returns processed spectrogram and consumed buffer.
        """
        pass

    def get_consumed_frames(self):
        pass


class TemporalMeshPreprocessor(BasePreprocessor):
    def process(self):
        if not self.is_ready():
            return None

        # Extract local_timestamp and data
        timestamps = np.array([data['local_timestamp'] for data in self.csi_buffer])
        t_seconds = (timestamps - timestamps[0]) / 1_000_000.0
        
        # Determine number of full 0.01s intervals (100Hz) available in the buffer
        max_t = t_seconds[-1]
        t_target = np.arange(0, max_t, 0.01)

        if len(t_target) < self.window_size:
            return None

        # Take strictly the required window size
        t_target = t_target[:self.window_size]

        raw_csi = np.array([data['data'] for data in self.csi_buffer], dtype=np.int32)
        real = raw_csi[:, [i * 2 for i in self.valid_subcarrier_index]]
        imag = raw_csi[:, [i * 2 - 1 for i in self.valid_subcarrier_index]]
        amplitude = np.sqrt(real**2 + imag**2).astype(np.float32)

        # Interpolate
        interpolator = interp1d(t_seconds, amplitude, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_amplitude = interpolator(t_target)

        # How many original frames are covered by the new t_target[-1]?
        # We find the index in t_seconds that is just past t_target[-1]
        consumed_idx = np.searchsorted(t_seconds, t_target[-1], side='right')
        
        # Prepare spectrogram (1, Channels, Height, Width) -> (1, 151, 52)
        spectrogram = interpolated_amplitude[np.newaxis, :, :]

        # Consumer will slice `self.csi_buffer[consumed_idx:]`
        self.consumed_idx = consumed_idx

        return spectrogram

    def consume_buffer(self):
        self.csi_buffer = self.csi_buffer[self.consumed_idx:]
