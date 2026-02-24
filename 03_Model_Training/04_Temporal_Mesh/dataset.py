import json
import os
from glob import glob

import pandas as pd
import numpy as np
import torch
import cv2
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


CSI_VALID_SUBCARRIER_INDEX = [i for i in range(6, 32)] + [i for i in range(33, 59)]
NUM_SUBCARRIERS = len(CSI_VALID_SUBCARRIER_INDEX)


class WificamDataset(Dataset):
    def __init__(self, base_dir, window_size):
        self.base_dir = base_dir
        self.window_size = window_size

        self.csi_amplitudes = []
        self.image_paths = []
        self.load_data()

    def load_data(self):
        data_paths = glob(os.path.join(self.base_dir, '**', 'csi.csv'), recursive=True)
        for data_path in data_paths:
            data_dir = os.path.dirname(data_path)
            # Use on_bad_lines=skip to ignore rows that got randomly corrupted or cut off in mid-stream
            df = pd.read_csv(data_path, on_bad_lines='skip', low_memory=False)
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
            df['local_timestamp'] = pd.to_numeric(df['local_timestamp'], errors='coerce')
            df = df.dropna(subset=['id', 'local_timestamp'])
            df = df.sort_values(by='id')

            # Safely parse JSON arrays, skipping broken lines
            valid_csi = []
            valid_ids = []
            valid_timestamps = []
            
            for index, row in df.iterrows():
                try:
                    csi_array = json.loads(row['data'])
                    if len(csi_array) == 256: # Ensure correct shape just in case
                        valid_csi.append(csi_array)
                        valid_ids.append(row['id'])
                        valid_timestamps.append(row['local_timestamp'])
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Skip malformed lines such as truncated data arrays
                    continue
            
            if not valid_csi:
                print(f'Warning: No valid CSI data in {data_dir}. Skipping.')
                continue
                
            raw_csi = np.array(valid_csi, dtype=np.int32)
            real = raw_csi[:, [i * 2 for i in CSI_VALID_SUBCARRIER_INDEX]]
            imag = raw_csi[:, [i * 2 - 1 for i in CSI_VALID_SUBCARRIER_INDEX]]
            amplitude = np.sqrt(real**2 + imag**2).astype(np.float32)

            all_image_files = glob(os.path.join(data_dir, '*.png'))
            readable_image_ids = []
            for f in all_image_files:
                test_img = cv2.imread(f)
                if test_img is not None:
                    try:
                        img_id = int(os.path.basename(f).split('.')[0])
                        readable_image_ids.append(img_id)
                    except ValueError:
                        continue

            if not readable_image_ids:
                print(f'Warning: No valid images found in {data_dir}. Skipping this directory.')
                continue

            readable_image_ids = np.array(sorted(readable_image_ids))

            # Timestamp processing and 100Hz interpolation
            # Timestamp processing and 100Hz interpolation using valid parsed frames
            timestamps = np.array(valid_timestamps)
            df_ids = np.array(valid_ids)
            
            # Convert timestamps to numeric relative seconds (assuming microsecond or millisecond, subtract first)
            # local_timestamp from ESP32 Rx is in microseconds (esp_timer_get_time())
            t_seconds = (timestamps - timestamps[0]) / 1_000_000.0
            
            # We want to interpolate at exactly 100Hz (0.01s intervals)
            t_target = np.arange(0, t_seconds[-1], 0.01)
            
            if len(t_target) < self.window_size:
                print(f'Warning: Not enough data for interpolation in {data_dir}. Skipping.')
                continue

            # Interpolate amplitude
            interpolator = interp1d(t_seconds, amplitude, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
            interpolated_amplitude = interpolator(t_target)

            # Interpolate id to map back to images
            # (Use 'nearest' kind for ID since we want actual image IDs, not fractional)
            id_interpolator = interp1d(t_seconds, df_ids, kind='nearest', bounds_error=False, fill_value='extrapolate')
            interpolated_ids = id_interpolator(t_target)

            num_samples = len(interpolated_amplitude) - self.window_size
            for i in range(num_samples):
                # The ID corresponding to the center of the window
                target_id = interpolated_ids[i + (self.window_size // 2)]
                
                # Find the closest image
                best_img_id = readable_image_ids[np.abs(readable_image_ids - target_id).argmin()]

                self.csi_amplitudes.append(interpolated_amplitude[i:i + self.window_size])
                self.image_paths.append(os.path.join(data_dir, f'{best_img_id}.png'))
        
        self.csi_amplitudes = np.array(self.csi_amplitudes, dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        spectrogram = torch.from_numpy(self.csi_amplitudes[index]).float()

        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return spectrogram, image
