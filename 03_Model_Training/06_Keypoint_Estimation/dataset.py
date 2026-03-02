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


class KeypointDataset(Dataset):
    def __init__(self, csi_base_dir, keypoint_base_dir, window_size=151):
        self.csi_base_dir = csi_base_dir
        self.keypoint_base_dir = keypoint_base_dir
        self.window_size = window_size

        self.csi_amplitudes = []
        self.labels_presence = []
        self.labels_keypoints = []
        self.paired_ids = []
        
        self.load_data()

    def load_data(self):
        # 1. Gather all label IDs available from Keypoint preprocessed directory
        json_paths = glob(os.path.join(self.keypoint_base_dir, '*_keypoints.json'))
        id_to_json = {}
        for path in json_paths:
            basename = os.path.basename(path)
            # format is [id]_keypoints.json
            try:
                img_id = int(basename.replace('_keypoints.json', ''))
                id_to_json[img_id] = path
            except ValueError:
                continue

        valid_label_ids = np.array(sorted(list(id_to_json.keys())))
        
        if len(valid_label_ids) == 0:
            print(f"Warning: No valid JSON labels found in {self.keypoint_base_dir}. Dataset will be empty.")
            return

        # 2. Find csi.csv files
        csi_csv_paths = glob(os.path.join(self.csi_base_dir, '**', 'csi.csv'), recursive=True)
        if not csi_csv_paths:
            print(f"No csi.csv found in {self.csi_base_dir}.")
            return
            
        for data_path in csi_csv_paths:
            data_dir = os.path.dirname(data_path)
            df = pd.read_csv(data_path, on_bad_lines='skip', low_memory=False)
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
            df['local_timestamp'] = pd.to_numeric(df['local_timestamp'], errors='coerce')
            df = df.dropna(subset=['id', 'local_timestamp'])
            df = df.sort_values(by='id')

            valid_csi = []
            valid_ids = []
            valid_timestamps = []
            
            for index, row in df.iterrows():
                try:
                    csi_array = json.loads(row['data'])
                    if len(csi_array) == 256:
                        valid_csi.append(csi_array)
                        valid_ids.append(row['id'])
                        valid_timestamps.append(row['local_timestamp'])
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            
            if not valid_csi:
                continue
                
            raw_csi = np.array(valid_csi, dtype=np.int32)
            real = raw_csi[:, [i * 2 for i in CSI_VALID_SUBCARRIER_INDEX]]
            imag = raw_csi[:, [i * 2 - 1 for i in CSI_VALID_SUBCARRIER_INDEX]]
            amplitude = np.sqrt(real**2 + imag**2).astype(np.float32)

            timestamps = np.array(valid_timestamps)
            df_ids = np.array(valid_ids)
            
            # Interpolation
            t_seconds = (timestamps - timestamps[0]) / 1_000_000.0
            t_target = np.arange(0, t_seconds[-1], 0.01)
            
            if len(t_target) < self.window_size:
                continue

            interpolator = interp1d(t_seconds, amplitude, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
            interpolated_amplitude = interpolator(t_target)

            id_interpolator = interp1d(t_seconds, df_ids, kind='nearest', bounds_error=False, fill_value='extrapolate')
            interpolated_ids = id_interpolator(t_target)

            num_samples = len(interpolated_amplitude) - self.window_size
            for i in range(num_samples):
                target_id = interpolated_ids[i + (self.window_size // 2)]
                
                # Match nearest valid label ID
                closest_idx = np.abs(valid_label_ids - target_id).argmin()
                best_label_id = valid_label_ids[closest_idx]
                
                # If target ID is reasonably close to an available label ID
                if abs(best_label_id - target_id) < 5: 
                    # Parse JSON
                    json_path = id_to_json[best_label_id]
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            
                        if data.get('people') and len(data['people']) > 0:
                            presence = 1.0
                            people_list = data['people']
                            
                            # Taking biggest human if multiple exist (same heuristic as pipeline.py)
                            biggest_box_idx = 0
                            max_scale = 0

                            for idx, person in enumerate(people_list):
                                kps = np.array(person['pose_keypoints_2d']).reshape(18, 3)
                                conf = kps[:, 2]
                                part = kps[:, :2][conf > 0, :]
                                if len(part) > 0:
                                    bbox = [min(part[:, 0]), min(part[:, 1]), max(part[:, 0]), max(part[:, 1])]
                                    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                                    if scale > max_scale:
                                        max_scale = scale
                                        biggest_box_idx = idx
                            
                            kps = np.array(people_list[biggest_box_idx]['pose_keypoints_2d'], dtype=np.float32)
                        else:
                            presence = 0.0
                            kps = np.zeros(54, dtype=np.float32)
                            
                        self.csi_amplitudes.append(interpolated_amplitude[i:i + self.window_size])
                        self.labels_presence.append([presence])
                        self.labels_keypoints.append(kps)
                        self.paired_ids.append(best_label_id)
                    except Exception as e:
                        print(f"Error parsing JSON {json_path}: {e}")
                        pass
        
        self.csi_amplitudes = np.array(self.csi_amplitudes, dtype=np.float32)
        self.labels_presence = np.array(self.labels_presence, dtype=np.float32)
        self.labels_keypoints = np.array(self.labels_keypoints, dtype=np.float32)
        self.paired_ids = np.array(self.paired_ids, dtype=np.int32)
        
        print(f"Loaded dataset: {len(self.csi_amplitudes)} samples.")

    def __len__(self):
        return len(self.csi_amplitudes)

    def __getitem__(self, index):
        spectrogram = torch.from_numpy(self.csi_amplitudes[index]).float()
        presence = torch.from_numpy(self.labels_presence[index]).float()
        keypoints = torch.from_numpy(self.labels_keypoints[index]).float()

        return spectrogram, presence, keypoints, self.paired_ids[index]
