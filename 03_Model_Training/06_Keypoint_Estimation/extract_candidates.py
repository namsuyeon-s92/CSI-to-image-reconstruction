import os
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from dataset import KeypointDataset
from tqdm import tqdm

current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent
project_root = current_folder.parent

csi_dir = os.path.join(project_root, '00_Datasets', 'data_20260220_2')
kp_dir = os.path.join(project_root, '00_Datasets', 'data_20260220_2_processed', 'openpose_outs')

window_size = 151
num_candidates = 100
output_path = os.path.join(current_folder, 'candidates', 'keypoint_candidates.npy')

def extract_candidates():
    print("Loading Dataset...")
    dataset = KeypointDataset(
        csi_base_dir=csi_dir,
        keypoint_base_dir=kp_dir,
        window_size=window_size
    )
    
    print("Filtering valid keypoints...")
    valid_kps = []
    seen_img_ids = set()
    # KeypointDataset returns (spectrogram, presence, keypoints, img_id)
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        _, presence, keypoints, img_id = dataset[i]
        img_id_val = img_id.item()
        if presence.item() > 0.5 and img_id_val not in seen_img_ids:
            valid_kps.append(keypoints.numpy())
            seen_img_ids.add(img_id_val)

    if not valid_kps:
        print("No valid candidates found!")
        return

    valid_kps = np.array(valid_kps)
    num_valid = len(valid_kps)
    print(f"Total valid keypoint samples: {num_valid}")
    
    n_candidates = min(num_candidates, num_valid)

    print(f"Extracting {n_candidates} distinct candidate keypoints using KMeans...")
    
    # We want to use only X, Y coordinates to compute similarities.
    # To account for prediction confidences (e.g. YOLO/OpenPose confidence),
    # we can weigh the coordinates by their confidences before clustering.
    # valid_kps is shape (N, 54). Reshape to (N, 18, 3)
    valid_kps_reshape = valid_kps.reshape(num_valid, 18, 3)
    
    # Extract x, y and confidence
    valid_kps_xy = valid_kps_reshape[:, :, :2] # (N, 18, 2)
    valid_kps_conf = valid_kps_reshape[:, :, 2:3] # (N, 18, 1)
    
    # Multiply x, y by confidence to give more weight to highly confident keypoints during distance calculation
    # Instead of direct multiplication which alters the space, we can use confidence as a feature
    # Or simply weight the distance calculation by flattening it: x*c, y*c
    # This ensures that keypoints with low confidence (likely 0,0 anyway or noisy) contribute less to distance
    valid_kps_weighted_xy = (valid_kps_xy * valid_kps_conf).reshape(num_valid, 36)
    
    kmeans = KMeans(n_clusters=n_candidates, random_state=42, n_init='auto')
    kmeans.fit(valid_kps_weighted_xy)
    centers = kmeans.cluster_centers_
    
    # Find the nearest ACTUAL observed keypoints to each cluster center
    closest_indices = pairwise_distances_argmin(centers, valid_kps_weighted_xy)
    
    # Select the actual 54D keypoints corresponding to those closest matches
    selected_candidates = valid_kps[closest_indices]
    
    # Save the selected candidates
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    np.save(output_path, selected_candidates)
    print(f"Saved {n_candidates} candidate keypoints to {output_path}")

if __name__ == '__main__':
    extract_candidates()

