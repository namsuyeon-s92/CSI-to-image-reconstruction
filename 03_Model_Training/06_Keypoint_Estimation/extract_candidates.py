import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from dataset import KeypointDataset
from tqdm import tqdm

def extract_candidates(args):
    print("Loading Dataset...")
    dataset = KeypointDataset(
        csi_base_dir=args.csi_dir,
        keypoint_base_dir=args.kp_dir,
        window_size=args.window_size
    )
    
    print("Filtering valid keypoints...")
    valid_kps = []
    # KeypointDataset returns (spectrogram, presence, keypoints, img_id)
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        _, presence, keypoints, _ = dataset[i]
        if presence.item() > 0.5:
            valid_kps.append(keypoints.numpy())
            
    if not valid_kps:
        print("No valid candidates found!")
        return

    valid_kps = np.array(valid_kps)
    num_valid = len(valid_kps)
    print(f"Total valid keypoint samples: {num_valid}")
    
    num_candidates = min(args.num_candidates, num_valid)

    print(f"Extracting {num_candidates} distinct candidate keypoints using KMeans...")
    
    # We want to use only X, Y coordinates to compute similarities to avoid
    # biases caused by prediction confidences (3rd element per keypoint).
    # valid_kps is shape (N, 54). Reshape to (N, 18, 3), extract x,y -> (N, 18, 2) -> (N, 36)
    valid_kps_xy = valid_kps.reshape(num_valid, 18, 3)[:, :, :2].reshape(num_valid, 36)
    
    kmeans = KMeans(n_clusters=num_candidates, random_state=42, n_init='auto')
    kmeans.fit(valid_kps_xy)
    centers = kmeans.cluster_centers_
    
    # Find the nearest ACTUAL observed keypoints to each cluster center
    closest_indices = pairwise_distances_argmin(centers, valid_kps_xy)
    
    # Select the actual 54D keypoints corresponding to those closest matches
    selected_candidates = valid_kps[closest_indices]
    
    # Save the selected candidates
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    np.save(args.output_path, selected_candidates)
    print(f"Saved {num_candidates} candidate keypoints to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csi_dir', type=str, default='../00_Datasets/data_20260220_2', help='Raw dataset directory containing csi.csv')
    parser.add_argument('--kp_dir', type=str, default='../00_Datasets/data_20260220_2_processed/openpose_outs', help='Preprocessed JSON directory')
    parser.add_argument('--window_size', type=int, default=100, help='CSI time window length')
    parser.add_argument('--num_candidates', type=int, default=1000, help='Number of unique keypoint candidates to extract')
    parser.add_argument('--output_path', type=str, default='candidates/keypoint_candidates.npy', help='Path to save candidate array')
    
    args = parser.parse_args()
    extract_candidates(args)
