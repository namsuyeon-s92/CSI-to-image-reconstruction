import os
import json
import glob
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import KeypointEstimator
from dataset import KeypointDataset, NUM_SUBCARRIERS


num_workers = 2
if torch.backends.mps.is_available():
    device = torch.device('mps')
    accelerator = 'mps'
elif torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = 'gpu'
else:
    device = torch.device('cpu')
    accelerator = 'cpu'

print(f"Using device: {device}")
print(f"Using accelerator: {accelerator}")

current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent
project_root = current_folder.parent
csi_dir = os.path.join(project_root, '00_Datasets', 'data_20260220_2')
kp_dir = os.path.join(project_root, '00_Datasets', 'data_20260220_2_processed', 'openpose_outs')
candidates_path = os.path.join(current_folder, 'candidates', 'keypoint_candidates.npy')

output_dir = os.path.join(current_folder, 'outputs')
save_dir = os.path.join(current_folder, 'results')
video_output_path = os.path.join(save_dir, 'skeleton_output.mp4')

# Model Configs
window_size = 151
batch_size = 1
persistent_workers = True if num_workers > 0 else False

# OpenPose COCO format 18 keypoints connection pairs
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), 
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), 
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]


def draw_skeleton(img, kps, threshold=0.1):
    res_img = img.copy()
    for pair in POSE_PAIRS:
        partA, partB = pair
        if kps[partA, 2] > threshold and kps[partB, 2] > threshold:
            pt1 = (int(kps[partA, 0]), int(kps[partA, 1]))
            pt2 = (int(kps[partB, 0]), int(kps[partB, 1]))
            cv2.line(res_img, pt1, pt2, (0, 255, 255), 2)
            cv2.circle(res_img, pt1, 4, (0, 0, 255), -1)
            cv2.circle(res_img, pt2, 4, (0, 0, 255), -1)
    return res_img


def test():
    # Setup Dataloader
    dataset = KeypointDataset(
        csi_base_dir=csi_dir,
        keypoint_base_dir=kp_dir,
        window_size=window_size
    )
    
    # Use split to get test set (replaying train.py split style)
    _, test_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=False)
    dataset_test = Subset(dataset, test_idx)
    dataloader_test = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    # Find and Load Model
    checkpoint_path = os.path.join(output_dir, 'epoch=58-val_loss=10.4195.ckpt')
    model = KeypointEstimator.load_from_checkpoint(checkpoint_path)
    
    model.to(device)
    model.eval()
    
    # Prepare Output Directories
    out_json_dir = os.path.join(save_dir, 'openpose_outs')
    out_img_dir = os.path.join(save_dir, 'imgs')
    os.makedirs(out_json_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    
    # Load Candidates for KMeans matching
    candidates = np.load(candidates_path) if os.path.exists(candidates_path) else None
    
    # Map image IDs to paths for visualization
    image_paths_map = {}
    for img_p in glob.glob(os.path.join(csi_dir, '**', '*.png'), recursive=True):
        basename = os.path.basename(img_p)
        if basename.split('.')[0].isdigit():
            image_paths_map[int(basename.split('.')[0])] = img_p

    video_writer = None
    step = 10
    idx = 0
    
    with torch.no_grad():
        for spectrogram, presence, keypoints_gt, img_id in tqdm(dataloader_test, desc="Testing"):
            idx += 1
            if idx % step != 0:
                continue

            spectrogram = spectrogram.to(device)
            img_id = img_id.item()
            
            presence_logit, kp_pred = model(spectrogram)
            
            presence_prob = torch.sigmoid(presence_logit).item()
            kp_out = kp_pred.squeeze().cpu().numpy()
            
            people_list = []
            if presence_prob > 0.5:
                # Shape is (54,) and we reshape to (18, 3)
                kp_out_reshaped = kp_out.reshape(18, 3)
                
                # Predicted keypoints (X, Y) and Confidence
                kp_out_xy = kp_out_reshaped[:, :2] # (18, 2)
                kp_out_conf = kp_out_reshaped[:, 2:3] # (18, 1)
                
                # Target candidates shape (Num_Candidates, 18, 3)
                candidates_reshaped = candidates.reshape(-1, 18, 3)
                candidates_xy = candidates_reshaped[:, :, :2] # (Num_Candidates, 18, 2)
                
                # Calculate weighted distance
                diff = candidates_xy - kp_out_xy # (Num_Candidates, 18, 2)
                squared_diff = diff ** 2
                dist_per_kp = np.sum(squared_diff, axis=2) # (Num_Candidates, 18)
                weighted_dist_per_cand = np.sum(dist_per_kp * kp_out_conf.T, axis=1) # (Num_Candidates,)
                
                best_cand_idx = np.argmin(weighted_dist_per_cand)
                kp_out = candidates[best_cand_idx].copy()

                people_list.append({"pose_keypoints_2d": kp_out.tolist()})
                
                # Boost confidence for visualization/JSON
                for i in range(2, 54, 3):
                    kp_out[i] = max(kp_out[i], 0.8)
            
            # Save JSON
            json_dict = {"version": 1.3, "people": people_list}
            with open(os.path.join(out_json_dir, f"{img_id}_keypoints.json"), 'w') as f:
                json.dump(json_dict, f)
                
            # Visualization
            if img_id in image_paths_map:
                orig_img = cv2.imread(image_paths_map[img_id])
                if orig_img is not None:
                    draw_img = draw_skeleton(orig_img, kp_out.reshape((18, 3))) if presence_prob > 0.5 else orig_img.copy()
                    cv2.imwrite(os.path.join(out_img_dir, f"{img_id}.jpg"), draw_img)
                    
                    if video_writer is None:
                        h, w = draw_img.shape[:2]
                        video_writer = cv2.VideoWriter(video_output_path, 
                                                     cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))
                    video_writer.write(draw_img)
                    
    if video_writer is not None: 
        video_writer.release()
    print(f"Testing finished! Outputs saved to {save_dir}")


if __name__ == '__main__':
    test()
