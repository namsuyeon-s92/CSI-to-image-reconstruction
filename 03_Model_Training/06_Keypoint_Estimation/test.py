import os
from pathlib import Path
import json
import torch
import cv2
import numpy as np
import glob
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import KeypointEstimator
from dataset import KeypointDataset, NUM_SUBCARRIERS

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
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    current_file_path = Path(__file__).resolve()
    current_folder = current_file_path.parent
    project_root = current_folder.parent

    # Configs
    csi_dir = os.path.join(project_root, '00_Datasets', 'data_20260220_2')
    kp_dir = os.path.join(project_root, '00_Datasets', 'data_20260220_2_processed', 'openpose_outs')
    checkpoint_path = os.path.join(current_folder, 'outputs', 'csv_logs', 'version_0', 'checkpoints', 'best_model.ckpt') # Adjusted path
    # Note: checkpoint path might vary based on Lightning versioning. Adjust if necessary.
    
    save_dir = os.path.join(current_folder, 'results')
    candidates_path = os.path.join(current_folder, 'candidates', 'keypoint_candidates.npy')
    window_size = 100

    test_dataset = KeypointDataset(
        csi_base_dir=csi_dir,
        keypoint_base_dir=kp_dir,
        window_size=window_size
    )
    
    # Use split to get test set (replaying train.py split)
    _, test_idx = train_test_split(list(range(len(test_dataset))), test_size=0.1, shuffle=True, random_state=42)
    test_subset = Subset(test_dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    # Try to load model
    if not os.path.exists(checkpoint_path):
        # Find any ckpt in outputs if specific one doesn't exist
        ckpts = glob.glob(os.path.join(current_folder, 'outputs', '**', '*.ckpt'), recursive=True)
        if ckpts:
            checkpoint_path = ckpts[0]
            print(f"Using found checkpoint: {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Attempting to load last .pth if exists...")
            checkpoint_path = os.path.join(current_folder, 'checkpoints', 'best_keypoint_model.pth')

    if checkpoint_path.endswith('.ckpt'):
        model = KeypointEstimator.load_from_checkpoint(checkpoint_path)
    else:
        model = KeypointEstimator(window_size=window_size, num_subcarriers=NUM_SUBCARRIERS)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    out_json_dir = os.path.join(save_dir, 'openpose_outs')
    out_img_dir = os.path.join(save_dir, 'imgs')
    os.makedirs(out_json_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    
    candidates = np.load(candidates_path) if os.path.exists(candidates_path) else None
    
    image_paths_map = {}
    for img_p in glob.glob(os.path.join(csi_dir, '**', '*.png'), recursive=True):
        basename = os.path.basename(img_p)
        if basename.split('.')[0].isdigit():
            image_paths_map[int(basename.split('.')[0])] = img_p

    video_writer = None
    
    with torch.no_grad():
        for spectrogram, presence, keypoints_gt, img_id in tqdm(test_loader, desc="Testing"):
            spectrogram = spectrogram.to(device)
            img_id = img_id.item()
            
            presence_logit, kp_pred = model(spectrogram)
            
            presence_prob = torch.sigmoid(presence_logit).item()
            kp_out = kp_pred.squeeze().cpu().numpy()
            
            people_list = []
            if presence_prob > 0.5:
                if candidates is not None:
                    kp_out_xy = kp_out.reshape(18, 3)[:, :2].flatten()
                    candidates_xy = candidates.reshape(-1, 18, 3)[:, :, :2].reshape(-1, 36)
                    distances = np.linalg.norm(candidates_xy - kp_out_xy, axis=1)
                    kp_out = candidates[np.argmin(distances)].copy()
                
                for i in range(2, 54, 3):
                    kp_out[i] = max(kp_out[i], 0.8)
                people_list.append({"pose_keypoints_2d": kp_out.tolist()})
            
            json_dict = {"version": 1.3, "people": people_list}
            with open(os.path.join(out_json_dir, f"{img_id}_keypoints.json"), 'w') as f:
                json.dump(json_dict, f)
                
            if img_id in image_paths_map:
                orig_img = cv2.imread(image_paths_map[img_id])
                if orig_img is not None:
                    draw_img = draw_skeleton(orig_img, kp_out.reshape((18, 3))) if presence_prob > 0.5 else orig_img.copy()
                    cv2.imwrite(os.path.join(out_img_dir, f"{img_id}.jpg"), draw_img)
                    
                    if video_writer is None:
                        h, w = draw_img.shape[:2]
                        video_writer = cv2.VideoWriter(os.path.join(save_dir, 'skeleton_output.mp4'), 
                                                     cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))
                    video_writer.write(draw_img)
                    
    if video_writer is not None: video_writer.release()
    print(f"Testing finished! Outputs saved to {save_dir}")

if __name__ == '__main__':
    test()
