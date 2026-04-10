import os
from pathlib import Path
import numpy as np
import torch
import cv2
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE 


num_workers = 2
torch.set_num_threads(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_folder = Path(__file__).resolve().parent
project_root = current_folder.parent
test_dir = os.path.join(project_root, 'data', '20260318_test_mesh')

experiment_name = '0318data/260409'
output_dir = os.path.join(current_folder, 'outputs', experiment_name)
output_video_file = os.path.join(output_dir, '7.mp4')

fps = 10
step = 10
size = (640, 480) # (가로, 세로)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
checkpoint_path = os.path.join(output_dir, 'epoch=7-val_loss=1.098134.ckpt')

window_size = 151
batch_size = 32

def test():
    dataset_test = WificamDataset(test_dir, window_size)
    
    #_, test_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=False)
    
    #dataset_test = Subset(dataset, test_idx)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    
    out = cv2.VideoWriter(output_video_file, fourcc, fps, size)
    if not out.isOpened():
        print("비디오 파일을 열 수 없습니다. 경로를 확인하세요.")
        return

    
    model = VAE.load_from_checkpoint(
        checkpoint_path,
        window_size=window_size,
        num_subcarriers=NUM_SUBCARRIERS,
        z_dim=256,     
        d_model=256,
        weights_only=True
    )

    model.to(device)
    model.eval()
    
    idx = 0
    for batch in tqdm(dataloader_test):
        spectrogram, image = batch
        spectrogram = spectrogram.to(device)

        with torch.no_grad():
            # encode와 decode를 거쳐 복원
            _, reconstruction = model(spectrogram)
            
        
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
        recon_np = reconstruction.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(len(recon_np)):
           
            data_content = (np.clip(image_np[i][..., ::-1], 0, 1) * 255).astype(np.uint8)
            pred_content = (np.clip(recon_np[i][..., ::-1], 0, 1) * 255).astype(np.uint8)
            pred_content = cv2.resize(pred_content, (data_content.shape[1], data_content.shape[0]))
            mask = np.any(pred_content > 10, axis=-1)
            data_content[mask] = pred_content[mask]

            idx += 1
            if idx % step == 0:
                
                final_frame = cv2.resize(data_content, size)
                out.write(final_frame)

    
    out.release()
    print(f"비디오 저장 완료: {output_video_file}")

if __name__ == '__main__':
    test()
