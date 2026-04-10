import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from pathlib import Path

import pytorch_lightning as L
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import Subset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE


num_workers = 2
torch.set_num_threads(4)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    accelerator = 'mps'
elif torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = 'gpu'
else:
    device = torch.device('cpu')
    accelerator = 'cpu'

current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent
project_root = current_folder.parent
data_dir = os.path.join(project_root, 'data', '20260318_train_mesh')

window_size = 151
batch_size = 32
epochs = 200
persistent_workers = True if num_workers > 0 else False


def train():
    dataset = WificamDataset(data_dir, window_size)

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=False)

    dataset_train = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, val_idx)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    output_dir = os.path.join(current_folder, 'outputs','0318data','260409')
    os.makedirs(output_dir, exist_ok=True)

    model = VAE(window_size=window_size, num_subcarriers=NUM_SUBCARRIERS)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",   
        patience=30,           
        mode="min",          
        verbose=True
    )

   
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss', 
            mode='min', 
            save_top_k=-1,
            save_last=True, 
            filename='{epoch}-{val_loss:.6f}',
            dirpath=output_dir,
            verbose=True
        )
        
    ]

    

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=1.0,
        logger=True,
        callbacks=callbacks,
        max_epochs=epochs
    )

    checkpoint_path = os.path.join(output_dir, 'last.ckpt')

    
    if os.path.exists(checkpoint_path):
        
        trainer.fit(model, dataloader_train, dataloader_val, ckpt_path=checkpoint_path)
    else:
        
        trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == '__main__':
    train()


