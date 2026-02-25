import torch
import torch.nn as nn
import pytorch_lightning as L

class KeypointEstimator(L.LightningModule):
    def __init__(self, window_size=100, num_subcarriers=56, num_keypoints=18, lr=1e-4):
        super(KeypointEstimator, self).__init__()
        self.save_hyperparameters()
        self.window_size = window_size
        self.num_subcarriers = num_subcarriers
        self.num_keypoints = num_keypoints
        self.lr = lr

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Flatten size: 128 * 4 * 4 = 2048
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.presence_head = nn.Linear(256, 1)
        # For each keypoint: x, y, confidence (3 values per keypoint)
        self.keypoints_head = nn.Linear(256, num_keypoints * 3)

        self.criterion_presence = nn.BCEWithLogitsLoss()
        self.criterion_kp = nn.MSELoss(reduction='none')

    def forward(self, x):
        # x is [B, W, F] -> [B, 1, W, F]
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        presence_logit = self.presence_head(x)
        keypoints = self.keypoints_head(x)
        
        return presence_logit, keypoints

    def __step(self, batch, batch_idx, stage):
        spectrogram, presence, keypoints_gt, _ = batch
        presence_logit, kp_pred = self.forward(spectrogram)
        
        loss_pres = self.criterion_presence(presence_logit, presence)
        
        # Keypoint loss only applies to samples where human is actually present
        mask = (presence == 1.0).squeeze(-1)
        if mask.sum() > 0:
            loss_kp = self.criterion_kp(kp_pred, keypoints_gt)
            loss_kp = loss_kp[mask].mean()
        else:
            loss_kp = torch.tensor(0.0).to(self.device)
            
        loss = loss_pres + 0.01 * loss_kp
        
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_pres_loss", loss_pres, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage}_kp_loss", loss_kp, on_epoch=True, prog_bar=False, logger=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
