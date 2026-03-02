import torch
import torch.nn as nn
import pytorch_lightning as L

class KeypointEstimator(L.LightningModule):
    def __init__(self, window_size=151, num_subcarriers=52, num_keypoints=18, lr=1e-4):
        super(KeypointEstimator, self).__init__()
        self.save_hyperparameters()
        self.window_size = window_size
        self.num_subcarriers = num_subcarriers
        self.num_keypoints = num_keypoints
        self.lr = lr

        self.subcarrier_encoder = nn.Sequential(
            nn.Linear(num_subcarriers, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 8),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(8 * window_size, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
        )
        
        self.presence_head = nn.Linear(256, 1)
        # For each keypoint: x, y, confidence (3 values per keypoint)
        self.keypoints_head = nn.Linear(256, num_keypoints * 3)

        self.criterion_presence = nn.BCEWithLogitsLoss()
        self.criterion_kp = nn.MSELoss(reduction='none')

    def forward(self, x):
        # x shape: [B, W, F]
        x = self.subcarrier_encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.latent_encoder(x)
        
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
