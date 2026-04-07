
import torch
import torch.nn as nn
import pytorch_lightning as L
import torch.nn.functional as F
from torch.distributions import Normal as Normal
from pytorch_msssim import ssim

z_dim = 256

#4_3 보다 l1가중치 내리고 ssim 가중치 올려서 세부 형체와 경계선 잡는 방향으로 조정해봅시다.

class VAE(L.LightningModule):
    def __init__(self, window_size, num_subcarriers):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.encoder = TokenEncoder(
            window_size=window_size,
            token_dim=num_subcarriers,
            d_model=256,
            z_dim=z_dim
            )
        self.decoder = Decoder()

        self.kl_start_epoch = 20   # 20에폭까지는 KL을 거의 안 씀
        self.kl_full_epoch = 80    # 80에폭에서 목표치(0.001) 도달
        self.target_kl_weight = 0.001

    def encode(self, x):
        mu, logvar = self.encoder.encode(x)
        logvar = torch.clamp(logvar, -10, 10)
        scale = torch.exp(0.5 * logvar)
        return Normal(mu, scale)

    def decode(self, qz_x):
        if self.training:
            z = qz_x.rsample()
        else:
            z = qz_x.loc

        x_hat = self.decoder.decode(z)
        return x_hat

    def forward(self, x):
        qz_x = self.encode(x)
        x_hat = self.decode(qz_x)
        return qz_x, x_hat
    
    def get_kl_weight(self):
        # 현재 에폭에 따른 KL 가중치 계산 (Linear Annealing)
        if self.current_epoch < self.kl_start_epoch:
            return 1e-8 # 0에 아주 가까운 값
        
        # 가중치를 서서히 증가
        progress = (self.current_epoch - self.kl_start_epoch) / (self.kl_full_epoch - self.kl_start_epoch)
        progress = min(1.0, max(0.0, progress))
        return progress * self.target_kl_weight
    
    def loss_function(self, x, qz_x, x_hat):
        # 1. 개별 로스 계산
        kl = self.calc_kl(qz_x)
        mse_loss = F.mse_loss(x_hat, x)  # MSE 추가: 전체적인 구도와 밝기 잡기
        l1_loss = F.l1_loss(x_hat, x)    # L1: 세부 형체와 경계선 잡기
        ssim_loss = 1 - ssim(x_hat, x, data_range=1.0, size_average=True) # 구조적 유사도
        
        
        # 2. KL Annealing 가중치 가져오기
        current_kl_w = self.get_kl_weight()

        # 3. 최종 가중치 합산 (기강 잡기 조합)
        
        total = (current_kl_w * kl) + (1.0 * mse_loss) + (5.0 * l1_loss) + (10.0 * ssim_loss)
    
        return {
            "loss": total, 
            "mse": mse_loss, # 로그 확인용 추가
            "kl": kl, 
            "l1": l1_loss, 
            "ssim": ssim_loss,
            "kl_weight": torch.tensor(current_kl_w)
        }

    def calc_kl(self, qz_x):
        variance = qz_x.scale**2
        kl = 0.5 * (variance + qz_x.loc**2 - 1 - torch.log(variance))
        return kl.mean(0).sum()

    def calc_mse(self, x, x_hat):
        diff = x - x_hat
        squared_diff = diff.pow(2)
        mse = squared_diff.mean(0).sum()
        return mse

    def __step(self, batch, batch_idx, stage):
        qz_x, x_hat = self.forward(batch[0])
        loss = self.loss_function(batch[1], qz_x, x_hat)
        for loss_n, loss_val in loss.items():
            self.log(f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=False)
        return loss["loss"]

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.__step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler,"monitor": "val_loss",},
        }


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        return x + self.block(x)
        
        
class Decoder(nn.Module):
    def __init__(self, z_dim=256):
        super(Decoder, self).__init__()

        
        # 6x8 -> 12x16 -> 24x32 -> 48x64 -> 96x128 -> 192x256 -> 384x512
        hidden_dims = [512, 256, 128, 64, 32, 16, 8] 
        
        # 시작 6x8 해상도, 512 채널
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 6 * 8)
        
        modules = []

        # 2. 6단계 업샘플링 (각 단계마다 채널은 절반으로, 해상도는 2배로)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.01),
                    ResBlock(hidden_dims[i + 1])
                )
            )

        # 3. 최종 출력층: 384x512 -> 480x640 강제 맞춤 및 RGB(3ch) 변환
        modules.append(
            nn.Sequential(
                nn.Upsample(size=(480, 640), mode='bilinear', align_corners=False),
                nn.Conv2d(hidden_dims[-1], 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def decode(self, z):
        # z: (Batch, z_dim)
        x = self.decoder_input(z)
        x = x.view(-1, 512, 6, 8)     # 초기 설정값 적용
        image = self.decoder(x)       # 최종 결과: (Batch, 3, 480, 640)
        return image
    

#TCN


class ResidualTCNBlock(nn.Module):
    def __init__(self, d_model, dilation, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout1d(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.norm2(self.conv2(out))
        out = self.dropout1(out)
        
        return self.relu(out + residual)

class TokenTCN(nn.Module):
    def __init__(self, d_model, layers=4,):
        super().__init__()
        blocks = []
        for i in range(layers):
            blocks.append(ResidualTCNBlock(d_model, dilation=2**i))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):

        x = x.permute(0, 2, 1) 
        x = self.network(x)
        x = x.permute(0, 2, 1)
        return x
        


#attention
class TokenAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=0.1
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
    
        attn_out, attn_weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out 

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x, attn_weights

class TokenEncoder(nn.Module):
    def __init__(self, window_size=151, token_dim=52, d_model=256, z_dim=256,n_layers=4):
        super().__init__()

        self.relu = nn.ReLU()
        self.token_embed = nn.Linear(token_dim, d_model)
        self.tcn = TokenTCN(d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        self.Transformer_layers = nn.ModuleList([
            TokenAttention(d_model, nhead=8) for _ in range(n_layers)
        ])


        self.fc = nn.Linear(d_model, 512)
        self.fc_dropout = nn.Dropout(0.1)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)

    def encode(self, x):
        # x: (B, 151, 52)
        x = self.token_embed(x) 
        x = x+ self.pos_encoding       # token projection
        x = self.tcn(x)          # positional
        
        for layer in self.Transformer_layers:
            x, _ = layer(x)

        x = x.mean(dim=1)
      
        x = self.relu(self.fc(x)) 
        x = self.fc_dropout(x)
        return self.mu(x), self.logvar(x)
    
