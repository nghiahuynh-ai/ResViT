import torch
import torch.nn as nn
from masking import RectangleMasking
from enc_dec import build_enc_dec
from bottleneck import build_bottleneck
from model.dataset import UNetReconstructDataset
from omegaconf import OmegaConf


class UNet(nn.Module):
    def __init__(self, cfg_filepath: str):
        super(UNet, self).__init__()
        
        cfg = OmegaConf.load(cfg_filepath)
        
        self.encoder, self.decoder = build_enc_dec(cfg.enc_dec)
        self.bottleneck = build_bottleneck(cfg.bottleneck)
        
        self.train_dataset = UNetReconstructDataset(cfg.train_dataset)
        self.validation_dataset = UNetReconstructDataset(cfg.validation_dataset)
        
        self.masking = RectangleMasking(cfg.rectangle_masking)
        
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)        
        return x
    
    def training_step(self, batch, batch_idx):
        x, x_masked = batch, batch.detach().clone()
        x_masked = self.masking(x_masked)
        x_reconstructed = self.forward(x_masked)
        loss = nn.functional.mse_loss(x_reconstructed, x)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, x_masked = batch, batch.detach().clone()
            x_masked = self.masking(x_masked)
            x_reconstructed = self.forward(x_masked)
            loss = nn.functional.mse_loss(x_reconstructed, x)
        
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    