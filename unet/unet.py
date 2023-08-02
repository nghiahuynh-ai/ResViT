import torch
import torch.nn as nn
from unet.masking import RectangleMasking
from unet.enc_dec import build_enc_dec
from unet.bottleneck import build_bottleneck
from unet.dataset import UNetReconstructDataset
from omegaconf import DictConfig
import lightning.pytorch as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure


class UNet(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(UNet, self).__init__()
        
        self.encoder, self.decoder = build_enc_dec(cfg.enc_dec)
        self.bottleneck = build_bottleneck(cfg.bottleneck)
        
        self.train_dataset = UNetReconstructDataset(cfg.train_dataset)
        self.validation_dataset = UNetReconstructDataset(cfg.validation_dataset)
        
        self.masking = RectangleMasking(cfg.rectangle_masking)
        
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, self.encoder.layers_outs)        
        return x
    
    def training_step(self, batch, batch_idx):
        x, sample_size = batch
        x_masked = x.detach().clone()
        x_masked = self.masking(x_masked)
        x_reconstructed = self.forward(x_masked)
        x_reconstructed = self.post_process(x_reconstructed, sample_size)
        loss = nn.functional.mse_loss(x_reconstructed, x)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, sample_size = batch
            x_masked = x.detach().clone()
            x_reconstructed = self.forward(x_masked)
            loss = nn.functional.mse_loss(x_reconstructed, x)
        
            ssim_score = self.ssim(x_reconstructed, x)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ssim_score", ssim_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
    def post_process(self, x, size):
        for i in range(x.shape[0]):
            hi, wi = size[i, 0], size[i, 1]
            x[i, :, hi:, wi:] = 0.0
            
        return x