import torch
import torch.nn as nn
from resvit.module.masking import NoiseMasking
from resvit.module.enc_dec import build_enc_dec
from resvit.module.bottleneck import build_bottleneck
from resvit.utils.dataset import ResViTSSLDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from omegaconf import DictConfig
import lightning.pytorch as pl


class ResViTSSL(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(ResViTSSL, self).__init__()
        
        self.encoder, self.decoder, self.out = build_enc_dec(cfg.enc_dec, out_layer=True)
        self.bottleneck = build_bottleneck(cfg.bottleneck)
        
        self.train_dataset = ResViTSSLDataset(cfg.train_dataset)
        self.validation_dataset = ResViTSSLDataset(cfg.validation_dataset)
        
        self.masking = NoiseMasking(cfg.masking)
        
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
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, sample_size = batch
        x_masked = x.detach().clone()
        x_masked = self.masking(x_masked)
        x_reconstructed = self.forward(x_masked)
        x_reconstructed = self.post_process(x_reconstructed, sample_size)
        loss = nn.functional.mse_loss(x_reconstructed, x)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, sample_size = batch
        x_masked = x.detach().clone()
        x_masked = self.masking(x_masked)
        x_reconstructed = self.forward(x_masked)
        x_reconstructed = self.post_process(x_reconstructed, sample_size)
        loss = nn.functional.mse_loss(x_reconstructed, x)
        
        ssim_score = self.ssim(x_reconstructed, x)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ssim_score", ssim_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return x_reconstructed
    
    def configure_optimizers(self):
        return self.optimizer
    
    def post_process(self, x, size):
        for i in range(x.shape[0]):
            hi, wi = size[i, 0], size[i, 1]
            x[i, :, hi:, :] = 0.0
            x[i, :, :, wi:] = 0.0
        return x