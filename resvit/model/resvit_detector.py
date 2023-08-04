import os
import torch
import torch.nn as nn
from resvit.module.masking import RectangleMasking, PixelMasking, NoiseMasking
from resvit.module.enc_dec import build_enc_dec
from resvit.module.bottleneck import build_bottleneck
from resvit.utils.dataset import UNetReconstructDataset
from omegaconf import DictConfig
import lightning.pytorch as pl


class ResViTDetector(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(ResViTDetector, self).__init__()
        
        self.encoder, self.decoder = build_enc_dec(cfg.enc_dec)
        self.bottleneck = build_bottleneck(cfg.bottleneck)

        if os.path.isfile(cfg.pretrain):
            pretrain = torch.load(cfg.pretrain, map_location=self.device)['state_dict']
            self.load_state_dict(pretrain)
            
        self.prob_producer = None
        self.thres_producer = None
        
        self.train_dataset = UNetReconstructDataset(cfg.train_dataset)
        self.validation_dataset = UNetReconstructDataset(cfg.validation_dataset)
        
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
        
    def forward(self, x):    
        return x
    
    def training_step(self, batch, batch_idx):
        x, sample_size = batch
        loss = None
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, sample_size = batch
        loss = None
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
    def post_process(self, x, size):
        for i in range(x.shape[0]):
            hi, wi = size[i, 0], size[i, 1]
            x[i, :, hi:, :] = 0.0
            x[i, :, :, wi:] = 0.0
        return x