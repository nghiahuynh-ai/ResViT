import os
import torch
import torch.nn as nn
from resvit.module.enc_dec import build_enc_dec
from resvit.module.bottleneck import build_bottleneck
from resvit.utils.dataset import UNetReconstructDataset
from omegaconf import DictConfig
import lightning.pytorch as pl


class ResViTDetector(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(ResViTDetector, self).__init__()
        
        self.encoder, self.decoder = build_enc_dec(cfg.enc_dec, out_layer=False)
        self.bottleneck = build_bottleneck(cfg.bottleneck)

        if os.path.isfile(cfg.pretrain):
            pretrain = torch.load(cfg.pretrain, map_location=self.device)['state_dict']
            self.load_state_dict(pretrain, strict=False)
            
        self.prob_producer = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.enc_dec.in_channels,
                out_channels=cfg.enc_dec.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )
        
        self.thres_producer = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.enc_dec.in_channels,
                out_channels=cfg.enc_dec.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )
        
        self.loss = {
            'ls': nn.BCELoss(),
            'lb': None,
            'lt': None,
        }
        
        self.train_dataset = UNetReconstructDataset(cfg.train_dataset)
        self.validation_dataset = UNetReconstructDataset(cfg.validation_dataset)
        self.test_dataset = UNetReconstructDataset(cfg.test_dataset)
        
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
        prob = self.prob_producer(x)
        thres = self.thres_producer(x)
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