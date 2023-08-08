import os
import torch
import torch.nn as nn
from resvit.module.enc_dec import build_enc_dec
from resvit.module.bottleneck import build_bottleneck
from resvit.utils.dataset import ResViTDetectorDataset
from resvit.module.loss import BalanceBCELoss, DiceLoss
from resvit.utils.scheduler import NoamScheduler
import torchmetrics.functional as metrics
import torchvision.transforms as T
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
        
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.enc_dec.in_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )
           
        # self.prob_producer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=cfg.enc_dec.in_channels,
        #         out_channels=cfg.enc_dec.in_channels,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ),
        #     nn.Sigmoid()
        # )
        
        # self.thres_producer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=cfg.enc_dec.in_channels,
        #         out_channels=cfg.enc_dec.in_channels,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ),
        #     nn.Sigmoid()
        # )
        
        self.loss = {
            'ls': nn.BCELoss(),
            'lb': DiceLoss(),
            'lt': nn.L1Loss(),
        }
        
        self.train_dataset = ResViTDetectorDataset(cfg.train_dataset)
        self.validation_dataset = ResViTDetectorDataset(cfg.validation_dataset)
        self.test_dataset = ResViTDetectorDataset(cfg.test_dataset)
        
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
        
        self.scheduler = NoamScheduler(
            optimizer=self.optimizer,
            factor=cfg.optim.factor,
            model_size=cfg.bottleneck.d_model,
            warmup_steps=cfg.optim.warmup_steps,
        )
        
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, 
        #     max_lr=cfg.optim.lr, 
        #     steps_per_epoch=cfg.optim.steps_per_epoch, 
        #     epochs=cfg.optim.epochs,
        #     anneal_strategy='linear'
        # )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, self.encoder.layers_outs)
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, gt = batch
        
        x_pred = self.forward(x)
        
        loss = self.loss['ls'](x_pred, gt)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, gt = batch

        x_pred = self.forward(x)
        loss = self.loss['ls'](x_pred, gt)
        x_pred = ((x_pred > 0.5) * 1.0).to(self.device)
        
        precision = metrics.precision(x_pred, gt, task='binary', num_classes=2)
        recall = metrics.recall(x_pred, gt, task='binary', num_classes=2)
        f1 = metrics.f1_score(x_pred, gt, task='binary', num_classes=2)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return x_pred
    
    def test_step(self, batch, batch_idx):
        x, gt = batch

        x_pred = self.forward(x)
        loss = self.loss['ls'](x_pred, gt)
        x_pred = ((x_pred > 0.5) * 1.0).to(gt.device)
        
        precision = metrics.precision(x_pred, gt, task='binary', num_classes=2)
        recall = metrics.recall(x_pred, gt, task='binary', num_classes=2)
        f1 = metrics.f1_score(x_pred, gt, task='binary', num_classes=2)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return x_pred
    
    def predict(self, image):
        pred = self.forward(image)
        pred = (pred > 0.5) * 1.0
        pred = pred.to(image.device)
        image = image * pred.unsqueeze(1)
        transform = T.ToPILImage()
        image = transform(image)
        image.show()
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }